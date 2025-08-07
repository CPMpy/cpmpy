#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## pumpkin.py
##
"""
    Interface to Pumpkin's API

    Pumpkin is a combinatorial optimisation solver developed by the ConSol Lab at TU Delft. 
    It is based on the (lazy clause generation) constraint programming paradigm.
    (see https://github.com/consol-lab/pumpkin)

    Always use :func:`cp.SolverLookup.get("pumpkin") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ===============
    Installation
    ===============

    The `pumpkin_solver_py` python package is currently not available on PyPI.
    It can be installed from source using the following steps:
     1. clone the repository from github: https://github.com/consol-lab/pumpkin
     2. install the "maturin" package to build the python bindings: :code:`pip install maturin`
     3. build and install the package: :code:`cd pumpkin/pumpkin-solver-py && maturin develop`

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_pumpkin

    ==============
    Module details
    ==============
"""
import warnings
import re
from typing import Optional
import pkg_resources

from os.path import join

import numpy as np

from cpmpy.exceptions import NotSupportedError
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar, boolvar
from ..expressions.utils import is_num, is_any_list, get_bounds
from ..transformations.get_variables import get_variables
from ..transformations.linearize import canonical_comparison
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.reification import reify_rewrite, only_bv_reifies, only_implies
from ..transformations.safening import no_partial_functions

import time


class CPM_pumpkin(SolverInterface):
    """
    Interface to Pumpkin's API

    Creates the following attributes (see parent constructor for more):

    - ``pum_solver``: the pumpkin.Model() object
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pumpkin_solver_py as psp
            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e


    @staticmethod
    def version() -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        try:
            # there is also a version of the solver itself in the Cargo.toml (/pumpkin-solver/Cargo.toml)
            # currently not accessible through the python api
            # dynamic = ["version"] in the pyproject.toml does not seem to get the right value?
            return pkg_resources.get_distribution('pumpkin-solver-py').version 
        except pkg_resources.DistributionNotFound:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: Model(), a CPMpy Model() (optional)
            subsolver: None, not used
        """
        if not self.supported():
            raise Exception("CPM_Pumpkin: Install the python package 'pumpkin_solver_py'")

        from pumpkin_solver_py import Model

        assert subsolver is None 

        # initialise the native solver object
        self.pum_solver = Model() 
        self.predicate_map = {} # cache predicates for reuse

        # for objective
        self._objective = None
        self.objective_is_min = True

        # initialise everything else and post the constraints/objective
        super().__init__(name="pumpkin", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.pum_solver


    def solve(self, time_limit=None, prove=False, proof_name="proof.drcp", proof_location=".", assumptions=None):
        """
        Call the Pumpkin solver

        Arguments:
            time_limit (float, optional):  maximum solve time in seconds 
            prove: whether to produce a DRCP proof (.lits file and .drcp proof file).
            proof_name: name for the the proof files.
            proof_location: location for the proof files (default to current working directory).
            assumptions: CPMpy Boolean variables (or their negation) that are assumed to be true.
                            For repeated solving, and/or for use with :func:`s.get_core() <cpmpy.solvers.pumpkin.CPM_pumpkin.get_core>`: if the model is UNSAT,
                            `get_core()` returns a small subset of assumption variables that are unsat together.
        """

        # Again, I don't know why this is necessary, but the PyO3 modules seem to be a bit wonky.
        from pumpkin_solver_py import BoolExpression as PumpkinBool, IntExpression as PumpkinInt
        from pumpkin_solver_py import SatisfactionResult, SatisfactionUnderAssumptionsResult
        from pumpkin_solver_py.optimisation import OptimisationResult, Direction

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # parse and dispatch the arguments
        if time_limit is not None and time_limit < 0:
            raise ValueError("Time limit cannot be negative, but got {time_limit}")
        kwargs = dict(timeout=time_limit)

        if self.has_objective():
            assert assumptions is None, "Optimization under assumptions is not supported"
            solve_func = self.pum_solver.optimise
            kwargs.update(proof=join(proof_location, proof_name) if prove else None,
                          objective=self.solver_var(self._objective),
                          direction=Direction.Minimise if self.objective_is_min else Direction.Maximise)

        elif assumptions is not None:
            assert not prove, "Proof-logging under assumptions is not supported"
            pum_assumptions = [self.to_predicate(a) for a in assumptions]
            self.assump_map = dict(zip(pum_assumptions, assumptions))
            solve_func = self.pum_solver.satisfy_under_assumptions
            kwargs.update(assumptions=pum_assumptions)

        else:
            solve_func = self.pum_solver.satisfy
            kwargs.update(proof=join(proof_location, proof_name) if prove else None)

        self._pum_core = None
        
        start_time = time.time() # when did solving start
        # call the solver, with parameters
        result = solve_func(**kwargs)

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = time.time() - start_time

        # translate solver exit status to CPMpy exit status
        if self.has_objective(): # check result after optimisation
            if isinstance(result, OptimisationResult.Optimal):
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            elif isinstance(result, OptimisationResult.Satisfiable):
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            elif  isinstance(result, OptimisationResult.Unsatisfiable):
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            elif isinstance(result, OptimisationResult.Unknown):
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            else:
                raise ValueError(f"Unknown Pumpkin-result: {result} of type {type(result)}, please report on github...")

        elif assumptions is not None: # check result under assumptions
            if isinstance(result, SatisfactionUnderAssumptionsResult.Satisfiable):
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            elif isinstance(result, SatisfactionUnderAssumptionsResult.Unsatisfiable):
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
                self._pum_core = [] # empty core, no required assumptions to prove UNSAT
            elif isinstance(result, SatisfactionUnderAssumptionsResult.UnsatisfiableUnderAssumptions):
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
                self._pum_core = result._0
            elif isinstance(result, SatisfactionUnderAssumptionsResult.Unknown):
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            else:
                raise ValueError(f"Unknown Pumpkin-result: {result} of type {type(result)}, please report on github...")

        else: # satisfaction result without assumptions
            if isinstance(result, SatisfactionResult.Satisfiable):
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            elif isinstance(result, SatisfactionResult.Unsatisfiable):
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            elif isinstance(result, SatisfactionResult.Unknown):
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            else:
                raise ValueError(f"Unknown Pumpkin-result: {result} of type {type(result)}, please report on github...")

        # translate solution (of user vars only)
        has_sol = self._solve_return(self.cpm_status)
        if has_sol:
            solution = result._0

            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                if isinstance(sol_var, PumpkinInt):
                    cpm_var._value = solution.int_value(sol_var)
                elif isinstance(sol_var, PumpkinBool):
                    cpm_var._value = solution.bool_value(sol_var)
                else:
                    raise NotSupportedError("Only boolean and integer variables are supported.")

            # translate solution values
            if self.has_objective():
                self.objective_value_ = solution.int_value(self.solver_var(self._objective))

        else: # wipe results
            for cpm_var in self.user_vars:
                cpm_var._value = None

        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """

        if is_num(cpm_var):
            return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return self.solver_var(cpm_var._bv).negate()

        # create if it does not exist
        if isinstance(cpm_var, _NumVarImpl):
            if cpm_var not in self._varmap:
                if isinstance(cpm_var, _BoolVarImpl):
                    revar = self.pum_solver.new_boolean_variable(name=str(cpm_var))
                elif isinstance(cpm_var, _IntVarImpl):
                    revar = self.pum_solver.new_integer_variable(cpm_var.lb, cpm_var.ub, name=str(cpm_var))
                else:
                    raise NotImplementedError("Not a known var {}".format(cpm_var))
                self._varmap[cpm_var] = revar

            # return from cache
            return self._varmap[cpm_var]
        
        # can also be a scaled variable (multiplication view)
        elif isinstance(cpm_var, Operator) and cpm_var.name == "mul":
            const, cpm_var = cpm_var.args
            if not is_num(const):
                raise ValueError(f"Cannot create view from non-constant multiplier {const} * {cpm_var}")
            return self.solver_var(cpm_var).scaled(const)
        
        raise ValueError(f"Not a known var {cpm_var}")



    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            Arguments:
                expr: Expression, the CPMpy expression that represents the objective function
                minimize: Bool, whether it is a minimization problem (True) or maximization problem (False)

            'objective()' can be called multiple times, only the last one is stored

            .. note::
                technical side note: any constraints created during conversion of the objective
                are premanently posted to the solver
        """
        # make objective function non-nested
        obj_var = intvar(*get_bounds(expr))
        self += expr == obj_var

        # make objective function or variable and post
        self._objective = obj_var
        self.objective_is_min = minimize

    def has_objective(self):
        return self._objective is not None

    # `__add__()` first calls `transform()`
    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the :ref:`Adding a new solver` docs on readthedocs for more information.

            :param cpm_expr: CPMpy expression, or list thereof
            :type cpm_expr: Expression or list of Expression

            :return: list of Expression
        """
        # apply transformations
        cpm_cons = toplevel_list(cpm_expr)
        supported = {"alldifferent", "cumulative", "table", "negative_table", "InDomain"
                     "min", "max", "element", "abs"}
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"element"}) # safen toplevel elements, assume total decomposition for partial functions
        cpm_cons = decompose_in_tree(cpm_cons, supported=supported, csemap=self._csemap)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
        supported_halfreif = {"or", "sum", "wsum", "sub", "mul", "div", "abs", "min", "max"}
        cpm_cons = reify_rewrite(cpm_cons, supported=supported_halfreif, csemap=self._csemap) # reified element not supported yet
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]),csemap=self._csemap)  # supports >, <, !=
        cpm_cons = canonical_comparison(cpm_cons) # ensure rhs is always a constant
        return cpm_cons

    def to_predicate(self, cpm_expr):
        """
            Convert a CPMpy expression to a Pumpkin predicate (comparison with constant)
        """
        from pumpkin_solver_py import Comparator, Predicate

        if isinstance(cpm_expr, _BoolVarImpl):
            if isinstance(cpm_expr, NegBoolView):
                lhs, comp, rhs = cpm_expr._bv, Comparator.LessThanOrEqual, 0
            else:
                lhs, comp, rhs = cpm_expr, Comparator.GreaterThanOrEqual, 1

        elif isinstance(cpm_expr, Comparison):

            lhs, rhs = cpm_expr.args
            assert is_num(rhs), "rhs of comparison must be a constant to be a predicate"

            if isinstance(lhs, Operator): # can be sum with single arg
                if lhs.name == "sum" and len(lhs.args) == 1:
                    lhs = lhs.args[0]
                elif lhs.name == "wsum" and len(lhs.args[0]) == 1:
                    lhs = lhs.args[0][0] * lhs.args[1][0]
                elif lhs.name == "mul" and is_num(lhs.args[0]):
                    lhs = lhs.args[0] * lhs.args[1]
                else:
                    raise ValueError(f"Lhs of predicate should be a sum with 1 argument, wsum with 1 arg, or mul with const, but got {lhs}")

            if cpm_expr.name == "==": comp = Comparator.Equal
            if cpm_expr.name == "<=": comp = Comparator.LessThanOrEqual
            if cpm_expr.name == ">=": comp = Comparator.GreaterThanOrEqual
            if cpm_expr.name == "!=": comp = Comparator.NotEqual
            if cpm_expr.name == "<":  comp, rhs = Comparator.LessThanOrEqual, rhs - 1
            if cpm_expr.name == ">":  comp, rhs = Comparator.GreaterThanOrEqual, rhs + 1
        else:
            raise ValueError(f"Cannot convert CPMpy expression {cpm_expr} to a predicate")
        
        # do we already have this predicate? 
        #  (actually, cse might already catch these...)
        if (lhs, comp, rhs) not in self.predicate_map:
            pred = Predicate(self.to_pum_ivar(lhs), comp, rhs)
            self.predicate_map[(lhs, comp, rhs)] = pred
        
        return self.predicate_map[(lhs, comp, rhs)]



    def to_pum_ivar(self, cpm_var):
        """
            Helper function to convert (boolean) variables and constants to Pumpkin integer expressions
        """
        if is_any_list(cpm_var):
            return [self.to_pum_ivar(v) for v in cpm_var]
        elif isinstance(cpm_var, _BoolVarImpl):
            return self.pum_solver.boolean_as_integer(self.solver_var(cpm_var))
        elif is_num(cpm_var):
            return self.solver_var(intvar(cpm_var, cpm_var))
        else:
            return self.solver_var(cpm_var)


    def _sum_args(self, expr, negate=False):
        """
            Helper function to convert CPMpy sum-like operators into pumpkin-compatible arguments.
            expr is expected to be a `sum`, `wsum` or `sub` operator.

            :return: Returns a list of Pumpkin integer expressions
        """
        args = []
        if isinstance(expr, Operator) and expr.name == "sum":
            for cpm_var in expr.args:
                pum_var = self.solver_var(cpm_var)
                if cpm_var.is_bool(): # have convert to integer
                    pum_var = self.pum_solver.boolean_as_integer(pum_var)
                args.append(pum_var.scaled(-1 if negate else 1))
        elif isinstance(expr, Operator) and expr.name == "wsum":
            for w, cpm_var in zip(*expr.args):
                if w == 0: continue # exclude
                pum_var = self.solver_var(cpm_var)
                if cpm_var.is_bool(): # have convert to integer
                    pum_var = self.pum_solver.boolean_as_integer(pum_var)
                args.append(pum_var.scaled(-w if negate else w))
        elif isinstance(expr, Operator) and expr.name == "sub":
            x, y = self.solver_vars(expr.args)
            if expr.args[0].is_bool():
                x = self.pum_solver.boolean_as_integer(x)
            if expr.args[1].is_bool():
                y = self.pum_solver.boolean_as_integer(y)
            args = [x.scaled(-1 if negate else 1), y.scaled(1 if negate else -1)]
        else:
            raise ValueError(f"Unknown expression to convert in sum-arguments: {expr}")
        return args
    
    def _is_predicate(self, cpm_expr):
        """
            Check if a CPMpy expression can be converted to a Pumpkin predicate
        """
        if isinstance(cpm_expr, _NumVarImpl):
            return True
        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == "sum" and len(cpm_expr.args) == 1:
                return True
            if cpm_expr.name == "wsum" and len(cpm_expr.args[0]) == 1:
                return True
            if cpm_expr.name == "mul" and is_num(cpm_expr.args[0]):
                return True
        return False

    def _get_constraint(self, cpm_expr):
        """
            Convert a CPMpy expression into a Pumpkin constraint
            Expects a transformed CPMpy expression, this logic is implemented as a separate function so we can support reification in `add()`
        """
        from pumpkin_solver_py import constraints

        if isinstance(cpm_expr, _BoolVarImpl):
            # base case, just var or ~var, post as clause
            return [constraints.Clause([self.solver_var(cpm_expr)])]
            
        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == "or": 
                return [constraints.Clause(self.solver_vars(cpm_expr.args))]

            raise NotImplementedError("Pumpkin: operator not (yet) supported", cpm_expr)

        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            assert isinstance(lhs, Expression), f"Expected a CPMpy expression on lhs but got {lhs} of type {type(lhs)}"

            if self._is_predicate(lhs):
                pred = self.to_predicate(cpm_expr)
                return [constraints.Clause([self.pum_solver.predicate_as_boolean(pred)])]

            if cpm_expr.name == "==":
                
                if "sum" in lhs.name or lhs.name == "sub":
                    return [constraints.Equals(self._sum_args(lhs), rhs)]
               
                pum_rhs = self.to_pum_ivar(rhs) # other operators require IntExpression
                if lhs.name == "div":
                    return [constraints.Division(*self.to_pum_ivar(lhs.args), pum_rhs)]
                elif lhs.name == "mul":
                    return [constraints.Times(*self.to_pum_ivar(lhs.args), pum_rhs)]
                elif lhs.name == "abs":
                    return [constraints.Absolute(self.to_pum_ivar(lhs.args[0]), pum_rhs)]
                elif lhs.name == "min":
                    return [constraints.Minimum(self.to_pum_ivar(lhs.args), pum_rhs)]
                elif lhs.name == "max":
                    return [constraints.Maximum(self.to_pum_ivar(lhs.args), pum_rhs)]
                elif lhs.name == "element":
                    arr, idx = lhs.args
                    return [constraints.Element(self.to_pum_ivar(idx), self.to_pum_ivar(arr), pum_rhs)]
                else:
                    raise NotImplementedError("Unknown lhs of comparison", cpm_expr)

            elif cpm_expr.name == "<=":
                return [constraints.LessThanOrEquals(self._sum_args(lhs), rhs)]
            elif cpm_expr.name == "<":
                return [constraints.LessThanOrEquals(self._sum_args(lhs), rhs-1)]
            elif cpm_expr.name == ">=":
                return [constraints.LessThanOrEquals(self._sum_args(lhs, negate=True), -rhs)]
            elif cpm_expr.name == ">":
                return [constraints.LessThanOrEquals(self._sum_args(lhs, negate=True), -rhs-1)]
            elif cpm_expr.name == "!=":
                return [constraints.NotEquals(self._sum_args(lhs), rhs)]

            raise ValueError("Unknown comparison", cpm_expr)

        elif isinstance(cpm_expr, GlobalConstraint):
            if cpm_expr.name == "alldifferent":
                return [constraints.AllDifferent(self.solver_vars(cpm_expr.args))]
            
            elif cpm_expr.name == "cumulative":
                start, dur, end, demand, cap = cpm_expr.args
                assert all(is_num(d) for d in dur), "Pumpkin only accepts Cumulative with fixed durations"
                assert all(is_num(d) for d in demand), "Pumpkin only accepts Cumulative with fixed demand"
                assert is_num(cap), "Pumpkin only accepts Cumulative with fixed capacity"

                return [constraints.Cumulative(self.solver_vars(start),dur, demand, cap)] + \
                        [self._get_constraint(c)[0] for c in self.transform([s + d == e for s,d,e in zip(start, dur, end)])]
            
            elif cpm_expr.name == "table":
                arr, table = cpm_expr.args
                return [constraints.Table(self.to_pum_ivar(arr), 
                                          np.array(table).tolist())] # ensure Python list
            
            elif cpm_expr.name == "negative_table":
                arr, table = cpm_expr.args
                return [constraints.NegativeTable(self.to_pum_ivar(arr), 
                                                  np.array(table).tolist())] # ensure Python list
            
            elif cpm_expr.name == "InDomain":
                val, domain = cpm_expr.args
                return [constraints.Table([self.to_pum_ivar(val)], 
                                          np.array(domain).tolist())] # ensure Python list
            
            
            else:
                raise NotImplementedError(f"Unknown global constraint {cpm_expr}")
                

        elif isinstance(cpm_expr, BoolVal): # unlikely base case
            a = boolvar() # dummy variable
            if cpm_expr.value() is True:
                return self._get_constraint(Operator("sum", [a]) >= 1)
            else:
                return self._get_constraint(Operator("sum", [a]) <= -1)

        else:
            raise ValueError("Unexpected constraint:", cpm_expr)


    def add(self, cpm_expr_orig):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

            :param cpm_expr: CPMpy expression, or list thereof
            :type cpm_expr: Expression or list of Expression

            :return: self
        """

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        for cpm_expr in self.transform(cpm_expr_orig):
            if isinstance(cpm_expr, Operator) and cpm_expr.name == "->": # found implication
                bv, subexpr = cpm_expr.args
                for pum_cons in self._get_constraint(subexpr):
                    self.pum_solver.add_implication(pum_cons, self.solver_var(bv))
            else:
                for pum_cons in self._get_constraint(cpm_expr):
                    self.pum_solver.add_constraint(pum_cons)

        return self
    __add__ = add # avoid redirect in superclass


    def get_core(self):
        """
           For use with :func:`s.solve(assumptions=[...]) <cpmpy.solvers.pumpkin.CPM_pumpkin.solve>`. Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

           CPMpy will return only those variables that are False (in the UNSAT core)

            .. note::

                Note that there is no guarantee that the core is minimal, though this interface does open up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!
        """

        assert self._pum_core is not None, "Can only get core if the last solve-call was unsatisfiable under assumptions"
        return [self.assump_map[pred] for pred in self._pum_core]

