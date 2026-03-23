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

    Requires that the 'pumpkin-solver' python package is installed:

    .. code-block:: console

        $ pip install pumpkin-solver

    The rest of this documentation is for advanced users

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
from typing import Optional, List
from os.path import join

import numpy as np
from packaging.version import Version

from cpmpy.exceptions import NotSupportedError
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.globalconstraints import Cumulative, GlobalConstraint
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar, boolvar
from ..expressions.utils import is_num, is_any_list, get_bounds
from ..transformations.get_variables import get_variables
from ..transformations.linearize import canonical_comparison
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree, decompose_objective
from ..transformations.flatten_model import flatten_constraint, get_or_make_var
from ..transformations.comparison import only_numexpr_equality
from ..transformations.reification import reify_rewrite, only_bv_reifies, only_implies
from ..transformations.safening import no_partial_functions, safen_objective

import time


class CPM_pumpkin(SolverInterface):
    """
    Interface to Pumpkin's API

    Creates the following attributes (see parent constructor for more):

    - ``pum_solver``: the pumpkin.Model() object
    """

    supported_global_constraints = frozenset({"alldifferent", "cumulative", "no_overlap", "table", "negative_table", "InDomain",
                                              "min", "max", "abs", "mul", "div", "element"})
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pumpkin_solver as psp
            pum_version = CPM_pumpkin.version()
            if Version(pum_version) < Version("0.3.0"):
                warnings.warn(f"CPMpy uses features only available from Pumpkin version >=0.3.0 "
                              f"but you have version {pum_version}")
                return False
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
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version('pumpkin-solver')
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None, proof=None, seed=None):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: Model(), a CPMpy Model() (optional)
            subsolver: None, not used
            proof (str, optional): path to the proof file
            seed (int, optional): random seed for the solver
        """
        if not self.supported():
            raise ModuleNotFoundError("CPM_pumpkin: Install the python package 'cpmpy[pumpkin]' to use this solver interface.")

        from pumpkin_solver import Model

        assert subsolver is None 

        # initialise the native solver object
        init_kwargs = dict()
        if proof is not None:
            init_kwargs['proof'] = proof
        if seed is not None:
            init_kwargs['seed'] = seed
        self._proof = proof
        self.pum_solver = Model(**init_kwargs)
        self.predicate_map = {} # cache predicates for reuse
        if proof is not None: # Table and friends are not supported when proof logging
            # see https://github.com/ConSol-Lab/Pumpkin/issues/354
            self.disabled_global_constraints = {"table", "negative_table", "InDomain"}
        else:
            self.disabled_global_constraints = set()

        # for objective
        self._objective = None
        self.objective_is_min = True

        # for solution hint
        self._solhint = None

        # initialise everything else and post the constraints/objective
        super().__init__(name="pumpkin", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.pum_solver

    def _unsat_at_rootlevel(self):
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        self.cpm_status.runtime = 0 # TODO: use post-time instead?

        for var in self.user_vars:
            var._value = None

        return self._solve_return(self.cpm_status)


    def solve(self, time_limit:Optional[float]=None, prove=False, assumptions:Optional[List[_BoolVarImpl]]=None, **kwargs):
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
        from pumpkin_solver import BoolExpression as PumpkinBool, IntExpression as PumpkinInt
        from pumpkin_solver import SatisfactionResult, SatisfactionUnderAssumptionsResult
        from pumpkin_solver.optimisation import OptimisationResult, Direction

        if "proof" in kwargs or "prove" in kwargs or "prove_location" in kwargs or "proof_name" in kwargs:
            raise ValueError("Proof-file should be supplied in the constructor, not as a keyword argument to solve."
                             "`cpmpy.SolverLookup.get('pumpkin', model, proof='path/to/proof.drcp')`")

        if self.pum_solver.is_inconsistent():
            return self._unsat_at_rootlevel()

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # parse and dispatch the arguments
        if time_limit is not None and time_limit < 0:
            raise ValueError("Time limit cannot be negative, but got {time_limit}")
        
        kwargs.update(timeout=time_limit)

        if self.has_objective():
            assert assumptions is None, "Optimization under assumptions is not supported"
            solve_func = self.pum_solver.optimise
            kwargs.update(objective=self.solver_var(self._objective),
                          direction=Direction.Minimise if self.objective_is_min else Direction.Maximise)
            if self._solhint is not None:
                kwargs.update(warm_start=self._solhint)

        elif assumptions is not None:
            assert self._proof is None, "Proof-logging under assumptions is not supported"
            pum_assumptions = [self.to_predicate(a) for a in assumptions]
            self.assump_map = dict(zip(pum_assumptions, assumptions))
            solve_func = self.pum_solver.satisfy_under_assumptions
            kwargs.update(assumptions=pum_assumptions)

        else:
            solve_func = self.pum_solver.satisfy

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

        # save user variables
        get_variables(expr, self.user_vars)

        # transform objective
        obj, safe_cons = safen_objective(expr)
        obj, decomp_cons = decompose_objective(obj,
                                               supported=self.supported_global_constraints,
                                               supported_reified=self.supported_reified_global_constraints,
                                               csemap=self._csemap)
        obj_var, obj_cons = get_or_make_var(obj) # do not pass csemap here, we will still transform obj_var == obj...
        if expr.is_bool():
            ivar = intvar(0,1)
            obj_cons += [ivar == obj_var]
            obj_var = ivar

        self.add(safe_cons + decomp_cons + obj_cons)

        # save objective function
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

        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"element", "div", "mod"}) # safen toplevel elements, assume total decomposition for partial functions
        cpm_cons = decompose_in_tree(cpm_cons,
                                     supported=self.supported_global_constraints - self.disabled_global_constraints,
                                     supported_reified=self.supported_reified_global_constraints - self.disabled_global_constraints,
                                     csemap=self._csemap)
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
        from pumpkin_solver import Comparator, Predicate

        if isinstance(cpm_expr, _BoolVarImpl):
            if isinstance(cpm_expr, NegBoolView):
                lhs, comp, rhs = cpm_expr._bv, Comparator.LessThanOrEqual, 0
            else:
                lhs, comp, rhs = cpm_expr, Comparator.GreaterThanOrEqual, 1

        elif isinstance(cpm_expr, Comparison):

            lhs, rhs = cpm_expr.args
            assert is_num(rhs), "rhs of comparison must be a constant to be a predicate"

            if isinstance(lhs, Operator):  # can be sum with single arg
                if lhs.name == "sum" and len(lhs.args) == 1:
                    lhs = lhs.args[0]
                elif lhs.name == "wsum" and len(lhs.args[0]) == 1:
                    lhs = lhs.args[0][0] * lhs.args[1][0]
                else:
                    raise ValueError(f"Lhs of predicate should be a sum with 1 argument or wsum with 1 arg, but got {lhs}")
            elif lhs.name == 'mul':
                if lhs.is_lhs_num:
                    lhs = lhs.args[0] * lhs.args[1]
                else:
                    raise ValueError(f"Lhs of predicate should be a mul with const, but got {lhs}")

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
            return self.solver_var(cpm_var).as_integer()
        elif is_num(cpm_var):
            return self.solver_var(intvar(cpm_var, cpm_var))
        # can also be a scaled variable (Multiplication with constant first)
        elif cpm_var.name == "mul" and cpm_var.is_lhs_num:
            const, cpm_var = cpm_var.args[0], cpm_var.args[1]
            if not is_num(const):
                raise ValueError(f"Cannot create view from non-constant multiplier {const} * {cpm_var}")
            return self.to_pum_ivar(cpm_var).scaled(const)
        else:
            return self.solver_var(cpm_var)


    def _sum_args(self, expr, negate=False, tag=None):
        """
            Helper function to convert CPMpy sum-like operators into pumpkin-compatible arguments.
            expr is expected to be a `sum`, `wsum` or `sub` operator.

            :return: Returns a list of Pumpkin integer expressions
        """
        if tag is None: raise ValueError("Expected tag to be provided but got None")
        if isinstance(expr, Operator) and expr.name == "sum":
            pum_vars = self.to_pum_ivar(expr.args)
            args = [pv.scaled(-1) if negate else pv for pv in pum_vars]
        elif isinstance(expr, Operator) and expr.name == "wsum":
            pum_vars = self.to_pum_ivar(expr.args[1])
            args = [pv.scaled(-w if negate else w) for w,pv in zip(expr.args[0], pum_vars) if w != 0]
        elif isinstance(expr, Operator) and expr.name == "sub":
            x, y = self.to_pum_ivar(expr.args)
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
        elif cpm_expr.name == 'mul' and cpm_expr.is_lhs_num:
            return True
        return False

    def _get_constraint(self, cpm_expr, tag=None):
        """
            Convert a CPMpy expression into a Pumpkin constraint
            Expects a transformed CPMpy expression, this logic is implemented as a separate function so we can support reification in `add()`
        """
        from pumpkin_solver import constraints
        if tag is None:
            tag = self.pum_solver.new_constraint_tag()
        if isinstance(cpm_expr, _BoolVarImpl):
            # base case, just var or ~var, post as clause
            return [constraints.Clause([self.solver_var(cpm_expr)], constraint_tag=tag)]
            
        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == "or": 
                return [constraints.Clause(self.solver_vars(cpm_expr.args), constraint_tag=tag)]

            raise NotImplementedError("Pumpkin: operator not (yet) supported", cpm_expr)

        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            assert isinstance(lhs, Expression), f"Expected a CPMpy expression on lhs but got {lhs} of type {type(lhs)}"

            if self._is_predicate(lhs):
                pred = self.to_predicate(cpm_expr)
                return [constraints.Clause([self.pum_solver.predicate_as_boolean(pred, tag=tag)], constraint_tag=tag)]

            if cpm_expr.name == "==":
                
                if "sum" in lhs.name or lhs.name == "sub":
                    return [constraints.Equals(self._sum_args(lhs, tag=tag), rhs, constraint_tag=tag)]
               
                pum_rhs = self.to_pum_ivar(rhs) # other operators require IntExpression
                if lhs.name == "div":
                    return [constraints.Division(*self.to_pum_ivar(lhs.args), pum_rhs, constraint_tag=tag)]
                elif lhs.name == "mul":
                    return [constraints.Times(*self.to_pum_ivar(lhs.args), pum_rhs, constraint_tag=tag)]
                elif lhs.name == "abs":
                    return [constraints.Absolute(self.to_pum_ivar(lhs.args[0]), pum_rhs, constraint_tag=tag)]
                elif lhs.name == "min":
                    return [constraints.Minimum(self.to_pum_ivar(lhs.args), pum_rhs, constraint_tag=tag)]
                elif lhs.name == "max":
                    return [constraints.Maximum(self.to_pum_ivar(lhs.args), pum_rhs, constraint_tag=tag)]
                elif lhs.name == "element":
                    arr, idx = lhs.args
                    return [constraints.Element(self.to_pum_ivar(idx),
                                                self.to_pum_ivar(arr),
                                                pum_rhs, constraint_tag=tag)]
                else:
                    raise NotImplementedError("Unknown lhs of comparison", cpm_expr)

            elif cpm_expr.name == "<=":
                return [constraints.LessThanOrEquals(self._sum_args(lhs, tag=tag), rhs, constraint_tag=tag)]
            elif cpm_expr.name == "<":
                return [constraints.LessThanOrEquals(self._sum_args(lhs, tag=tag), rhs-1,constraint_tag=tag)]
            elif cpm_expr.name == ">=":
                return [constraints.LessThanOrEquals(self._sum_args(lhs, negate=True, tag=tag), -rhs, constraint_tag=tag)]
            elif cpm_expr.name == ">":
                return [constraints.LessThanOrEquals(self._sum_args(lhs, negate=True, tag=tag), -rhs-1, constraint_tag=tag)]
            elif cpm_expr.name == "!=":
                return [constraints.NotEquals(self._sum_args(lhs, tag=tag), rhs, constraint_tag=tag)]

            raise ValueError("Unknown comparison", cpm_expr)

        elif isinstance(cpm_expr, GlobalConstraint):
            if cpm_expr.name == "alldifferent":
                return [constraints.AllDifferent(self.solver_vars(cpm_expr.args), constraint_tag=tag)]
            
            elif cpm_expr.name == "cumulative":
                start, dur, end, demand, cap = cpm_expr.args
                assert all(is_num(d) for d in dur), "Pumpkin only accepts Cumulative with fixed durations"
                assert all(is_num(d) for d in demand), "Pumpkin only accepts Cumulative with fixed demand"
                assert is_num(cap), "Pumpkin only accepts Cumulative with fixed capacity"

                pum_cons = [constraints.Cumulative(self.solver_vars(start),dur, demand, cap, constraint_tag=tag)]
                if end is not None:
                    pum_cons += [self._get_constraint(c, tag=tag)[0] for c in self.transform([s + d == e for s,d,e in zip(start, dur, end)])]
                return pum_cons

            elif cpm_expr.name == "no_overlap":
                start, dur, end = cpm_expr.args
                return self._get_constraint(Cumulative(start, dur, end, demand=1, capacity=1), tag=tag)

            elif cpm_expr.name == "table":
                arr, table = cpm_expr.args
                return [constraints.Table(self.to_pum_ivar(arr),
                                          np.array(table).tolist(), # ensure Python list
                                          constraint_tag=tag)
                        ]
            
            elif cpm_expr.name == "negative_table":
                arr, table = cpm_expr.args
                return [constraints.NegativeTable(self.to_pum_ivar(arr),
                                                  np.array(table).tolist(),# ensure Python list
                                                  constraint_tag=tag)
                        ]
            
            elif cpm_expr.name == "InDomain":
                val, domain = cpm_expr.args
                return [constraints.Table(self.to_pum_ivar([val]),
                                          [[d] for d in domain], # each domain value is its own row
                                          constraint_tag=tag)
                        ]
            
            
            else:
                raise NotImplementedError(f"Unknown global constraint {cpm_expr}")
                

        elif isinstance(cpm_expr, BoolVal): # unlikely base case
            a = boolvar() # dummy variable
            if cpm_expr.value() is True:
                return self._get_constraint(Operator("sum", [a]) >= 1, tag=tag)
            else:
                return self._get_constraint(Operator("sum", [a]) <= -1, tag=tag)

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
        if self.pum_solver.is_inconsistent():
            return self # cannot post any more constraints once inconsistency is reached

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        try:
            for cpm_expr in self.transform(cpm_expr_orig):
                if isinstance(cpm_expr, Operator) and cpm_expr.name == "->": # found implication
                    bv, subexpr = cpm_expr.args
                    for pum_cons in self._get_constraint(subexpr):
                        self.pum_solver.add_implication(pum_cons, self.solver_var(bv))
                else:
                    for pum_cons in self._get_constraint(cpm_expr):
                        self.pum_solver.add_constraint(pum_cons)
            return self

        except RuntimeError as e:
            # Can happen when conflict is found with just root level propagation
            if self.pum_solver.is_inconsistent():
                return self
            raise e # something else happened

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

    def solution_hint(self, cpm_vars: List[_NumVarImpl], vals: List[int]):
        """
        Pumpkin supports warmstarting the solver with a (in)feasible solution.
        The provided value will affect branching heurstics during solving, making it more likely the final solution will contain the provided assignment.
        Technical side-node: only used during optimization.

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
        if self.pum_solver.is_inconsistent() is False: # otherwise, not guaranteed all variables are known
            self._solhint = {self.solver_var(v) : val for v, val in zip(cpm_vars, vals)} # store for later use in solve
