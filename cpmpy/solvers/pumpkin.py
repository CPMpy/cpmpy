#!/usr/bin/env python
import warnings
import re

from os.path import join

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

"""
    Interface to Pumpkin's API

    Pumpkin is a combinatorial optimisation solver developed by the ConSol Lab at TU Delft. 
    It is based on the (lazy clause generation) constraint programming paradigm.
    (see https://github.com/consol-lab/pumpkin)

    Always use :func:`cp.SolverLookup.get("pumpkin") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ===============
    Installation
    ===============

    The `pumpkin_py` python package is currently not available on PyPI.
    It can be installed from source using the following steps:
     1. clone the repository from github: https://github.com/consol-lab/pumpkin
     2. install the "maturin" package to build the python bindings: $ pip install maturin
     3. build and install the package: $ maturin develop

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
            import pumpkin_py as gp
            return True
        except ImportError:
            return False


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: None, not used
        """
        if not self.supported():
            raise Exception("CPM_Pumpkin: Install the python package 'pumpkin_py'")

        from pumpkin_py import Model

        assert subsolver is None 

        # initialise the native solver object
        self.pum_solver = Model() 

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
            - time_limit:  maximum solve time in seconds (float, optional)
            - prove: whether to produce a DRCP proof (.lits file and .drcp proof file).
            - proof_name: name for the the proof files.
            - proof_location: location for the proof files (default to current working directory).
            - assumptions: CPMpy Boolean variables (or their negation) that are assumed to be true.
                           For repeated solving, and/or for use with s.get_core(): if the model is UNSAT,
                           get_core() returns a small subset of assumption variables that are unsat together.
        """

        # Again, I don't know why this is necessary, but the PyO3 modules seem to be a bit wonky.
        from pumpkin_py import BoolExpression as PumpkinBool, IntExpression as PumpkinInt
        from pumpkin_py import SatisfactionResult, SatisfactionUnderAssumptionsResult
        from pumpkin_py.optimisation import OptimisationResult, Direction

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        if time_limit is not None:
            raise ValueError("Time limits are currently not supported by Pumpkin")

        # parse and dispatch the arguments
        kwargs = dict()

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
            elif  isinstance(result, SatisfactionUnderAssumptionsResult.UnsatisfiableUnderAssumptions):
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

    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            - expr: Expression, the CPMpy expression that represents the objective function
            - minimize: Bool, whether it is a minimization problem (True) or maximization problem (False)

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

            See the 'Adding a new solver' docs on readthedocs for more information.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        # apply transformations
        cpm_cons = toplevel_list(cpm_expr)
        supported = {"alldifferent", "cumulative", 
                     "min", "max", "element"}
        cpm_cons = decompose_in_tree(cpm_cons, supported=supported)
        # safening after decompose here, need to safen toplevel elements too
        #   which come from decomposition of other global constraints...
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"element"})
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_implies(cpm_cons)
        supported_halfreif = {"or", "sum", "wsum", "sub", "mul", "div", "abs", "min", "max"}
        cpm_cons = reify_rewrite(cpm_cons, supported=supported_halfreif) # reified element not supported yet (TODO?)
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        cpm_cons = canonical_comparison(cpm_cons) # ensure rhs is always a constant
        return cpm_cons

    def to_predicate(self, cpm_expr):
        """
            Convert a CPMpy expression to a Pumpkin predicate (comparison with constant)
        """
        from pumpkin_py import Comparator, Predicate

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
                else:
                    raise ValueError("Lhs of predicate should be a sum with 1 argument") # TODO: also wsum with 1 arg/mul with const?

            assert isinstance(lhs, _NumVarImpl), "lhs should be variable by now"

            if cpm_expr.name == "==": comp = Comparator.Equal
            if cpm_expr.name == "<=": comp = Comparator.LessThanOrEqual
            if cpm_expr.name == ">=": comp = Comparator.GreaterThanOrEqual
            if cpm_expr.name == "!=": comp = Comparator.NotEqual
            if cpm_expr.name == "<":  comp, rhs = Comparator.LessThanOrEqual, rhs - 1
            if cpm_expr.name == ">":  comp, rhs = Comparator.GreaterThanOrEqual, rhs + 1
        else:
            raise ValueError(f"Cannot convert CPMpy expression {cpm_expr} to a predicate")

        var = self._ivars(lhs)
        return Predicate(var, comp, rhs)


    def _ivars(self, cpm_var):
        """
            Helper function to convert (boolean) variables and constants to Pumpkin integer expressions
        """
        if is_any_list(cpm_var):
            return [self._ivars(v) for v in cpm_var]
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
        elif isinstance(Operator, expr) and expr.name == "sub":
            x, y = self.solver_vars(expr.args)
            if expr.args[0].is_bool():
                x = self.pum_solver.boolean_as_integer(x)
            if expr.args[1].is_bool():
                y = self.pum_solver.boolean_as_integer(y)
            args = [x.scaled(-1 if negate else 1), y.scaled(1 if negate else -1)]
        else:
            raise ValueError(f"Unknown expression to convert in sum-arguments: {expr}")
        return args

    def _get_constraint(self, cpm_expr):
        """
            Get a solver's constraint by a supported CPMpy constraint

            :param cpm_expr: CPMpy expression
            :type cpm_expr: Expression
        """
        from pumpkin_py import constraints

        if isinstance(cpm_expr, _BoolVarImpl):
            # base case, just var or ~var
            forced_sum = Operator("sum", [cpm_expr])
            return [constraints.Equals(self._sum_args(forced_sum), 1)]

        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == "or": 
                # bit of a hack: clauses cannot be tagged in the proof, use sum instead
                summed_args = Operator("sum", cpm_expr.args)
                return [constraints.LessThanOrEquals(self._sum_args(summed_args, negate=True), -1)]

            raise NotImplementedError("Pumpkin: operator not (yet) supported", cpm_expr)

        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            assert isinstance(lhs, Expression), f"Expected a CPMpy expression on lhs but got {lhs} of type {type(lhs)}"

            if cpm_expr.name == "==":
                if "sum" in lhs.name or lhs.name == "sub":
                    return [constraints.Equals(self._sum_args(lhs), rhs)]
               
                pum_rhs = self._ivars(rhs) # other operators require IntExpression
                if lhs.name == "div":
                    return [constraints.Division(*self._ivars(lhs.args), pum_rhs)]
                elif lhs.name == "mul":
                    return [constraints.Times(*self._ivars(lhs.args), pum_rhs)]
                elif lhs.name == "abs":
                    return [constraints.Absolute(self.solver_var(lhs), pum_rhs)]
                elif lhs.name == "min":
                    return [constraints.Minimum(self._ivars(lhs.args), pum_rhs)]
                elif lhs.name == "max":
                    return [constraints.Maximum(self._ivars(lhs.args), pum_rhs)]
                elif lhs.name == "element":
                    arr, idx = lhs.args
                    return [constraints.Element(self._ivars(idx), self._ivars(arr), pum_rhs)]
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
        from pumpkin_py import constraints

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
           For use with s.solve(assumptions=[...]). Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

           CPMpy will return only those variables that are False (in the UNSAT core)

           Note that there is no guarantee that the core is minimal, though this interface does open up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!
        """

        assert self._pum_core is not None, "Can only get core if the last solve-call was unsatisfiable under assumptions"
        return [self.assump_map[pred] for pred in self._pum_core]

