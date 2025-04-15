#!/usr/bin/env python
import warnings
import re

from cpmpy.exceptions import NotSupportedError
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.python_builtins import any as cpm_any
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar, boolvar
from ..expressions.utils import is_num, is_any_list, is_boolexpr, eval_comparison, flip_comparison, get_bounds
from ..transformations.get_variables import get_variables
from ..transformations.linearize import canonical_comparison
from ..transformations.normalize import toplevel_list
from ..transformations.negation import push_down_negation
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.comparison import only_numexpr_equality
from ..transformations.reification import reify_rewrite, only_bv_reifies, only_implies
from ..transformations.safening import no_partial_functions

import time

"""
    Interface to Pumpkin's API

    Pumpkin is a combinatorial optimisation solver based on lazy clause 
    generation and constraint programming.

    Documentation of the solver's own Python API:
      - https://github.com/consol-lab/pumpkin

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_pumpkin
"""

class CPM_pumpkin(SolverInterface):
    """
    Interface to Pumpkin's API

    Requires that the 'pumpkin_py' python package is installed:
    $ pip install pumpkin_py

    Creates the following attributes (see parent constructor for more):
    - tpl_model: object, Pumpkin's model object
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
        - subsolver: str, name of a subsolver (optional)
        """
        if not self.supported():
            raise Exception("CPM_Pumpkin: Install the python package 'pumpkin_py'")

        from pumpkin_py import Model

        assert subsolver is None # unless you support subsolvers, see pysat or minizinc

        # initialise the native solver object
        self.pum_solver = Model() 
        self.prefix = None

        # a dictionary for constant variables, so they can be re-used
        self._constantvars = dict()

        # for objective
        self._objective = None
        self.objective_is_min = True

        # dictionary of tags to user constraints
        self.user_cons = dict()

        # initialise everything else and post the constraints/objective
        # [GUIDELINE] this superclass call should happen AFTER all solver-native objects are created.
        #           internally, the constructor relies on __add__ which uses the above solver native object(s)
        super().__init__(name="Pumpkin", cpm_model=cpm_model)


    def solve(self, time_limit=None, proof=None, assumptions=None, **kwargs):
        """
            Call the Pumpkin solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - proof:       path to a proof file
            - assumptions: CPMpy Boolean variables (or their negation) that are assumed to be true.
                           For repeated solving, and/or for use with s.get_core(): if the model is UNSAT,
                           get_core() returns a small subset of assumption variables that are unsat together.
            - kwargs:      any keyword argument, sets parameters of solver object


            Arguments that correspond to solver parameters:
            # [GUIDELINE] Please document key solver arguments that the user might wish to change
            #       for example: assumptions=[x,y,z], log_output=True, var_ordering=3, num_cores=8, ...
            # [GUIDELINE] Add link to documentation of all solver parameters
        """

        # Again, I don't know why this is necessary, but the PyO3 modules seem to be a bit wonky.
        from pumpkin_py import BoolExpression as PumpkinBool, IntExpression as PumpkinInt
        from pumpkin_py import SatisfactionResult, SatisfactionUnderAssumptionsResult
        from pumpkin_py.optimisation import OptimisationResult, Direction

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # [GUIDELINE] if your solver supports solving under assumptions, add `assumptions` as argument in header
        #       e.g., def solve(self, time_limit=None, assumptions=None, **kwargs):
        #       then translate assumptions here; assumptions are a list of Boolean variables or NegBoolViews

        # call the solver, with parameters
        start_time = time.time() # when did solving start

        if proof is not None:
            self.prefix = proof

        if self.has_objective():
            assert assumptions is None, "Optimization under assumptions is not supported"
            solve_func = self.pum_solver.optimise
            kwargs.update(proof=proof,
                          objective=self.solver_var(self._objective),
                          direction=Direction.Minimise if self.objective_is_min else Direction.Maximise)

        elif assumptions is not None:
            assert proof is None, "Proof-logging under assumptions is not supported"
            pum_assumptions = [self.to_predicate(a) for a in assumptions]
            self.assump_map = dict(zip(pum_assumptions, assumptions))
            solve_func = self.pum_solver.satisfy_under_assumptions
            kwargs.update(assumptions=pum_assumptions)

        else:
            solve_func = self.pum_solver.satisfy
            kwargs.update(proof=proof)

        self._pum_core = None
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
                raise ValueError("Unexpected optimisation result:", result)

        elif assumptions is not None: # check result under assumptions
            if isinstance(result, SatisfactionUnderAssumptionsResult.Satisfiable):
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            elif isinstance(result, SatisfactionUnderAssumptionsResult.Unsatisfiable):
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
                self._pum_core = [] # empty core, no required assumptions to prove UNSAT
            elif  isinstance(result, SatisfactionUnderAssumptionsResult.UnsatisfiableUnderAssumptions):
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
                self._pum_core = result._0
            elif isinstance(result, SatisfactionUnderAssumptionsResult.Unknown):
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            else:
                raise ValueError("Unexpected result:", result)

        else: # satisfaction result without assumptions
            if isinstance(result, SatisfactionResult.Satisfiable):
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
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
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL

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
            if not cpm_var in self._constantvars:
                self._constantvars[cpm_var] = self.pum_solver.new_integer_variable(cpm_var, cpm_var, name=str(cpm_var))

            return self._constantvars[cpm_var]

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

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective

            are permanently posted to the solver)
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
        supported_halfreif = {"or", "sum", "wsum", "sub", "mul", "div", "abs", "min", "max", "cumulative"}
        cpm_cons = reify_rewrite(cpm_cons, supported=supported_halfreif) # reified element not supported yet (TODO?)
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        cpm_cons = canonical_comparison(cpm_cons) # ensure rhs is always a constant
        return cpm_cons

    def to_predicate(self, cpm_expr):
        """

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

            assert isinstance(lhs, _NumVarImpl), "lhs should be variable"

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

        from pumpkin_py import constraints

        if isinstance(cpm_expr, _BoolVarImpl):
            # base case, just var or ~var
            forced_sum = Operator("sum", [cpm_expr])
            return [constraints.Equals(self._sum_args(forced_sum), 1)]

        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == "or":
                hacked = Operator("sum", cpm_expr.args)
                return [constraints.LessThanOrEquals(self._sum_args(hacked, negate=True), -1)]
                return [constraints.Clause(self.solver_vars(cpm_expr.args))]

            raise NotImplementedError("Pumpkin: operator not (yet) supported", cpm_expr)

        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            assert isinstance(lhs, Expression), f"Expected a CPMpy expression on lhs but got {lhs} of type {type(lhs)}"

            # if isinstance(lhs, Operator) and lhs.name == "sum" and len(lhs.args) == 1:
            #     # a predicate
            #     return [constraints.Clause([self.pum_solver.predicate_as_boolean(self.to_predicate(cpm_expr))])]

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
                return self._get_constraint(Operator("sum", [a]) >= 0)
            else:
                return self._get_constraint(Operator("sum", [a]) <= -1)

        else:
            raise ValueError("Unexpected constraint:", cpm_expr)


    def __add__(self, cpm_expr_orig):
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

        # transform and post the constraints
        for orig_expr in toplevel_list(cpm_expr_orig, merge_and=True):
            if orig_expr in set(self.user_cons.values()):
                continue
            else:
                constraint_tag = len(self.user_cons)+1
                self.user_cons[constraint_tag] = orig_expr
            for cpm_expr in self.transform(orig_expr):
                tag = constraint_tag
                if isinstance(cpm_expr, Operator) and cpm_expr.name == "->": # found implication

                    bv, subexpr = cpm_expr.args
                    for cons in self._get_constraint(subexpr):
                        if isinstance(cons, constraints.Clause):    
                            raise ValueError("_get_constraint should not return clauses")
                        self.pum_solver.add_implication(cons, self.solver_var(bv), tag=tag)
                else:
                    solver_constraints = self._get_constraint(cpm_expr)

                    for cons in solver_constraints:
                        if isinstance(cons, constraints.Clause):
                            raise ValueError("_get_constraint should not return clauses")
                        self.pum_solver.add_constraint(cons,tag=tag)

        return self

    # Other functions from SolverInterface that you can overwrite:
    # solveAll, solution_hint, get_core

    def get_core(self):
        """
           For use with s.solve(assumptions=[...]). Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

           CPMpy will return only those variables that are False (in the UNSAT core)

           Note that there is no guarantee that the core is minimal, though this interface does open up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!
        """

        assert self._pum_core is not None, "Can only get core if the last solve-call was unsatisfiable under assumptions"
        return [self.assump_map[pred] for pred in self._pum_core]

