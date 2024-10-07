#!/usr/bin/env python
from cpmpy.exceptions import NotSupportedError
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl
from ..expressions.utils import is_num, is_any_list, is_boolexpr
from ..transformations.get_variables import get_variables
from ..transformations.linearize import canonical_comparison
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.comparison import only_numexpr_equality
from ..transformations.reification import reify_rewrite, only_bv_reifies, only_implies

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

        # a dictionary for constant variables, so they can be re-used
        self._constantvars = dict()

        # initialise everything else and post the constraints/objective
        # [GUIDELINE] this superclass call should happen AFTER all solver-native objects are created.
        #           internally, the constructor relies on __add__ which uses the above solver native object(s)
        super().__init__(name="Pumpkin", cpm_model=cpm_model)


    def solve(self, time_limit=None, proof=None, **kwargs):
        """
            Call the Pumpkin solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - proof:       path to a proof file
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            # [GUIDELINE] Please document key solver arguments that the user might wish to change
            #       for example: assumptions=[x,y,z], log_output=True, var_ordering=3, num_cores=8, ...
            # [GUIDELINE] Add link to documentation of all solver parameters
        """

        # Again, I don't know why this is necessary, but the PyO3 modules seem to be a bit wonky.
        from pumpkin_py import BoolExpression as PumpkinBool, IntExpression as PumpkinInt, SatisfactionResult

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # [GUIDELINE] if your solver supports solving under assumptions, add `assumptions` as argument in header
        #       e.g., def solve(self, time_limit=None, assumptions=None, **kwargs):
        #       then translate assumptions here; assumptions are a list of Boolean variables or NegBoolViews

        # call the solver, with parameters
        start_time = time.time() # when did solving start
        result = self.pum_solver.satisfy(proof=proof, **kwargs)

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = time.time() - start_time

        # translate solver exit status to CPMpy exit status
        match result:
            case SatisfactionResult.Satisfiable(solution):
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE

                # fill in variable values
                for cpm_var in self.user_vars:
                    sol_var = self.solver_var(cpm_var)

                    if isinstance(sol_var, PumpkinInt):
                        cpm_var._value = solution.int_value(sol_var)
                    elif isinstance(sol_var, PumpkinBool):
                        cpm_var._value = solution.bool_value(sol_var)
                    else:
                        raise NotSupportedError("Only boolean and integer variables are supported.")


            case SatisfactionResult.Unsatisfiable():
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE

            case SatisfactionResult.Unknown():
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN

        # translate solution values (of user specified variables only)
        self.objective_value_ = None

        return self._solve_return(self.cpm_status)


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """

        if is_num(cpm_var):
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


    # [GUIDELINE] if Pumpkin does not support objective functions, you can delete this function definition
    # def objective(self, expr, minimize=True):
    #     """
    #         Post the given expression to the solver as objective to minimize/maximize
    #
    #         'objective()' can be called multiple times, only the last one is stored
    #
    #         (technical side note: any constraints created during conversion of the objective
    #
    #         are permanently posted to the solver)
    #     """
    #     # make objective function non-nested
    #     (flat_obj, flat_cons) = flatten_objective(expr)
    #     self += flat_cons # add potentially created constraints
    #     self.user_vars.update(get_variables(flat_obj)) # add objvars to vars
    #
    #     # make objective function or variable and post
    #     obj = self._make_numexpr(flat_obj)
    #     # [GUIDELINE] if the solver interface does not provide a solver native "numeric expression" object,
    #     #         _make_numexpr may be removed and an objective can be posted as:
    #     #           self.pum_solver.MinimizeWeightedSum(obj.args[0], self.solver_vars(obj.args[1]) or similar
    #
    #     if minimize:
    #         self.pum_solver.Minimize(obj)
    #     else:
    #         self.pum_solver.Maximize(obj)

    # def has_objective(self):
    #     return False # TODO
    #     return self.pum_solver.hasObjective()

    def _make_numexpr(self, cpm_expr):
        """
            Converts a numeric CPMpy 'flat' expression into a solver-specific numeric expression

            Primarily used for setting objective functions, and optionally in constraint posting
        """

        # [GUIDELINE] not all solver interfaces have a native "numerical expression" object.
        #       in that case, this function may be removed and a case-by-case analysis of the numerical expression
        #           used in the constraint at hand is required in __add__
        #       For an example of such solver interface, check out solvers/choco.py or solvers/exact.py

        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # any solver-native numerical expression
        if isinstance(cpm_expr, Operator):
           if cpm_expr.name == 'sum':
               return self.pum_solver.sum(self.solver_vars(cpm_expr.args))
           elif cpm_expr.name == 'wsum':
               weights, vars = cpm_expr.args
               return self.pum_solver.weighted_sum(weights, self.solver_vars(vars))
           # [GUIDELINE] or more fancy ones such as max
           #        be aware this is not the Maximum CONSTRAINT, but rather the Maximum NUMERICAL EXPRESSION
           elif cpm_expr.name == "max":
               return self.pum_solver.maximum_of_vars(self.solver_vars(cpm_expr.args))
           # ...
        raise NotImplementedError("Pumpkin: Not a known supported numexpr {}".format(cpm_expr))


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
        cpm_cons = decompose_in_tree(cpm_cons, supported={"alldifferent", "cumulative"})
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_implies(cpm_cons)
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        cpm_cons = canonical_comparison(cpm_cons) # ensure rhs is always a constant
        return cpm_cons

    def _ivars(self, cpm_var):
        if is_any_list(cpm_var):
            return [self._ivars(v) for v in cpm_var]
        elif isinstance(cpm_var, _BoolVarImpl):
            return self.pum_solver.boolean_as_integer(self.solver_var(cpm_var))
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
            return [constraints.Clause([self.solver_var(cpm_expr)])]

        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == "or":
                return [constraints.Clause(self.solver_vars(cpm_expr.args))]

            raise NotImplementedError("Pumpkin: operator not (yet) supported", cpm_expr)

        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            assert isinstance(lhs, Expression), f"Expected a CPMpy expression on lhs but got {lhs} of type {type(lhs)}"
            pum_rhs = self.solver_var(rhs)
            if isinstance(rhs, _BoolVarImpl):
                pum_rhs = self.pum_solver.boolean_as_integer(pum_rhs)

            if cpm_expr.name == "==":
                if "sum" in lhs.name or lhs.name == "sub":
                    return [constraints.Equals(self._sum_args(lhs), rhs)]
                if lhs.name == "div":
                    return [constraints.Division(*self.solver_vars(lhs.args), pum_rhs)]
                if lhs.name == "mul":
                    return [constraints.Times(*self.solver_vars(lhs.args), pum_rhs)]
                if lhs.name == "abs":
                    return [constraints.Absolute(self.solver_var(lhs), pum_rhs)]
                elif lhs.name == "minimum":
                    return [constraints.Minimum(self.solver_vars(lhs.args), pum_rhs)]
                elif lhs.name == "maximum":
                    return [constraints.Maximum(self.solver_vars(lhs.args), pum_rhs)]
                elif lhs.name == "element":
                    return [constraints.Element(*self.solver_vars(lhs.args), pum_rhs)]

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

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        for tag, orig_expr in enumerate(toplevel_list(cpm_expr_orig, merge_and=True)):
            for cpm_expr in self.transform(orig_expr):

                if isinstance(cpm_expr, Operator) and cpm_expr.name == "->": # found implication
                    bv, subexpr = cpm_expr.args
                    for cons in self._get_constraint(subexpr):
                        self.pum_solver.add_implication(cons, self.solver_var(bv), tag=None)
                else:
                    solver_constraints = self._get_constraint(cpm_expr)
                    for cons in solver_constraints:
                        self.pum_solver.add_constraint(cons, tag=None)

        return self

    # Other functions from SolverInterface that you can overwrite:
    # solveAll, solution_hint, get_core

    # def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
    #     """
    #         A shorthand to (efficiently) compute all (optimal) solutions, map them to CPMpy and optionally display the solutions.
    #
    #         If the problem is an optimization problem, returns only optimal solutions.
    #
    #        Arguments:
    #             - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
    #                     default/None: nothing displayed
    #             - time_limit: stop after this many seconds (default: None)
    #             - solution_limit: stop after this many solutions (default: None)
    #             - call_from_model: whether the method is called from a CPMpy Model instance or not
    #             - any other keyword argument
    #
    #         Returns: number of solutions found
    #     """
    #
    #     # check if objective function
    #     if self.has_objective():
    #         raise NotSupportedError("Pumpkin does not support finding all optimal solutions")
    #
    #     # A. Example code if solver supports callbacks
    #     if is_any_list(display):
    #         callback = lambda : print([var.value() for var in display])
    #     else:
    #         callback = display
    #
    #     self.solve(time_limit, callback=callback, enumerate_all_solutions=True, **kwargs)
    #     return self.pum_solver.SolutionCount()
    #
    #     # B. Example code if solver does not support callbacks
    #     self.solve(time_limit, enumerate_all_solutions=True, **kwargs)
    #     solution_count = 0
    #     for solution in self.pum_solver.GetAllSolutions():
    #         solution_count += 1
    #         # Translate solution to variables
    #         for cpm_var in self.user_vars:
    #             cpm_var._value = solution.value(solver_var)
    #
    #         if display is not None:
    #             if isinstance(display, Expression):
    #                 print(display.value())
    #             elif isinstance(display, list):
    #                 print([v.value() for v in display])
    #             else:
    #                 display()  # callback
    #
    #     return solution_count
