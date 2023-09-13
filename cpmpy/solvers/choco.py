#!/usr/bin/env python
## choco.py
##
"""
    Interface to Choco solver's Python API

    ...

    Documentation of the solver's own Python API:
    https://pypi.org/project/pychoco/

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_choco
"""
import sys  # for stdout checking
import numpy as np
from cpmpy.transformations.normalize import toplevel_list

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView, boolvar
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.utils import is_num, is_any_list, eval_comparison
from ..transformations.decompose_global import decompose_global, decompose_in_tree
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.reification import only_bv_implies, reify_rewrite
from ..transformations.comparison import only_numexpr_equality
from ..transformations.linearize import canonical_comparison

class CPM_choco(SolverInterface):
    """
    Interface to the Choco solver python API

    Requires that the 'pychoco' python package is installed:
    $ pip install pychoco

    See detailed installation instructions at:
    https://pypi.org/project/pychoco/

    Creates the following attributes (see parent constructor for more):
    chc_model: the pychoco.Model() created by _model()
    chc_solver: the choco Model().get_solver() instance used in solve()

    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pychoco as chc
            return True
        except ImportError:
            return False


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        choco model and solver object (chc_model and chc_solver)

        chc_model and chc_solver can both be modified externally before
        calling solve(), a prime way to use more advanced solver features

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: None
        """
        if not self.supported():
            raise Exception("Install the python 'pychoco' package to use this solver interface")

        import pychoco as chc

        assert(subsolver is None)

        # initialise the native solver objects
        self.chc_model = chc.Model()
        self.chc_solver = chc.Model().get_solver()
        self.helper_var = self.chc_model.intvar(0,2)
        # for solving with assumption variables, TO-CHECK

        # initialise everything else and post the constraints/objective
        super().__init__(name="choco", cpm_model=cpm_model)


    def solve(self, time_limit=None, **kwargs):
        """
            Call the Choco solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Additional keyword arguments:
            The ortools solver parameters are defined in its 'sat_parameters.proto' description:
            https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto

            Arguments that correspond to solver parameters:
            <Please document key solver arguments that the user might wish to change
             for example: log_output=True, var_ordering=3, num_cores=8, ...>
            <Add link to documentation of all solver parameters>
        """
        import pychoco as chc

        # call the solver, with parameters
        self.chc_solver = self.chc_model.get_solver()
        self.chc_status = self.chc_solver.solve()

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        #self.cpm_status.runtime = self.chc_status.WallTime()

        """
        # translate exit status
        if self.ort_status == ort.FEASIBLE:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif self.ort_status == ort.OPTIMAL:
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        elif self.ort_status == ort.INFEASIBLE:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif self.ort_status == ort.MODEL_INVALID:
            raise Exception("OR-Tools says: model invalid:", self.ort_model.Validate())
        elif self.ort_status == ort.UNKNOWN:
            # can happen when timeout is reached...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:  # another?
            raise NotImplementedError(self.ort_status)  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                cpm_var._value = self.ort_solver.Value(self.solver_var(cpm_var))
                if isinstance(cpm_var, _BoolVarImpl):
                    cpm_var._value = bool(cpm_var._value) # ort value is always an int

            # translate objective
            if self.has_objective():
                self.objective_value_ = self.ort_solver.ObjectiveValue()
        """
        return True

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            A shorthand to (efficiently) compute all solutions, map them to CPMpy and optionally display the solutions.

            It is just a wrapper around the use of `OrtSolutionPrinter()` in fact.

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - solution_limit: stop after this many solutions (default: None)
                - call_from_model: whether the method is called from a CPMpy Model instance or not

            Returns: number of solutions found
        """
        if self.has_objective():
            raise NotSupportedError("OR-tools does not support finding all optimal solutions.")

        cb = OrtSolutionPrinter(self, display=display, solution_limit=solution_limit)
        self.solve(enumerate_all_solutions=True, solution_callback=cb, time_limit=time_limit, **kwargs)
        return cb.solution_count()

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var):  # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return self.solver_var(cpm_var._bv).Not()

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.chc_model.boolvar(name=str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.chc_model.intvar(cpm_var.lb, cpm_var.ub, name=str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        return self._varmap[cpm_var]


    def objective(self, expr, minimize):
        """
            Post the given expression to the solver as objective to minimize/maximize

            - expr: Expression, the CPMpy expression that represents the objective function
            - minimize: Bool, whether it is a minimization problem (True) or maximization problem (False)

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective
            are premanently posted to the solver)
        """
        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr)
        self += flat_cons  # add potentially created constraints
        get_variables(flat_obj, collect=self.user_vars)  # add objvars to vars

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        #if minimize:
        #    self.ort_model.Minimize(obj)
        #else:
        #    self.ort_model.Maximize(obj)

    def has_objective(self):
        pass
        #return self.chc_model.
        #return self.ort_model.HasObjective()

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function

            Accepted by ORTools:
            - Decision variable: Var
            - Linear: sum([Var])                                   (CPMpy class 'Operator', name 'sum')
                      wsum([Const],[Var])                          (CPMpy class 'Operator', name 'wsum')
        """

        lhs = cpm_expr.args[0]
        rhs = cpm_expr.args[1]
        op = cpm_expr.name
        if op == "==": op = "="

        if is_num(lhs):    #TODO     can this happen to be num in lhs?? I think no
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(lhs, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.chc_model.arithm(self.solver_var(lhs), op, self.solver_var(rhs))

        print("cpm_expr: ", cpm_expr)
        # sum or weighted sum
        if isinstance(lhs, Operator):
            if lhs.name == 'sum':
                self.chc_model.sum(self.solver_vars(lhs.args), op, self.solver_var(rhs))
            elif lhs.name == "sub":
                a,b = self.solver_vars(lhs.args)
                return self.chc_model.arithm(a, "-", b, op, self.solver_var(rhs))
            elif lhs.name == 'wsum':
                w = lhs.args[0]
                x = self.solver_vars(lhs.args[1])
                return self.chc_model.scalar(x, w, op, self.solver_var(rhs))

        raise NotImplementedError("ORTools: Not a known supported numexpr {}".format(cpm_expr))


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

        cpm_cons = toplevel_list(cpm_expr)
        supported = {"min", "max", "abs", "element", "alldifferent", "xor", "table", "cumulative", "circuit", "inverse"}
        cpm_cons = decompose_in_tree(cpm_cons, supported)
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = canonical_comparison(cpm_cons)
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        cpm_cons = only_bv_implies(cpm_cons) # everything that can create
                                             # reified expr must go before this

        return cpm_cons

    def __add__(self, cpm_expr):
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
        get_variables(cpm_expr, collect=self.user_vars)

        # transform and post the constraints
        for con in self.transform(cpm_expr):
            self._post_constraint(con)

        return self

    # only 3 constraints support it (and,or,sum),
    # we can just add reified support for those and not need `reifiable` or returning the constraint
    # then we can remove _post_constraint and have its code inside the for loop of __add__
    # like for other solvers
    def _post_constraint(self, cpm_expr):
        """
            Post a supported CPMpy constraint directly to the underlying solver's API

            What 'supported' means depends on the solver capabilities, and in effect on what transformations
            are applied in `transform()`.

            Returns the posted ortools 'Constraint', so that it can be used in reification
            e.g. self._post_constraint(smth, reifiable=True).onlyEnforceIf(self.solver_var(bvar))

        :param cpm_expr: CPMpy expression
        :type cpm_expr: Expression

        :param reifiable: if True, will throw an error if cpm_expr can not be reified by ortools (for safety)
        """
        import pychoco as chc

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        if isinstance(cpm_expr, Operator):
            # 'and'/n, 'or'/n, '->'/2
            if cpm_expr.name == 'and':
                return self.chc_model.and_(self.solver_vars(cpm_expr.args)).post()
            elif cpm_expr.name == 'or':
                return self.chc_model.or_(self.solver_vars(cpm_expr.args)).post()
            elif cpm_expr.name == '->':
                assert(isinstance(cpm_expr.args[0], _BoolVarImpl))  # lhs must be boolvar
                lhs = self.solver_var(cpm_expr.args[0])
                if isinstance(cpm_expr.args[1], _BoolVarImpl):
                    # bv -> bv
                    return lhs.implies(self.solver_var(cpm_expr.args[1]))
            else:
                raise NotImplementedError("Not a known supported Choco Operator '{}' {}".format(
                        cpm_expr.name, cpm_expr))

        # Comparisons: only numeric ones as the `only_bv_implies()` transformation #TODO: Choco allows == in bools
        # has removed the '==' reification for Boolean expressions
        # numexpr `comp` bvar|const
        elif isinstance(cpm_expr, Comparison):
            lhs = cpm_expr.args[0]
            chcrhs = self.solver_var(cpm_expr.args[1])

            if isinstance(lhs, _NumVarImpl):
                # both are variables, do python comparison over ORT variables
                print("Arithm")
                return self.chc_model.arithm(self.solver_var(lhs), cpm_expr.name, chcrhs).post()
            elif isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum' or lhs.name == "sub"):
                # a BoundedLinearExpression LHS, special case, like in objective
                chc_numexpr = self._make_numexpr(cpm_expr)
                return chc_numexpr.post()
            elif cpm_expr.name == '==':
                # NumExpr == IV, supported by ortools (thanks to `only_numexpr_equality()` transformation)
                if lhs.name == 'min':
                    return self.chc_model.min(chcrhs, self.solver_vars(lhs.args))
                elif lhs.name == 'max':
                    return self.chc_model.max(chcrhs, self.solver_vars(lhs.args))
                elif lhs.name == 'abs':
                    return self.chc_model.absolute(chcrhs, self.solver_var(lhs.args[0]))
                elif lhs.name == 'mul':
                    return self.chc_model.times(self.solver_vars(lhs.args[0]), self.solver_vars(lhs.args[1]), chcrhs)
                elif lhs.name == 'div':
                    return self.chc_model.div(self.solver_vars(lhs.args[0]), self.solver_vars(lhs.args[1]), chcrhs)
                elif lhs.name == 'element':
                    # arr[idx]==rvar (arr=arg0,idx=arg1), ort: (idx,arr,target)
                    return self.chc_model.element(self.solver_var(lhs.args[1]),
                                                     self.solver_vars(lhs.args[0]), chcrhs)
                elif lhs.name == 'mod':
                    # catch tricky-to-find ortools limitation
                    divisor = lhs.args[1]
                    if not is_num(divisor):
                        if divisor.lb <= 0 and divisor.ub >= 0:
                            raise Exception(
                                    f"Expression '{lhs}': or-tools does not accept a 'modulo' operation where '0' is in the domain of the divisor {divisor}:domain({divisor.lb}, {divisor.ub}). Even if you add a constraint that it can not be '0'. You MUST use a variable that is defined to be higher or lower than '0'.")
                    return self.ort_model.AddModuloEquality(ortrhs, *self.solver_vars(lhs.args))
                elif lhs.name == 'pow':
                    # only `POW(b,2) == IV` supported, post as b*b == IV
                    assert (lhs.args[1] == 2), "Ort: 'pow', only var**2 supported, no other exponents"
                    b = self.solver_var(lhs.args[0])
                    return self.ort_model.AddMultiplicationEquality(ortrhs, [b,b])
            raise NotImplementedError(
                        "Not a known supported ORTools left-hand-side '{}' {}".format(lhs.name, cpm_expr))

        # base (Boolean) global constraints
        elif isinstance(cpm_expr, GlobalConstraint):

            if cpm_expr.name == 'alldifferent':
                return self.ort_model.AddAllDifferent(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == 'table':
                assert (len(cpm_expr.args) == 2)  # args = [array, table]
                array, table = self.solver_vars(cpm_expr.args)
                return self.ort_model.AddAllowedAssignments(array, table)
            elif cpm_expr.name == "cumulative":
                start, dur, end, demand, cap = self.solver_vars(cpm_expr.args)
                if is_num(demand):
                    demand = [demand] * len(start)
                intervals = [self.ort_model.NewIntervalVar(s,d,e,f"interval_{s}-{d}-{e}") for s,d,e in zip(start,dur,end)]
                return self.ort_model.AddCumulative(intervals, demand, cap)
            elif cpm_expr.name == "circuit":
                # ortools has a constraint over the arcs, so we need to create these
                # when using an objective over arcs, using these vars direclty is recommended
                # (see PCTSP-path model in the future)
                x = cpm_expr.args
                N = len(x)
                arcvars = boolvar(shape=(N,N), name="circuit_arcs")
                # post channeling constraints from int to bool
                self += [b == (x[i] == j) for (i,j),b in np.ndenumerate(arcvars)]
                # post the global constraint
                # when posting arcs on diagonal (i==j), it would do subcircuit
                ort_arcs = [(i,j,self.solver_var(b)) for (i,j),b in np.ndenumerate(arcvars) if i != j]
                return self.ort_model.AddCircuit(ort_arcs)
            elif cpm_expr.name == 'inverse':
                assert len(cpm_expr.args) == 2, "inverse() expects two args: fwd, rev"
                fwd, rev = self.solver_vars(cpm_expr.args)
                return self.ort_model.AddInverse(fwd, rev)
            elif cpm_expr.name == 'xor':
                return self.ort_model.AddBoolXOr(self.solver_vars(cpm_expr.args))
            else:
                raise NotImplementedError(f"Unknown global constraint {cpm_expr}, should be decomposed! If you reach this, please report on github.")

        # unlikely base case: Boolean variable
        elif isinstance(cpm_expr, _BoolVarImpl):
            print(cpm_expr)
            return self.chc_model.and_([self.solver_var(cpm_expr)]).post()

        # unlikely base case: True or False
        elif isinstance(cpm_expr, BoolVal):
            if cpm_expr:
                return self.chc_model.arithm(self.helper_var, ">=", 0).post()
            else:
                return self.chc_model.arithm(self.helper_var, "<", 0).post()

        # a direct constraint, pass to solver
        elif isinstance(cpm_expr, DirectConstraint):
            return cpm_expr.callSolver(self, self.chc_model)

        # else
        raise NotImplementedError(cpm_expr)  # if you reach this... please report on github


    def solution_hint(self, cpm_vars, vals):
        """
        or-tools supports warmstarting the solver with a feasible solution

        More specifically, it will branch that variable on that value first if possible. This is known as 'phase saving' in the SAT literature, but then extended to integer variables.

        The solution hint does NOT need to satisfy all constraints, it should just provide reasonable default values for the variables. It can decrease solving times substantially, especially when solving a similar model repeatedly

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
        self.ort_model.ClearHints() # because add just appends
        for (cpm_var, val) in zip(cpm_vars, vals):
            self.ort_model.AddHint(self.solver_var(cpm_var), val)


    def get_core(self):
        from ortools.sat.python import cp_model as ort
        """
            For use with s.solve(assumptions=[...]). Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

            CPMpy will return only those variables that are False (in the UNSAT core)

            Note that there is no guarantee that the core is minimal, though this interface does upon up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!

            For pure or-tools example, see http://github.com/google/or-tools/blob/master/ortools/sat/samples/assumptions_sample_sat.py

            Requires or-tools >= 8.2!!!
        """
        assert (self.ort_status == ort.INFEASIBLE), "get_core(): solver must return UNSAT"
        assert (self.assumption_dict is not None),  "get_core(): requires a list of assumption variables, e.g. s.solve(assumptions=[...])"

        # use our own dict because of VarIndexToVarProto(0) bug in ort 8.2
        assum_idx = self.ort_solver.SufficientAssumptionsForInfeasibility()

        # return cpm_variables corresponding to ort_assum vars in UNSAT core
        return [self.assumption_dict[i] for i in assum_idx]

    @classmethod
    def tunable_params(cls):
        return {
            'cp_model_probing_level': [0, 1, 2],
            'preferred_variable_order': [0, 1, 2],
            'linearization_level': [0, 1, 2],
            'symmetry_level': [0, 1, 2],
            'minimization_algorithm': [0, 1, 2],
            'search_branching': [0, 1, 2, 3, 4, 5, 6],
            'optimize_with_core': [False, True],
            'use_erwa_heuristic': [False, True]
        }

    @classmethod
    def default_params(cls):
        return {
            'cp_model_probing_level': 2,
            'preferred_variable_order': 0,
            'linearization_level': 1,
            'symmetry_level': 2,
            'minimization_algorithm': 2,
            'search_branching': 0,
            'optimize_with_core': False,
            'use_erwa_heuristic': False
        }


# solvers are optional, so this file should be interpretable
# even if ortools is not installed...
try:
    from ortools.sat.python import cp_model as ort
    import time


    class OrtSolutionCounter(ort.CpSolverSolutionCallback):
        """
        Native or-tools callback for solution counting.

        It is based on ortools' built-in `ObjectiveSolutionPrinter`
        but with output printing being optional

        use with CPM_ortools as follows:
        `cb = OrtSolutionCounter()`
        `s.solve(enumerate_all_solutions=True, solution_callback=cb)`

        then retrieve the solution count with `cb.solution_count()`

        Arguments:
            - verbose whether to print info on every solution found (bool, default: False)
    """

        def __init__(self, verbose=False):
            super().__init__()
            self.__solution_count = 0
            self.__verbose = verbose
            if self.__verbose:
                self.__start_time = time.time()

        def on_solution_callback(self):
            """Called on each new solution."""
            if self.__verbose:
                current_time = time.time()
                obj = self.ObjectiveValue()
                print('Solution %i, time = %0.2f s, objective = %i' %
                      (self.__solution_count, current_time - self.__start_time, obj))
            self.__solution_count += 1

        def solution_count(self):
            """Returns the number of solutions found."""
            return self.__solution_count

    class OrtSolutionPrinter(OrtSolutionCounter):
        """
            Native or-tools callback for solution printing.

            Subclasses OrtSolutionCounter, see those docs too

            use with CPM_ortools as follows:
            `cb = OrtSolutionPrinter(s, display=vars)`
            `s.solve(enumerate_all_solutions=True, solution_callback=cb)`

            for multiple variabes (single or NDVarArray), use:
            `cb = OrtSolutionPrinter(s, display=[v, x, z])`

            for a custom print function, use for example:
            ```def myprint():
        print(f"x0={x[0].value()}, x1={x[1].value()}")
        cb = OrtSolutionPrinter(s, printer=myprint)```

            optionally retrieve the solution count with `cb.solution_count()`

            Arguments:
                - verbose: whether to print info on every solution found (bool, default: False)
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                            default/None: nothing displayed
                - solution_limit: stop after this many solutions (default: None)
        """
        def __init__(self, solver, display=None, solution_limit=None, verbose=False):
            super().__init__(verbose)
            self._solution_limit = solution_limit
            # we only need the cpmpy->solver varmap from the solver
            self._varmap = solver._varmap
            # identify which variables to populate with their values
            self._cpm_vars = []
            self._display = display
            if isinstance(display, (list,Expression)):
                self._cpm_vars = get_variables(display)
            elif callable(display):
                # might use any, so populate all (user) variables with their values
                self._cpm_vars = solver.user_vars

        def on_solution_callback(self):
            """Called on each new solution."""
            super().on_solution_callback()
            if len(self._cpm_vars):
                # populate values before printing
                for cpm_var in self._cpm_vars:
                    # it might be an NDVarArray
                    if hasattr(cpm_var, "flat"):
                        for cpm_subvar in cpm_var.flat:
                            cpm_subvar._value = self.Value(self._varmap[cpm_subvar])
                    elif isinstance(cpm_var, _BoolVarImpl):
                        cpm_var._value = bool(self.Value(self._varmap[cpm_var]))
                    else:
                        cpm_var._value = self.Value(self._varmap[cpm_var])

                if isinstance(self._display, Expression):
                    print(self._display.value())
                elif isinstance(self._display, list):
                    # explicit list of expressions to display
                    print([v.value() for v in self._display])
                else: # callable
                    self._display()

            # check for count limit
            if self.solution_count() == self._solution_limit:
                self.StopSearch()

except ImportError:
    pass  # Ok, no ortools installed...
