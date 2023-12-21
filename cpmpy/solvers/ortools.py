#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## ortools.py
##
"""
    Interface to ortools' CP-SAT Python API

    Google OR-Tools is open source software for combinatorial optimization, which seeks
    to find the best solution to a problem out of a very large set of possible solutions.
    The OR-Tools CP-SAT solver is an award-winning constraint programming solver
    that uses SAT (satisfiability) methods and lazy-clause generation.

    Documentation of the solver's own Python API:
    https://google.github.io/or-tools/python/ortools/sat/python/cp_model.html

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_ortools
"""
import sys  # for stdout checking
import numpy as np

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView, boolvar
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.utils import is_num, is_any_list, eval_comparison, flatlist
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from ..transformations.comparison import only_numexpr_equality

class CPM_ortools(SolverInterface):
    """
    Interface to the python 'ortools' CP-SAT API

    Requires that the 'ortools' python package is installed:
    $ pip install ortools

    See detailed installation instructions at:
    https://developers.google.com/optimization/install

    Creates the following attributes (see parent constructor for more):
    ort_model: the ortools.sat.python.cp_model.CpModel() created by _model()
    ort_solver: the ortools cp_model.CpSolver() instance used in solve()

    The `DirectConstraint`, when used, calls a function on the `ort_model` object.
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import ortools
            return True
        except ImportError:
            return False


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        or-tools model and solver object (ort_model and ort_solver)

        ort_model and ort_solver can both be modified externally before
        calling solve(), a prime way to use more advanced solver features

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: None
        """
        if not self.supported():
            raise Exception("Install the python 'ortools' package to use this solver interface")

        from ortools.sat.python import cp_model as ort

        assert(subsolver is None)

        # initialise the native solver objects
        self.ort_model = ort.CpModel()
        self.ort_solver = ort.CpSolver()

        # for solving with assumption variables,
        # need to store mapping from ORTools Index to CPMpy variable
        self.assumption_dict = None

        # initialise everything else and post the constraints/objective
        super().__init__(name="ortools", cpm_model=cpm_model)


    def solve(self, time_limit=None, assumptions=None, solution_callback=None, **kwargs):
        """
            Call the CP-SAT solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - assumptions: list of CPMpy Boolean variables (or their negation) that are assumed to be true.
                           For repeated solving, and/or for use with s.get_core(): if the model is UNSAT,
                           get_core() returns a small subset of assumption variables that are unsat together.
                           Note: the or-tools interface is stateless, so you can incrementally call solve() with assumptions, but or-tools will always start from scratch...
            - solution_callback: an `ort.CpSolverSolutionCallback` object. CPMpy includes its own, namely `OrtSolutionCounter`. If you want to count all solutions, don't forget to also add the keyword argument 'enumerate_all_solutions=True'.

            Additional keyword arguments:
            The ortools solver parameters are defined in its 'sat_parameters.proto' description:
            https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto

            You can use any of these parameters as keyword argument to `solve()` and they will
            be forwarded to the solver. Examples include:
                - num_search_workers=8          number of parallel workers (default: 8)
                - log_search_progress=True      to log the search process to stdout (default: False)
                - cp_model_presolve=False       to disable presolve (default: True, almost always beneficial)
                - cp_model_probing_level=0      to disable probing (default: 2, also valid: 1, maybe 3, etc...)
                - linearization_level=0         to disable linearisation (default: 1, can also set to 2)
                - optimize_with_core=True       to do max-sat like lowerbound optimisation (default: False)
                - use_branching_in_lp=True      to generate more info in lp propagator (default: False)
                - polish_lp_solution=True       to spend time in lp propagator searching integer values (default: False)
                - symmetry_level=1              only do symmetry breaking in presolve (default: 2, also possible: 0)

            example:
            o.solve(num_search_workers=8, log_search_progress=True)

        """
        from ortools.sat.python import cp_model as ort

        # set time limit?
        if time_limit is not None:
            self.ort_solver.parameters.max_time_in_seconds = float(time_limit)

        if assumptions is not None:
            ort_assum_vars = self.solver_vars(assumptions)
            # dict mapping ortools vars to CPMpy vars
            self.assumption_dict = {ort_var.Index(): cpm_var for (cpm_var, ort_var) in zip(assumptions, ort_assum_vars)}
            self.ort_model.ClearAssumptions()  # because add just appends
            self.ort_model.AddAssumptions(ort_assum_vars)
            # workaround for a presolve with assumptions bug in ortools
            # https://github.com/google/or-tools/issues/2649
            # still present in v9.0
            self.ort_solver.parameters.keep_all_feasible_solutions_in_presolve = True

        # set additional keyword arguments in sat_parameters.proto
        for (kw, val) in kwargs.items():
            setattr(self.ort_solver.parameters, kw, val)

        if 'log_search_progress' in kwargs and hasattr(self.ort_solver, "log_callback") \
                and (sys.stdout != sys.__stdout__):
            # ortools>9.0, for IPython use, force output redirecting
            # see https://github.com/google/or-tools/issues/1903
            # but only if a nonstandard stdout, otherwise duplicate output
            # see https://github.com/CPMpy/cpmpy/issues/84
            self.ort_solver.log_callback = print

        # call the solver, with parameters
        self.ort_status = self.ort_solver.Solve(self.ort_model, solution_callback=solution_callback)

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.ort_solver.WallTime()

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
                try:
                    cpm_var._value = self.ort_solver.Value(self.solver_var(cpm_var))
                    if isinstance(cpm_var, _BoolVarImpl):
                        cpm_var._value = bool(cpm_var._value) # ort value is always an int
                except IndexError:
                    cpm_var._value = None  # probably got optimized away by our transformations

            # translate objective
            if self.has_objective():
                self.objective_value_ = self.ort_solver.ObjectiveValue()

        return has_sol

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
                revar = self.ort_model.NewBoolVar(str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.ort_model.NewIntVar(cpm_var.lb, cpm_var.ub, str(cpm_var))
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
        if minimize:
            self.ort_model.Minimize(obj)
        else:
            self.ort_model.Maximize(obj)

    def has_objective(self):
        return self.ort_model.HasObjective()

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
        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # sum or weighted sum
        if isinstance(cpm_expr, Operator):
            if cpm_expr.name == 'sum':
                return sum(self.solver_vars(cpm_expr.args))  # OR-Tools supports this
            elif cpm_expr.name == "sub":
                a,b = self.solver_vars(cpm_expr.args)
                return a - b
            elif cpm_expr.name == 'wsum':
                w = cpm_expr.args[0]
                x = self.solver_vars(cpm_expr.args[1])
                return sum(wi*xi for wi,xi in zip(w,x))  # XXX is there a more direct way?

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
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_implies(cpm_cons)  # everything that can create
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

    # TODO: 'reifiable' is an artefact from the early days
    # only 3 constraints support it (and,or,sum),
    # we can just add reified support for those and not need `reifiable` or returning the constraint
    # then we can remove _post_constraint and have its code inside the for loop of __add__
    # like for other solvers
    def _post_constraint(self, cpm_expr, reifiable=False):
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

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        if isinstance(cpm_expr, Operator):
            # 'and'/n, 'or'/n, '->'/2
            if cpm_expr.name == 'and':
                return self.ort_model.AddBoolAnd(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == 'or':
                return self.ort_model.AddBoolOr(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == '->':
                assert(isinstance(cpm_expr.args[0], _BoolVarImpl))  # lhs must be boolvar
                lhs = self.solver_var(cpm_expr.args[0])
                if isinstance(cpm_expr.args[1], _BoolVarImpl):
                    # bv -> bv
                    return self.ort_model.AddImplication(lhs, self.solver_var(cpm_expr.args[1]))
                else:
                    # bv -> boolexpr
                    # the `reify_rewrite()` transformation ensures that only
                    # the natively reifiable 'and', 'or' and 'sum' remain here
                    return self._post_constraint(cpm_expr.args[1], reifiable=True).OnlyEnforceIf(lhs)
            else:
                raise NotImplementedError("Not a known supported ORTools Operator '{}' {}".format(
                        cpm_expr.name, cpm_expr))

        # Comparisons: only numeric ones as the `only_implies()` transformation
        # has removed the '==' reification for Boolean expressions
        # numexpr `comp` bvar|const
        elif isinstance(cpm_expr, Comparison):
            lhs = cpm_expr.args[0]
            ortrhs = self.solver_var(cpm_expr.args[1])

            if isinstance(lhs, _NumVarImpl):
                # both are variables, do python comparison over ORT variables
                return self.ort_model.Add(eval_comparison(cpm_expr.name, self.solver_var(lhs), ortrhs))
            elif isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum' or lhs.name == "sub"):
                # a BoundedLinearExpression LHS, special case, like in objective
                ortlhs = self._make_numexpr(lhs)
                # ortools accepts sum(x) >= y over ORT variables
                return self.ort_model.Add(eval_comparison(cpm_expr.name, ortlhs, ortrhs))
            elif cpm_expr.name == '==':
                # NumExpr == IV, supported by ortools (thanks to `only_numexpr_equality()` transformation)
                if lhs.name == 'min':
                    return self.ort_model.AddMinEquality(ortrhs, self.solver_vars(lhs.args))
                elif lhs.name == 'max':
                    return self.ort_model.AddMaxEquality(ortrhs, self.solver_vars(lhs.args))
                elif lhs.name == 'abs':
                    return self.ort_model.AddAbsEquality(ortrhs, self.solver_var(lhs.args[0]))
                elif lhs.name == 'mul':
                    return self.ort_model.AddMultiplicationEquality(ortrhs, self.solver_vars(lhs.args))
                elif lhs.name == 'div':
                    return self.ort_model.AddDivisionEquality(ortrhs, *self.solver_vars(lhs.args))
                elif lhs.name == 'element':
                    # arr[idx]==rvar (arr=arg0,idx=arg1), ort: (idx,arr,target)
                    return self.ort_model.AddElement(self.solver_var(lhs.args[1]),
                                                     self.solver_vars(lhs.args[0]), ortrhs)
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
                intervals = [self.ort_model.NewIntervalVar(s,d,e,f"interval_{s}-{d}-{e}") for s,d,e in zip(start,dur,end)]
                return self.ort_model.AddCumulative(intervals, demand, cap)
            elif cpm_expr.name == "circuit":
                # ortools has a constraint over the arcs, so we need to create these
                # when using an objective over arcs, using these vars direclty is recommended
                # (see PCTSP-path model in the future)
                x = cpm_expr.args
                N = len(x)
                arcvars = boolvar(shape=(N,N))
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
            return self.ort_model.AddBoolOr([self.solver_var(cpm_expr)])

        # unlikely base case: True or False
        elif isinstance(cpm_expr, BoolVal):
            return self.ort_model.Add(cpm_expr.args[0])

        # a direct constraint, pass to solver
        elif isinstance(cpm_expr, DirectConstraint):
            return cpm_expr.callSolver(self, self.ort_model)

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

        cpm_vars = flatlist(cpm_vars)
        vals = flatlist(vals)
        assert (len(cpm_vars) == len(vals)), "Variables and values must have the same size for hinting"
        for (cpm_var, val) in zip(cpm_vars, vals):
            self.ort_model.AddHint(self.solver_var(cpm_var), val)


    def get_core(self):
        from ortools.sat.python import cp_model as ort
        """
            For use with s.solve(assumptions=[...]). Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

            CPMpy will return only those variables that are False (in the UNSAT core)

            Note that there is no guarantee that the core is minimal, though this interface does open up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!

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
        """
            Suggestion of tunable hyperparameters of the solver.
            List compiled based on a conversation with OR-tools' Laurent Perron (issue #138).
        """
        return {
            'use_branching_in_lp': [False, True],
            'optimize_with_core' : [False, True],
            'search_branching': [0,1,2,3,4,5,6],
            'boolean_encoding_level' : [0,1,2,3],
            'linearization_level': [0, 1, 2],
            'core_minimization_level' : [0,1,2], # new in OR-tools>=v9.8
            'cp_model_probing_level': [0, 1, 2, 3],
            'cp_model_presolve' : [False, True],
            'clause_cleanup_ordering' : [0,1],
            'binary_minimization_algorithm' : [0,1,2,3,4],
            'minimization_algorithm' : [0,1,2,3],
            'use_phase_saving' : [False, True]
        }

    @classmethod
    def default_params(cls):
        return {
            'use_branching_in_lp': False,
            'optimize_with_core': False,
            'search_branching': 0,
            'boolean_encoding_level': 1,
            'linearization_level': 1,
            'core_minimization_level': 2,# new in OR-tools>=v9.8
            'cp_model_probing_level': 2,
            'cp_model_presolve': True,
            'clause_cleanup_ordering': 0,
            'binary_minimization_algorithm': 1,
            'minimization_algorithm': 2,
            'use_phase_saving': True
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
