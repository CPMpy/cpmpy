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

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView
from ..expressions.utils import is_num, is_any_list
from ..transformations.get_variables import get_variables_model, get_variables
from ..transformations.flatten_model import flatten_model, flatten_constraint, flatten_objective, get_or_make_var, negated_normal
from ..transformations.reification import only_bv_implies

class CPM_ortools(SolverInterface):
    """
    Interface to the python 'ortools' CP-SAT API

    Requires that the 'ortools' python package is installed:
    $ pip install ortools

    See detailed installation instructions at:
    https://developers.google.com/optimization/install
    and if you are on Apple M1: https://cpmpy.readthedocs.io/en/latest/installation_M1.html

    Creates the following attributes (see parent constructor for more):
    ort_model: the ortools.sat.python.cp_model.CpModel() created by _model()
    ort_solver: the ortools cp_model.CpSolver() instance used in solve()
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import ortools
            return True
        except ImportError as e:
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
            raise Exception("Install the python 'ortools' package to use this '{}' solver interface".format(name))

        from ortools.sat.python import cp_model as ort

        assert(subsolver is None)

        # initialise the native solver objects
        self.ort_model = ort.CpModel()
        self.ort_solver = ort.CpSolver()

        # initialize assumption dict to None
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
                           Note: the or-tools interace is stateless, so you can incrementally call solve() with assumptions, but or-tools will always start from scratch...
            - solution_callback: an `ort.CpSolverSolutionCallback` object. CPMpy includes its own, namely `OrtSolutionCounter`. If you want to count all solutions, don't forget to also add the keyword argument 'enumerate_all_solutions=True'.

            Additional keyword arguments:
            The ortools solver parameters are defined in its 'sat_parameters.proto' description:
            https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto

            You can use any of these parameters as keyword argument to `solve()` and they will
            be forwarded to the solver. Examples include:
                - num_search_workers=8          number of parallel workers (default: 1)
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
                cpm_var._value = self.ort_solver.Value(self.solver_var(cpm_var))
                if isinstance(cpm_var, _BoolVarImpl):
                    cpm_var._value = bool(cpm_var._value) # ort value is always an int

            # translate objective
            if self.ort_model.HasObjective():
                self.objective_value_ = self.ort_solver.ObjectiveValue()

        return has_sol

    def solveAll(self, display=None, time_limit=None, solution_limit=None, **kwargs):
        """
            A shorthand to (efficiently) compute all solutions, map them to CPMpy and optionally display the solutions.

            It is just a wrapper around the use of `OrtSolutionPrinter()` in fact.

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - solution_limit: stop after this many solutions (default: None)

            Returns: number of solutions found
        """
        # XXX: check that no objective function??
        cb = OrtSolutionPrinter(self, display=display, solution_limit=solution_limit)
        self.solve(enumerate_all_solutions=True, solution_callback=cb, time_limit=time_limit, **kwargs)
        return cb.solution_count()


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var): # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return self.solver_var(cpm_var._bv).Not()

        # create if it does not exit
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.ort_model.NewBoolVar(str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.ort_model.NewIntVar(cpm_var.lb, cpm_var.ub, str(cpm_var))
            else:
                raise NotImplementedError("Not a know var {}".format(cpm_var))
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
        self += flat_cons # add potentially created constraints
        self.user_vars.update(get_variables(flat_obj)) # add objvars to vars

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        if minimize:
            self.ort_model.Minimize(obj)
        else:
            self.ort_model.Maximize(obj)

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
            elif cpm_expr.name == 'wsum':
                w = cpm_expr.args[0]
                x = self.solver_vars(cpm_expr.args[1])
                return sum(wi*xi for wi,xi in zip(w,x)) # XXX is there more direct way?

        raise NotImplementedError("ORTools: Not a know supported numexpr {}".format(cpm_expr))


    def __add__(self, cpm_con):
        """
        Post a (list of) CPMpy constraints(=expressions) to the solver

        Note that we don't store the constraints in a cpm_model,
        we first transform the constraints into primitive constraints,
        then post those primitive constraints directly to the native solver

        :param cpm_con CPMpy constraint, or list thereof
        :type cpm_con (list of) Expression(s)
        """
        # add new user vars to the set
        self.user_vars.update(get_variables(cpm_con))

        # apply transformations, then post internally
        cpm_cons = only_bv_implies(flatten_constraint(cpm_con))
        for con in cpm_cons:
            self._post_constraint(con)

        return self


    def _post_constraint(self, cpm_expr, reifiable=False):
        """
            Post a primitive CPMpy constraint to the native solver API

            What 'primitive' means depends on the solver capabilities,
            more specifically on the transformations applied in `__add__()`

            While the normal form is divided in 'base', 'comparison' and 'reified', we
            here regroup the implementation per CPMpy class

            Returns the posted ortools 'Constraint', so that it can be used in reification
            e.g. self._post_constraint(smth, reifiable=True).onlyEnforceIf(self.solver_var(bvar))
            
            - reifiable: if True, will throw an error if cpm_expr can not be reified
        """
        # Base case: Boolean variable
        if isinstance(cpm_expr, _BoolVarImpl):
            return self.ort_model.AddBoolOr([self.solver_var(cpm_expr)])

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        elif isinstance(cpm_expr, Operator):
            # 'and'/n, 'or'/n, 'xor'/n, '->'/2
            if cpm_expr.name == 'and':
                return self.ort_model.AddBoolAnd(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == 'or':
                return self.ort_model.AddBoolOr(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == 'xor':
                return self.ort_model.AddBoolXOr(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == '->':
                assert(isinstance(cpm_expr.args[0], _BoolVarImpl)) # lhs must be boolvar
                lhs = self.solver_var(cpm_expr.args[0])
                if isinstance(cpm_expr.args[1], _BoolVarImpl):
                    # bv -> bv
                    return self.ort_model.AddImplication(lhs, self.solver_var(cpm_expr.args[1]))
                else:
                    # bv -> boolexpr, natively supported by or-tools
                    # actually, only supported for and, or, and linear expression (not xor nor any global)
                    # XXX should we check/assert that here??
                    # TODO: and if rhs is a global, first decompose it and reify that??
                    assert(cpm_expr.args[1].is_bool())
                    # Special case for 'xor', which is not natively reifiable in ortools
                    # TODO: make xor a global constraint (so it can be decomposed generally) and get rid of this special case here
                    if isinstance(cpm_expr.args[1], Operator) and cpm_expr.args[1].name == 'xor':
                        if len(cpm_expr.args) == 2:
                            return self._post_constraint((sum(cpm_expr.args[1].args) == 1), reifiable=True).OnlyEnforceIf(lhs)
                        else:
                            raise NotImplementedError("ORT: reified n-ary XOR not yet supported, make an issue on github if you need it")
                    # TODO: and if something like b.implies(min(x) >= 10) that it splits up in
                    # b.implies( aux >= 10) & (min(x) == aux)
                    return self._post_constraint(cpm_expr.args[1], reifiable=True).OnlyEnforceIf(lhs)
            else:
                raise NotImplementedError("Not a know supported ORTools Operator '{}' {}".format(
                        cpm_expr.name, cpm_expr))


        # Comparisons: only numeric ones as 'only_bv_implies()' has removed the '==' reification for Boolean expressions
        # numexpr `comp` bvar|const
        elif isinstance(cpm_expr, Comparison):
            lhs = cpm_expr.args[0]
            rvar = self.solver_var(cpm_expr.args[1])

            # TODO: this should become a transformation!!
            if cpm_expr.name != '==' and not is_num(lhs) and not isinstance(lhs, _NumVarImpl)\
                    and not lhs.name == "wsum" and not lhs.name == "sum":
                # functional globals only exist for equality in ortools
                # example: min(x) > 10 :: min(x) == aux, aux > 10
                # create the equality and overwrite lhs with auxiliary (will handle appropriate bounds)
                (lhs, cons) = get_or_make_var(lhs)
                self += cons

            # all but '==' now only have as lhs: const|ivar|sum|wsum
            # translate ivar|sum|wsum so they can be posted directly below
            if isinstance(lhs, _NumVarImpl):
                lhs = self.solver_var(lhs)
            elif isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum'):
                # a BoundedLinearExpression LHS, special case, like in objective
                lhs = self._make_numexpr(lhs)
                # assumes that ortools accepts sum(x) >= y without further simplification

            # post the comparison
            if cpm_expr.name == '<=':
                return self.ort_model.Add(lhs <= rvar)
            elif cpm_expr.name == '<':
                return self.ort_model.Add(lhs < rvar)
            elif cpm_expr.name == '>=':
                return self.ort_model.Add(lhs >= rvar)
            elif cpm_expr.name == '>':
                return self.ort_model.Add(lhs > rvar)
            elif cpm_expr.name == '!=':
                return self.ort_model.Add(lhs != rvar)
            elif cpm_expr.name == '==':
                if not isinstance(lhs, Expression): 
                    # base cases: const|ivar|sum|wsum with prepped lhs above
                    return self.ort_model.Add(lhs == rvar)
                elif lhs.name == 'min':
                    return self.ort_model.AddMinEquality(rvar, self.solver_vars(lhs.args))
                elif lhs.name == 'max':
                    return self.ort_model.AddMaxEquality(rvar, self.solver_vars(lhs.args))
                elif lhs.name == 'abs':
                    return self.ort_model.AddAbsEquality(rvar, self.solver_var(lhs.args[0]))
                elif lhs.name == 'mul':
                    return self.ort_model.AddMultiplicationEquality(rvar, self.solver_vars(lhs.args))
                elif lhs.name == 'div':
                    return self.ort_model.AddDivisionEquality(rvar, *self.solver_vars(lhs.args))
                elif lhs.name == 'element':
                    # arr[idx]==rvar (arr=arg0,idx=arg1), ort: (idx,arr,target)
                    return self.ort_model.AddElement(self.solver_var(lhs.args[1]),
                                                     self.solver_vars(lhs.args[0]), rvar)
                elif lhs.name == 'mod':
                    # catch tricky-to-find ortools limitation
                    divisor = lhs.args[1]
                    if not is_num(divisor):
                        if divisor.lb <= 0 and divisor.ub >= 0:
                            raise Exception(
                                    f"Expression '{lhs}': or-tools does not accept a 'modulo' operation where '0' is in the domain of the divisor {divisor}:domain({divisor.lb}, {divisor.ub}). Even if you add a constraint that it can not be '0'. You MUST use a variable that is defined to be higher or lower than '0'.")
                    return self.ort_model.AddModuloEquality(rvar, *self.solver_vars(lhs.args))
                elif lhs.name == 'pow':
                    # translate to multiplications
                    # TODO: perhaps this should be a transformation too? pow to (binary) mult
                    x = self.solver_var(lhs.args[0])
                    y = lhs.args[1]
                    assert is_num(y), f"Ort: 'pow' only supports constants as power, not {y}"
                    if y == 0:
                        return 1
                    elif y == 1:
                        return self.ort_model.Add(x == rvar)
                    # mul([x,x,x,...]) with 'y' elements
                    assert (y == 2), "Ort: 'pow' with an exponent larger than 2 has lead to crashes..."
                    return self.ort_model.AddMultiplicationEquality(rvar, [x] * y)
            raise NotImplementedError(
                        "Not a know supported ORTools left-hand-side '{}' {}".format(lhs.name, cpm_expr))


        # rest: base (Boolean) global constraints
        elif cpm_expr.name == 'alldifferent':
            return self.ort_model.AddAllDifferent(self.solver_vars(cpm_expr.args))
        elif cpm_expr.name == 'table':
            assert (len(cpm_expr.args) == 2)  # args = [array, table]
            array, table = self.solver_vars(cpm_expr.args)
            return self.ort_model.AddAllowedAssignments(array, table)
        else:
            # TODO: NOT YET MAPPED: Automaton, Circuit, Cumulative,
            #    ForbiddenAssignments, Inverse?, NoOverlap, NoOverlap2D,
            #    ReservoirConstraint, ReservoirConstraintWithActive
            
            # global constraint not known, try posting generic decomposition
            self += cpm_expr.decompose() # assumes a decomposition exists...
            # TODO: dynamic mapping of cpm_expr.name to API call? see #74
            return None # will throw error if used in reification
        
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
            self.ort_model.AddHint(self.ort_var(cpm_var), val)


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
