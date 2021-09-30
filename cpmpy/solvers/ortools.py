#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## ortools.py
##
"""
    Interface to ortools' Python API

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_ortools
"""
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView
from ..expressions.utils import is_num, is_any_list
from ..transformations.get_variables import get_variables_model, get_variables
from ..transformations.flatten_model import flatten_model, flatten_constraint, flatten_objective, get_or_make_var, negated_normal

class CPM_ortools(SolverInterface):
    """
    Interface to the python 'ortools' API

    Requires that the 'ortools' python package is installed:
    $ pip install ortools

    Creates the following attributes:
    user_vars: variables in the original (unflattened) model, incl objective
    ort_model: the ortools.sat.python.cp_model.CpModel() created by _model()
    ort_solver: the ortools cp_model.CpSolver() instance used in solve()
    ort_status: the ortools 'status' instance returned by ort_solver.Solve()
    cpm_status: the corresponding CPMpy status
    """

    @staticmethod
    def supported():
        try:
            import ortools
            return True
        except ImportError as e:
            return False

    def __init__(self, cpm_model=None):
        """
        Constructor of the solver object

        Requires a CPMpy model as input, and will create the corresponding
        or-tools model and solver object (ort_model and ort_solver)

        ort_model and ort_solver can both be modified externally before
        calling solve(), a prime way to use more advanced solver features
        """
        if not self.supported():
            raise Exception("Install the python 'ortools' package to use this '{}' solver interface".format(self.name))
        from ortools.sat.python import cp_model as ort

        super().__init__()
        self.name = "ortools"

        if cpm_model is None:
            self.user_vars = []
            self.ort_model = ort.CpModel()
            self.varmap = dict() # cppy var -> solver var
        else:
            # store original vars and objective (before flattening)
            self.user_vars = get_variables_model(cpm_model)

            # create model (includes conversion to flat normal form)
            self.ort_model = self.make_model(cpm_model)

        # create the solver instance
        # (so its params can still be changed before calling solve)
        self.ort_solver = ort.CpSolver()


    def __add__(self, cons):
        """
        Direct solver access constraint addition,
        avoids having to flatten the model of the constructor again
        when calling s.solve() repeatedly.

        Note that we don't store the resulting cpm_model, we translate
        directly to the ort_model

        :param cpm_cons list of CPMpy constraints
        :type cpm_cons list of Expressions
        """
        # store new user vars
        new_user_vars = get_variables(cons)
        for v in frozenset(new_user_vars)-frozenset(self.user_vars):
            self.user_vars.append(v)

        flat_cons = flatten_constraint(cons)
        # add new (auxiliary) variables
        for var in get_variables(flat_cons):
            if not var in self.varmap:
                self.add_to_varmap(var)
        # add constraints
        for cpm_con in flat_cons:
            self.post_constraint(cpm_con)

        return self

    def minimize(self, expr):
        """
            Minimize the given objective function

            `minimize()` can be called multiple times, only the last one is stored
        """
        # flatten
        (flat_obj, flat_cons) = flatten_objective(expr)

        # add constraints if needed
        if len(flat_cons) != 0:
            self += flat_cons

        obj = self.ort_numexpr(flat_obj)
        self.ort_model.Minimize(obj)

    def maximize(self, expr):
        """
            Maximize the given objective function

            `maximize()` can be called multiple times, only the last one is stored
        """
        # flatten
        (flat_obj, flat_cons) = flatten_objective(expr)

        # add constraints if needed
        if len(flat_cons) != 0:
            self += flat_cons

        obj = self.ort_numexpr(flat_obj)
        self.ort_model.Maximize(obj)


    def solution_hint(self, cpm_vars, vals):
        """
        or-tools supports warmstarting the solver with a feasible solution

        More specifically, it will branch that variable on that value first if possible. This is known as 'phase saving' in the SAT literature, but then extended to integer variables.

        The solution hint does NOT need to satisfy all constraints, it should just provide reasonable default values for the variables. It can decrease solving times substantially, especially when solving a similar model repeatedly

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
        for (cpm_var, val) in zip(cpm_vars, vals):
            self.ort_model.ClearHints() # because add just appends
            self.ort_model.AddHint(self.ort_var(cpm_var), val)


    def solve(self, time_limit=None, assumptions=None, solution_callback=None, **kwargs):
        """
            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - assumptions: list of CPMpy Boolean variables (or their negation) that are assumed to be true.
                           For use with s.get_core(): if the model is UNSAT, get_core() returns a small subset of assumption variables that are unsat together.
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
            ort_assum_vars = [self.ort_var(v) for v in assumptions]
            # this is fucked up... the ort_var()'s index does not seem
            # to match ort_model.VarIndexToVarProto(index)...
            # yet, SufficientAssum... will return that index, so keep own map
            #
            # oh, actually... its a bug that I already reported earlier for
            # VarIndexToVarProto(0) and that Laurent then fixed...
            # Until version 8.3 is released, I'm sticking to own dict
            self.assumption_dict = dict( (ort_var.Index(), cpm_var) for (cpm_var, ort_var) in zip(assumptions, ort_assum_vars) )
            self.ort_model.ClearAssumptions() # because add just appends
            self.ort_model.AddAssumptions(ort_assum_vars)
            # workaround for a presolve with assumptions bug in ortools
            # https://github.com/google/or-tools/issues/2649
            self.ort_solver.parameters.keep_all_feasible_solutions_in_presolve = True

        # set additional keyword arguments in sat_parameters.proto
        for (kw, val) in kwargs.items():
            setattr(self.ort_solver.parameters, kw, val)

        if 'log_search_progress' in kwargs and hasattr(self.ort_solver, "log_callback"):
            # ortools>9.0, for IPython use, force output redirecting
            # see https://github.com/google/or-tools/issues/1903
            self.ort_solver.log_callback = print

        if solution_callback is None:
            ort_status = self.ort_solver.Solve(self.ort_model)
        else:
            try:
                # from ortools 9.0 onwards
                ort_status = self.ort_solver.Solve(self.ort_model, solution_callback=solution_callback)
            except TypeError:
                # ortools < 9.0
                ort_status = self.ort_solver.SolveWithSolutionCallback(self.ort_model, solution_callback)

        return self._after_solve(ort_status)

    def _after_solve(self, ort_status):
        """
            To be called immediately after calling or-tools Solve() or SolveWithSolutionCallBack() or SearchForAllSolutions()
            Translate or-tools status, variable values and objective value to CPMpy corresponding things

            - ort_status, an or-tools 'status' value
        """
        from ortools.sat.python import cp_model as ort

        self.ort_status = ort_status

        # translate exit status
        self.cpm_status = SolverStatus(self.name)
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
        else: # another?
            raise NotImplementedError(self.ort_status) # a new status type was introduced, please report on github

        # translate runtime
        self.cpm_status.runtime = self.ort_solver.WallTime()

        # translate solution values (of original vars only)
        if self.ort_status == ort.FEASIBLE or self.ort_status == ort.OPTIMAL:
            # fill in variables
            for var in self.user_vars:
                var._value = self.ort_solver.Value(self.varmap[var])

        # translate objective
        objective_value = None
        if self.ort_model.HasObjective():
            objective_value = self.ort_solver.ObjectiveValue()

        return self._solve_return(self.cpm_status, objective_value)


    def get_core(self):
        from ortools.sat.python import cp_model as ort
        """
            For use with s.solve(assumptions=[...]). Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

            CPMpy will return only those variables that are False (in the UNSAT core)

            Note that there is no guarantee that the core is minimal, though this interface does upon up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!

            For pure or-tools example, see http://github.com/google/or-tools/blob/master/ortools/sat/samples/assumptions_sample_sat.py

            Requires or-tools >= 8.2!!!
        """
        assert (self.assumption_dict is not None), "get_core(): requires a list of assumption variables, e.g. s.solve(assumptions=[...])"
        assert (self.ort_status == ort.INFEASIBLE), "get_core(): solver must return UNSAT"

        # use our own dict because of VarIndexToVarProto(0) bug in ort 8.2
        assum_idx = self.ort_solver.SufficientAssumptionsForInfeasibility()

        return [self.assumption_dict[i] for i in assum_idx]

    def make_model(self, cpm_model):
        """
            Makes the ortools.sat.python.cp_model formulation out of 
            a CPMpy model (will do flattening and other transformations)

            Typically only needed for internal use
        """
        from ortools.sat.python import cp_model as ort

        # Constraint programming engine
        self.ort_model = ort.CpModel()

        # Transform into flattened model
        flat_model = flatten_model(cpm_model)

        # Create corresponding solver variables
        self.varmap = dict() # cppy var -> solver var
        for var in get_variables_model(flat_model):
            self.add_to_varmap(var)

        # Post the (flat) constraint expressions to the solver
        for con in flat_model.constraints:
            self.post_constraint(con)

        # Post the objective
        if flat_model.objective is None:
            pass # no objective, satisfaction problem
        else:
            obj = self.ort_numexpr(flat_model.objective)
            if flat_model.objective_max:
                self.ort_model.Maximize(obj)
            else:
                self.ort_model.Minimize(obj)

        return self.ort_model


    def add_to_varmap(self, cpm_var):
        """
        Add the CPMpy variable to the 'varmap' mapping,
        which maps CPMpy variables to or-tools variables

        Typically only needed for internal use
        """
        if isinstance(cpm_var, _BoolVarImpl):
            revar = self.ort_model.NewBoolVar(str(cpm_var))
        elif isinstance(cpm_var, _IntVarImpl):
            revar = self.ort_model.NewIntVar(cpm_var.lb, cpm_var.ub, str(cpm_var))
        self.varmap[cpm_var] = revar


    def post_constraint(self, cpm_expr, reifiable=False):
        """
            Constraints are expected to be in 'flat normal form' (see flatten_model.py)

            While the normal form is divided in 'base', 'comparison' and 'reified', we
            here regroup it per CPMpy class

            Returns the posted ortools 'Constraint', so that it can be used in reification
            e.g. self.post_constraint(smth, reifiable=True).onlyEnforceIf(self.ort_var(bvar))
            
            - reifiable: ensures only constraints that support reification are returned

            Typically only needed for internal use
        """
        # Base case: Boolean variable
        if isinstance(cpm_expr, _BoolVarImpl):
            return self.ort_model.AddBoolOr( [self.ort_var(cpm_expr)] )
        
        # Comparisons: including base (vars), numeric comparison and reify/imply comparison
        elif isinstance(cpm_expr, Comparison):
            lhs,rhs = cpm_expr.args

            if isinstance(lhs, _BoolVarImpl) and cpm_expr.name == '==':
                # base: bvar == bvar|const
                lvar,rvar = map(self.ort_var, (lhs,rhs))
                return self.ort_model.Add(lvar == rvar)

            elif lhs.is_bool() and cpm_expr.name == '==':
                assert (not reifiable), "can not reify a reification"
                # reified case: boolexpr == var, split into two implications
                lexpr = cpm_expr.args[0]
                rvar = cpm_expr.args[1]
                # split in boolexpr -> var and var -> boolexpr
                self.post_constraint(lexpr.implies(rvar))
                self.post_constraint(rvar.implies(lexpr))

            else:
                # numeric (non-reify) comparison case
                rvar = self.ort_var(rhs)
                # lhs can be numexpr
                if isinstance(lhs, _NumVarImpl):
                    # simplest LHS case, a var
                    newlhs = self.ort_var(lhs)
                else:
                    if isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum'):
                        # a BoundedLinearExpression LHS, special case, like in objective
                        newlhs = self.ort_numexpr(lhs) 
                    elif cpm_expr.name == '==' and not reifiable:
                        newlhs = None
                        if lhs.name == 'abs':
                            return self.ort_model.AddAbsEquality(rvar, self.ort_var(lhs.args[0]))
                        elif lhs.name == 'mul':
                            return self.ort_model.AddMultiplicationEquality(rvar, self.ort_var_or_list(lhs.args))
                        elif lhs.name == 'mod':
                            # catch tricky-to-find ortools limitation
                            divisor = lhs.args[1]
                            if not is_num(divisor):
                                if divisor.lb <= 0 and divisor.ub >= 0:
                                    raise Exception(f"Expression '{lhs}': or-tools does not accept a 'modulo' operation where '0' is in the domain of the divisor {divisor}:domain({divisor.lb}, {divisor.ub}). Even if you add a constraint that it can not be '0'. You MUST use a variable that is defined to be higher or lower than '0'.")
                            return self.ort_model.AddModuloEquality(rvar, *self.ort_var_or_list(lhs.args))
                        elif lhs.name == 'div':
                            return self.ort_model.AddDivisionEquality(rvar, *self.ort_var_or_list(lhs.args))
                        elif lhs.name == 'min':
                            return self.ort_model.AddMinEquality(rvar, self.ort_var_or_list(lhs.args))
                        elif lhs.name == 'max':
                            return self.ort_model.AddMaxEquality(rvar, self.ort_var_or_list(lhs.args))
                        elif lhs.name == 'element':
                            # arr[idx]==rvar (arr=arg0,idx=arg1), ort: (idx,arr,target)
                            return self.ort_model.AddElement(self.ort_var(lhs.args[1]), self.ort_var_or_list(lhs.args[0]), rvar)
                        else:
                            raise NotImplementedError("Not a know supported ORTools left-hand-side '{}' {}".format(lhs.name, cpm_expr))
                    else:
                        # other equality than == 
                        # example: x*y > 10 :: x*y == aux, aux > 10
                        # creat the equality (will handle appropriate bounds)
                        (newvar, cons) = get_or_make_var(lhs)
                        self.add_to_varmap(newvar)
                        for con in cons:
                            # post the flattened constraints, including the 'lhs == newvar' one
                            # if this contains new auxiliary variables we will crash
                            self.post_constraint(con)
                        newlhs = self.ort_var(newvar)

                if newlhs is None:
                    pass # is already posted directly, eg a '=='
                elif cpm_expr.name == '==':
                    return self.ort_model.Add( newlhs == rvar)
                elif cpm_expr.name == '!=':
                    return self.ort_model.Add( newlhs != rvar )
                elif cpm_expr.name == '<=':
                    return self.ort_model.Add( newlhs <= rvar )
                elif cpm_expr.name == '<':
                    return self.ort_model.Add( newlhs < rvar )
                elif cpm_expr.name == '>=':
                    return self.ort_model.Add( newlhs >= rvar )
                elif cpm_expr.name == '>':
                    return self.ort_model.Add( newlhs > rvar )

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == '->' and \
             (not isinstance(cpm_expr.args[0], _BoolVarImpl) or \
              not isinstance(cpm_expr.args[1], _BoolVarImpl)):
                # reified case: var -> boolexpr, boolexpr -> var
                if isinstance(cpm_expr.args[0], _BoolVarImpl):
                    # var -> boolexpr, natively supported by or-tools
                    bvar = self.ort_var(cpm_expr.args[0])
                    # Special case for 'xor', which is not reifiable in ortools
                    if isinstance(cpm_expr.args[1], Operator) and cpm_expr.args[1].name == 'xor':
                        return self.post_constraint((sum(cpm_expr.args[1].args) == 1), reifiable=True).OnlyEnforceIf(bvar)
                    return self.post_constraint(cpm_expr.args[1], reifiable=True).OnlyEnforceIf(bvar)
                else:
                    # boolexpr -> var, have to convert to ~var -> ~boolexpr
                    negbvar = self.ort_var(cpm_expr.args[1]).Not()
                    negleft = negated_normal(cpm_expr.args[0])
                    return self.post_constraint(negleft, reifiable=True).OnlyEnforceIf(negbvar)

            else:
                # base 'and'/n, 'or'/n, 'xor'/n, '->'/2
                args = [self.ort_var(v) for v in cpm_expr.args]

                if cpm_expr.name == 'and':
                    return self.ort_model.AddBoolAnd(args)
                elif cpm_expr.name == 'or':
                    return self.ort_model.AddBoolOr(args)
                elif cpm_expr.name == 'xor':
                    return self.ort_model.AddBoolXOr(args)
                elif cpm_expr.name == '->':
                    return self.ort_model.AddImplication(args[0],args[1])
                else:
                    raise NotImplementedError("Not a know supported ORTools Operator '{}' {}".format(cpm_expr.name, cpm_expr))

        # rest: base (Boolean) global constraints
        else:
            args = [self.ort_var_or_list(v) for v in cpm_expr.args]

            if cpm_expr.name == 'alldifferent':
                return self.ort_model.AddAllDifferent(args) 
            elif cpm_expr.name == 'table':
                assert(len(args) == 2) # args = [array, table]
                return self.ort_model.AddAllowedAssignments(args[0], args[1])
            # TODO: NOT YET MAPPED: Automaton, Circuit, Cumulative,
            #    ForbiddenAssignments, Inverse?, NoOverlap, NoOverlap2D,
            #    ReservoirConstraint, ReservoirConstraintWithActive
            else:
                # global constraint not known, try generic decomposition
                dec = cpm_expr.decompose()
                if not dec is None:
                    flatdec = flatten_constraint(dec)

                    # collect and create new variables
                    for var in get_variables(flatdec):
                        if not var in self.varmap:
                            self.add_to_varmap(var)
                    # post decomposition
                    for con in flatdec:
                        self.post_constraint(con)
                    # XXX how to deal with reification of such a global??
                    # TODO: we would have to catch this at the time of the reification... (outer call)
                    return None # will throw error if used in reification...
                else:
                    raise NotImplementedError(cpm_expr) # if you reach this... please report on github

    def ort_var(self, cpm_var):
        """
            Uses 'varmap' to return the corresponding or-tools variable
            (or a constant)

            Typically only needed for internal use
        """
        if is_num(cpm_var):
            return cpm_var

        # decision variables, check in varmap
        if isinstance(cpm_var, NegBoolView):
            return self.varmap[cpm_var._bv].Not()
        elif isinstance(cpm_var, _NumVarImpl): # _BoolVarImpl is subclass of _NumVarImpl
            return self.varmap[cpm_var]

        raise NotImplementedError("Not a know var {}".format(cpm_var))

    def ort_var_or_list(self, cpm_expr):
        """
            like ort_var() but also works on lists of variables

            Typically only needed for internal use
        """
        if is_any_list(cpm_expr):
            return [self.ort_var_or_list(sub) for sub in cpm_expr]
        return self.ort_var(cpm_expr)


    def ort_numexpr(self, cpm_expr):
        """
            converts CPMpy subexpression into corresponding
            ORTools subexpressions (for in objective function and comparisons)
            Accepted by ORTools:
            - Decision variable: Var
            - Linear: sum([Var])                                   (CPMpy class 'Operator', name 'sum')
                      wsum([Const],[Var])                          (CPMpy class 'Operator', name 'wsum')

            Typically only needed for internal use
        """
        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl): # _BoolVarImpl is subclass of _NumVarImpl
            return self.ort_var(cpm_expr)

        # sum or (to be implemented: wsum)
        if isinstance(cpm_expr, Operator):
            args = [self.ort_var(v) for v in cpm_expr.args]
            if cpm_expr.name == 'sum':
                return sum(args) # OR-Tools supports this

        raise NotImplementedError("Not a know supported ORTools expression {}".format(cpm_expr))

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
except ImportError:
    pass # Ok, no ortools installed...
