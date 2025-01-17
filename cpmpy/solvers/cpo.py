import docplex.cp.parameters
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from .. import DirectConstraint
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl
from ..expressions.utils import is_num, is_any_list, is_boolexpr, eval_comparison, argval, argvals
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.reification import reify_rewrite, only_bv_reifies

"""
    Interface to CP Optimizers API

    <some information on the solver>

    Documentation of the solver's own Python API: (all modeling functions)
    https://ibmdecisionoptimization.github.io/docplex-doc/cp/docplex.cp.modeler.py.html#module-docplex.cp.modeler

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_template
"""

class CPM_cpo(SolverInterface):
    """
    Interface to TEMPLATE's API

    Requires that the 'TEMPLATEpy' python package is installed:
    $ pip install TEMPLATEpy

    See detailed installation instructions at:
    <URL to detailed solver installation instructions, if any>

    Creates the following attributes (see parent constructor for more):
    - tpl_model: object, TEMPLATE's model object
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import docplex.cp as docp
            return True
        except ModuleNotFoundError:
            return False


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: str, name of a subsolver (optional)
        """
        if not self.supported():
            raise Exception("CPM_cpo: Install the python package 'docplex'")

        import docplex.cp.model
        assert subsolver is None

        # initialise the native solver object
        # [GUIDELINE] we commonly use 3-letter abbrivations to refer to native objects:
        #           OR-tools uses ort_solver, Gurobi grb_solver, Exact xct_solver...
        self.cpo_model = docplex.cp.model.CpoModel()

        # initialise everything else and post the constraints/objective
        # [GUIDELINE] this superclass call should happen AFTER all solver-native objects are created.
        #           internally, the constructor relies on __add__ which uses the above solver native object(s)
        super().__init__(name="cpo", cpm_model=cpm_model)


    def solve(self, time_limit=None, **kwargs):
        """
            Call the CP Optimizer solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            # [GUIDELINE] Please document key solver arguments that the user might wish to change
            #       for example: assumptions=[x,y,z], log_output=True, var_ordering=3, num_cores=8, ...
            # [GUIDELINE] Add link to documentation of all solver parameters
        """

        # [GUIDELINE] if your solver supports solving under assumptions, add `assumptions` as argument in header
        #       e.g., def solve(self, time_limit=None, assumptions=None, **kwargs):
        #       then translate assumptions here; assumptions are a list of Boolean variables or NegBoolViews

        # call the solver, with parameters
        self.cpo_result = self.cpo_model.solve(LogVerbosity='Quiet', TimeLimit=time_limit, **kwargs)
        # [GUIDELINE] consider saving the status as self.TPL_status so that advanced CPMpy users can access the status object.
        #       This is mainly useful when more elaborate information about the solve-call is saved into the status

        # new status, translate runtime
        self.cpo_status = self.cpo_result.get_solve_status()
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.cpo_result.get_solve_time() # wallclock time in (float) seconds

        # translate solver exit status to CPMpy exit status
        if self.cpo_status == "Feasible":
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif self.cpo_status == "Infeasible":
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif self.cpo_status == "Unknown":
            # can happen when timeout is reached...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        elif self.cpo_status == "Optimal":
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        elif self.cpo_status == "JobFailed":
            self.cpm_status.exitstatus = ExitStatus.ERROR
        elif self.cpo_status == "JobAborted":
            self.cpm_status.exitstatus = ExitStatus.NOT_RUN
        else:  # another?
            raise NotImplementedError(self.cpo_status)  # a new status type was introduced, please report on GitHub

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                if isinstance(cpm_var,_BoolVarImpl):
                    # because cp optimizer does not have boolean variables we use an integer var x with domain 0, 1
                    # and then replace a boolvar by x == 1
                    sol_var = sol_var.children[0]
                    cpm_var._value = bool(self.cpo_result.get_var_solution(sol_var).get_value())
                else:
                    cpm_var._value = self.cpo_result.get_var_solution(sol_var).get_value()

            # translate objective, for optimisation problems only
            if self.has_objective():
                self.objective_value_ = self.cpo_result.get_objective_value()

        else:  # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var.clear()

        return has_sol

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            A shorthand to (efficiently) compute all (optimal) solutions, map them to CPMpy and optionally display the solutions.

            If the problem is an optimization problem, returns only optimal solutions.

           Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - call_from_model: whether the method is called from a CPMpy Model instance or not
                - any other keyword argument

            Returns: number of solutions found
        """

        #cpo_solver = self.cpo_model.start_search(TimeLimit=time_limit, SolutionLimit=solution_limit, LogVerbosity='Quiet', **kwargs)
        solution_count = 0
        if solution_limit is None:
            solution_limit = 99999
        while solution_count < solution_limit:
            cpo_result = self.solve(time_limit=time_limit, **kwargs)
            if not cpo_result:
                break
            solution_count += 1
            '''if cpo_result.is_solution_optimal():
                print('optimal')
            if not cpo_result.is_new_solution():
                break
            solution_count += 1'''
            if self.has_objective():
                # only find all optimal solutions
                self.cpo_model.add(self.cpo_model.get_objective_expression().children[0] == self.objective_value_)
            solvars = []
            vals = []
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                cpm_value = cpm_var._value
                if isinstance(cpm_var, _BoolVarImpl):
                    # because cp optimizer does not have boolean variables we use an integer var x with domain 0, 1
                    # and then replace a boolvar by x == 1
                    sol_var = sol_var.children[0]
                    cpm_value = int(cpm_value)
                solvars.append(sol_var)
                vals.append(cpm_value)
            self.cpo_model.add(docplex.cp.modeler.forbidden_assignments(solvars, [vals]))
            if display is not None:
                if isinstance(display, Expression):
                    print(argval(display))
                elif isinstance(display, list):
                    print(argvals(display))
                else:
                    display()  # callback
        return solution_count
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
            return (self.solver_var(cpm_var._bv) == 0)

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                # note that a binary var is an integer var with domain (0,1), you cannot do boolean operations on it.
                # we should add == 1 to turn it into a boolean expression
                revar = docplex.cp.expression.binary_var(str(cpm_var)) == 1
            elif isinstance(cpm_var, _IntVarImpl):
                revar = docplex.cp.expression.integer_var(min=cpm_var.lb, max=cpm_var.ub, name=str(cpm_var))
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
        if self.has_objective():
            self.cpo_model.remove(self.cpo_model.get_objective_expression())
        expr = self._cpo_expr(expr)
        if minimize:
            self.cpo_model.add(docplex.cp.modeler.minimize(expr))
        else:
            self.cpo_model.add(docplex.cp.modeler.maximize(expr))

    def has_objective(self):
        return self.cpo_model.get_objective() is not None

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
               return self.TPL_solver.sum(self.solver_vars(cpm_expr.args))
           elif cpm_expr.name == 'wsum':
               weights, vars = cpm_expr.args
               return self.TPL_solver.weighted_sum(weights, self.solver_vars(vars))
           # [GUIDELINE] or more fancy ones such as max
           #        be aware this is not the Maximum CONSTRAINT, but rather the Maximum NUMERICAL EXPRESSION
           elif cpm_expr.name == "max":
               return self.TPL_solver.maximum_of_vars(self.solver_vars(cpm_expr.args))
           # ...
        raise NotImplementedError("TEMPLATE: Not a known supported numexpr {}".format(cpm_expr))


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
        # count is only supported with a constant to be counted, so we decompose
        supported = {"alldifferent", 'inverse', 'nvalue', 'element', 'table', 'indomain',
                     "negative_table", "gcc", 'max', 'min', 'abs'}
        supported_reified = {"alldifferent", 'nvalue', 'element', 'table', 'indomain', 'max', 'min',
                     "negative_table", 'abs'}
        cpm_cons = decompose_in_tree(cpm_cons, supported=supported, supported_reified=supported_reified)
        '''cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !='''
        # ...
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

        for cpm_con in self.transform(cpm_expr):
            # translate each expression tree, then post straight away
            cpo_con = self._cpo_expr(cpm_con)
            self.cpo_model.add(cpo_con)

        return self

    def _cpo_expr(self, cpm_con):
        """
            CP Optimizer supports nested expressions,
            so we recursively translate our expressions to theirs.

            Accepts single constraints or a list thereof, return type changes accordingly.

        """
        if is_any_list(cpm_con):
            # arguments can be lists
            return [self._cpo_expr(con) for con in cpm_con]

        elif isinstance(cpm_con, BoolVal):
            return cpm_con.args[0]

        elif is_num(cpm_con):
            return cpm_con

        elif isinstance(cpm_con, _NumVarImpl):
            return self.solver_var(cpm_con)

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        elif isinstance(cpm_con, Operator):
            arity, _ = Operator.allowed[cpm_con.name]
            # 'and'/n, 'or'/n, '->'/2
            if cpm_con.name == 'and':
                return docplex.cp.modeler.logical_and(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == 'or':
                return docplex.cp.modeler.logical_or(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == '->':
                return docplex.cp.modeler.if_then(*self._cpo_expr(cpm_con.args))
            elif cpm_con.name == 'not':
                return docplex.cp.modeler.logical_not(self._cpo_expr(cpm_con.args[0]))

            # 'sum'/n, 'wsum'/2
            elif cpm_con.name == 'sum':
                return docplex.cp.modeler.sum(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == 'wsum':
                w = cpm_con.args[0]
                x = self._cpo_expr(cpm_con.args[1])
                return docplex.cp.modeler.scal_prod(w,x)

            # 'sub'/2, 'mul'/2, 'div'/2, 'pow'/2, 'm2od'/2
            elif arity == 2 or cpm_con.name == "mul":
                assert len(cpm_con.args) == 2, "Currently only support multiplication with 2 vars"
                x, y = self._cpo_expr(cpm_con.args)
                if cpm_con.name == 'sub':
                    return x - y
                elif cpm_con.name == "mul":
                    return x * y
                elif cpm_con.name == "div":
                    return x // y
                elif cpm_con.name == "pow":
                    return x ** y
                elif cpm_con.name == "mod":
                    return x % y
            # '-'/1
            elif cpm_con.name == "-":
                return -self._cpo_expr(cpm_con.args[0])

            else:
                raise NotImplementedError(f"Operator {cpm_con} not (yet) implemented for CP Optimizer, "
                                          f"please report on github if you need it")

        # Comparisons (just translate the subexpressions and re-post)
        elif isinstance(cpm_con, Comparison):
            lhs, rhs = self._cpo_expr(cpm_con.args)

            # post the comparison
            return eval_comparison(cpm_con.name, lhs, rhs)
        # rest: base (Boolean) global constraints
        elif isinstance(cpm_con, GlobalConstraint):
            if cpm_con.name == 'alldifferent':
                return docplex.cp.modeler.all_diff(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == "gcc":
                vars, vals, occ = self._cpo_expr(cpm_con.args)
                cons = [docplex.cp.modeler.distribute(occ, vars, vals)]
                if cpm_con.closed:  # not supported by cpo, so post separately
                    cons += [docplex.cp.modeler.allowed_assignments(v, vals) for v in vars]
                return cons
            elif cpm_con.name == "inverse":
                x, y = self._cpo_expr(cpm_con.args)
                return docplex.cp.modeler.inverse(x, y)
            elif cpm_con.name == "table":
                arr, table = self._cpo_expr(cpm_con.args)
                return docplex.cp.modeler.allowed_assignments(arr, table)
            elif cpm_con.name == "indomain":
                expr, arr = self._cpo_expr(cpm_con.args)
                return docplex.cp.modeler.allowed_assignments(expr, arr)
            elif cpm_con.name == "negative_table":
                arr, table = self._cpo_expr(cpm_con.args)
                return docplex.cp.modeler.forbidden_assignments(arr, table)
            # a direct constraint, make with cpo (will be posted to it by calling function)
            elif isinstance(cpm_con, DirectConstraint):
                return cpm_con.callSolver(self, self.cpo_model)

            else:
                try:
                    cpo_global = getattr(docplex.cp.modeler, cpm_con.name)
                    return cpo_global(self._cpo_expr(cpm_con.args))  # works if our naming is the same
                except AttributeError:
                    raise ValueError(f"Global constraint {cpm_con} not known in CP Optimizer, please report on github.")

        elif isinstance(cpm_con, GlobalFunction):
            if cpm_con.name == "element":
                return docplex.cp.modeler.element(*self._cpo_expr(cpm_con.args))
            elif cpm_con.name == "min":
                return docplex.cp.modeler.min(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == "max":
                return docplex.cp.modeler.max(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == "abs":
                return docplex.cp.modeler.abs(self._cpo_expr(cpm_con.args)[0])
            elif cpm_con.name == "nvalue":
                return docplex.cp.modeler.count_different(self._cpo_expr(cpm_con.args))



        raise NotImplementedError("CP Optimizer: constraint not (yet) supported", cpm_con)
