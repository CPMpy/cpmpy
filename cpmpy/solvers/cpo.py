"""
    Interface to CP Optimizer's Python API.

    CP Optimizer, also a feature of IBM ILOG Optimization Studio, is a software library of constraint programming tools 
    supporting constraint propagation, domain reduction, and highly optimized solution search.

    Always use :func:`cp.SolverLookup.get("cpo") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'docplex' python package is installed:

    .. code-block:: console
    
        $ pip install docplex

    docplex documentation:
    https://ibmdecisionoptimization.github.io/docplex-doc/

    You will also need to install CPLEX Optimization Studio from IBM's website,
    and add the location of the CP Optimizer binary to your path.
    There is a free community version available.
    https://www.ibm.com/products/ilog-cplex-optimization-studio

    See detailed installation instructions at:
    https://www.ibm.com/docs/en/icos/22.1.2?topic=2212-installing-cplex-optimization-studio

    Academic license:
    https://community.ibm.com/community/user/ai-datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
       :nosignatures:

        CPM_cpo
"""

import time
from typing import Optional
import warnings
import pkg_resources

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from .. import DirectConstraint
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.utils import is_num, is_any_list, eval_comparison, argval, argvals, get_bounds
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.safening import no_partial_functions



class CPM_cpo(SolverInterface):
    """
    Interface to CP Optimizer's Python API.

    Creates the following attributes (see parent constructor for more):

    - ``cpo_model``: object, CP Optimizers model object

    Documentation of the solver's own Python API: (all modeling functions)
    https://ibmdecisionoptimization.github.io/docplex-doc/cp/docplex.cp.modeler.py.html#module-docplex.cp.modeler

    """

    _docp = None  # Static attribute to hold the docplex.cp module

    @classmethod
    def get_docp(cls):
        if cls._docp is None:
            import docplex.cp as docp  # Import only once
            cls._docp = docp
        return cls._docp

    @staticmethod
    def supported():
        return CPM_cpo.installed() and CPM_cpo.license_ok()

    @staticmethod
    def installed():
        # try to import the package
        try:
            import docplex.cp as docp
            return True
        except ModuleNotFoundError:
            return False

    @staticmethod
    def license_ok():
        if not CPM_cpo.installed():
            warnings.warn(f"License check failed, python package 'docplex' is not installed! Please check 'CPM_cpo.installed()' before attempting to check license.")
            return False
        else:
            try:
                from docplex.cp.model import CpoModel
                mdl = CpoModel()
                mdl.solve(LogVerbosity='Quiet')
                return True
            except:
                return False
            
    @staticmethod
    def version() -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.

        For CPO, two version numbers get returned: ``<docplex version>/<solver version>``
        """
        try:
            import docplex.cp as docp
            s = docp.solver.solver.CpoSolver(docp.model.CpoModel())
            return f"{pkg_resources.get_distribution('docplex').version}/{s.get_solver_version()}"
        except (pkg_resources.DistributionNotFound, ModuleNotFoundError):
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: Model(), a CPMpy Model() (optional)
            subsolver: str, name of a subsolver (optional)
        """
        if not self.installed():
            raise Exception("CPM_cpo: Install the python package 'docplex'")

        if not self.license_ok():
            raise Exception("You need to install the CPLEX Optimization Studio to use this solver. "
                            "Also make sure that the binary is in your path")

        docp = self.get_docp()
        assert subsolver is None
        self.cpo_model = docp.model.CpoModel()
        super().__init__(name="cpo", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.cpo_model
    
    def solve(self, time_limit=None, solution_callback=None, **kwargs):
        """
            Call the CP Optimizer solver

            Arguments:
                time_limit (float, optional):   maximum solve time in seconds 
                solution_callback (an `docplex.cp.solver.solver_listener.CpoSolverListener` object):   CPMpy includes its own, namely `CpoSolutionCounter`. If you want to count all solutions, 
                                                                                                        don't forget to also add the keyword argument 'enumerate_all_solutions=True'.
                kwargs:                         any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:


            =============================   ============
            Argument                        Description
            =============================   ============
            LogVerbosity                    Determines the verbosity of the search log. Choose a value from  ['Quiet', 'Terse', 'Normal', 'Verbose']. Default value is 'Quiet'.
            OptimalityTolerance             This parameter sets an absolute tolerance on the objective value for optimization models. The value is a positive float. Default value is 1e-09.
            RelativeOptimalityTolerance     This parameter sets a relative tolerance on the objective value for optimization models. The optimality of a solution is proven if either of the two parameters' criteria is fulfilled.
            Presolve                        This parameter controls the presolve of the model to produce more compact formulations and to achieve more domain reduction. Possible values for this parameter are On (presolve is activated) and Off (presolve is deactivated).
                                            The value is a symbol in ['On', 'Off']. Default value is 'On'.
            Workers                         This parameter sets the number of workers to run in parallel to solve your model. The value is a positive integer. Default value is Auto. (Auto = use all available CPU cores)
            =============================   ============

            All solver parameters are documented here: https://ibmdecisionoptimization.github.io/docplex-doc/cp/docplex.cp.parameters.py.html#docplex.cp.parameters.CpoParameters

        """
        docp = self.get_docp()

        # ensure all vars are known to solver        
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)

        # call the solver, with parameters
        if 'LogVerbosity' not in kwargs:
            kwargs['LogVerbosity'] = 'Quiet'
        
        # set time limit
        if time_limit is not None and time_limit <= 0:
            raise ValueError("Time limit must be positive")
        
        # create solver object
        self.cpo_solver = docp.solver.solver.CpoSolver(
            self.cpo_model,
            TimeLimit=time_limit, 
            **kwargs, 
            listeners=[solution_callback] if solution_callback is not None else None
        )

        self.cpo_result = self.cpo_solver.solve()

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
                display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping.
                         Default is None, meaning nothing is displayed.
                time_limit: Stop after this many seconds. Default is None.
                solution_limit: Stop after this many solutions. Default is None.
                call_from_model: Whether the method is called from a CPMpy Model instance or not.
                **kwargs: Any other keyword arguments.

            Returns:
                int: Number of solutions found.
        """

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)

        docp = self.get_docp()
        solution_count = 0

        # TODO: convert to use 'start_search' and solution callback handlers
        start = time.time()
        while ((time_limit is None) or (time_limit > 0)) and self.solve(time_limit=time_limit, **kwargs):

            # display if needed
            if display is not None:
                if isinstance(display, Expression):
                    print(argval(display))
                elif isinstance(display, list):
                    print(argvals(display))
                else:
                    display()  # callback

            # count and stop
            solution_count += 1
            if solution_count == solution_limit:
                break

            if self.has_objective():
                # only find all optimal solutions
                self.cpo_model.add(self.cpo_model.get_objective_expression().children[0] == self.objective_value_)

            # add nogood on the user variables
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
            self.cpo_model.add(docp.modeler.forbidden_assignments(solvars, [vals]))

            if time_limit is not None: # update remaining time
                time_limit -= self.status().runtime
        end = time.time()

        # update solver status
        self.cpm_status.runtime = end - start
        if solution_count:
            if solution_count == solution_limit:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            elif self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE:
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        # else: <- is implicit since nothing needs to update
        #     if self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE:
        #         self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        #     elif self.cpm_status.exitstatus == ExitStatus.UNKNOWN:
        #         self.cpm_status.exitstatus = ExitStatus.UNKNOWN

        return solution_count

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        docp = self.get_docp()
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
                revar = docp.expression.binary_var(str(cpm_var)) == 1
            elif isinstance(cpm_var, _IntVarImpl):
                revar = docp.expression.integer_var(min=cpm_var.lb, max=cpm_var.ub, name=str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar
            self.cpo_model.add(revar >= cpm_var.lb) # ensure the model also has the variable

        # return from cache
        return self._varmap[cpm_var]

    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            ``objective()`` can be called multiple times, only the last one is stored

            .. note::
            
                technical side note: any constraints created during conversion of the objective are permanently posted to the solver
        """
        dom = self.get_docp().modeler
        if self.has_objective():
            self.cpo_model.remove(self.cpo_model.get_objective_expression())
        expr = self._cpo_expr(expr)
        if minimize:
            self.cpo_model.add(dom.minimize(expr))
        else:
            self.cpo_model.add(dom.maximize(expr))

    def has_objective(self):
        return self.cpo_model.get_objective() is not None

    # `add()` first calls `transform()`
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
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel=frozenset({}))
        # count is only supported with a constant to be counted, so we decompose
        supported = {"alldifferent", 'inverse', 'nvalue', 'element', 'table', 'indomain',
                     "negative_table", "gcc", 'max', 'min', 'abs', 'cumulative', 'no_overlap'}
        supported_reified = {"alldifferent", 'table', 'indomain', "negative_table"} # global functions by default here
        cpm_cons = decompose_in_tree(cpm_cons, supported=supported, supported_reified=supported_reified, csemap=self._csemap)
        # no flattening required
        return cpm_cons

    def add(self, cpm_expr):
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
    __add__ = add  # avoid redirect in superclass

    def _cpo_expr(self, cpm_con):
        """
            CP Optimizer supports nested expressions,
            so we recursively translate our expressions to theirs.

            Accepts single constraints or a list thereof, return type changes accordingly.

        """
        dom = self.get_docp().modeler
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
                return dom.logical_and(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == 'or':
                return dom.logical_or(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == '->':
                return dom.if_then(*self._cpo_expr(cpm_con.args))
            elif cpm_con.name == 'not':
                return dom.logical_not(self._cpo_expr(cpm_con.args[0]))

            # 'sum'/n, 'wsum'/2
            elif cpm_con.name == 'sum':
                return dom.sum(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == 'wsum':
                w = cpm_con.args[0]
                x = self._cpo_expr(cpm_con.args[1])
                return dom.scal_prod(w,x)

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
                return dom.all_diff(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == "gcc":
                vars, vals, occ = self._cpo_expr(cpm_con.args)
                cons = [dom.distribute(occ, vars, vals)]
                if cpm_con.closed:  # not supported by cpo, so post separately
                    cons += [dom.allowed_assignments(v, vals) for v in vars]
                return cons
            elif cpm_con.name == "inverse":
                x, y = self._cpo_expr(cpm_con.args)
                return dom.inverse(x, y)
            elif cpm_con.name == "table":
                arr, table = self._cpo_expr(cpm_con.args)
                return dom.allowed_assignments(arr, table)
            elif cpm_con.name == "indomain":
                expr, arr = self._cpo_expr(cpm_con.args)
                return dom.allowed_assignments(expr, arr)
            elif cpm_con.name == "negative_table":
                arr, table = self._cpo_expr(cpm_con.args)
                return dom.forbidden_assignments(arr, table)
            elif cpm_con.name == "cumulative":
                start, dur, end, height, capacity = cpm_con.args
                docp = self.get_docp()
                total_usage = []
                cons = []
                for s, d, e, h in zip(start, dur, end, height):
                    bounds_d = get_bounds(d)
                    # Special case for tasks with duration 0
                    # -> cpo immediately returns UNSAT if done through tasks
                    if bounds_d[1] == bounds_d[0] == 0:
                        cpo_s, cpo_e = self.solver_vars([s, e])
                        cons += [cpo_s == cpo_e] # enforce 0 duration
                        # no restrictions on height due to zero duration and thus no contribution to capacity
                        continue
                    # Normal setting
                    cpo_s, cpo_d, cpo_e, cpo_h = self.solver_vars([s, d, e, h])                   
                    task = docp.expression.interval_var(start=get_bounds(s), size=get_bounds(d), end=get_bounds(e))
                    task_height = dom.pulse(task, get_bounds(h))
                    cons += [dom.start_of(task) == cpo_s, dom.size_of(task) == cpo_d, dom.end_of(task) == cpo_e]
                    cons += [cpo_h == dom.height_at_start(task, task_height)]
                    total_usage.append(task_height)
                cons += [dom.sum(total_usage) <= self.solver_var(capacity)]
                return cons
            elif cpm_con.name == "no_overlap":
                start, dur, end  = cpm_con.args
                docp = self.get_docp()
                cons = []
                tasks = []
                for s, d, e in zip(start, dur, end):
                    cpo_s, cpo_d, cpo_e = self.solver_vars([s, d, e])
                    task = docp.expression.interval_var(start=get_bounds(s), size=get_bounds(d), end=get_bounds(e))
                    tasks.append(task)
                    cons += [dom.start_of(task) == cpo_s, dom.size_of(task) == cpo_d, dom.end_of(task) == cpo_e]
                return cons + [dom.no_overlap(tasks)]
            # a direct constraint, make with cpo (will be posted to it by calling function)
            elif isinstance(cpm_con, DirectConstraint):
                return cpm_con.callSolver(self, self.cpo_model)

            else:
                try:
                    cpo_global = getattr(dom, cpm_con.name)
                    return cpo_global(self._cpo_expr(cpm_con.args))  # works if our naming is the same
                except AttributeError:
                    raise ValueError(f"Global constraint {cpm_con} not known in CP Optimizer, please report on github.")

        elif isinstance(cpm_con, GlobalFunction):
            if cpm_con.name == "element":
                return dom.element(*self._cpo_expr(cpm_con.args))
            elif cpm_con.name == "min":
                return dom.min(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == "max":
                return dom.max(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == "abs":
                return dom.abs(self._cpo_expr(cpm_con.args)[0])
            elif cpm_con.name == "nvalue":
                return dom.count_different(self._cpo_expr(cpm_con.args))

        raise NotImplementedError("CP Optimizer: constraint not (yet) supported", cpm_con)


# solvers are optional, so this file should be interpretable
# even if cpo is not installed...
try:
    from docplex.cp.solver.solver_listener import CpoSolverListener
    import time

    class CpoSolutionCounter(CpoSolverListener):
        """
        Native CP Optimizer callback for solution counting.

        It is based on cpo's built-in `CpoSolverListener`.

        use with CPM_cpo as follows:

        .. code-block:: python
            
            cb = CpoSolutionCounter()
            s.solve(solution_callback=cb)

        then retrieve the solution count with ``cb.solution_count()``

        Arguments:
            verbose (bool, default: False): whether to print info on every solution found 
    """

        def __init__(self, verbose=False):
            super().__init__()
            self.__solution_count = 0
            self.__verbose = verbose
            if self.__verbose:
                self.__start_time = time.time()

        def result_found(self, solver, sres):
            """Called on each new solution."""
            if self.__verbose:
                current_time = time.time()
                obj = sres.get_objective_value()
                print('Solution %i, time = %0.2f s, objective = %i' %
                      (self.__solution_count, current_time - self.__start_time, obj))
            self.__solution_count += 1

        def solution_count(self):
            """Returns the number of solutions found."""
            return self.__solution_count

    class CpoSolutionPrinter(CpoSolutionCounter):
        """
            Native CP Optimizer callback for solution printing.

            Subclasses :class:`CpoSolutionCounter`, see those docs too.

            Use with :class:`CPM_cpo` as follows:

            .. code-block:: python

                cb = CpoSolutionPrinter(s, display=vars)
                s.solve(solution_callback=cb)

            For multiple variables (single or NDVarArray), use:
            ``cb = CpoSolutionPrinter(s, display=[v, x, z])``.

            For a custom print function, use for example:
            
            .. code-block:: python

                def myprint():
                    print(f"x0={x[0].value()}, x1={x[1].value()}")
                
                cb = CpoSolutionPrinter(s, printer=myprint)

            Optionally retrieve the solution count with ``cb.solution_count()``.

            Arguments:
                verbose (bool, default = False): whether to print info on every solution found 
                display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                            default/None: nothing displayed
                solution_limit (default = None): stop after this many solutions 
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

        def result_found(self, solver, sres):
            """Called on each new solution."""
            if len(self._cpm_vars):
                # populate values before printing
                for cpm_var in self._cpm_vars:
                    # it might be an NDVarArray
                    if hasattr(cpm_var, "flat"):
                        for cpm_subvar in cpm_var.flat:
                            sol_var = self._varmap[cpm_subvar]
                            if isinstance(cpm_var,_BoolVarImpl):
                                sol_var = sol_var.children[0]
                                cpm_var._value = bool(sres.get_var_solution(sol_var).get_value())
                            else:
                                cpm_var._value = sres.get_var_solution(sol_var).get_value()
                    elif isinstance(cpm_var, _BoolVarImpl):
                        sol_var = self._varmap[cpm_subvar].children[0]
                        cpm_var._value = bool(sres.get_var_solution(sol_var).get_value())
                    else:
                        sol_var = self._varmap[cpm_subvar]
                        cpm_var._value = sres.get_var_solution(sol_var).get_value()

                if isinstance(self._display, Expression):
                    print(argval(self._display))
                elif isinstance(self._display, list):
                    # explicit list of expressions to display
                    print(argvals(self._display))
                else: # callable
                    self._display()

            # check for count limit
            if self.solution_count() == self._solution_limit:
                self.end_solve()

except ImportError:
    pass  # Ok, no cpo installed...
