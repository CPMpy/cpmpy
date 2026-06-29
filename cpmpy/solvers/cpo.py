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

from typing import Optional
import warnings
import time


from .solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback
from .. import DirectConstraint
from ..expressions.core import Expression, Comparison, Operator, BoolVal, NestedBoolExprLike
from ..expressions.globalconstraints import Cumulative, CumulativeOptional, GlobalConstraint, NoOverlap, NoOverlapOptional
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.utils import is_num, is_int, is_any_list, eval_comparison, argval, argvals, get_bounds, get_nonneg_args, implies
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree, decompose_objective
from ..transformations.safening import no_partial_functions



class CPM_cpo(SolverInterface):
    """
    Interface to CP Optimizer's Python API.

    Creates the following attributes (see parent constructor for more):

    - ``cpo_model``: object, CP Optimizers model object

    Documentation of the solver's own Python API: (all modeling functions)
    https://ibmdecisionoptimization.github.io/docplex-doc/cp/docplex.cp.modeler.py.html#module-docplex.cp.modeler

    """

    supported_global_constraints = frozenset({"alldifferent", 'inverse', 'table', 'indomain', "negative_table", "gcc",
                                              'cumulative', 'cumulative_optional', 'no_overlap', 'no_overlap_optional',
                                              "min", "max", "abs", "mul", "div", "mod", "pow", "element", "nvalue"})
    supported_reified_global_constraints = frozenset({"alldifferent", "table", "indomain", "negative_table"})

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
        from importlib.metadata import version, PackageNotFoundError
        try:
            import docplex.cp as docp
            cpo_version = docp.solver.solver.CpoSolver(docp.model.CpoModel()).get_solver_version()
            docplex_version = version("docplex")
            return f"{docplex_version}/{cpo_version}"
        except (PackageNotFoundError, ModuleNotFoundError):
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: Model(), a CPMpy Model() (optional)
            subsolver: str, name of a subsolver (optional)
        """
        if not self.installed():
            raise ModuleNotFoundError("CPM_cpo: Install the python package 'cpmpy[cpo]' to use this solver interface.")

        if not self.license_ok():
            raise ModuleNotFoundError("CPM_cpo: You also need to install the CPLEX Optimization Studio to use this solver. "
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
    
    def solve(self, time_limit:Optional[float]=None, solution_callback=None, display:Optional[Callback]=None, **kwargs):
        """
            Call the CP Optimizer solver

            Arguments:
                time_limit (float, optional):   maximum solve time in seconds
                solution_callback:              a ``docplex.cp.solver.solver_listener.CpoSolverListener`` or ``docplex.cp.solver.cpo_callback.CpoCallback`` object, or a list thereof
                                                takes precedence over ``display`` when both are set.
                display:                        generic solution callback for use during optimization.
                                                either a list of CPMpy expressions, OR a callback function which
                                                gets called after the variable-value mapping of the intermediate solution.
                                                default/None: nothing is displayed
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
        )

        callback = None
        if solution_callback is not None:
            callback = solution_callback
            if not isinstance(callback, list):
                callback = [callback]
        elif display is not None:
            callback = [CpoSolutionPrinter(self, display)]

        if callback is not None:
            for cb in callback:
                if isinstance(cb, CpoSolverListener):
                    self.cpo_solver.add_listener(cb)
                    # By default `solve()` only notifies listeners once with the final/best result.
                    # Enable search_next mode so listeners are warned about every intermediate solution.
                    self.cpo_solver.set_solve_with_search_next(True)
                if isinstance(cb, CpoCallback):
                    self.cpo_solver.add_callback(cb)

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

    def solveAll(self, display:Optional[Callback]=None, time_limit:Optional[float]=None, solution_limit:Optional[int]=None, call_from_model=False, **kwargs):
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
            self.print_display(display)

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
            or returns a constant if the variable is a constant
        """

        if isinstance(cpm_var, _NumVarImpl):
            name = cpm_var.name
            revar = self._varmap.get(name)
            if revar is not None:
                return revar

            # not yet created, make a new solver var
            docp = self.get_docp()
            if cpm_var.is_bool():
                if isinstance(cpm_var, NegBoolView):
                    raise ValueError("Negative literals cannot be created in `solver_var()` for cpo, use `_cpo_expr()` instead")
                else:
                    # note that a binary var is an integer var with domain (0,1), you cannot do boolean operations on it.
                    revar = docp.expression.binary_var(name)
            else:
                revar = docp.expression.integer_var(min=cpm_var.lb, max=cpm_var.ub, name=name)
            
            self._varmap[name] = revar
             # ensure the model also has the variable, just creating the variable is not enough
            self.cpo_model.add(revar >= cpm_var.lb) 
            return revar

        if is_int(cpm_var):  # shortcut, eases posting constraints
            return cpm_var

        raise NotImplementedError("Not a known var {}".format(cpm_var))

    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            ``objective()`` can be called multiple times, only the last one is stored

            .. note::
            
                technical side note: any constraints created during conversion of the objective are permanently posted to the solver
        """

        # save user variables
        get_variables(expr, self.user_vars)

        obj, decomp_cons = decompose_objective(expr,
                                               supported=self.supported_global_constraints,
                                               supported_reified=self.supported_reified_global_constraints,
                                               csemap=self._csemap)
        self.add(decomp_cons)

        dom = self.get_docp().modeler
        if self.has_objective():
            self.cpo_model.remove(self.cpo_model.get_objective_expression())

        cpo_obj = self._cpo_expr(obj)
        if minimize:
            self.cpo_model.add(dom.minimize(cpo_obj))
        else:
            self.cpo_model.add(dom.maximize(cpo_obj))

    def has_objective(self):
        return self.cpo_model.get_objective() is not None

    # `add()` first calls `transform()`
    def transform(self, cpm_expr: NestedBoolExprLike) -> list[Expression]:
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the :ref:`Adding a new solver` docs on readthedocs for more information.

            :param cpm_expr: CPMpy expression, or list thereof
            :type cpm_expr: NestedBoolExprLike

            :return: list of Expression
        """
        # apply transformations
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel=frozenset({"nd_element"}))
        cpm_cons = decompose_in_tree(cpm_cons,
                                     supported=self.supported_global_constraints,
                                     supported_reified=self.supported_reified_global_constraints,
                                     csemap=self._csemap)
        # no flattening required
        return cpm_cons

    def add(self, cpm_expr: NestedBoolExprLike) -> "CPM_cpo":
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

            :param cpm_expr: CPMpy expression, or list thereof
            :type cpm_expr: NestedBoolExprLike

            :return: self
        """

        # add new user vars to the set
        get_variables(cpm_expr, collect=self.user_vars)

        for cpm_con in self.transform(cpm_expr):
            # translate each expression tree, then post straight away
            cpo_con = self._cpo_expr(cpm_con, boolexpr=True)
            self.cpo_model.add(cpo_con)

        return self
    __add__ = add  # avoid redirect in superclass

    def _cpo_expr(self, cpm_con, boolexpr=False):
        """
            CP Optimizer supports nested expressions,
            so we recursively translate our expressions to theirs.

            Accepts single constraints or a list thereof, return type changes accordingly.

        """
        dom = self.get_docp().modeler
        if is_any_list(cpm_con):
            # arguments can be lists, assume boolexpr is same for all arguments
            return [self._cpo_expr(con, boolexpr=boolexpr) for con in cpm_con]

        elif isinstance(cpm_con, BoolVal):
            return cpm_con.args[0]

        elif is_num(cpm_con):
            return cpm_con

        elif isinstance(cpm_con, _NumVarImpl): # handle variables
            if isinstance(cpm_con, NegBoolView):
                if boolexpr:
                    return self.solver_var(cpm_con._bv) == 0
                else:
                    return 1 - self.solver_var(cpm_con._bv)
            elif cpm_con.is_bool() and boolexpr:
                return self.solver_var(cpm_con) == 1
            # else: integer variable, or boolean variable used as integer
            return self.solver_var(cpm_con)

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        elif isinstance(cpm_con, Operator):
            arity, _ = Operator.allowed[cpm_con.name]
            # 'and'/n, 'or'/n, '->'/2
            if cpm_con.name == 'and':
                return dom.logical_and(self._cpo_expr(cpm_con.args, boolexpr=True))
            elif cpm_con.name == 'or':
                return dom.logical_or(self._cpo_expr(cpm_con.args, boolexpr=True))
            elif cpm_con.name == '->':
                return dom.if_then(*self._cpo_expr(cpm_con.args, boolexpr=True))
            elif cpm_con.name == 'not':
                return dom.logical_not(self._cpo_expr(cpm_con.args[0], boolexpr=True))

            # 'sum'/n, 'wsum'/2
            elif cpm_con.name == 'sum':
                return dom.sum(self._cpo_expr(cpm_con.args))
            elif cpm_con.name == 'wsum':
                w = cpm_con.args[0]
                x = self._cpo_expr(cpm_con.args[1])
                return dom.scal_prod(w,x)

            # 'sub'/2
            elif cpm_con.name == 'sub':
                x, y = self._cpo_expr(cpm_con.args)
                return x - y
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
            elif cpm_con.name == "cumulative" or cpm_con.name == "cumulative_optional":
                if cpm_con.name == "cumulative":
                    is_present = None
                    if len(cpm_con.args) == 4:
                        start, dur, demand, capacity = cpm_con.args
                        end = None
                    else:
                        start, dur, end, demand, capacity = cpm_con.args
                
                elif cpm_con.name == "cumulative_optional":
                    if len(cpm_con.args) == 5:
                        start, dur, demand, capacity, is_present = cpm_con.args
                        end = None
                    else:
                        start, dur, end, demand, capacity, is_present = cpm_con.args

                tasks, cons = self._make_tasks(start, dur, end, is_present)

                # usage constraints
                demand_lst, demand_cons = get_nonneg_args(demand, is_present)
                cons += self._cpo_expr(demand_cons)

                total_usage = []
                for i, (task, h) in enumerate(zip(tasks, demand_lst)):
                    if task is None: # can happen with 0 duration tasks
                        continue
                    else:
                        task_demand = dom.pulse(task, get_bounds(h))
                        if is_present is not None:
                            cons += [dom.if_then(self._cpo_expr(is_present[i], boolexpr=True),
                                                 self._cpo_expr(h) == dom.height_at_start(task, task_demand))]
                        else:
                            cons += [self._cpo_expr(h) == dom.height_at_start(task, task_demand)]
                        total_usage.append(task_demand)
               
                cons += [dom.sum(total_usage) <= self._cpo_expr(capacity)]
                return cons
            elif cpm_con.name == "no_overlap":
                if len(cpm_con.args) == 2:
                    start, dur = cpm_con.args
                    end = None
                else:
                    start, dur, end = cpm_con.args
                tasks, cons = self._make_tasks(start, dur, end, None)
                return cons + [dom.no_overlap(tasks)]
            
            elif cpm_con.name == "no_overlap_optional":
                if len(cpm_con.args) == 3:
                    start, dur, is_present = cpm_con.args
                    end = None
                else:
                    start, dur, end, is_present = cpm_con.args

                tasks, cons = self._make_tasks(start, dur, end, is_present)
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
            elif cpm_con.name == "mul":
                x, y = self._cpo_expr(cpm_con.args)
                return x * y
            elif cpm_con.name == "div":
                x,y = self._cpo_expr(cpm_con.args)
                return x // y
            elif cpm_con.name == "mod":
                x,y = self._cpo_expr(cpm_con.args)
                return x % y
            elif cpm_con.name == "pow":
                x,y = self._cpo_expr(cpm_con.args)
                return x ** y

        raise NotImplementedError("CP Optimizer: constraint not (yet) supported", cpm_con)

    def _make_tasks(self, start, dur, end, is_present):
        """
            Helper function to create list of task objects and additional constraints enforcing task-relation
        """
        if end is None:
            end = [None for _ in range(len(start))] # easier to handle the task-making below

        # CPO crashes if size of interval is negative
        dur, dur_cons = get_nonneg_args(dur, is_present)
        extra_cons = self._cpo_expr(dur_cons)

        if is_present is None:
            is_present = [None] * len(start) # eases handling below

        tasks = []
        for s, d, e, p in zip(start, dur, end, is_present):
            task, task_cons = self._make_task(s, d, e, p)
            tasks.append(task)
            extra_cons += task_cons
        return tasks, extra_cons

    def _make_task(self, start, dur, end, is_present):
        """
            Helper function to create a task object and additional constraints enforcing task-relation
        """
        assert get_bounds(dur)[0] >= 0, "cpo does not support intervals with negative duration, use `utils.get_nonneg_args` first"
        dom = self.get_docp().modeler
        docp = self.get_docp()
        is_optional = is_present is not None
        if not is_optional:
            is_present = BoolVal(True) # eases handling below

        lb, ub = get_bounds(dur)
        extra_cons = []
        if lb == 0 == ub:
            if end is None: # nothing to enforce
                return None, []
            return None, extra_cons + self._cpo_expr([implies(is_present, start == end)]) # no task, just enforce 0 duration

        # Normal setting
        if end is None: # no end provided by user
            task = docp.expression.interval_var(start=get_bounds(start), size=get_bounds(dur), end=get_bounds(start+dur), optional=is_optional)
            extra_cons += [dom.if_then(dom.presence_of(task), dom.start_of(task) == self._cpo_expr(start)),
                           dom.if_then(dom.presence_of(task), dom.size_of(task) == self._cpo_expr(dur))]
            if is_optional: # enforce presence of task
                extra_cons += [dom.presence_of(task) == self._cpo_expr(is_present, boolexpr=True)]
            return task, extra_cons
        else:
            task = docp.expression.interval_var(start=get_bounds(start), size=get_bounds(dur), end=get_bounds(end), optional=is_optional)
            extra_cons += [dom.if_then(dom.presence_of(task), dom.start_of(task) == self._cpo_expr(start)),
                           dom.if_then(dom.presence_of(task), dom.size_of(task) ==  self._cpo_expr(dur)),
                           dom.if_then(dom.presence_of(task), dom.end_of(task) == self._cpo_expr(end))]
            if is_optional: # enforce presence of task
                extra_cons += [dom.presence_of(task) == self._cpo_expr(is_present, boolexpr=True)]
            return task, extra_cons


# solvers are optional, so this file should be interpretable
# even if cpo is not installed...
try:
    from docplex.cp.solver.cpo_callback import CpoCallback, EVENT_SOLUTION
    from docplex.cp.solver.solver_listener import CpoSolverListener
    from docplex.cp.solver.solver import CpoSolver, CpoSolveResult
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

        def result_found(self, solver: CpoSolver, sres: CpoSolveResult):
            """Keep track of solution count, invoked for each solution"""
            if not sres.is_new_solution():
                return # ignore non-new solutions

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
        def __init__(self, solver: CPM_cpo, display:Optional[Callback]=None, solution_limit:Optional[int]=None, verbose:bool=False):
            super().__init__(verbose)
            self._solution_limit = solution_limit
            # we only need the cpmpy->solver varmap from the solver
            self._varmap = solver._varmap
            # identify which variables to populate with their values
            self._cpm_vars = []
            self._display = display
            if isinstance(display, Expression) or is_any_list(display):
                self._cpm_vars = get_variables(display)
            elif callable(display):
                # might use any, so populate all (user) variables with their values
                self._cpm_vars = solver.user_vars
            self._cpm_solver = solver

        def result_found(self, solver:CpoSolver, sres:CpoSolveResult):
            
            if not sres.is_new_solution():
                return # ignore non-new solutions

            if len(self._cpm_vars):
                # populate values before printing
                for cpm_var in self._cpm_vars:
                    sol_var = self._varmap[cpm_var.name]
                    if isinstance(cpm_var, _BoolVarImpl):
                        cpm_var._value = bool(sres.get_var_solution(sol_var).get_value())
                    elif isinstance(cpm_var, _IntVarImpl):
                        cpm_var._value = sres.get_var_solution(sol_var).get_value()
                    else:
                        raise NotImplementedError(f"Unexpected variable type {type(cpm_var)}")

                self._cpm_solver.print_display(self._display)

            # check for count limit
            if self.solution_count() == self._solution_limit:
                self.end_solve()

except ImportError:
    pass  # Ok, no cpo installed...
