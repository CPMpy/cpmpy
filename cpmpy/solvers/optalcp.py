"""
    Interface to OptalCP's Python API.

    OptalCP is a scheduling-oriented constraint programming solver with a native Python API.

    Always use :func:`cp.SolverLookup.get("optalcp") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires the `optalcp` Python package and an OptalCP solver binary.

    Using OptalCP through CPMpy requires a full or academic version, which can be installed through:

    .. code-block:: console

        $ pip install git+https://github.com/ScheduleOpt/optalcp-py-bin-academic@latest

    Visit https://dev.vilim.eu/docs/Quick%20Start/editions#installation-1 for more information on how to obtain non-preview versions of OptalCP.

    Documentation:
    https://optalcp.com/docs/

    ===============
    List of classes
    ===============

    .. autosummary::
       :nosignatures:

        CPM_optalcp
"""

from typing import Optional
import warnings

from .solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback
from .. import DirectConstraint
from ..exceptions import NotSupportedError
from ..expressions.core import Comparison, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.utils import is_num, is_np_int, is_np_bool, is_any_list, eval_comparison, get_bounds, get_nonneg_args, implies
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree, decompose_objective
from ..transformations.safening import no_partial_functions


class CPM_optalcp(SolverInterface):
    """
    Interface to OptalCP's Python API.

    Creates the following attributes (see parent constructor for more):

    - ``opt_model``: object, OptalCP model object

    Documentation of the solver's own Python API: (all modeling functions)
        https://dev.vilim.eu/python-api/index.html
    """

    supported_global_constraints = frozenset({
        "cumulative", "cumulative_optional", "no_overlap", "no_overlap_optional",
        "min", "max", "abs", "mul", "div",
    })
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def installed():
        # try to import the package
        try:
            import optalcp
            return True
        except ModuleNotFoundError:
            return False
        
    @staticmethod
    def license_ok():
        if not CPM_optalcp.installed():
            warnings.warn(
                "License check failed, python package 'optalcp' is not installed! "
                "Please check 'CPM_optalcp.installed()' before attempting to check license."
            )
            return False
        try:
            import optalcp
            mdl = optalcp.Model()
            bv = mdl.bool_var(name="dummy")
            mdl.enforce(bv)
            result = mdl.solve({"printLog": False})
            if result.nb_solutions == 0:
                return False
            # Without a valid (non-preview) license, OptalCP returns masked/garbage values.
            # A bool var enforced to True must be 1 if the license is valid.
            return result.solution.get_value(bv) == 1
        except Exception:
            return False

    @staticmethod
    def supported():
        return CPM_optalcp.installed() and CPM_optalcp.license_ok()

    @staticmethod
    def version() -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.

        The version does not include whether the user has a preview version or not.
        """
        try:
            import optalcp
            return optalcp.__version__
        except ModuleNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: Model(), a CPMpy Model() (optional)
        """
        if not self.installed():
            raise ModuleNotFoundError(
                "CPM_optalcp: Install the 'optalcp' Python package and an OptalCP binary to use this solver interface."
            )
        
        if not self.license_ok():
            raise ModuleNotFoundError("CPM_optalcp: In CPMpy we only support non-preview versions of OptalCP." \
            "You can request an academic/full version: https://dev.vilim.eu/docs/Quick%20Start/editions")

        import optalcp

        assert subsolver is None
        self.opt_model = optalcp.Model()
        self._objective_is_min = None
        super().__init__(name="optalcp", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.opt_model

    def solve(self, time_limit: Optional[float] = None, **kwargs):
        """
            Call the OptalCP solver

            Arguments:
                time_limit (float, optional):   maximum solve time in seconds 

                kwargs:                         any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:


            =============================   ============
            Argument                        Description
            =============================   ============
            printLog                        This parameter controls the verbosity. It is a boolean. The default value is False.
            absoluteGapTolerance            This parameter sets an absolute tolerance on the objective value for optimization models. The value is a positive float. Default value is 0.
            relativeGapTolerance            This parameter sets a relative tolerance on the objective value for optimization models. The value is a positive float. Default value is 0.0001. This parameter works together with Parameters.relativeGapTolerance as an OR condition: the search stops when either the absolute gap or the relative gap is within tolerance.
            seachType                       This parameter controls the type of search the solver uses. Possible values are:
                                            - 'Auto': Automatically determined (Default)
                                            - 'LNS': Large Neighbourhood Search
                                            - 'FDS': Failure-Directed Search 
                                            - 'FDSDual': Failure-Directed Search working on objective bounds
                                            - 'SetTimes': Depth-first set-times search
            nbWorkers                       This parameter sets the number of workers to run in parallel to solve your model. The value is a positive integer. Default value is 0. (0 = use environment variable OPTALCP_NB_WORKERS or use all available CPU cores)
            =============================   ============

            All solver parameters are documented here: https://dev.vilim.eu/python-api/api.html#optalcp.Parameters
        """
        # ensure all vars are known to solver 
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)

        # actual default value is True, but we do not want the log to be on by default in CPMpy
        if "printLog" not in kwargs:
            kwargs["printLog"] = False

        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            kwargs["timeLimit"] = time_limit

        self.opt_result = self.opt_model.solve(kwargs or None)
        
        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.opt_result.duration # wallclock time in (float) seconds

        # translate solver exit status to CPMpy exit status
        if self.opt_result.nb_solutions == 0:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE if self.opt_result.proof else ExitStatus.UNKNOWN
        elif self.has_objective():
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL if self.opt_result.proof else ExitStatus.FEASIBLE
        else:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        if has_sol:
            # fill the variable values
            for cpm_var in self.user_vars:
                opt_var = self.solver_var(cpm_var)
                value = self.opt_result.solution.get_value(opt_var)
                if isinstance(cpm_var, _BoolVarImpl):
                    cpm_var._value = bool(value)
                else:
                    cpm_var._value = value
            
            # translate objective, for optimisation problems only
            if self.has_objective():
                self.objective_value_ = self.opt_result.objective

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var.clear()

        return has_sol

    def solveAll(self, display: Optional[Callback] = None, time_limit: Optional[float] = None,
                 solution_limit: Optional[int] = None, call_from_model=False, **kwargs):
        """
            A shorthand to compute all (optimal) solutions, map them to CPMpy and optionally display the solutions.

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
        try:
            return super().solveAll(display=display, time_limit=time_limit,
                                    solution_limit=solution_limit, call_from_model=call_from_model, **kwargs)
        except Exception as exc:
            raise exc

    
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
            return ~self.solver_var(cpm_var._bv)

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.opt_model.bool_var(name=str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.opt_model.int_var(min=cpm_var.lb, max=cpm_var.ub, name=str(cpm_var))
            else:
                raise NotImplementedError(f"Not a known var {cpm_var}")
            self._varmap[cpm_var] = revar

        # return from cache
        return self._varmap[cpm_var]

    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            ``objective()`` can be called multiple times, only the last one is stored

            .. note::
            
                technical side note: any constraints created during conversion of the objective are permanently posted to the solver
        """
        # save user variables
        get_variables(expr, self.user_vars)

        obj, decomp_cons = decompose_objective(
            expr,
            supported=self.supported_global_constraints,
            supported_reified=self.supported_reified_global_constraints,
            csemap=self._csemap
        )
        self.add(decomp_cons)

        opt_obj = self._opt_expr(obj)
        self._objective_is_min = minimize
        if minimize:
            self.opt_model.minimize(opt_obj)
        else:
            self.opt_model.maximize(opt_obj)

    def has_objective(self):
        return self._objective_is_min is not None

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
        cpm_cons = decompose_in_tree(
            cpm_cons,
            supported=self.supported_global_constraints,
            supported_reified=self.supported_reified_global_constraints,
            csemap=self._csemap
        )
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
            opt_con = self._opt_expr(cpm_con)
            if is_any_list(opt_con):
                self.opt_model.enforce(opt_con)
            else:
                self.opt_model.enforce(opt_con)

        return self

    __add__ = add # avoid redirect in superclass

    def _opt_expr(self, cpm_con):
        """
            OptalCP supports nested expressions,
            so we recursively translate our expressions to theirs.

            Accepts a single constraint or a list thereof; return type changes accordingly.
        """
        if is_any_list(cpm_con):
            # arguments can be lists
            return [self._opt_expr(con) for con in cpm_con]

        if isinstance(cpm_con, BoolVal):
            return cpm_con.args[0]
        
        if is_np_int(cpm_con):
            return int(cpm_con)
        
        if is_np_bool(cpm_con):
            return bool(cpm_con)

        if is_num(cpm_con):
            return cpm_con

        if isinstance(cpm_con, _NumVarImpl):
            return self.solver_var(cpm_con)
        
        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        if isinstance(cpm_con, Operator):
            if cpm_con.name == "and":
                args = self._opt_expr(cpm_con.args)
                if not args:
                    return True
                cur = args[0]
                for arg in args[1:]:
                    cur = self.opt_model.and_(cur, arg)
                return cur
            if cpm_con.name == "or":
                args = self._opt_expr(cpm_con.args)
                if not args:
                    return False
                cur = args[0]
                for arg in args[1:]:
                    cur = self.opt_model.or_(cur, arg)
                return cur
            if cpm_con.name == "->":
                lhs, rhs = self._opt_expr(cpm_con.args)
                return self.opt_model.implies(lhs, rhs)
            if cpm_con.name == "not":
                return self.opt_model.not_(self._opt_expr(cpm_con.args[0]))
            if cpm_con.name == "sum":
                return self.opt_model.sum(self._opt_expr(cpm_con.args))
            if cpm_con.name == "wsum":
                weights = cpm_con.args[0]
                exprs = self._opt_expr(cpm_con.args[1])
                return self.opt_model.sum([w * x for w, x in zip(weights, exprs)])
            if cpm_con.name == "sub":
                lhs, rhs = self._opt_expr(cpm_con.args)
                return lhs - rhs
            if cpm_con.name == "-":
                return -self._opt_expr(cpm_con.args[0])
            raise NotImplementedError(f"Operator {cpm_con} not implemented for OptalCP")

        # Comparisons (just translate the subexpressions and re-post)
        if isinstance(cpm_con, Comparison):
            lhs, rhs = self._opt_expr(cpm_con.args)
            # post the comparison
            return eval_comparison(cpm_con.name, lhs, rhs)
        # rest: base (Boolean) global constraints
        if isinstance(cpm_con, GlobalConstraint):
            if cpm_con.name in ("cumulative", "cumulative_optional"):
                return self._opt_cumulative(cpm_con)
            if cpm_con.name == "no_overlap":
                if len(cpm_con.args) == 2:
                    start, dur = cpm_con.args
                    end = None
                else:
                    start, dur, end = cpm_con.args
                tasks, cons = self._make_tasks(start, dur, end, None)
                task_list = [task for task in tasks if task is not None]
                if len(task_list) > 1:
                    cons.append(self.opt_model.no_overlap(task_list))
                return cons
            if cpm_con.name == "no_overlap_optional":
                if len(cpm_con.args) == 3:
                    start, dur, is_present = cpm_con.args
                    end = None
                else:
                    start, dur, end, is_present = cpm_con.args
                tasks, cons = self._make_tasks(start, dur, end, is_present)
                task_list = [task for task in tasks if task is not None]
                if len(task_list) > 1:
                    cons.append(self.opt_model.no_overlap(task_list))
                return cons
            if isinstance(cpm_con, DirectConstraint):
                return cpm_con.callSolver(self, self.opt_model)
            raise NotImplementedError(f"Global constraint {cpm_con} not supported by OptalCP backend")

        if isinstance(cpm_con, GlobalFunction):
            if cpm_con.name == "min":
                return self.opt_model.min(self._opt_expr(cpm_con.args))
            if cpm_con.name == "max":
                return self.opt_model.max(self._opt_expr(cpm_con.args))
            if cpm_con.name == "abs":
                return self.opt_model.abs(self._opt_expr(cpm_con.args)[0])
            if cpm_con.name == "mul":
                lhs, rhs = self._opt_expr(cpm_con.args)
                return lhs * rhs
            if cpm_con.name == "div":
                lhs, rhs = self._opt_expr(cpm_con.args)
                return lhs // rhs
            raise NotImplementedError(f"Global function {cpm_con} not supported by OptalCP backend")

        raise NotImplementedError("OptalCP: constraint not (yet) supported", cpm_con)

    def _opt_cumulative(self, cpm_con):
        """
            Helper function to translate CPMpy's Cumulative and CumulativeOptional constraints
            to OptalCP's pulse-based cumulative expressions.

            Creates interval variables for each task via `_make_tasks`, then posts
            the sum of pulses <= capacity constraint.

            OptalCP requires the capacity expression to be 'present' (non-optional).
            If capacity is a complex expression (e.g. involving boolean variables),
            it is reified into a fresh auxiliary intvar to satisfy this requirement.
        """
        # unpack arguments depending on constraint type and whether end times are provided
        if cpm_con.name == "cumulative":
            is_present = None
            if len(cpm_con.args) == 4:
                start, dur, demand, capacity = cpm_con.args
                end = None
            else:
                start, dur, end, demand, capacity = cpm_con.args
        else: # cumulative_optional
            if len(cpm_con.args) == 5:
                start, dur, demand, capacity, is_present = cpm_con.args
                end = None
            else:
                start, dur, end, demand, capacity, is_present = cpm_con.args

        # create interval variables and linking constraints for each task
        tasks, cons = self._make_tasks(start, dur, end, is_present)

        # get_nonneg_args strips the presence multiplication from demand expressions
        # (e.g. bv * [2,3,4] -> [2,3,4]) since the optional interval var already
        # contributes 0 pulse when absent — no need to encode it in the demand
        demand_lst, demand_cons = get_nonneg_args(demand, is_present)
        cons += self._opt_expr(demand_cons)

        # build a pulse for each present (non-zero-duration) task
        pulses = []
        for task, h in zip(tasks, demand_lst):
            if task is not None:
                pulses.append(task.pulse(self._opt_expr(h)))

        if pulses:
            # OptalCP requires maxCapacity to be a "present" (non-optional) expression.
            # If capacity is a complex expression (e.g. 3*bv[0]), reify it into a
            # fresh auxiliary intvar that is guaranteed present.
            if is_num(capacity) or isinstance(capacity, _IntVarImpl):
                opt_cap = self._opt_expr(capacity)
            else:
                lb, ub = get_bounds(capacity)
                cap_aux = intvar(lb, ub)
                self.user_vars.add(cap_aux)
                opt_cap = self.solver_var(cap_aux)
                cons += self._opt_expr([cap_aux == capacity])

            cons.append(self.opt_model.sum(pulses) <= opt_cap)
        return cons

    def _make_tasks(self, start, dur, end, is_present):
        """
            Helper function to create a list of OptalCP interval variables and additional
            constraints enforcing the relationship between CPMpy start/duration/end variables
            and the native interval variables.

            Arguments:
                start:      list of CPMpy integer expressions for task start times
                dur:        list of CPMpy integer expressions for task durations
                end:        list of CPMpy integer expressions for task end times, or None
                is_present: list of CPMpy boolean variables indicating task presence, or None

            Returns:
                tasks:      list of OptalCP interval variables (None for zero-duration tasks)
                cons:       list of OptalCP constraints linking the interval variables to
                            the CPMpy start/duration/end/presence variables
        """
        if end is None:
            end = [None for _ in range(len(start))] # easier to handle the task-making below

        # OptalCP crashes if size of interval is negative
        dur, dur_cons = get_nonneg_args(dur, is_present)
        extra_cons = self._opt_expr(dur_cons)

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
            Helper function to create a single OptalCP interval variable and additional
            constraints enforcing the relationship between CPMpy start/duration/end variables
            and the native interval variable.

            Zero-duration tasks (lb == ub == 0) are handled as a special case: no interval
            variable is created; instead, a constraint start == end is posted (if end is given).

            For optional tasks (is_present is not None), the interval variable is created
            with optional=True, and its presence is tied to the CPMpy boolean variable
            via a presence() == is_present constraint. Start, length, and end are linked
            through implication constraints conditioned on presence.

            Arguments:
                start:      CPMpy integer expression for the task start time
                dur:        CPMpy integer expression for the task duration (must be non-negative)
                end:        CPMpy integer expression for the task end time, or None
                is_present: CPMpy boolean variable indicating task presence, or None

            Returns:
                task:       OptalCP interval variable, or None for zero-duration tasks
                cons:       list of OptalCP constraints linking the interval variable to
                            the CPMpy start/duration/end/presence variables
        """
        assert get_bounds(dur)[0] >= 0, "optalcp does not support intervals with negative duration, use `utils.get_nonneg_args` first"

        is_optional = is_present is not None
        if not is_optional:
            is_present = BoolVal(True) # eases handling below

        lb, ub = get_bounds(dur)
        extra_cons = []
        if lb == 0 == ub:
            if end is None: # nothing to enforce
                return None, []
            return None, self._opt_expr([implies(is_present, start == end)])

        # create the interval variable with domain bounds derived from the CPMpy expressions;
        # mark it optional when a presence variable was provided
        task = self.opt_model.interval_var(
            start=get_bounds(start),
            end=get_bounds(end) if end is not None else None,
            length=get_bounds(dur),
            optional=is_optional
        )

        if is_optional:
            # tie the interval's presence status to the CPMpy boolean variable
            extra_cons.append(task.presence() == self.solver_var(is_present))
            # when present, link start/length/end to the CPMpy expressions via implications
            # so that the solver correctly propagates values from interval to CPMpy variables
            extra_cons.append(self.opt_model.implies(task.presence(), task.start() == self._opt_expr(start)))
            extra_cons.append(self.opt_model.implies(task.presence(), task.length() == self._opt_expr(dur)))
            if end is not None:
                extra_cons.append(self.opt_model.implies(task.presence(), task.end() == self._opt_expr(end)))
        else:
            # mandatory task: link unconditionally
            extra_cons.append(task.start() == self._opt_expr(start))
            extra_cons.append(task.length() == self._opt_expr(dur))
            if end is not None:
                extra_cons.append(task.end() == self._opt_expr(end))

        return task, extra_cons
