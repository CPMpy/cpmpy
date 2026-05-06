"""
    Interface to OptalCP's Python API.

    OptalCP is a scheduling-oriented constraint programming solver with a native Python API.

    Always use :func:`cp.SolverLookup.get("optalcp") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires the `optalcp` Python package and an OptalCP solver binary.

    Public preview installation currently works via:

    .. code-block:: console

        $ pip install git+https://github.com/ScheduleOpt/optalcp-py-bin-preview@latest

    The preview edition solves models and reports objective values, but masks decision-variable assignments.
    Academic and full editions expose variable values as well.

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

from .solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback
from .. import DirectConstraint
from ..exceptions import MaskedSolverValueError, NotSupportedError
from ..expressions.core import Comparison, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.utils import is_num, is_any_list, eval_comparison, get_bounds, get_nonneg_args, implies
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree, decompose_objective
from ..transformations.safening import no_partial_functions


class CPM_optalcp(SolverInterface):
    """
    Interface to OptalCP's Python API.

    Creates the following attributes (see parent constructor for more):

    - ``opt_model``: object, OptalCP model object
    """

    supported_global_constraints = frozenset({
        "cumulative", "cumulative_optional", "no_overlap", "no_overlap_optional",
        "min", "max", "abs", "mul", "div",
    })
    supported_reified_global_constraints = frozenset()

    _masked_value_message = (
        "OptalCP Preview solved the model but masks decision-variable values. "
        "Install an Academic or Full OptalCP edition to retrieve variable assignments."
    )

    @staticmethod
    def installed():
        try:
            import optalcp
            return True
        except ModuleNotFoundError:
            return False

    @staticmethod
    def supported():
        if not CPM_optalcp.installed():
            return False
        try:
            import optalcp
            optalcp.Solver.find_solver({})
            return True
        except Exception:
            return False

    @staticmethod
    def version() -> Optional[str]:
        try:
            import optalcp
            return optalcp.__version__
        except ModuleNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        if not self.installed():
            raise ModuleNotFoundError(
                "CPM_optalcp: Install the 'optalcp' Python package and an OptalCP binary to use this solver interface."
            )

        import optalcp

        assert subsolver is None
        self.opt_model = optalcp.Model()
        self._objective_is_min = None
        self._values_masked = False
        super().__init__(name="optalcp", cpm_model=cpm_model)

    @property
    def native_model(self):
        return self.opt_model

    def solve(self, time_limit: Optional[float] = None, **kwargs):
        self.solver_vars(list(self.user_vars))

        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)

        if "printLog" not in kwargs:
            kwargs["printLog"] = False

        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            kwargs["timeLimit"] = time_limit

        self.opt_result = self.opt_model.solve(kwargs or None)

        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.opt_result.duration

        if self.opt_result.nb_solutions == 0:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE if self.opt_result.proof else ExitStatus.UNKNOWN
        elif self.has_objective():
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL if self.opt_result.proof else ExitStatus.FEASIBLE
        else:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE

        has_sol = self._solve_return(self.cpm_status)
        self.objective_value_ = None
        self._values_masked = False

        if has_sol:
            self._populate_values(self.opt_result.solution)
            if self.has_objective():
                self.objective_value_ = self.opt_result.objective
        else:
            for cpm_var in self.user_vars:
                cpm_var.clear()

        return has_sol

    def solveAll(self, display: Optional[Callback] = None, time_limit: Optional[float] = None,
                 solution_limit: Optional[int] = None, call_from_model=False, **kwargs):
        try:
            return super().solveAll(display=display, time_limit=time_limit,
                                    solution_limit=solution_limit, call_from_model=call_from_model, **kwargs)
        except MaskedSolverValueError as exc:
            raise NotSupportedError(
                "OptalCP solveAll() requires visible decision-variable values. "
                "The Preview edition masks them."
            ) from exc

    def _populate_values(self, solution):
        if solution is None:
            for cpm_var in self.user_vars:
                cpm_var.clear()
            return

        masked = False
        values = {}
        for cpm_var in self.user_vars:
            opt_var = self.solver_var(cpm_var)
            value = solution.get_value(opt_var)
            values[cpm_var] = value
            if value is None:
                masked = True

        if masked:
            self._values_masked = True
            for cpm_var in self.user_vars:
                cpm_var._mask_value(self._masked_value_message)
            return

        for cpm_var, value in values.items():
            cpm_var.clear()
            if isinstance(cpm_var, _BoolVarImpl):
                cpm_var._value = bool(value)
            else:
                cpm_var._value = value

    def solver_var(self, cpm_var):
        if is_num(cpm_var):
            return cpm_var

        if isinstance(cpm_var, NegBoolView):
            return ~self.solver_var(cpm_var._bv)

        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.opt_model.bool_var(name=str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.opt_model.int_var(min=cpm_var.lb, max=cpm_var.ub, name=str(cpm_var))
            else:
                raise NotImplementedError(f"Not a known var {cpm_var}")
            self._varmap[cpm_var] = revar

        return self._varmap[cpm_var]

    def objective(self, expr, minimize=True):
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

    def transform(self, cpm_expr):
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel=frozenset({}))
        cpm_cons = decompose_in_tree(
            cpm_cons,
            supported=self.supported_global_constraints,
            supported_reified=self.supported_reified_global_constraints,
            csemap=self._csemap
        )
        return cpm_cons

    def add(self, cpm_expr):
        get_variables(cpm_expr, collect=self.user_vars)

        for cpm_con in self.transform(cpm_expr):
            opt_con = self._opt_expr(cpm_con)
            if is_any_list(opt_con):
                self.opt_model.enforce(opt_con)
            else:
                self.opt_model.enforce(opt_con)

        return self

    __add__ = add

    def _opt_expr(self, cpm_con):
        if is_any_list(cpm_con):
            return [self._opt_expr(con) for con in cpm_con]

        if isinstance(cpm_con, BoolVal):
            return cpm_con.args[0]

        if is_num(cpm_con):
            return cpm_con

        if isinstance(cpm_con, _NumVarImpl):
            return self.solver_var(cpm_con)

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

        if isinstance(cpm_con, Comparison):
            lhs, rhs = self._opt_expr(cpm_con.args)
            return eval_comparison(cpm_con.name, lhs, rhs)

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
        if cpm_con.name == "cumulative":
            is_present = None
            if len(cpm_con.args) == 4:
                start, dur, demand, capacity = cpm_con.args
                end = None
            else:
                start, dur, end, demand, capacity = cpm_con.args
        else:
            if len(cpm_con.args) == 5:
                start, dur, demand, capacity, is_present = cpm_con.args
                end = None
            else:
                start, dur, end, demand, capacity, is_present = cpm_con.args

        tasks, cons = self._make_tasks(start, dur, end, is_present)
        demand_lst, demand_cons = get_nonneg_args(demand, is_present)
        cons += self._opt_expr(demand_cons)

        pulses = []
        for task, h in zip(tasks, demand_lst):
            if task is not None:
                pulses.append(task.pulse(self._opt_expr(h)))

        if pulses:
            cons.append(self.opt_model.sum(pulses) <= self._opt_expr(capacity))
        return cons

    def _make_tasks(self, start, dur, end, is_present):
        if end is None:
            end = [None for _ in range(len(start))]

        dur, dur_cons = get_nonneg_args(dur, is_present)
        extra_cons = self._opt_expr(dur_cons)

        if is_present is None:
            is_present = [None] * len(start)

        tasks = []
        for s, d, e, p in zip(start, dur, end, is_present):
            task, task_cons = self._make_task(s, d, e, p)
            tasks.append(task)
            extra_cons += task_cons
        return tasks, extra_cons

    def _make_task(self, start, dur, end, is_present):
        assert get_bounds(dur)[0] >= 0, "optalcp does not support intervals with negative duration, use `utils.get_nonneg_args` first"

        is_optional = is_present is not None
        if not is_optional:
            is_present = BoolVal(True)

        lb, ub = get_bounds(dur)
        extra_cons = []
        if lb == 0 == ub:
            if end is None:
                return None, []
            return None, self._opt_expr([implies(is_present, start == end)])

        task = self.opt_model.interval_var(
            start=get_bounds(start),
            end=get_bounds(end) if end is not None else None,
            length=get_bounds(dur),
            optional=is_optional
        )

        if is_optional:
            extra_cons.append(task.presence() == self.solver_var(is_present))
            extra_cons.append(self.opt_model.implies(task.presence(), task.start() == self._opt_expr(start)))
            extra_cons.append(self.opt_model.implies(task.presence(), task.length() == self._opt_expr(dur)))
            if end is not None:
                extra_cons.append(self.opt_model.implies(task.presence(), task.end() == self._opt_expr(end)))
        else:
            extra_cons.append(task.start() == self._opt_expr(start))
            extra_cons.append(task.length() == self._opt_expr(dur))
            if end is not None:
                extra_cons.append(task.end() == self._opt_expr(end))

        return task, extra_cons
