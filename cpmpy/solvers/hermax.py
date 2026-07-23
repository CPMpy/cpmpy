#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## hermax.py
##
"""
    Interface to Hermax's high-level modeling API

    Hermax is a Python library of incremental MaxSAT solvers with a CP-style
    modeling layer (typed variables, linear/pseudo-Boolean constraints, and
    some global constraints). See https://github.com/josalhor/hermax

    Always use :func:`cp.SolverLookup.get("hermax") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'hermax' python package is installed:

    .. code-block:: console

        $ pip install hermax

    See detailed installation instructions at:
    https://hermax.readthedocs.io/

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_hermax
"""
import time
from typing import Optional

from cpmpy.expressions.globalfunctions import GlobalFunction

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal, NestedBoolExprLike
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.variables import _BoolVarImpl, NegBoolView, _NumVarImpl, intvar
from ..expressions.utils import is_num, is_int, is_boolexpr
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list
from ..transformations.safening import no_partial_functions, safen_objective
from ..transformations.flatten_model import flatten_constraint, flatten_objective, get_or_make_var
from ..transformations.comparison import only_numexpr_equality
from ..transformations.negation import push_down_negation, push_down_negation_objective
from ..transformations.reification import reify_rewrite, only_bv_reifies, only_implies
from ..transformations.linearize import (
    decompose_linear,
    decompose_linear_objective,
    linearize_reified_variables,
)


class CPM_hermax(SolverInterface):
    """
    Interface to Hermax's modeling API (``hermax.model.Model``).

    Creates the following attributes (see parent constructor for more):

    - ``her_model``: the underlying ``hermax.model.Model`` object

    Documentation of the solver's own Python API:
    https://hermax.readthedocs.io/
    """

    # CP-level globals Hermax encodes natively
    supported_global_constraints = frozenset({
        "alldifferent", "cumulative", "increasing", "lex_less", "min", "max",
    })
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        try:
            import hermax  # noqa: F401
            return True
        except ModuleNotFoundError:
            return False

    @staticmethod
    def version() -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("hermax")
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: Model(), a CPMpy Model() (optional)
            subsolver: None, not used
        """
        if not self.supported():
            raise ModuleNotFoundError("CPM_hermax: Install the python package 'hermax' to use this solver interface.")

        from hermax.model import Model

        assert subsolver is None, "Hermax does not support subsolvers yet"

        self.her_model = Model()
        self._objective = None
        self._objective_posted = False
        self._assumption_map = {}  # dimacs lit -> CPMpy assumption expr
        self._last_core = None
        self._bool_intmap = {}  # boolvar name -> channeled {0,1} intvar

        super().__init__(name="hermax", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.her_model

    def solve(self, time_limit: Optional[float] = None, assumptions=None, **kwargs):
        """
        Call the Hermax solver

        Arguments:
            time_limit (float, optional): maximum solve time in seconds.
                Negative values raise ``ValueError``. When supported by the
                underlying PySAT/backend solver, search is interrupted after
                the limit.
            assumptions: iterable of Boolean variables (or their negation) assumed true
            **kwargs: forwarded to ``hermax.model.Model.solve`` (e.g. ``sat_solver_name``,
                ``maxsat_backend``, ``incremental``, ``solver``)

        Returns:
            whether a solution was found
        """
        if time_limit is not None and time_limit < 0:
            raise ValueError(f"Time limit cannot be negative, but got {time_limit}")

        self.solver_vars(list(self.user_vars))

        hm_assumptions = None
        self._assumption_map = {}
        self._last_core = None
        if assumptions is not None:
            assumptions = list(assumptions)
            hm_assumptions = []
            for a in assumptions:
                hm_a = self.solver_var(a)
                hm_assumptions.append(hm_a)
                # signed DIMACS literal (polarity=False -> negative id)
                dimacs = int(hm_a.id) if hm_a.polarity else -int(hm_a.id)
                self._assumption_map[dimacs] = a

        start = time.time()
        timer = None
        if time_limit is not None and time_limit > 0:
            from threading import Timer

            def _interrupt():
                sat = getattr(self.her_model._inc_state, "sat_solver", None)
                if sat is not None and hasattr(sat, "interrupt"):
                    try:
                        sat.interrupt()
                    except Exception:
                        pass
                ip = getattr(self.her_model._inc_state, "ip_solver", None)
                if ip is not None and hasattr(ip, "interrupt"):
                    try:
                        ip.interrupt()
                    except Exception:
                        pass

            timer = Timer(time_limit, _interrupt)
            timer.daemon = True
            timer.start()

        try:
            result = self.her_model.solve(assumptions=hm_assumptions, **kwargs)
        finally:
            if timer is not None:
                timer.cancel()
                sat = getattr(self.her_model._inc_state, "sat_solver", None)
                if sat is not None and hasattr(sat, "clear_interrupt"):
                    try:
                        sat.clear_interrupt()
                    except Exception:
                        pass

        runtime = time.time() - start

        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = runtime

        status = str(result.status).lower()
        if status in ("optimum", "optimal"):
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        elif status in ("sat", "satisfiable", "feasible"):
            self.cpm_status.exitstatus = (
                ExitStatus.OPTIMAL if self.has_objective() else ExitStatus.FEASIBLE
            )
        elif status in ("unsat", "unsatisfiable"):
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            # extract unsat core from underlying PySAT solver when available
            sat = getattr(self.her_model._inc_state, "sat_solver", None)
            if sat is not None and hasattr(sat, "get_core"):
                try:
                    dimacs_core = sat.get_core() or []
                    core = []
                    for lit in dimacs_core:
                        if lit in self._assumption_map:
                            cpm_a = self._assumption_map[lit]
                            if cpm_a not in core:
                                core.append(cpm_a)
                    self._last_core = core
                except Exception:
                    self._last_core = None
        elif status in ("unknown", "timeout", "interrupted"):
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:
            if result.ok:
                self.cpm_status.exitstatus = (
                    ExitStatus.OPTIMAL if self.has_objective() else ExitStatus.FEASIBLE
                )
            else:
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN

        has_sol = self._solve_return(self.cpm_status)
        self.objective_value_ = None
        if has_sol:
            for cpm_var in self.user_vars:
                cpm_var._value = result.assignment[self.solver_var(cpm_var)]
            if self.has_objective():
                self.objective_value_ = int(self._objective.value())
        else:
            for cpm_var in self.user_vars:
                cpm_var._value = None

        return has_sol

    def get_core(self):
        """
        For use with :func:`s.solve(assumptions=[...]) <cpmpy.solvers.hermax.CPM_hermax.solve>`.
        Only meaningful if the solver returned UNSAT. Returns a subset of the
        assumption literals that are unsatisfiable together.

        .. note::
            Core extraction uses the underlying PySAT SAT backend. It may be
            unavailable when solving through a MaxSAT backend.
        """
        if self._last_core is None:
            raise NotSupportedError(
                "Hermax: no unsat core available (solve under assumptions with a SAT backend first)"
            )
        return list(self._last_core)

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
            if cpm_var.is_bool():
                if isinstance(cpm_var, NegBoolView):
                    # special case, negative-bool-view: work directly on var inside the view
                    revar = ~self.solver_var(cpm_var._bv)
                else:
                    revar = self.her_model.bool(name)
            else:
                revar = self.her_model.int(name, int(cpm_var.lb), int(cpm_var.ub))

            self._varmap[name] = revar
            return revar

        if is_int(cpm_var):  # shortcut, eases posting constraints
            return int(cpm_var)

        raise NotImplementedError("Not a known var {}".format(cpm_var))

    def _bool_as_int(self, cpm_bool):
        """
        Return a {0,1} intvar channeled to ``cpm_bool`` (one intvar per underlying bool).
        """
        if isinstance(cpm_bool, NegBoolView):
            cpm_bool = cpm_bool._bv

        name = cpm_bool.name
        if name in self._bool_intmap:
            return self._bool_intmap[name]

        iv = intvar(0, 1, name=f"_hm_{name}")
        self._bool_intmap[name] = iv
        hm_bv = self.solver_var(cpm_bool)
        hm_iv = self.solver_var(iv)
        self.her_model &= hm_bv.implies(hm_iv == 1)
        self.her_model &= (~hm_bv).implies(hm_iv == 0)
        return iv

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function

            Supports sum, wsum , sub operators and single decision variables.
        """
        if is_num(cpm_expr):
            return int(cpm_expr)

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl): 
            if cpm_expr.is_bool():
                # Boolvars are not integer for Hermax, channel to 0-1 intvar
                if isinstance(cpm_expr, NegBoolView):
                    return 1 - self.solver_var(self._bool_as_int(cpm_expr._bv))
                return self.solver_var(self._bool_as_int(cpm_expr))
            return self.solver_var(cpm_expr)

        if isinstance(cpm_expr, Operator):
            if cpm_expr.name == "sum":
                return sum(self._make_numexpr(a) for a in cpm_expr.args)
            if cpm_expr.name == "wsum":
                weights, vars_ = cpm_expr.args
                return sum(w * self._make_numexpr(v) for w, v in zip(weights, vars_))
            if cpm_expr.name == "sub":
                return self._make_numexpr(cpm_expr.args[0]) - self._make_numexpr(cpm_expr.args[1])
            if cpm_expr.name == "-":
                return (-1) * self._make_numexpr(cpm_expr.args[0])
            if cpm_expr.name == "mul":
                a, b = cpm_expr.args
                if is_num(a):
                    return int(a) * self._make_numexpr(b)
                if is_num(b):
                    return self._make_numexpr(a) * int(b)
                raise NotSupportedError(f"Hermax: non-linear mul {cpm_expr}")

            raise NotSupportedError(f"Hermax: unsupported operator {cpm_expr}")

        raise NotImplementedError("Hermax: Not a known supported numexpr {}".format(cpm_expr))

    def objective(self, expr: Expression, minimize: bool = True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            ``objective()`` can be called multiple times; only the last one is stored.
            Subsequent calls replace the Hermax objective via ``model.obj.replace_with(...)``.

            Arguments:
                expr: Expression, the CPMpy expression that represents the objective function
                minimize: Bool, whether it is a minimization problem (True) or maximization (False)
        """
        get_variables(expr, self.user_vars)

        obj, safe_cons = safen_objective(expr)
        obj = push_down_negation_objective(obj)
        obj, decomp_cons = decompose_linear_objective(
            obj,
            supported=self.supported_global_constraints,
            supported_reified=self.supported_reified_global_constraints,
            csemap=self._csemap,
        )
        obj, flat_cons = flatten_objective(obj, csemap=self._csemap)
        obj_var, obj_cons = get_or_make_var(obj)
        if expr.is_bool():
            ivar = intvar(0, 1)
            obj_cons.append(ivar == obj_var)
            obj_var = ivar

        self.add(safe_cons + decomp_cons + flat_cons + obj_cons)
        self._objective = obj_var

        hm_obj = self.solver_var(obj_var)
        # Hermax soft IntVar terms currently require a non-negative domain
        lb, ub = int(obj_var.lb), int(obj_var.ub)
        if minimize:
            hm_expr = hm_obj - lb  # shift to [0, ub-lb]
        else:
            hm_expr = ub - hm_obj  # maximize via minimizing (ub - x)
        if self._objective_posted:
            self.her_model.obj.replace_with(hm_expr)
        else:
            self.her_model.obj += hm_expr
            self._objective_posted = True

    def has_objective(self):
        return self._objective is not None

    def transform(self, cpm_expr: NestedBoolExprLike) -> list[Expression]:
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Arguments:
                cpm_expr (NestedBoolExprLike): CPMpy expression, or list thereof

            Returns:
                list[Expression]: transformed constraints
        """
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons)
        cpm_cons = push_down_negation(cpm_cons)
        cpm_cons = decompose_linear(
            cpm_cons,
            supported=self.supported_global_constraints,
            supported_reified=self.supported_reified_global_constraints,
            csemap=self._csemap,
        )
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)
        cpm_cons = reify_rewrite(
            cpm_cons,
            supported=frozenset({"or", "sum", "wsum", "sub"}),
            csemap=self._csemap,
        )
        cpm_cons = only_numexpr_equality(
            cpm_cons,
            supported=frozenset({"sum", "wsum", "sub"}),
            csemap=self._csemap,
        )
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap)
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
        return cpm_cons

    def add(self, cpm_expr: NestedBoolExprLike) -> "CPM_hermax":
        """
            Eagerly add a constraint to the underlying solver.

            Arguments:
                cpm_expr (NestedBoolExprLike): CPMpy expression, or list thereof

            Returns:
                self
        """
        get_variables(cpm_expr, collect=self.user_vars)

        for cpm_con in self.transform(cpm_expr):
            if isinstance(cpm_con, Operator) and cpm_con.name == "and":
                for arg in cpm_con.args:
                    self.add(arg)
                continue

            hm_con = self._hermax_expr(cpm_con)
            if hm_con is not None:
                self.her_model &= hm_con

        return self

    __add__ = add

    def _hermax_expr(self, cpm_con):
        """
            Translate a flat CPMpy constraint to a Hermax constraint/expression.
        """
        if isinstance(cpm_con, BoolVal):
            return bool(cpm_con.value())

        if isinstance(cpm_con, _BoolVarImpl):
            return self.solver_var(cpm_con)

        if isinstance(cpm_con, Operator):
            if cpm_con.name == "or":
                args = self.solver_vars(cpm_con.args)
                clause = args[0]
                for a in args[1:]:
                    clause = clause | a
                return clause
            if cpm_con.name == "->":
                bv, subexpr = cpm_con.args
                hm_bv = self.solver_var(bv)
                if isinstance(subexpr, _BoolVarImpl):
                    return hm_bv.implies(self.solver_var(subexpr))
                elif isinstance(subexpr, Operator) and subexpr.name == "or":
                    # encode as bigger clause
                    return ~hm_bv | self._hermax_expr(subexpr)
                if isinstance(subexpr, Comparison):
                    hm_sub = self._hermax_expr(subexpr)
                    # Hermax may constant-fold a comparison to True/False once domains
                    # are fixed by earlier posts; literals have .only_if(), bools do not.
                    if isinstance(hm_sub, bool):
                        return True if hm_sub else ~hm_bv
                    return hm_sub.only_if(hm_bv)
                raise NotImplementedError(f"Hermax: unsupported implication {cpm_con}")
            raise NotImplementedError(f"Hermax: unsupported operator {cpm_con}")

        if isinstance(cpm_con, Comparison):
            lhs, rhs = cpm_con.args
        
            # Global functions in equality form: min/max(...) == rhs
            if isinstance(lhs, GlobalFunction):
                assert cpm_con.name == "==", "Hermax: only equality comparisons are supported for global functions"
                if lhs.name == "min":
                    return self.her_model.min(self.solver_vars(lhs.args)) == self.solver_var(rhs)
                if lhs.name == "max":
                    return self.her_model.max(self.solver_vars(lhs.args)) == self.solver_var(rhs)
                raise NotImplementedError(f"Hermax: unsupported global function {lhs}")

            hm_lhs = self._make_numexpr(lhs)
            hm_rhs = self._make_numexpr(rhs)
            if cpm_con.name == "==":
                return hm_lhs == hm_rhs
            if cpm_con.name == "!=":
                return hm_lhs != hm_rhs
            if cpm_con.name == "<=":
                return hm_lhs <= hm_rhs
            if cpm_con.name == "<":
                return hm_lhs < hm_rhs
            if cpm_con.name == ">=":
                return hm_lhs >= hm_rhs
            if cpm_con.name == ">":
                return hm_lhs > hm_rhs
            raise NotImplementedError(f"Hermax: unsupported comparison {cpm_con}")

        if isinstance(cpm_con, GlobalConstraint):
            if cpm_con.name == "alldifferent":
                hm_vars = [self.solver_var(v) for v in cpm_con.args]
                return self.her_model.vector(hm_vars).all_different()
            if cpm_con.name == "increasing":
                hm_vars = [self.solver_var(v) for v in cpm_con.args]
                return self.her_model.vector(hm_vars).increasing()
            if cpm_con.name == "lex_less":
                x, y = cpm_con.args
                hx = self.her_model.vector([self.solver_var(v) for v in x])
                hy = self.her_model.vector([self.solver_var(v) for v in y])
                return hx.lexicographic_less_than(hy)
            if cpm_con.name == "cumulative":
                if len(cpm_con.args) == 4:
                    start, dur, demand, cap = cpm_con.args
                else:
                    start, dur, end, demand, cap = cpm_con.args
                    for s, d, e in zip(start, dur, end):
                        self.add(s + d == e)

                if not all(is_num(d) for d in dur):
                    raise NotSupportedError("Hermax Cumulative only supports fixed durations")
                if not all(is_num(h) for h in demand):
                    raise NotSupportedError("Hermax Cumulative only supports fixed demands")
                if not is_num(cap):
                    raise NotSupportedError("Hermax Cumulative only supports fixed capacity")

                hm_starts = [self.solver_var(s) for s in start]
                self.her_model.cumulative(
                    hm_starts,
                    [int(d) for d in dur],
                    [int(h) for h in demand],
                    int(cap),
                )
                return None
            raise NotImplementedError(f"Hermax: unsupported global {cpm_con}")

        raise NotImplementedError(f"Hermax: unexpected constraint {cpm_con}")
