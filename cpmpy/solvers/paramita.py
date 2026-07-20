#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
Interface to Paramita's Python API (modelling layer).

Paramita is an extensible SAT/MaxSAT framework with a high-level modelling API
(Boolean formulas, pseudo-Boolean constraints, linear objectives) and plugin-based
solver back-ends.

Always use :func:`cp.SolverLookup.get("paramita") <cpmpy.solvers.utils.SolverLookup.get>`
to instantiate the solver object.

============
Installation
============

Requires the ``paramita`` package and at least one compiled solver plugin:

.. code-block:: console

    $ pip install paramita
    $ paramita plugins build cmake '#sat-solvers/cadical-3.0.0'
    $ paramita plugins build cmake '#maxsat-solvers/mse22-EvalMaxSAT2022'

Linux only (see Paramita docs). Detailed documentation:

- https://ulog.udl.cat/static/doc/paramita/

The rest of this documentation is for advanced users.

===============
List of classes
===============

.. autosummary::
    :nosignatures:

    CPM_paramita

==============
Module details
==============
"""

import time
import warnings
from pathlib import Path
from threading import Timer
from typing import Iterable, Optional, Set

from ..exceptions import NotSupportedError
from ..expressions.core import Expression, BoolVal, Comparison, Operator, NestedBoolExprLike
from ..expressions.utils import is_int, is_bool
from ..expressions.variables import NegBoolView, _BoolVarImpl, _NumVarImpl
from ..transformations.flatten_model import flatten_constraint
from ..transformations.get_variables import get_variables
from ..transformations.int2bool import _decide_encoding, _encode_int_var, int2bool, replace_int_user_vars
from ..transformations.linearize import linearize_constraint, linearize_reified_variables, decompose_linear
from ..transformations.normalize import simplify_boolean, toplevel_list
from ..transformations.negation import push_down_negation
from ..transformations.reification import only_bv_reifies, only_implies
from ..transformations.safening import no_partial_functions
from ..transformations.to_cnf import to_cnf_objective
from .solver_interface import ExitStatus, SolverInterface, SolverStatus


def _paramita_plugins_by_iface():
    """Return {iface_class: [(implementation_name, path), ...]} for installed plugins."""
    from paramita.plugins import find_plugins, read_interface_info, default_plugin_search_paths

    found = {}
    for directory in default_plugin_search_paths():
        if not Path(directory).exists():
            continue
        for iface, paths in find_plugins(directory).items():
            for path in paths:
                info = read_interface_info(path)
                found.setdefault(iface, []).append((info.implementation_name, path))
    return found


class CPM_paramita(SolverInterface):
    """
    Interface to Paramita via its modelling API.

    Creates the following attributes (see parent constructor for more):

    - ``paramita_solver``: underlying ``SatSolver`` or ``MaxSatSolver`` plugin
    - ``pool``: ``DimacsVariablePool`` mapping CPMpy Bool names to DIMACS ids
    - ``ivarmap``: a mapping from integer variables to their encoding for ``int2bool``
    - ``encoding``: the encoding used for ``int2bool``, choose from (``"auto"``, ``"direct"``, ``"order"``, or ``"binary"``). Set to ``"auto"`` but can be changed in the solver object.
    - ``objective_``: posted CPMpy objective expression, or ``None``
    - ``is_maxsat``: whether the active plugin is a ``MaxSatSolver`` (may become ``True`` after posting an objective)
    - ``hard_paramita``: buffered hard modelling expressions, re-posted when switching to MaxSAT
    - ``assumption_vars``: assumptions of the last ``solve(assumptions=[...])``, or ``None``

    Documentation of Paramita:

    - https://ulog.udl.cat/static/doc/paramita/
    """

    supported_global_constraints = frozenset()
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        try:
            import paramita  # noqa: F401
            from paramita.solvers import SatSolver, MaxSatSolver

            plugins = _paramita_plugins_by_iface()
            return bool(plugins.get(SatSolver) or plugins.get(MaxSatSolver))
        except ModuleNotFoundError:
            return False

    @staticmethod
    def solvernames(**kwargs):
        """Return Paramita SatSolver and MaxSatSolver plugin names."""
        if not CPM_paramita.supported():
            warnings.warn("Paramita is not installed or no solver plugins are available.")
            return []
        from paramita.solvers import SatSolver, MaxSatSolver

        plugins = _paramita_plugins_by_iface()
        names = []
        for iface in (SatSolver, MaxSatSolver):
            for name, _ in plugins.get(iface, []):
                names.append(name)
        return sorted(set(names))

    @staticmethod
    def solverversion(subsolver: str) -> Optional[str]:
        return None

    @staticmethod
    def version() -> Optional[str]:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("paramita")
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        if not self.supported():
            raise ModuleNotFoundError(
                "CPM_paramita: Install `cpmpy[paramita]` and build at least one solver plugin, e.g.\n"
                "  paramita plugins build cmake '#sat-solvers/cadical-3.0.0'\n"
                "  paramita plugins build cmake '#maxsat-solvers/mse22-EvalMaxSAT2022'\n"
                "See https://ulog.udl.cat/static/doc/paramita/"
            )

        from paramita.modelling import DimacsVariablePool
        from paramita.solvers import SatSolver, MaxSatSolver

        plugins = _paramita_plugins_by_iface()
        sat_names = {n for n, _ in plugins.get(SatSolver, [])}
        max_names = {n for n, _ in plugins.get(MaxSatSolver, [])}

        if subsolver is None or subsolver == "paramita":
            # Prefer a SatSolver for decision/assumptions/core; MaxSAT when only that is available
            if "Cadical300" in sat_names:
                subsolver = "Cadical300"
            elif sat_names:
                subsolver = sorted(sat_names)[0]
            elif "EvalMaxSAT2022" in max_names:
                subsolver = "EvalMaxSAT2022"
            elif max_names:
                subsolver = sorted(max_names)[0]
            else:
                raise ModuleNotFoundError("CPM_paramita: no SatSolver or MaxSatSolver plugins found")
        elif subsolver.startswith("paramita:"):
            subsolver = subsolver[len("paramita:"):]

        self.is_maxsat = subsolver in max_names
        if self.is_maxsat:
            self.paramita_solver = MaxSatSolver.from_name(subsolver)
        elif subsolver in sat_names:
            self.paramita_solver = SatSolver.from_name(subsolver)
        else:
            raise ValueError(
                f"CPM_paramita: unknown subsolver '{subsolver}', "
                f"choose from {sorted(sat_names | max_names)}"
            )

        self.pool = DimacsVariablePool()
        self.ivarmap = dict()
        self.encoding = "auto"
        self.objective_ = None
        self.assumption_vars = None
        self.hard_paramita = []  # buffered hard modelling exprs (for switching to MaxSAT)

        super().__init__(name="paramita:" + subsolver, cpm_model=cpm_model)

    @property
    def native_model(self):
        return self.paramita_solver

    def has_objective(self) -> bool:
        return self.objective_ is not None

    def _int2bool_user_vars(self) -> Set[_BoolVarImpl]:
        for cpm_var in self.user_vars:
            if isinstance(cpm_var, _NumVarImpl) and not cpm_var.is_bool():
                if cpm_var.name not in self.ivarmap:
                    _, cons = _encode_int_var(
                        self.ivarmap, cpm_var, _decide_encoding(cpm_var, None, encoding=self.encoding)
                    )
                    for cpm_expr in self.transform(cons):
                        self._post_paramita(self._to_paramita(cpm_expr))
                for bv in self.ivarmap[cpm_var.name].vars().flatten():
                    self.solver_var(bv)
            else:
                self.solver_var(cpm_var)
        return replace_int_user_vars(self.user_vars, self.ivarmap)

    def solver_var(self, cpm_var):
        if isinstance(cpm_var, _NumVarImpl):
            assert cpm_var.is_bool(), f"CPM_paramita.solver_var only supports Boolean variables, not {cpm_var}"
            if isinstance(cpm_var, NegBoolView):
                return -self.pool.get_variable(cpm_var._bv.name)
            return self.pool.get_variable(cpm_var.name)

        if is_int(cpm_var):
            return cpm_var

        raise NotImplementedError(f"CPM_paramita: variable {cpm_var} not supported")

    def transform(self, cpm_expr: NestedBoolExprLike) -> list[Expression]:
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons)
        cpm_cons = push_down_negation(cpm_cons)
        cpm_cons = decompose_linear(
            cpm_cons,
            supported=self.supported_global_constraints,
            supported_reified=self.supported_reified_global_constraints,
            csemap=self._csemap,
        )
        cpm_cons = simplify_boolean(cpm_cons)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap, ivarmap=self.ivarmap)
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
        cpm_cons = linearize_constraint(
            cpm_cons,
            supported=frozenset({"sum", "wsum", "->", "and", "or"}) | self.supported_global_constraints,
            csemap=self._csemap,
        )
        cpm_cons = int2bool(cpm_cons, self.ivarmap, encoding=self.encoding, csemap=self._csemap)
        return cpm_cons

    def _to_paramita(self, cpm_expr):
        """Map a transformed CPMpy expression to a Paramita modelling expression."""
        from paramita.modelling import Bool, If, Iff

        if isinstance(cpm_expr, bool) or cpm_expr is True or cpm_expr is False:
            return bool(cpm_expr)

        elif isinstance(cpm_expr, BoolVal):
            return bool(cpm_expr.args[0])

        elif is_int(cpm_expr):
            return int(cpm_expr)

        elif isinstance(cpm_expr, NegBoolView):
            return ~Bool(cpm_expr._bv.name)

        elif isinstance(cpm_expr, _BoolVarImpl):
            return Bool(cpm_expr.name)

        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == "and":
                args = [self._to_paramita(a) for a in cpm_expr.args]
                expr = args[0]
                for a in args[1:]:
                    expr = expr & a
                return expr
            elif cpm_expr.name == "or":
                args = [self._to_paramita(a) for a in cpm_expr.args]
                expr = args[0]
                for a in args[1:]:
                    expr = expr | a
                return expr
            elif cpm_expr.name == "not":
                return ~self._to_paramita(cpm_expr.args[0])
            elif cpm_expr.name == "->":
                return If(self._to_paramita(cpm_expr.args[0]), self._to_paramita(cpm_expr.args[1]))
            elif cpm_expr.name == "sum":
                args = [self._to_paramita(a) for a in cpm_expr.args]
                expr = args[0]
                for a in args[1:]:
                    expr = expr + a
                return expr
            elif cpm_expr.name == "wsum":
                weights, lits = cpm_expr.args
                terms = [int(w) * self._to_paramita(x) for w, x in zip(weights, lits)]
                expr = terms[0]
                for t in terms[1:]:
                    expr = expr + t
                return expr
            # remaining Operator.allowed names ("-", "sub") should be gone after transform
            raise NotSupportedError(f"CPM_paramita: operator {cpm_expr} should have been transformed")

        elif isinstance(cpm_expr, Comparison):
            lhs = self._to_paramita(cpm_expr.args[0])
            rhs = self._to_paramita(cpm_expr.args[1])
            if cpm_expr.name == "==":
                # Bool <-> Bool uses Iff when both sides are Boolean-shaped; otherwise PB/arith EQ
                if is_bool(cpm_expr.args[0]) and is_bool(cpm_expr.args[1]):
                    return Iff(lhs, rhs)
                return lhs == rhs
            elif cpm_expr.name == "!=":
                return ~(lhs == rhs) if not (is_bool(cpm_expr.args[0]) and is_bool(cpm_expr.args[1])) else ~Iff(lhs, rhs)
            elif cpm_expr.name == "<=":
                return lhs <= rhs
            elif cpm_expr.name == "<":
                return lhs < rhs
            elif cpm_expr.name == ">=":
                return lhs >= rhs
            elif cpm_expr.name == ">":
                return lhs > rhs

        raise NotSupportedError(f"CPM_paramita: cannot map expression to Paramita modelling: {cpm_expr}")

    def _post_paramita(self, paramita_expr):
        from paramita.modelling import add_to_container, add_to_weighted_container, to_cnf

        if self.is_maxsat:
            add_to_weighted_container(0, paramita_expr, to_cnf, self.paramita_solver, self.pool)
        else:
            # buffer until/if we switch to MaxSAT for an objective
            self.hard_paramita.append(paramita_expr)
            add_to_container(paramita_expr, to_cnf, self.paramita_solver, self.pool)

    def _ensure_maxsat(self):
        """Switch from SatSolver to a MaxSatSolver, re-posting buffered hard constraints."""
        if self.is_maxsat:
            return
        from paramita.solvers import MaxSatSolver
        from paramita.modelling import add_to_weighted_container, to_cnf

        max_names = {n for n, _ in _paramita_plugins_by_iface().get(MaxSatSolver, [])}
        if "EvalMaxSAT2022" in max_names:
            name = "EvalMaxSAT2022"
        elif max_names:
            name = sorted(max_names)[0]
        else:
            raise NotSupportedError(
                "CPM_paramita: objectives require a MaxSatSolver plugin "
                "(e.g. build '#maxsat-solvers/mse22-EvalMaxSAT2022')"
            )
        self.paramita_solver = MaxSatSolver.from_name(name)
        self.is_maxsat = True
        self.name = "paramita:" + name
        for expr in self.hard_paramita:
            add_to_weighted_container(0, expr, to_cnf, self.paramita_solver, self.pool)

    def add(self, cpm_expr: NestedBoolExprLike) -> "CPM_paramita":
        get_variables(cpm_expr, collect=self.user_vars)
        for con in self.transform(cpm_expr):
            self._post_paramita(self._to_paramita(con))
        return self

    __add__ = add

    def objective(self, expr, minimize):
        self._ensure_maxsat()
        if self.objective_ is not None:
            raise NotSupportedError("CPM_paramita: objective can only be set once")

        from paramita.modelling import Bool, add_to_weighted_container, to_cnf

        get_variables(expr, collect=self.user_vars)
        self.objective_ = expr

        # MaxSAT soft clauses prefer satisfied literals (maximize satisfied weight)
        obj_expr = -expr if minimize else expr
        weights, xs, _, extra_cons = to_cnf_objective(
            obj_expr,
            encoding=self.encoding,
            csemap=self._csemap,
            ivarmap=self.ivarmap,
            supported=self.supported_global_constraints,
            supported_reified=self.supported_reified_global_constraints,
        )
        self.add(extra_cons)

        if len(weights) == 0:
            return

        terms = []
        for w, x in zip(weights, xs):
            if isinstance(x, NegBoolView):
                lit = ~Bool(x._bv.name)
            else:
                lit = Bool(x.name)
            terms.append(int(w) * lit)
        paramita_obj = terms[0]
        for t in terms[1:]:
            paramita_obj = paramita_obj + t
        add_to_weighted_container(paramita_obj, True, to_cnf, self.paramita_solver, self.pool)

    def solve(self, time_limit: Optional[float] = None, assumptions: Optional[Iterable[_BoolVarImpl]] = None):
        from paramita.solvers import MAXSAT_ANSWER

        self.user_vars = self._int2bool_user_vars()

        if assumptions is None:
            assum_lits = []
            self.assumption_vars = None
        else:
            assumptions = list(assumptions)
            self.assumption_vars = assumptions
            assum_lits = self.solver_vars(assumptions)

        t0 = time.time()
        timer = None
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            if hasattr(self.paramita_solver, "interrupt"):
                timer = Timer(time_limit, lambda s: s.interrupt(), [self.paramita_solver])
                timer.start()

        result = self.paramita_solver.solve(assumptions=assum_lits)

        if timer is not None:
            timer.cancel()

        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = time.time() - t0

        if self.is_maxsat:
            if result == MAXSAT_ANSWER.UNSATISFIABLE:
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            elif result == MAXSAT_ANSWER.UNKNOWN:
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            elif self.has_objective() and result == MAXSAT_ANSWER.OPTIMAL:
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            elif result in (MAXSAT_ANSWER.OPTIMAL, MAXSAT_ANSWER.SATISFIABLE):
                # decision problem (or suboptimal): report FEASIBLE
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            else:
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:
            if result is True:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            elif result is False:
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            else:
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN

        has_sol = self._solve_return(self.cpm_status)

        if has_sol:
            # EvalMaxSAT often lacks val_lit; read a full model instead
            sol = set(self.paramita_solver.model()) if self.is_maxsat else None
            for cpm_var in self.user_vars:
                if isinstance(cpm_var, NegBoolView):
                    vid = self.pool.get_variable(cpm_var._bv.name)
                    if sol is not None:
                        cpm_var._value = vid not in sol
                    else:
                        val = self.paramita_solver.val_lit(vid)
                        # unspecified free vars (e.g. after tautology) default to False
                        cpm_var._value = (not val) if val is not None else True
                elif isinstance(cpm_var, _BoolVarImpl):
                    vid = self.pool.get_variable(cpm_var.name)
                    if sol is not None:
                        cpm_var._value = vid in sol
                    else:
                        val = self.paramita_solver.val_lit(vid)
                        cpm_var._value = val if val is not None else False
                else:
                    raise ValueError(
                        f"Integer variables should have been encoded using int2bool, got {cpm_var}"
                    )
            for enc in self.ivarmap.values():
                enc._x._value = enc.decode()
            if self.has_objective():
                self.objective_value_ = self.objective_.value()
        else:
            for cpm_var in self.user_vars:
                cpm_var._value = None
            for enc in self.ivarmap.values():
                enc._x._value = None
            self.objective_value_ = None

        return has_sol

    def get_core(self):
        assert self.assumption_vars is not None, "get_core(): requires solve(assumptions=[...])"
        assert self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE, "get_core(): solver must return UNSAT"
        if self.is_maxsat:
            raise NotSupportedError("CPM_paramita: UNSAT cores are only supported with SatSolver plugins")
        core_lits = set(self.paramita_solver.core())
        return [v for v in self.assumption_vars if self.solver_var(v) in core_lits]