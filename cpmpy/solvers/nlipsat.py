#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## nlipsat.py
##
"""
    Interface to the NLIPSat toolkit.

    NLIPSat encodes a bounded polynomial integer program into weighted partial
    MaxSAT and solves it with RC2 or an external MaxSAT binary (MaxHS, WMaxCDCL,
    or Open-WBO). Products and constant powers are posted as polynomial
    monomials. Every other CPMpy constraint is decomposed and linearized first.

    The toolkit has no incremental native model: every ``solve()`` rebuilds the
    polynomial problem and the WCNF encoding. ``solveAll()`` therefore uses the
    generic nogood loop.

    Always use :func:`cp.SolverLookup.get("nlipsat") <cpmpy.solvers.utils.SolverLookup.get>`
    to instantiate the solver object.

    ============
    Installation
    ============

    The ``nlipsat`` package loads its encoders from the repository checkout, so
    it has to be installed editable:

    .. code-block:: console

        $ pip install -e "git+https://github.com/ZhenglingYangli/NLIPSat-Toolkit.git#egg=nlipsat"

    That also installs PySAT (with ``pblib``) and ``z3-solver``, which the
    toolkit declares. RC2 needs no extra binary. ``maxhs``, ``wmaxcdcl`` and
    ``openwbo`` are used only when the corresponding executable from the
    toolkit's ``solvers/maxsat`` directory is present.

    Subsolvers are selected with ``nlipsat:<name>``, for example
    ``model.solve(solver="nlipsat:rc2")``. The encoding (``BIN``, ``OH`` or
    ``UNA``) and order-decomposition options can be set on the constructor or
    on ``solve()``.

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_nlipsat
"""
import math
import os
import time
import warnings
from threading import Timer
from fractions import Fraction
from typing import Optional

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal, NestedBoolExprLike
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.variables import _BoolVarImpl, _NumVarImpl, NegBoolView, boolvar
from ..expressions.utils import is_num, eval_comparison
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list
from ..transformations.safening import no_partial_functions, safen_objective
from ..transformations.negation import push_down_negation, push_down_negation_objective
from ..transformations.linearize import (
    decompose_linear,
    decompose_linear_objective,
    linearize_constraint,
    linearize_reified_variables,
    only_positive_bv,
    only_positive_bv_wsum,
)
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.reification import reify_rewrite, only_bv_reifies, only_implies
from ..transformations.comparison import only_numexpr_equality


_ENCODINGS = frozenset({"OH", "UNA", "BIN"})
_DECOMP_STRATEGIES = frozenset({"sequential", "binary_tree"})
_SUBSOLVERS = ("rc2", "maxhs", "wmaxcdcl", "openwbo")
# Names kept as polynomial monomials, plus the linear operators linearize already emits.
_LINEARIZE_SUPPORTED = frozenset({"sum", "wsum", "sub", "mul", "pow", "->"})
_REIFY_SUPPORTED = frozenset({"sum", "wsum", "sub", "mul", "pow"})
_OBJECTIVE_SUPPORTED = frozenset({"sum", "wsum", "mul", "pow"})
_SOLVE_OPTIONS = frozenset({
    "encoding", "use_decomposition", "decomp_threshold", "decomp_strategy",
    "decomp_exact", "decomp_shared",
})


def _as_int(value):
    """Integer coefficient or constant. Rejects floats so they are not truncated."""
    if isinstance(value, BoolVal):
        return int(bool(value.value()))
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    # numpy integers, but not numpy floats
    if type(value).__name__ in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
        return int(value)
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
    except ModuleNotFoundError:
        pass
    raise NotSupportedError(f"CPM_nlipsat: expected an integer coefficient, got {value!r}")


def _poly_add(left, right):
    out = dict(left)
    for key, coeff in right.items():
        out[key] = out.get(key, 0) + coeff
        if out[key] == 0:
            del out[key]
    return out


def _poly_scale(poly, factor):
    if factor == 0:
        return {}
    return {key: coeff * factor for key, coeff in poly.items()}


def _poly_mul(left, right):
    if not left or not right:
        return {}
    out = {}
    for key_l, coeff_l in left.items():
        for key_r, coeff_r in right.items():
            degrees = {}
            for name, exp in key_l + key_r:
                degrees[name] = degrees.get(name, 0) + exp
            key = tuple(sorted((name, exp) for name, exp in degrees.items() if exp > 0))
            coeff = coeff_l * coeff_r
            if coeff == 0:
                continue
            out[key] = out.get(key, 0) + coeff
            if out[key] == 0:
                del out[key]
    return out


def _poly_pow(poly, exponent):
    exponent = _as_int(exponent)
    if exponent < 0:
        raise NotSupportedError("CPM_nlipsat: only nonnegative integer powers are supported")
    result = {(): 1}
    for _ in range(exponent):
        result = _poly_mul(result, poly)
    return result


def _pow_bounds(lb, ub, exponent):
    """Exact bounds of ``x**exponent`` on an integer interval."""
    if exponent == 0:
        return 1, 1
    values = [lb ** exponent, ub ** exponent]
    if exponent % 2 == 0 and lb <= 0 <= ub:
        values.append(0)
    return min(values), max(values)


def _mul_bounds(left, right):
    products = (left[0] * right[0], left[0] * right[1], left[1] * right[0], left[1] * right[1])
    return min(products), max(products)


class CPM_nlipsat(SolverInterface):
    """
    Interface to the NLIPSat toolkit.

    Creates the following attributes (see parent constructor for more):

    - ``nlip_problem``: the polynomial problem dict built by the last ``solve()``
    - ``encoding``: ``"BIN"`` (default), ``"OH"`` or ``"UNA"``
    - ``subsolver``: ``"rc2"`` (default), ``"maxhs"``, ``"wmaxcdcl"`` or ``"openwbo"``

    ``native_model`` returns ``nlip_problem``. NLIPSat exposes no solver object
    that accepts posted constraints, so :class:`~cpmpy.expressions.globalconstraints.DirectConstraint`
    is not supported.

    Documentation of the toolkit:
    https://github.com/ZhenglingYangli/NLIPSat-Toolkit
    """

    supported_global_constraints = frozenset({"mul", "pow"})
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        # try to import the package
        try:
            from nlipsat import EncodingConfig
            EncodingConfig()  # importing the config also loads codes/solver
            return True
        except ModuleNotFoundError:
            return False
        except ImportError as exc:
            warnings.warn(
                "NLIPSat is installed but its encoders could not be imported "
                f"({exc}). Install the checkout editable: "
                'pip install -e "git+https://github.com/ZhenglingYangli/NLIPSat-Toolkit.git#egg=nlipsat"'
            )
            return False

    @classmethod
    def version(cls) -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("nlipsat")
        except PackageNotFoundError:
            return None

    @staticmethod
    def solvernames(installed: bool = True):
        """
            Returns MaxSAT backends supported by NLIPSat.

            Arguments:
                installed (boolean): whether to filter the solvernames to those installed on your system (default True)

            Returns:
                list of solver names
        """
        if not CPM_nlipsat.supported():
            warnings.warn("NLIPSat is not installed or not supported on this system.")
            return []
        if not installed:
            return list(_SUBSOLVERS)
        from nlipsat import EncodingConfig
        cfg = EncodingConfig()
        paths = {
            "maxhs": cfg.maxhs_path,
            "wmaxcdcl": cfg.wmaxcdcl_path,
            "openwbo": cfg.openwbo_path,
        }
        names = ["rc2"]
        for name, path in paths.items():
            if os.path.isfile(path) and os.access(path, os.X_OK):
                names.append(name)
        return names

    @classmethod
    def solverversion(cls, subsolver: str) -> Optional[str]:
        """
        Returns the version of the requested subsolver.

        Arguments:
            subsolver (str): name of the subsolver

        Returns:
            Version number of the subsolver if installed, else None
        """
        name = subsolver.lower()
        if name not in _SUBSOLVERS:
            raise ValueError(f"Subsolver '{subsolver}' not found in the list of available solvers.")
        if name not in CPM_nlipsat.solvernames(installed=True):
            return None
        # External MaxSAT binaries do not expose a version string; report the toolkit version.
        return cls.version()

    def __init__(self, cpm_model=None, subsolver=None, encoding: str = "BIN",
                 use_decomposition: bool = False, decomp_threshold: int = 3,
                 decomp_strategy: str = "sequential", decomp_exact: bool = True,
                 decomp_shared: bool = True):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: str, one of ``rc2`` (default), ``maxhs``, ``wmaxcdcl``, ``openwbo``
        - encoding: ``BIN`` (default), ``OH`` or ``UNA``
        - use_decomposition: enable order decomposition for high-degree BIN terms
        - decomp_threshold: minimum degree that triggers decomposition
        - decomp_strategy: ``sequential`` or ``binary_tree``
        - decomp_exact: exact multiplication semantics (``False`` is the toolkit's relaxed mode)
        - decomp_shared: share decomposed subexpressions
        """
        if not self.supported():
            raise ModuleNotFoundError(
                "CPM_nlipsat: Install the python package 'cpmpy[nlipsat]' and the editable "
                "NLIPSat checkout. See the installation instructions in this module."
            )

        self.subsolver = self._check_subsolver(subsolver)
        self.encoding = self._check_encoding(encoding)
        self.use_decomposition = use_decomposition
        self.decomp_threshold = decomp_threshold
        self.decomp_strategy = self._check_strategy(decomp_strategy)
        self.decomp_exact = decomp_exact
        self.decomp_shared = decomp_shared

        # accumulated transformed constraints; the WCNF is rebuilt on every solve
        self._constraints: list[Expression] = []
        self._objective_expr: Optional[Expression] = None
        self._objective_flat: Optional[Expression | int] = None
        self._objective_is_min = True
        self._false_bv = None
        self.nlip_problem: dict = {}

        # initialise everything else and post the constraints/objective
        super().__init__(name="nlipsat", cpm_model=cpm_model)

    @staticmethod
    def _check_subsolver(subsolver):
        if subsolver is None:
            return "rc2"
        name = subsolver.lower()
        if name not in _SUBSOLVERS:
            raise ValueError(f"CPM_nlipsat: unknown subsolver '{subsolver}', choose from {list(_SUBSOLVERS)}")
        if name not in CPM_nlipsat.solvernames(installed=True):
            raise ModuleNotFoundError(f"CPM_nlipsat: subsolver '{name}' is not installed on this system")
        return name

    @staticmethod
    def _check_encoding(encoding):
        name = encoding.upper()
        if name not in _ENCODINGS:
            raise ValueError(f"CPM_nlipsat: unknown encoding '{encoding}', choose from {sorted(_ENCODINGS)}")
        return name

    @staticmethod
    def _check_strategy(strategy):
        if strategy not in _DECOMP_STRATEGIES:
            raise ValueError(
                f"CPM_nlipsat: unknown decomp_strategy '{strategy}', choose from {sorted(_DECOMP_STRATEGIES)}"
            )
        return strategy

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.nlip_problem

    def solver_var(self, cpm_var):
        """
           Creates solver variable for cpmpy variable
           or returns from cache if previously created
           or returns a constant if the variable is a constant
        """
        if isinstance(cpm_var, NegBoolView):
            key = cpm_var.name
            if key not in self._varmap:
                self._varmap[key] = ("neg", self.solver_var(cpm_var._bv))
            return self._varmap[key]
        if isinstance(cpm_var, _NumVarImpl):
            return self._name_for(cpm_var)
        if is_num(cpm_var):
            return _as_int(cpm_var)
        raise NotImplementedError(f"CPM_nlipsat: unknown variable {cpm_var}")

    def _name_for(self, cpm_var: _NumVarImpl) -> str:
        """Stable NLIPSat variable name. CPMpy names can be missing or duplicated."""
        if isinstance(cpm_var, NegBoolView):
            raise ValueError("CPM_nlipsat: internal error, NegBoolView has no polynomial variable")
        key = cpm_var.name
        if key not in self._varmap or not isinstance(self._varmap[key], str):
            self._varmap[key] = f"v{sum(isinstance(v, str) and v.startswith('v') for v in self._varmap.values())}"
        return self._varmap[key]

    def _register(self, cpm_var: _NumVarImpl, domains: dict, revmap: dict) -> str:
        name = self._name_for(cpm_var)
        lb, ub = cpm_var.get_bounds()
        lb, ub = _as_int(lb), _as_int(ub)
        if lb > ub:
            raise ValueError(f"CPM_nlipsat: variable {cpm_var} has empty domain [{lb}, {ub}]")
        domains[name] = (lb, ub)
        revmap[name] = cpm_var
        return name

    def has_objective(self) -> bool:
        return self._objective_expr is not None

    def objective(self, expr: Expression, minimize: bool = True) -> None:
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored
        """
        self._objective_expr = expr
        self._objective_is_min = minimize
        get_variables(expr, self.user_vars)

        obj, safe_cons = safen_objective(expr)
        obj = push_down_negation_objective(obj)
        obj, decomp_cons = decompose_linear_objective(
            obj,
            supported=self.supported_global_constraints,
            supported_reified=self.supported_reified_global_constraints,
            csemap=self._csemap,
        )
        obj, flat_cons = flatten_objective(obj, supported=_OBJECTIVE_SUPPORTED, csemap=self._csemap)
        self.add(safe_cons + decomp_cons + flat_cons)

        if is_num(obj):
            self._objective_flat = _as_int(obj)
        else:
            self._objective_flat = only_positive_bv_wsum(obj)

    def transform(self, cpm_expr: NestedBoolExprLike) -> list[Expression]:
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the :ref:`Adding a new solver` docs on readthedocs for more information.

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
        cpm_cons = reify_rewrite(cpm_cons, supported=_REIFY_SUPPORTED, csemap=self._csemap)
        cpm_cons = only_numexpr_equality(cpm_cons, supported=_REIFY_SUPPORTED, csemap=self._csemap)
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap)
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
        cpm_cons = linearize_constraint(cpm_cons, supported=_LINEARIZE_SUPPORTED, csemap=self._csemap)
        cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)
        return cpm_cons

    def add(self, cpm_expr: NestedBoolExprLike) -> "CPM_nlipsat":
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

            Arguments:
                cpm_expr (NestedBoolExprLike): CPMpy expression, or list thereof

            Returns:
                self
        """
        get_variables(cpm_expr, collect=self.user_vars)
        self._constraints.extend(self.transform(cpm_expr))
        return self

    __add__ = add  # avoid redirect in superclass

    def solve(self, time_limit: Optional[float] = None, **kwargs):
        """
            Call the NLIPSat solver

            Arguments:
                time_limit:  maximum solve time in seconds (float, optional).
                             Limits both the encoding phase and the MaxSAT backend.
                kwargs:      encoding options, overriding the values set on the constructor:

            - encoding: ``BIN`` (default), ``OH`` or ``UNA``
            - use_decomposition: order decomposition for high-degree binary terms
            - decomp_threshold: minimum degree that triggers decomposition
            - decomp_strategy: ``sequential`` or ``binary_tree``
            - decomp_exact: exact (default) or relaxed multiplication semantics
            - decomp_shared: share decomposed subexpressions across terms
        """
        if time_limit is not None and time_limit <= 0:
            raise ValueError(f"Time limit must be positive but was {time_limit}")
        unknown = set(kwargs) - _SOLVE_OPTIONS
        if unknown:
            raise ValueError(f"CPM_nlipsat: unknown solver arguments {sorted(unknown)}")

        encoding = self._check_encoding(kwargs.get("encoding", self.encoding))
        use_decomposition = kwargs.get("use_decomposition", self.use_decomposition)
        decomp_threshold = kwargs.get("decomp_threshold", self.decomp_threshold)
        decomp_strategy = self._check_strategy(kwargs.get("decomp_strategy", self.decomp_strategy))
        decomp_exact = kwargs.get("decomp_exact", self.decomp_exact)
        decomp_shared = kwargs.get("decomp_shared", self.decomp_shared)

        for cpm_var in self.user_vars:
            cpm_var.clear()
        self.objective_value_ = None

        t0 = time.time()
        self.cpm_status = SolverStatus(self.name)
        deadline = None if time_limit is None else t0 + time_limit

        problem = self._build_problem()
        self.nlip_problem = problem

        from nlipsat import EncodingConfig, build_wcnf
        _install_nlipsat_workarounds()
        cfg = EncodingConfig(
            encoding=encoding,
            use_decomposition=use_decomposition,
            decomp_threshold=decomp_threshold,
            decomp_strategy=decomp_strategy,
            decomp_exact=decomp_exact,
            decomp_shared=decomp_shared,
            # Nonzero lower bounds are shifted inside preprocessing. The encoders
            # themselves treat a domain as 0..ub unless that shift has been applied.
            enable_preprocess=True,
            encoding_deadline=0.0 if deadline is None else deadline,
        )
        if time_limit is not None:
            cfg.external_solver_timeout = max(1, math.ceil(time_limit))

        try:
            wcnf, name2idx, vpool = build_wcnf(problem, encoding, cfg)
        except Exception as exc:
            if exc.__class__.__name__ == "EncodingTimeoutError":
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
                self.cpm_status.solve_time = time.time() - t0
                return self._solve_return(self.cpm_status)
            raise

        remaining = None if deadline is None else deadline - time.time()
        if remaining is not None and remaining <= 0:
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            self.cpm_status.solve_time = time.time() - t0
            return self._solve_return(self.cpm_status)

        if self.subsolver == "rc2":
            exitstatus, cost, assignment, _solve_time = self._solve_rc2(wcnf, remaining)
        else:
            exitstatus, cost, assignment, _solve_time = self._solve_external(wcnf, cfg, remaining)

        self.cpm_status.exitstatus = exitstatus
        self.cpm_status.solve_time = time.time() - t0
        has_sol = self._solve_return(self.cpm_status)
        if not has_sol:
            return False

        self._assign_values(assignment, encoding, name2idx, wcnf, vpool)
        if self.has_objective():
            self.objective_value_ = self._objective_from_assignment(wcnf, cost)
        return True

    def _solve_rc2(self, wcnf, remaining):
        from pysat.examples.rc2 import RC2

        solver = RC2(wcnf, adapt=True, exhaust=True, trim=3)
        # Core exhaustion calls the SAT oracle with expect_interrupt=False, so a
        # time limit never aborts that call. Forward the flag from compute() and
        # stop once interrupt() has been requested. The oracle is deleted afterwards.
        oracle_call = solver._call_oracle

        def _call_oracle(assumptions=[], expect_interrupt=False):
            if solver.interrupted:
                return False
            return oracle_call(
                assumptions,
                expect_interrupt=bool(solver.expect_interrupt or expect_interrupt),
            )

        solver._call_oracle = _call_oracle
        timer = None
        interrupted = False
        t0 = time.time()
        try:
            if remaining is None:
                solution = solver.compute()
            else:
                timer = Timer(remaining, solver.interrupt)
                timer.start()
                solution = solver.compute(expect_interrupt=True)
                interrupted = bool(getattr(solver, "interrupted", False))
            if solution is None:
                return (
                    ExitStatus.UNKNOWN if interrupted else ExitStatus.UNSATISFIABLE,
                    None,
                    None,
                    time.time() - t0,
                )
            cost = solver.cost
            assignment = list(solution)
        finally:
            if timer is not None:
                timer.cancel()
            solver.delete()

        if interrupted:
            status = ExitStatus.FEASIBLE
        elif self.has_objective():
            status = ExitStatus.OPTIMAL
        else:
            status = ExitStatus.FEASIBLE
        return status, cost, assignment, time.time() - t0

    def _solve_external(self, wcnf, cfg, remaining):
        import subprocess
        from solver.solve import solve_external

        paths = {
            "maxhs": cfg.maxhs_path,
            "wmaxcdcl": cfg.wmaxcdcl_path,
            "openwbo": cfg.openwbo_path,
        }
        path = paths[self.subsolver]
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            raise ModuleNotFoundError(f"CPM_nlipsat: subsolver '{self.subsolver}' was not found at {path}")
        timeout = 0 if remaining is None else max(1, math.ceil(remaining))
        t0 = time.time()
        try:
            cost, assignment, status = solve_external(wcnf, cfg.workdir, path, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ExitStatus.UNKNOWN, None, None, time.time() - t0

        if status == "OPTIMAL":
            exitstatus = ExitStatus.OPTIMAL if self.has_objective() else ExitStatus.FEASIBLE
        elif status == "FEASIBLE":
            exitstatus = ExitStatus.FEASIBLE
        elif status == "UNSAT":
            exitstatus = ExitStatus.UNSATISFIABLE
        else:
            exitstatus = ExitStatus.UNKNOWN
        if exitstatus in (ExitStatus.UNSATISFIABLE, ExitStatus.UNKNOWN) and not assignment:
            return exitstatus, None, None, time.time() - t0
        return exitstatus, cost, assignment, time.time() - t0

    def _assign_values(self, assignment, encoding, name2idx, wcnf, vpool):
        from tools.verify import decode_assignment

        processed = getattr(wcnf, "_processed_problem", None) or self.nlip_problem
        decoded = decode_assignment(assignment or [], encoding, name2idx, processed, vpool)
        meta = getattr(wcnf, "_preprocess_meta", None) or {}
        shifts = meta.get("variable_shifts", {}) or {}
        for name, shift in shifts.items():
            if name in decoded:
                decoded[name] = _as_int(decoded[name]) + _as_int(shift)
        probed = (meta.get("objective", {}) or {}).get("probed_variables", {}) or {}
        for name, value in probed.items():
            decoded[name] = _as_int(value) + _as_int(shifts.get(name, 0))

        domains, revmap = {}, {}
        for cpm_var in self.user_vars:
            self._register(cpm_var, domains, revmap)
        for name, cpm_var in revmap.items():
            if name not in decoded:
                continue
            value = _as_int(decoded[name])
            cpm_var._value = bool(value) if cpm_var.is_bool() else value

    def _objective_from_assignment(self, wcnf, cost):
        """Prefer the value of the posted CPMpy objective; fall back to the toolkit's cost decoding."""
        if self._objective_expr is not None:
            value = self._objective_expr.value()
            if value is not None:
                value = _as_int(value) if not isinstance(value, float) else value
                if isinstance(value, float) and value == int(value):
                    return int(value)
                return value
        if cost is None or cost < 0:
            return None
        return _recover_objective(wcnf, cost)

    def _build_problem(self):
        domains, revmap = {}, {}
        for cpm_var in self.user_vars:
            self._register(cpm_var, domains, revmap)

        constraints = []
        for expr in self._constraints:
            self._emit(expr, constraints, domains, revmap)

        if self._objective_flat is None:
            objective = {"sense": "max", "terms": []}
        else:
            poly = self._to_poly(self._objective_flat, domains, revmap)
            objective = {
                "sense": "min" if self._objective_is_min else "max",
                "terms": _poly_terms(poly, keep_constant=True),
            }

        variables = {name: {"lb": lb, "ub": ub} for name, (lb, ub) in domains.items()}
        return {"variables": variables, "constraints": constraints, "objective": objective}

    def _emit(self, expr, constraints, domains, revmap):
        if isinstance(expr, BoolVal):
            if not expr.value():
                constraints.append(self._false_constraint(domains, revmap))
            return
        if isinstance(expr, Comparison):
            self._emit_comparison(expr, None, constraints, domains, revmap)
            return
        if isinstance(expr, Operator) and expr.name == "->":
            cond, sub = expr.args
            if not isinstance(sub, Comparison):
                raise NotImplementedError(f"CPM_nlipsat: cannot post implication {expr}")
            positive = True
            if isinstance(cond, NegBoolView):
                positive = False
                cond = cond._bv
            if not isinstance(cond, _BoolVarImpl):
                raise NotImplementedError(f"CPM_nlipsat: implication condition must be a Boolean variable, got {cond}")
            self._emit_comparison(sub, (cond, positive), constraints, domains, revmap)
            return
        if isinstance(expr, DirectConstraint):
            raise NotSupportedError("CPM_nlipsat does not support DirectConstraint")
        raise NotImplementedError(f"CPM_nlipsat: cannot post {expr}")

    def _false_constraint(self, domains, revmap):
        """A non-empty polynomial constraint that no assignment satisfies.

        NLIPSat drops constraints whose term list is empty, so a constant
        contradiction has to mention a variable.
        """
        if self._false_bv is None:
            self._false_bv = boolvar(name=f"nlip_false_{id(self)}")
        name = self._register(self._false_bv, domains, revmap)
        return {"terms": [{"c": 1, "vars": {name: 1}}], "sense": "<=", "rhs": -1}

    def _emit_comparison(self, expr, indicator, constraints, domains, revmap):
        lhs, rhs = expr.args
        poly = _poly_add(self._to_poly(lhs, domains, revmap), _poly_scale(self._to_poly(rhs, domains, revmap), -1))
        sense = expr.name
        if sense not in ("<=", ">=", "=="):
            raise NotImplementedError(f"CPM_nlipsat: comparison {expr} was not normalized")

        const = poly.get((), 0)
        var_poly = {key: coeff for key, coeff in poly.items() if key != ()}
        # var_poly + const  sense  0    <=>    var_poly  sense  -const
        rhs_value = -const

        if indicator is None:
            self._append_constraint(var_poly, sense, rhs_value, constraints, domains)
            return

        cond, positive = indicator
        bname = self._register(cond, domains, revmap)
        if sense == "==":
            self._emit_implied(var_poly, "<=", rhs_value, bname, positive, constraints, domains)
            self._emit_implied(var_poly, ">=", rhs_value, bname, positive, constraints, domains)
        else:
            self._emit_implied(var_poly, sense, rhs_value, bname, positive, constraints, domains)

    def _append_constraint(self, var_poly, sense, rhs_value, constraints, domains):
        if not var_poly:
            if not eval_comparison(sense, 0, rhs_value):
                constraints.append(self._false_constraint(domains, {}))
            return
        constraints.append({"terms": _poly_terms(var_poly, keep_constant=False), "sense": sense, "rhs": int(rhs_value)})

    def _emit_implied(self, var_poly, sense, rhs_value, bname, positive, constraints, domains):
        """Post ``b -> (poly sense rhs)`` as a Big-M polynomial inequality."""
        lb, ub = _poly_bounds(var_poly, domains)
        status = _constraint_status(lb, ub, sense, rhs_value)
        if status == "true":
            return
        if status == "false":
            # the indicator itself must be false
            if positive:
                constraints.append({"terms": [{"c": 1, "vars": {bname: 1}}], "sense": "<=", "rhs": 0})
            else:
                constraints.append({"terms": [{"c": 1, "vars": {bname: 1}}], "sense": ">=", "rhs": 1})
            return

        if sense == "<=":
            big_m = max(0, ub - rhs_value)
            # positive:    q <= K + M*(1-b)  <=>  q + M*b <= K+M
            # negated:     q <= K + M*b      <=>  q - M*b <= K
            coeff = big_m if positive else -big_m
            rhs = rhs_value + (big_m if positive else 0)
        else:  # >=
            big_m = max(0, rhs_value - lb)
            # positive:    q >= K - M*(1-b)  <=>  q - M*b >= K-M
            # negated:     q >= K - M*b      <=>  q + M*b >= K
            coeff = -big_m if positive else big_m
            rhs = rhs_value - (big_m if positive else 0)

        implied = dict(var_poly)
        key = ((bname, 1),)
        implied[key] = implied.get(key, 0) + coeff
        if implied[key] == 0:
            del implied[key]
        self._append_constraint(implied, sense, rhs, constraints, domains)

    def _to_poly(self, expr, domains, revmap):
        if isinstance(expr, BoolVal):
            return {(): _as_int(expr)}
        if isinstance(expr, NegBoolView):
            # ~b = 1 - b
            name = self._register(expr._bv, domains, revmap)
            return {(): 1, ((name, 1),): -1}
        if isinstance(expr, _NumVarImpl):
            name = self._register(expr, domains, revmap)
            return {((name, 1),): 1}
        if is_num(expr):
            return {(): _as_int(expr)}
        if not isinstance(expr, Expression):
            raise NotImplementedError(f"CPM_nlipsat: cannot encode {expr}")

        name = expr.name
        if name == "sum":
            acc = {}
            for arg in expr.args:
                acc = _poly_add(acc, self._to_poly(arg, domains, revmap))
            return acc
        if name == "wsum":
            weights, args = expr.args
            acc = {}
            for weight, arg in zip(weights, args):
                acc = _poly_add(acc, _poly_scale(self._to_poly(arg, domains, revmap), _as_int(weight)))
            return acc
        if name == "sub":
            left, right = expr.args
            return _poly_add(
                self._to_poly(left, domains, revmap),
                _poly_scale(self._to_poly(right, domains, revmap), -1),
            )
        if name == "-":
            return _poly_scale(self._to_poly(expr.args[0], domains, revmap), -1)
        if name == "mul":
            acc = {(): 1}
            for arg in expr.args:
                acc = _poly_mul(acc, self._to_poly(arg, domains, revmap))
            return acc
        if name == "pow":
            base, exponent = expr.args
            if isinstance(exponent, Expression):
                raise NotSupportedError(f"CPM_nlipsat: variable exponents are not supported in {expr}")
            return _poly_pow(self._to_poly(base, domains, revmap), exponent)
        raise NotImplementedError(f"CPM_nlipsat: cannot encode {expr} as a polynomial")


def _poly_terms(poly, keep_constant: bool):
    terms = []
    for key, coeff in poly.items():
        if coeff == 0:
            continue
        if key == ():
            if keep_constant:
                terms.append({"c": int(coeff), "vars": {}})
            continue
        terms.append({"c": int(coeff), "vars": {name: exp for name, exp in key}})
    return terms


def _poly_bounds(poly, domains):
    lb_total, ub_total = 0, 0
    for key, coeff in poly.items():
        if key == () or coeff == 0:
            continue
        bounds = (coeff, coeff)
        for name, exponent in key:
            if name not in domains:
                raise ValueError(f"CPM_nlipsat: variable {name} has no domain")
            bounds = _mul_bounds(bounds, _pow_bounds(domains[name][0], domains[name][1], exponent))
        lb_total += bounds[0]
        ub_total += bounds[1]
    return lb_total, ub_total


def _install_nlipsat_workarounds():
    """Correct two NLIPSat behaviours that change the set of models.

    Bound tightening ignores every nonlinear term and propagates the linear
    part on its own, so a feasible product constraint can be tightened to empty.
    Shifts of nonzero lower bounds still run; only that propagation is skipped.

    Unit propagation in the WCNF cleanup deletes a clause that became empty, so
    a contradiction is reported as a model. Its subsumption pass is also
    quadratic in the number of hard clauses. Unit propagation that keeps the
    empty clause is enough; subsumption does not change the set of models.
    """
    import tools.preprocessing as nlip_pre
    nlip_pre._bound_tightening = lambda *args, **kwargs: 0

    import solver.solve as nlip_solve

    def _preprocess_wcnf(wcnf):
        if not wcnf.hard:
            return
        units = {clause[0] for clause in wcnf.hard if len(clause) == 1}
        if not units:
            return
        new_hard = []
        for clause in wcnf.hard:
            if any(lit in units for lit in clause):
                if len(clause) == 1:
                    new_hard.append(clause)
                continue
            reduced = [lit for lit in clause if -lit not in units]
            # An empty clause is a contradiction. The toolkit dropped it.
            new_hard.append(reduced)
        wcnf.hard = new_hard

    nlip_solve._preprocess_wcnf = _preprocess_wcnf


def _constraint_status(lb, ub, sense, rhs):
    if sense == "<=":
        if ub <= rhs:
            return "true"
        if lb > rhs:
            return "false"
    elif sense == ">=":
        if lb >= rhs:
            return "true"
        if ub < rhs:
            return "false"
    elif sense == "==":
        if lb == ub == rhs:
            return "true"
        if ub < rhs or lb > rhs:
            return "false"
    return "maybe"


def _recover_objective(wcnf, cost):
    """Undo the toolkit's soft-weight and preprocessing transforms.

    Mirrors ``codes/solver/solve.py`` (``build_and_solve``): the encoded
    MaxSAT cost is not the NLIP objective until the GCD, negative-weight
    offset, constant term and preprocessing scale/sense flip are reversed.
    """
    total_soft = sum(wcnf.wght) if wcnf.wght else 0
    neg_offset = getattr(wcnf, "_neg_soft_total", 0)
    weight_gcd = getattr(wcnf, "_weight_gcd", 1) or 1
    value = ((total_soft - cost) - neg_offset) * weight_gcd
    value += getattr(wcnf, "_objective_constant", 0)

    meta = getattr(wcnf, "_preprocess_meta", None)
    if not meta:
        return int(value)
    obj_meta = meta.get("objective", {}) or {}
    const_shift = obj_meta.get("constant_shift", 0)
    sense_flip = obj_meta.get("sense_flip", False)
    scale_lcm = obj_meta.get("scale_lcm", 1) or 1
    scale_gcd = obj_meta.get("scale_gcd", 1) or 1

    if scale_lcm > 1 or scale_gcd > 1:
        scaled_const = int(Fraction(const_shift) * Fraction(scale_lcm, scale_gcd))
    else:
        scaled_const = int(const_shift)
    value += scaled_const
    if sense_flip:
        value = -value
    if scale_lcm > 1 or scale_gcd > 1:
        value = float(Fraction(value) * Fraction(scale_gcd, scale_lcm))
        if value == int(value):
            value = int(value)
    return value
