"""
Generic CPMpy solution checker.

Given a CPMpy Model, a CPMpy ``ExitStatus``, and optionally a variable-to-value
mapping (dict of name -> int/bool), verifies the solution across four stages:

  1. Completeness:  all model variables have an assigned value in var_map.
  2. Domain:        every assigned value lies within the variable's declared [lb, ub].
  3. Constraints:   every model constraint evaluates to True under the assignment.
  4. Objective:     (COP only) the computed objective value lies within the
                    theoretical bounds returned by get_bounds() on the objective
                    expression.

Only ``ExitStatus.FEASIBLE`` and ``ExitStatus.OPTIMAL`` have a solution to check.
Sub-optimal solutions (e.g. solver reported SUB-OPTIMAL or FEASIBLE when optimising)
are mapped to ``FEASIBLE`` by the output parsers and are verified like any other solution.
All other statuses (UNSATISFIABLE, UNKNOWN, NOT_RUN, ERROR) are skipped immediately
with ``CheckResult.skipped == True``.

The check is non-destructive: variable _value attributes are saved and restored
after each call.

Example usage::

    import cpmpy as cp
    from cpmpy.solvers.solver_interface import ExitStatus
    from cpmpy.tools.solution_checker import check_solution

    x = cp.intvar(0, 5, name="x")
    y = cp.intvar(0, 5, name="y")
    model = cp.Model([x + y == 7], minimize=x)

    result = check_solution(model, ExitStatus.FEASIBLE, {"x": 3, "y": 4})
    print(result)         # VALID, objective = 3
    print(result.valid)   # True

    result = check_solution(model, ExitStatus.FEASIBLE, {"x": 3, "y": 5})
    print(result)         # INVALID (constraint: ...)

    result = check_solution(model, ExitStatus.UNSATISFIABLE)
    print(result)         # SKIPPED (UNSATISFIABLE)
    print(result.skipped) # True
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cpmpy.solvers.solver_interface import ExitStatus


def _to_python_scalar(val):
    """Convert variable value to Python int or bool for assignment; avoid numpy in _value."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    return val


# ExitStatus values that carry a solution to verify
_SOLUTION_STATUSES = frozenset({ExitStatus.FEASIBLE, ExitStatus.OPTIMAL})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CheckViolation:
    """One specific violation found during solution checking."""

    stage: int    # 1 = completeness, 2 = domain, 3 = constraint, 4 = objective
    kind: str     # "unassigned" | "domain" | "constraint" | "constraint_error"
                  # | "objective" | "objective_mismatch" | "objective_error"
    message: str
    context: Any = None  # the constraint object, variable name, etc.

    def __str__(self) -> str:
        return f"[stage {self.stage} / {self.kind}] {self.message}"


@dataclass
class CheckResult:
    """Aggregated result of checking a solution against a CPMpy model."""

    exit_status: ExitStatus
    valid: bool                # False only when FEASIBLE/OPTIMAL and a violation was found
    violations: List[CheckViolation] = field(default_factory=list)
    objective_value: Optional[float] = None
    # Non-fatal notices about things the checker could not fully verify.
    # A non-empty warnings list means the result should be treated with caution
    # even when valid=True.
    warnings: List[str] = field(default_factory=list)

    # --- convenience views ---

    @property
    def skipped(self) -> bool:
        """True when the exit status carries no solution (nothing was checked)."""
        return self.exit_status not in _SOLUTION_STATUSES

    @property
    def errors(self) -> List[str]:
        """All violation messages as strings."""
        return [v.message for v in self.violations]

    def violations_by_stage(self, stage: int) -> List[CheckViolation]:
        return [v for v in self.violations if v.stage == stage]

    # --- summary ---

    def summary(self) -> str:
        if self.skipped:
            return f"SKIPPED ({self.exit_status.name})"
        if self.valid:
            obj_str = (
                f", objective = {self.objective_value}"
                if self.objective_value is not None
                else ""
            )
            warn_str = (
                f" [WARNINGS: {len(self.warnings)}]"
                if self.warnings
                else ""
            )
            return f"VALID{obj_str}{warn_str}"
        counts: Dict[str, int] = {}
        for v in self.violations:
            counts[v.kind] = counts.get(v.kind, 0) + 1
        parts = [f"{k}: {n}" for k, n in counts.items()]
        return "INVALID (" + ", ".join(parts) + ")"

    def __str__(self) -> str:
        lines = [self.summary()]
        for v in self.violations:
            lines.append(f"  {v}")
        for w in self.warnings:
            lines.append(f"  [WARNING] {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------

def check_solution(
    model,
    exit_status: ExitStatus,
    var_map: Optional[Dict[str, Any]] = None,
    *,
    expected_objective: Optional[int] = None,
    stop_on_unassigned: bool = True,
    stop_on_domain_error: bool = False,
    ignore_aux_vars: bool = True,
) -> CheckResult:
    """Check a solution given as a variable-to-value mapping against a CPMpy model.

    Arguments:
        model:                CPMpy Model instance.
        exit_status:          ``ExitStatus`` reported by the solver.
                              Only ``FEASIBLE`` and ``OPTIMAL`` trigger an actual
                              check; all other statuses (UNSATISFIABLE, UNKNOWN,
                              NOT_RUN, ERROR) return immediately with
                              ``CheckResult.skipped == True``.
        var_map:              dict mapping variable name (str) to int / bool value.
                              Required for FEASIBLE / OPTIMAL; ignored otherwise.
        expected_objective:   If provided (COP only), the objective value that the
                              solver declared (e.g. from a ``cost`` attribute or
                              ``o`` line).  Stage 4 will flag a violation when the
                              computed objective does not exactly match this value.
        stop_on_unassigned:   If True (default), abort after stage 1 when any
                              variables are unassigned.  This avoids misleading
                              cascade errors in stage 3.
        stop_on_domain_error: If True, abort after stage 2 on domain violations.
        ignore_aux_vars:      If True (default), auxiliary CPMpy variables (whose
                              names start with the internal prefixes ``IV`` or
                              ``BV``) that are missing from var_map are silently
                              ignored rather than reported as unassigned.

    Returns:
        CheckResult
    """
    if exit_status not in _SOLUTION_STATUSES:
        return CheckResult(exit_status=exit_status, valid=True)

    if var_map is None:
        raise ValueError(
            f"var_map is required when exit_status is {exit_status.name}"
        )
    from cpmpy.transformations.get_variables import get_variables_model, get_variables
    from cpmpy.expressions.utils import get_bounds, is_any_list
    from cpmpy.expressions.variables import _IV_PREFIX, _BV_PREFIX

    result = CheckResult(exit_status=exit_status, valid=True)

    def _is_aux_name(name: str) -> bool:
        return name.startswith(_IV_PREFIX) or name.startswith(_BV_PREFIX)

    def _missing_vars(expr) -> List[Any]:
        """Return vars used in expr that are not assigned in var_map."""
        return [v for v in get_variables(expr) if v.name not in var_map]

    def _only_aux_missing(expr) -> bool:
        """True if expr has missing vars and all missing vars are auxiliary IV/BV."""
        missing = _missing_vars(expr)
        return bool(missing) and ignore_aux_vars and all(_is_aux_name(v.name) for v in missing)

    # Collect all variables (ordered, unique) appearing in constraints + objective
    variables = get_variables_model(model)

    # ------------------------------------------------------------------ #
    # Stage 1 — completeness                                               #
    # ------------------------------------------------------------------ #
    for var in variables:
        if var.name not in var_map:
            is_aux = var.name.startswith(_IV_PREFIX) or var.name.startswith(_BV_PREFIX)
            if ignore_aux_vars and is_aux:
                continue
            result.violations.append(
                CheckViolation(
                    stage=1,
                    kind="unassigned",
                    message=f"Variable '{var.name}' has no assigned value in var_map",
                    context=var.name,
                )
            )
            result.valid = False

    if not result.valid and stop_on_unassigned:
        return result

    # ------------------------------------------------------------------ #
    # Stage 2 — domain                                                     #
    # ------------------------------------------------------------------ #
    for var in variables:
        if var.name not in var_map:
            continue
        val = var_map[var.name]
        lb, ub = var.get_bounds()
        try:
            if not (lb <= int(val) <= ub):
                result.violations.append(
                    CheckViolation(
                        stage=2,
                        kind="domain",
                        message=(
                            f"Variable '{var.name}' = {val} is outside "
                            f"declared domain [{lb}, {ub}]"
                        ),
                        context=var.name,
                    )
                )
                result.valid = False
        except (TypeError, ValueError) as exc:
            result.violations.append(
                CheckViolation(
                    stage=2,
                    kind="domain",
                    message=f"Variable '{var.name}': cannot convert value {val!r} to int: {exc}",
                    context=var.name,
                )
            )
            result.valid = False

    if not result.valid and stop_on_domain_error:
        return result

    # ------------------------------------------------------------------ #
    # Assign values; save old values for restoration                        #
    # ------------------------------------------------------------------ #
    saved = {var: var._value for var in variables}
    try:
        for var in variables:
            if var.name in var_map:
                var._value = _to_python_scalar(var_map[var.name])
            # else: leave as None (was already None or from a previous solve)

        # -------------------------------------------------------------- #
        # Stage 3 — constraint satisfaction                                #
        # -------------------------------------------------------------- #
        def _constraint_val_safe(val):
            """Convert constraint value to a type safe for None/truth check; avoid ambiguous array truth."""
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    return None  # empty array -> treat as None to avoid "truth of empty array" error
                if val.size == 1:
                    return _to_python_scalar(val.flat[0])
                return val  # multiple elements: leave as-is; truth check may still fail
            return _to_python_scalar(val)

        def _check_expr(c) -> None:
            if is_any_list(c):
                for sub in c:
                    _check_expr(sub)
                return

            # If only auxiliary vars are missing, skip this constraint check when
            # ignore_aux_vars=True. Some solver outputs do not report IV/BV vars.
            if _only_aux_missing(c):
                return

            try:
                val = c.value()
            except Exception as exc:
                result.violations.append(
                    CheckViolation(
                        stage=3,
                        kind="constraint_error",
                        message=f"Error evaluating constraint '{c}': {exc}",
                        context=c,
                    )
                )
                result.valid = False
                return

            val = _constraint_val_safe(val)

            # val is None if some variable in c has no value
            if val is None:
                result.violations.append(
                    CheckViolation(
                        stage=3,
                        kind="constraint_error",
                        message=(
                            f"Constraint '{c}' evaluated to None "
                            f"(some variable may be unassigned)"
                        ),
                        context=c,
                    )
                )
                result.valid = False
            elif isinstance(val, np.ndarray):
                # Avoid "truth value of array is ambiguous"
                result.violations.append(
                    CheckViolation(
                        stage=3,
                        kind="constraint_error",
                        message=f"Constraint '{c}' evaluated to array (ambiguous truth value)",
                        context=c,
                    )
                )
                result.valid = False
            elif not val:
                result.violations.append(
                    CheckViolation(
                        stage=3,
                        kind="constraint",
                        message=f"Constraint violated: {c}",
                        context=c,
                    )
                )
                result.valid = False

        for c in model.constraints:
            _check_expr(c)

        # -------------------------------------------------------------- #
        # Stage 4 — objective bounds (COP only)                            #
        # -------------------------------------------------------------- #
        if model.has_objective():
            try:
                # If objective depends only on missing auxiliary vars, we cannot recompute
                # it from the var_map.  Still validate the declared objective against
                # theoretical bounds, and record it.
                if _only_aux_missing(model.objective_):
                    if expected_objective is not None:
                        result.objective_value = expected_objective
                        lb, ub = get_bounds(model.objective_)
                        if lb is not None:
                            lb = _to_python_scalar(lb)
                        if ub is not None:
                            ub = _to_python_scalar(ub)
                        if lb is not None and ub is not None and not (lb <= expected_objective <= ub):
                            result.violations.append(
                                CheckViolation(
                                    stage=4,
                                    kind="objective",
                                    message=(
                                        f"Declared objective {expected_objective} is outside "
                                        f"theoretical bounds [{lb}, {ub}] "
                                        f"(objective expression uses auxiliary variables only)"
                                    ),
                                )
                            )
                            result.valid = False
                        else:
                            result.warnings.append(
                                f"Objective expression uses auxiliary variables not reported by the solver; "
                                f"declared value {expected_objective} is within theoretical bounds "
                                f"but could not be independently recomputed from the variable assignment"
                            )
                    else:
                        result.warnings.append(
                            "Objective expression uses auxiliary variables not reported by the solver "
                            "and no declared objective value (no 'o' line / 'cost=' attribute) was found; "
                            "the objective could not be computed or verified at all"
                        )
                    return result

                obj_val = model.objective_.value()
                obj_val = _to_python_scalar(obj_val) if obj_val is not None else None
                result.objective_value = obj_val

                lb, ub = get_bounds(model.objective_)
                if lb is not None:
                    lb = _to_python_scalar(lb)
                if ub is not None:
                    ub = _to_python_scalar(ub)

                if obj_val is None:
                    result.violations.append(
                        CheckViolation(
                            stage=4,
                            kind="objective_error",
                            message="Objective expression evaluated to None",
                        )
                    )
                    result.valid = False
                elif not (lb <= obj_val <= ub):
                    result.violations.append(
                        CheckViolation(
                            stage=4,
                            kind="objective",
                            message=(
                                f"Objective value {obj_val} is outside "
                                f"theoretical bounds [{lb}, {ub}]"
                            ),
                        )
                    )
                    result.valid = False

                if obj_val is not None and expected_objective is None:
                    result.warnings.append(
                        f"COP instance: no declared objective value found in solver output "
                        f"(no 'o' line or 'cost=' attribute); computed objective from variable "
                        f"assignment is {obj_val}, but there is nothing to compare it against"
                    )

                if obj_val is not None and expected_objective is not None and int(obj_val) != expected_objective:
                    result.violations.append(
                        CheckViolation(
                            stage=4,
                            kind="objective_mismatch",
                            message=(
                                f"Computed objective {obj_val} does not match "
                                f"declared objective {expected_objective}"
                            ),
                        )
                    )
                    result.valid = False
            except Exception as exc:
                result.violations.append(
                    CheckViolation(
                        stage=4,
                        kind="objective_error",
                        message=f"Error evaluating objective: {exc}",
                    )
                )
                result.valid = False

    finally:
        # Always restore original variable values
        for var in variables:
            var._value = saved[var]

    return result
