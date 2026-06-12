import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.tools.solution_checker import check_solution
from cpmpy.transformations.get_variables import get_variables_model

from ..runner.runner import Runner
from .base import Observer


_RE_ORDER = re.compile(r"^(.+)_ge_(\d+)$")
_RE_DIRECT = re.compile(r"^(.+)_eq_(\d+)$")
_RE_BINARY = re.compile(r"^(.+)_bit(\d+)$")
_CHECKABLE_STATUSES = frozenset({ExitStatus.FEASIBLE, ExitStatus.OPTIMAL})


def _canonical_cp_name(name: str) -> str:
    """Map VeriPB-safe indexed names back to CPMpy style."""
    return name.replace("][", ",")


def decode_veripb_assignment(raw: dict[str, bool]) -> dict[str, int]:
    """Decode annotated PB booleans back to CP integer assignments."""
    order: dict[str, dict[int, bool]] = defaultdict(dict)
    direct: dict[str, dict[int, bool]] = defaultdict(dict)
    binary: dict[str, dict[int, bool]] = defaultdict(dict)

    for name, value in raw.items():
        if name.startswith("_"):
            continue
        match = _RE_ORDER.match(name)
        if match:
            order[_canonical_cp_name(match.group(1))][int(match.group(2))] = value
            continue
        match = _RE_DIRECT.match(name)
        if match:
            direct[_canonical_cp_name(match.group(1))][int(match.group(2))] = value
            continue
        match = _RE_BINARY.match(name)
        if match:
            binary[_canonical_cp_name(match.group(1))][int(match.group(2))] = value

    decoded: dict[str, int] = {}
    for source, thresholds in order.items():
        true_thresholds = [threshold for threshold, value in thresholds.items() if value]
        decoded[source] = max(true_thresholds) if true_thresholds else min(thresholds) - 1

    for source, values in direct.items():
        true_values = [value for value, selected in values.items() if selected]
        if true_values:
            decoded[source] = true_values[0]

    for source, bits in binary.items():
        decoded[source] = sum(2**bit for bit, selected in bits.items() if selected)

    return decoded


def _translated_sidecar_path(instance_path: Path) -> Optional[Path]:
    instance_str = str(instance_path)
    candidates = [Path(instance_str + ".meta.json")]
    if instance_path.suffix == ".xz":
        candidates.append(Path(instance_str[:-len(".xz")] + ".meta.json"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _reader_for_dataset(dataset_name: str) -> Optional[Callable[[str], Any]]:
    if dataset_name == "psplib":
        from cpmpy.tools.io.rcpsp import load_rcpsp

        return load_rcpsp
    if dataset_name == "jsplib":
        from cpmpy.tools.io.jsplib import load_jsplib

        return load_jsplib
    if dataset_name == "nurserostering":
        from cpmpy.tools.io.nurserostering import load_nurserostering

        return load_nurserostering
    return None


def _resolve_source_instance(sidecar: dict[str, Any], source_root: Path) -> Optional[Path]:
    dataset = sidecar.get("dataset") if isinstance(sidecar.get("dataset"), dict) else {}
    dataset_name = dataset.get("name")
    source_file = sidecar.get("source_file")
    if not isinstance(dataset_name, str) or not isinstance(source_file, str):
        return None

    source_path = Path(source_file)
    if source_path.is_absolute() and source_path.exists():
        return source_path

    category = sidecar.get("category") if isinstance(sidecar.get("category"), dict) else {}
    candidates = [
        source_root / source_path,
        source_root / dataset_name / source_path,
    ]
    if dataset_name == "psplib":
        variant = category.get("variant", "rcpsp")
        family = category.get("family")
        if isinstance(variant, str) and isinstance(family, str):
            candidates.insert(0, source_root / dataset_name / variant / family / source_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _payload_from_result(
    result,
    *,
    checker: str,
    translated_instance: Path,
    source_instance: Optional[Path],
    decoded_variable_count: int,
    sidecar_path: Optional[Path],
    warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
    return {
        "checker": checker,
        "valid": result.valid,
        "skipped": result.skipped,
        "summary": result.summary(),
        "objective_value": result.objective_value,
        "warnings": [*result.warnings, *(warnings or [])],
        "violations": [
            {
                "stage": violation.stage,
                "kind": violation.kind,
                "message": violation.message,
                "context": str(violation.context) if violation.context is not None else None,
            }
            for violation in result.violations
        ],
        "violation_count": len(result.violations),
        "decoded_variable_count": decoded_variable_count,
        "translated_instance": str(translated_instance),
        "source_instance": str(source_instance) if source_instance is not None else None,
        "sidecar": str(sidecar_path) if sidecar_path is not None else None,
    }


class PBCPSolutionCheckerObserver(Observer):
    """
    Decode a VeriPB-annotated OPB solution and check it against the source CP model.
    """

    def __init__(
        self,
        source_root: str = "data",
        max_violations_to_print: int = 5,
        **kwargs,
    ):
        self.source_root = Path(source_root)
        self.max_violations_to_print = max_violations_to_print

    def _set_payload(self, runner: Runner, payload: dict[str, Any]) -> None:
        setattr(runner, "solution_checker", payload)
        if hasattr(runner, "runner_metadata") and isinstance(runner.runner_metadata, dict):
            runner.runner_metadata["solution_checker"] = payload

    def _skipped_payload(
        self,
        runner: Runner,
        translated_instance: Path,
        exit_status: ExitStatus,
        reason: str,
    ) -> dict[str, Any]:
        payload = {
            "checker": self.__class__.__name__,
            "valid": True,
            "skipped": True,
            "summary": f"SKIPPED ({exit_status.name})",
            "objective_value": None,
            "warnings": [reason],
            "violations": [],
            "violation_count": 0,
            "decoded_variable_count": 0,
            "translated_instance": str(translated_instance),
            "source_instance": None,
            "sidecar": None,
        }
        payload["reason"] = reason
        return payload

    def observe_end(self, runner: Runner):
        translated_instance = Path(getattr(runner, "instance_path", None) or getattr(runner, "_instance_path", ""))
        if not translated_instance:
            return
        if getattr(runner, "model", None) is None or getattr(runner, "s", None) is None:
            # RunExec parent replay has only relayed metadata. The checker must run in the child
            # process where the solved model and solver assignment are available.
            return

        try:
            status_obj = runner.s.status() if getattr(runner, "s", None) is not None else None
            exit_status = status_obj.exitstatus if status_obj is not None else ExitStatus.UNKNOWN
        except Exception:
            exit_status = ExitStatus.UNKNOWN

        has_solution = bool(getattr(runner, "is_sat", False))
        if not has_solution:
            payload = self._skipped_payload(
                runner,
                translated_instance,
                exit_status,
                f"No CP-level solution check for status {exit_status.name}",
            )
            self._set_payload(runner, payload)
            runner.print_comment(f"PB-to-CP solution checker: {payload['summary']}")
            return

        sidecar_path = _translated_sidecar_path(translated_instance)
        if sidecar_path is None:
            payload = self._skipped_payload(
                runner,
                translated_instance,
                exit_status,
                "Translated OPB sidecar not found",
            )
            self._set_payload(runner, payload)
            runner.print_comment("PB-to-CP solution checker skipped: sidecar not found")
            return

        try:
            sidecar = _load_json(sidecar_path)
            source_instance = _resolve_source_instance(sidecar, self.source_root)
            dataset_name = (sidecar.get("dataset") or {}).get("name")
            reader = _reader_for_dataset(dataset_name)
            if source_instance is None:
                raise ValueError("Could not resolve original CP source instance")
            if reader is None:
                raise ValueError(f"No source reader registered for dataset {dataset_name!r}")
            source_model = reader(str(source_instance))
        except Exception as exc:
            payload = self._skipped_payload(
                runner,
                translated_instance,
                exit_status,
                f"Could not load original CP model: {exc}",
            )
            self._set_payload(runner, payload)
            runner.print_comment(f"PB-to-CP solution checker skipped: {exc}")
            return

        try:
            pb_assignment = {
                str(var.name): bool(var.value())
                for var in get_variables_model(runner.model)
                if var.value() is not None
            }
            decoded = decode_veripb_assignment(pb_assignment)
            check_status = exit_status if exit_status in _CHECKABLE_STATUSES else ExitStatus.FEASIBLE
            result = check_solution(source_model, check_status, var_map=decoded)
        except Exception as exc:
            payload = self._skipped_payload(
                runner,
                translated_instance,
                exit_status,
                f"Could not decode/check PB solution: {exc}",
            )
            self._set_payload(runner, payload)
            runner.print_comment(f"PB-to-CP solution checker failed: {exc}")
            return

        payload = _payload_from_result(
            result,
            checker=self.__class__.__name__,
            translated_instance=translated_instance,
            source_instance=source_instance,
            decoded_variable_count=len(decoded),
            sidecar_path=sidecar_path,
        )
        payload["solver_exit_status"] = exit_status.name
        if exit_status not in _CHECKABLE_STATUSES:
            payload["warnings"].append(
                f"Solver exit status was {exit_status.name}, but runner.is_sat=True; checked incumbent as FEASIBLE"
            )
        self._set_payload(runner, payload)
        runner.print_comment(f"PB-to-CP solution checker: {result.summary()}")

        for warning in result.warnings:
            runner.print_comment(f"PB-to-CP solution checker warning: {warning}")
        if not result.valid and not result.skipped:
            for violation in result.violations[: self.max_violations_to_print]:
                runner.print_comment(f"PB-to-CP solution checker violation: {violation}")
            remaining = len(result.violations) - self.max_violations_to_print
            if remaining > 0:
                runner.print_comment(
                    f"PB-to-CP solution checker: {remaining} additional violations omitted"
                )
