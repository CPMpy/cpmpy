from functools import partial
import lzma
import os
import re
from typing import Optional

from cpmpy.tools.benchmark.cpbenchy.adapter._base import InstanceAdapter
from cpmpy.tools.benchmark.cpbenchy.observer import (
    DIMACSPrintingObserver,
    HandlerObserver,
    RuntimeObserver,
    ResourceLimitObserver,
    SolverArgsObserver,
)
from cpmpy.tools.benchmark.opb import solution_opb
from cpmpy.tools.io.opb import read_opb


OPB_HIGHS_INTSIZE_LIMIT = 32
OPB_INTSIZE_RE = re.compile(r"\bintsize=\s*(\d+)\b")


def _open_opb(instance):
    """Open OPB instance, handling .xz compression."""
    p = str(instance)
    if p.endswith(".xz"):
        return lzma.open(instance, mode="rt", encoding="utf-8")
    return open(instance, "rt")


def _opb_header_intsize(instance) -> Optional[int]:
    """Read only the OPB header line and extract its intsize value."""
    with _open_opb(instance) as opb_file:
        header = opb_file.readline()
    match = OPB_INTSIZE_RE.search(header)
    if match is None:
        return None
    return int(match.group(1))


def _is_highs_intsize_unsupported(instance, solver: str) -> tuple[bool, Optional[int]]:
    if str(solver).lower() != "highs":
        return False, None
    intsize = _opb_header_intsize(instance)
    return (
        intsize is not None and intsize > OPB_HIGHS_INTSIZE_LIMIT,
        intsize,
    )


class OPBCompetitionPrintingObserver(DIMACSPrintingObserver):
    """OPB competition-style output printer."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__(solution_printer=solution_opb, verbose=verbose, **kwargs)


class OPBAdapter(InstanceAdapter):
    solution_printer = staticmethod(solution_opb)

    default_observers = [
        OPBCompetitionPrintingObserver,
        RuntimeObserver,
        HandlerObserver,
        SolverArgsObserver,
        ResourceLimitObserver,
    ]

    reader = staticmethod(partial(read_opb, open=_open_opb))

    def cmd(self, instance: str, solver: str = "ortools", output_file: str = None, **kwargs):
        cmd = self.base_cmd(instance)
        if solver is not None:
            cmd.append("--solver")
            cmd.append(solver)
        if output_file is not None:
            cmd.append("--output_file")
            cmd.append(output_file)
        return cmd

    def print_comment(self, comment: str):
        print("c" + chr(32) + comment.rstrip("\n"), end="\r\n", flush=True)

    def run(self, instance: str, solver: str = "ortools", output_file: str = None, **kwargs):
        unsupported, intsize = _is_highs_intsize_unsupported(instance, solver)
        if unsupported:
            lines = [
                f"c Skipping HiGHS: OPB intsize {intsize} exceeds 32-bit threshold",
                "s UNSUPPORTED",
            ]
            for line in lines:
                print(line, end="\r\n", flush=True)
            if output_file is not None:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("\r\n".join(lines) + "\r\n")
            return None
        return super().run(instance=instance, solver=solver, output_file=output_file, **kwargs)


def main():
    runner = OPBAdapter()
    parser = runner.argparser()
    args = parser.parse_args()
    runner.run(**vars(args))


if __name__ == "__main__":
    main()
