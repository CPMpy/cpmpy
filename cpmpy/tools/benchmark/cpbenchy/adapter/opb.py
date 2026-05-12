from functools import partial
import lzma

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


def _open_opb(instance):
    """Open OPB instance, handling .xz compression."""
    p = str(instance)
    if p.endswith(".xz"):
        return lzma.open(instance, mode="rt", encoding="utf-8")
    return open(instance, "rt")


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

    reader = partial(read_opb, open=_open_opb)

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


def main():
    runner = OPBAdapter()
    parser = runner.argparser()
    args = parser.parse_args()
    runner.run(**vars(args))


if __name__ == "__main__":
    main()
