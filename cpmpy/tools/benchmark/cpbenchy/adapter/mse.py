from functools import partial
import lzma

from cplab.adapter._base import InstanceAdapter
from cplab.observer import (
    DIMACSPrintingObserver,
    HandlerObserver,
    RuntimeObserver,
    ResourceLimitObserver,
    SolverArgsObserver,
    SolutionCheckerObserver,
)
from cpmpy.tools.io.wcnf import read_wcnf


def solution_mse_wcnf(solver):
    """
    Convert a CPMpy WCNF model solution into the MSE solution string format.
    WCNF variables are named x1, x2, ... (cf. read_wcnf); filters for those.
    """
    variables = [
        var
        for var in solver.user_vars
        if var.name.startswith("x") and var.name[1:].isdigit()
    ]
    variables = sorted(variables, key=lambda v: int(v.name[1:]))
    return " ".join(str(1 if var.value() else 0) for var in variables)


def _open_wcnf(instance):
    """Open WCNF instance, handling .xz compression."""
    p = str(instance)
    if p.endswith(".xz"):
        return lzma.open(instance, mode="rt", encoding="utf-8")
    return open(instance, "rt")


class MSECompetitionPrintingObserver(DIMACSPrintingObserver):
    """MSE (MaxSAT Evaluation) competition-style output printer."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__(solution_printer=solution_mse_wcnf, verbose=verbose, **kwargs)


class MSEAdapter(InstanceAdapter):

    default_observers = [
        MSECompetitionPrintingObserver,
        RuntimeObserver,
        HandlerObserver,
        SolverArgsObserver,
        SolutionCheckerObserver,
        ResourceLimitObserver,
    ]

    reader = partial(read_wcnf, open=_open_wcnf)

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
    runner = MSEAdapter()
    parser = runner.argparser()
    args = parser.parse_args()
    runner.run(**vars(args))


if __name__ == "__main__":
    main()
