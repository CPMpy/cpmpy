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
from cpmpy.tools.io.wcnf import read_wcnf


def solution_mse_wcnf(solver, max_var=None):
    """
    Convert a CPMpy WCNF model solution into the MSE solution string format.
    WCNF variables are named x1, x2, ... (cf. read_wcnf); filters for those.
    """
    variables = {
        int(var.name[1:]): var
        for var in solver.user_vars
        if var.name.startswith("x") and var.name[1:].isdigit()
    }
    if max_var is None:
        max_var = max(variables, default=0)

    return "".join(
        str(1 if variables.get(i) is not None and variables[i].value() else 0)
        for i in range(1, max_var + 1)
    )


def _open_wcnf(instance, mode="rt", *args, **kwargs):
    """Open WCNF instance, handling .xz compression."""
    p = str(instance)
    if p.endswith(".xz"):
        kwargs.setdefault("encoding", "utf-8")
        return lzma.open(instance, mode=mode, *args, **kwargs)
    return open(instance, mode, *args, **kwargs)


class MSECompetitionPrintingObserver(DIMACSPrintingObserver):
    """MSE (MaxSAT Evaluation) competition-style output printer."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__(solution_printer=solution_mse_wcnf, verbose=verbose, **kwargs)

    def print_result(self, s, runner=None):
        max_var = getattr(getattr(runner, "model", None), "wcnf_max_var", None)
        if max_var is None:
            return super().print_result(s, runner)

        original_printer = self.solution_printer
        self.solution_printer = lambda solver: solution_mse_wcnf(solver, max_var=max_var)
        try:
            return super().print_result(s, runner)
        finally:
            self.solution_printer = original_printer


class MSEAdapter(InstanceAdapter):

    default_observers = [
        MSECompetitionPrintingObserver,
        RuntimeObserver,
        HandlerObserver,
        SolverArgsObserver,
            ResourceLimitObserver,
    ]

    reader = staticmethod(partial(read_wcnf, open=_open_wcnf))

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
