from cplab.adapter._base import InstanceAdapter
from cplab.observer import (
    DIMACSPrintingObserver,
    HandlerObserver,
    RuntimeObserver,
    ResourceLimitObserver,
    SolverArgsObserver,
    SolutionCheckerObserver,
)
from cpmpy.tools.benchmark.jsplib import solution_psplib
from cpmpy.tools.io.jsplib import read_jsplib


class JSPLibCompetitionPrintingObserver(DIMACSPrintingObserver):
    """JSPLib competition-style output printer."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__(solution_printer=solution_psplib, verbose=verbose, **kwargs)


class JSPLibAdapter(InstanceAdapter):

    default_observers = [
        JSPLibCompetitionPrintingObserver,
        RuntimeObserver,
        HandlerObserver,
        SolverArgsObserver,
        SolutionCheckerObserver,
        ResourceLimitObserver,
    ]

    reader = staticmethod(read_jsplib)

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
    runner = JSPLibAdapter()
    parser = runner.argparser()
    args = parser.parse_args()
    runner.run(**vars(args))


if __name__ == "__main__":
    main()
