from cpmpy.tools.benchmark.cpbenchy.adapter._base import InstanceAdapter
from cpmpy.tools.benchmark.cpbenchy.observer import (
    DIMACSPrintingObserver,
    HandlerObserver,
    RuntimeObserver,
    ResourceLimitObserver,
    SolverArgsObserver,
)
from cpmpy.tools.io.rcpsp import read_rcpsp


def solution_psplib(model):
    """
    Convert a CPMpy model solution into the solution string format.

    Arguments:
        model (cp.solvers.SolverInterface): The solver-specific model for which to print its solution

    Returns:
        str: formatted solution string.
    """
    variables = {var.name: var.value() for var in model.user_vars if var.name[:2] not in ["IV", "BV", "B#"]} # dirty workaround for all missed aux vars in user vars TODO fix with Ignace
    return str(variables)


class PSPLibCompetitionPrintingObserver(DIMACSPrintingObserver):
    """PSPLib competition-style output printer."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__(solution_printer=solution_psplib, verbose=verbose, **kwargs)


class PSPLibAdapter(InstanceAdapter):

    default_observers = [
        PSPLibCompetitionPrintingObserver,
        RuntimeObserver,
        HandlerObserver,
        SolverArgsObserver,
            ResourceLimitObserver,
    ]

    reader = staticmethod(read_rcpsp)

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
    runner = PSPLibAdapter()
    parser = runner.argparser()
    args = parser.parse_args()
    runner.run(**vars(args))


if __name__ == "__main__":
    main()
