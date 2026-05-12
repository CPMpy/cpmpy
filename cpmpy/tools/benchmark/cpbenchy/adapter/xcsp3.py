from functools import partial
import lzma

from cpmpy.tools.benchmark.cpbenchy.adapter._base import InstanceAdapter
from cpmpy.tools.benchmark.cpbenchy.observer import HandlerObserver, RuntimeObserver, ResourceLimitObserver, SolverArgsObserver
from cpmpy.tools.benchmark.cpbenchy.observer.dimacs_printing import DIMACSPrintingObserver
from cpmpy.tools.benchmark.xcsp3 import solution_xcsp3
from cpmpy.tools.xcsp3.parser import read_xcsp3


class XCSP3CompetitionPrintingObserver(DIMACSPrintingObserver):
    """XCSP3 competition-style output printer using DIMACS format."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__(solution_printer=solution_xcsp3, verbose=verbose, **kwargs)



class XCSP3Adapter(InstanceAdapter):
    solution_printer = staticmethod(solution_xcsp3)

    default_observers = [
        XCSP3CompetitionPrintingObserver,
        RuntimeObserver,
        HandlerObserver,
        SolverArgsObserver,
            ResourceLimitObserver,
    ]

    reader = partial(read_xcsp3, open= lambda instance: lzma.open(instance, mode='rt', encoding='utf-8') if str(instance).endswith(".lzma") else open(instance))

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
        print('c' + chr(32) + comment.rstrip('\n'), end="\r\n", flush=True)


def main():
    runner = XCSP3Adapter()

    parser = runner.argparser()
    args = parser.parse_args()
    
    runner.run(**vars(args))

if __name__ == "__main__":
    main()