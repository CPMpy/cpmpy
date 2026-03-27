from functools import partial
import lzma


from cpmpy.tools.benchmark.test.instance_runner import InstanceRunner
from cpmpy.tools.benchmark.test.observer import CompetitionPrintingObserver, HandlerObserver, RuntimeObserver, ResourceLimitObserver, Runner, SolverArgsObserver, SolutionCheckerObserver
from cpmpy.tools.xcsp3.parser import read_xcsp3




class XCSP3InstanceRunner(InstanceRunner):

    default_observers = [
        CompetitionPrintingObserver,
        RuntimeObserver,
        HandlerObserver,
        SolverArgsObserver,
        SolutionCheckerObserver,
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
    runner = XCSP3InstanceRunner()

    parser = runner.argparser()
    args = parser.parse_args()
    
    runner.run(**vars(args))

if __name__ == "__main__":
    main()