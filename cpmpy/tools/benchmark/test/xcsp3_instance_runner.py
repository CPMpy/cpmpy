from functools import partial
import lzma
from pathlib import Path
from cpmpy.tools.benchmark.test.instance_runner import InstanceRunner
import os, sys

from cpmpy.tools.benchmark.test.runner import CompetitionPrintingObserver, HandlerObserver, ProfilingObserver, ResourceLimitObserver, Runner, SolverArgsObserver, SolutionCheckerObserver
from cpmpy.tools.xcsp3.parser import read_xcsp3

class XCSP3InstanceRunner(InstanceRunner):

    this_file_path = os.path.abspath(__file__)
    this_python = sys.executable

    def cmd(self, instance: str, solver: str = "ortools", output_file: str = None, **kwargs):
        cmd = [
            self.this_python,
            self.this_file_path,
            instance,
        ]
        if solver is not None:
            cmd.append("--solver")
            cmd.append(solver)
        if output_file is not None:
            cmd.append("--output_file")
            cmd.append(output_file)
        return cmd

    def print_comment(self, comment: str):
        print('c' + chr(32) + comment.rstrip('\n'), end="\r\n", flush=True)

    def run(self, instance: str, solver: str = "ortools", output_file: str = None, **kwargs):

        if output_file is None:
            output_file = f"results/{solver}_{instance}.txt"
        else:
            output_file = f"results/{output_file}"

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        runner = Runner(reader=partial(read_xcsp3, open= lambda instance: lzma.open(instance, mode='rt', encoding='utf-8') if str(instance).endswith(".lzma") else open(instance)))

        runner.register_observer(CompetitionPrintingObserver())
        runner.register_observer(ProfilingObserver())
        runner.register_observer(HandlerObserver())
        runner.register_observer(SolverArgsObserver())
        runner.register_observer(SolutionCheckerObserver())
        runner.register_observer(ResourceLimitObserver()) # Don't enforce any limits, just observe / capture exceptions
        
        # Register any additional observers that were added programmatically
        for observer in self.get_additional_observers():
            runner.register_observer(observer)

        runner.run(instance=instance, solver=solver, output_file=output_file, **kwargs)


def main():
    runner = XCSP3InstanceRunner()

    parser = runner.argparser()
    args = parser.parse_args()
    
    runner.run(**vars(args))

if __name__ == "__main__":
    main()