"""
Deprecated: Use XCSP3InstanceRunner instead
"""
import argparse
import lzma
from pathlib import Path
from functools import partial

from cpmpy.tools.benchmark.test.runner import Runner, CompetitionPrintingObserver, ProfilingObserver, HandlerObserver, SolverArgsObserver, SolutionCheckerObserver, WriteToFileObserver
from cpmpy.tools.dataset.model.xcsp3 import XCSP3Dataset
from cpmpy.tools.xcsp3 import read_xcsp3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=str)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--solver", type=str, default="ortools")
    parser.add_argument("--time_limit", type=int, default=None)
    parser.add_argument("--mem_limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--intermediate", action="store_true", default=False)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    # parser.add_argument("--kwargs", type=str, default="")
    parser.add_argument("--observers", type=list[str], default=None)

    args = parser.parse_args()


    if args.output_file is None:
        args.output_file = f"results/{args.solver}_{args.instance}.txt"
    else:
        args.output_file = f"results/{args.output_file}"

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)


    # dataset = XCSP3Dataset(root="./data", year=2024, track="CSP24", download=True)

    runner = Runner(reader=partial(read_xcsp3, open= lambda instance: lzma.open(instance, mode='rt', encoding='utf-8') if str(instance).endswith(".lzma") else open(instance)))
    # runner.register_observer(LoggerObserver())
    runner.register_observer(CompetitionPrintingObserver())
    runner.register_observer(ProfilingObserver())
    # runner.register_observer(ResourceLimitObserver(time_limit=args.time_limit, mem_limit=args.mem_limit))
    runner.register_observer(HandlerObserver())
    runner.register_observer(SolverArgsObserver())
    runner.register_observer(SolutionCheckerObserver())
    #runner.register_observer(WriteToFileObserver(file_path=args.output_file))

    for observer in args.observers:
        pass


    print(vars(args))
    runner.run(**vars(args))

if __name__ == "__main__":
    main()