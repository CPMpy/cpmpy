import argparse
import lzma
from pathlib import Path
from functools import partial

from cpmpy.tools.benchmark.test.runner import Runner, CompetitionPrintingObserver, ProfilingObserver, HandlerObserver, SolverArgsObserver, SolutionCheckerObserver, WriteToFileObserver
from cpmpy.tools.dataset.model.xcsp3 import XCSP3Dataset
from cpmpy.tools.xcsp3 import read_xcsp3

class InstanceRunner:
    
    def __init__(self):
        self.additional_observers = []
    
    def cmd(self, instance: str):
        pass

    def argparser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("instance", type=str)
        parser.add_argument("--solver", type=str, default="ortools")
        parser.add_argument("--output_file", type=str, default=None)
        parser.add_argument("--verbose", action="store_true", default=False)
        parser.add_argument("--time_limit", type=float, default=None)
        parser.add_argument("--mem_limit", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--intermediate", action="store_true", default=False)
        parser.add_argument("--cores", type=int, default=None)
        parser.add_argument("--observers", type=list[str], default=None)
        return parser

    def print_comment(self, comment: str):
        pass
    
    def register_observer(self, observer):
        """Register an observer to be added when run() is called."""
        self.additional_observers.append(observer)
    
    def get_additional_observers(self):
        """Get the list of additional observers that should be registered."""
        return self.additional_observers

