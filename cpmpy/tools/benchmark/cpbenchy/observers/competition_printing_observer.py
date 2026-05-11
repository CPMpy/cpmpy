from cpmpy.solvers.solver_interface import ExitStatus as CPMStatus
from cpmpy.tools.benchmark.opb import solution_opb

from ..runner import Runner
from .base import Observer


class CompetitionPrintingObserver(Observer):

    def __init__(self, verbose: bool = False, **kwargs):
        self.verbose = verbose
    
    def print_comment(self, comment: str):
        # Comment is already formatted by Runner.print_comment() before being passed to observers
        # So just print it as-is
        print(comment.rstrip('\n'), end="\r\n", flush=True)

    def observe_post_solve(self, runner: Runner):
        self.print_result(runner.s)

    def observe_intermediate(self, runner: Runner = None, objective: int = None, elapsed_seconds=None, **kwargs):
        self.print_intermediate(objective)

    def print_status(self, status: str):
        print('s' + chr(32) + status, end="\n", flush=True)

    def print_value(self, value: str):
        print('v' + chr(32) + value, end="\n", flush=True)

    def print_objective(self, objective: int):
        print('o' + chr(32) + str(objective), end="\n", flush=True)

    def print_intermediate(self, objective: int):
        self.print_objective(objective)

    def print_result(self, s):
        if s.status().exitstatus == CPMStatus.OPTIMAL:
            self.print_objective(s.objective_value())
            self.print_value(solution_opb(s))
            self.print_status("OPTIMAL" + chr(32) + "FOUND")
        elif s.status().exitstatus == CPMStatus.FEASIBLE:
            self.print_objective(s.objective_value())
            self.print_value(solution_opb(s))
            self.print_status("SATISFIABLE")
        elif s.status().exitstatus == CPMStatus.UNSATISFIABLE:
            self.print_status("UNSATISFIABLE")
        else:
            self.print_comment("Solver did not find any solution within the time/memory limit")
            self.print_status("UNKNOWN")