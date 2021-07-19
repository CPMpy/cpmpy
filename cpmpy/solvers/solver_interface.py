"""
    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        SolverInterface
        SolverStatus
        ExitStatus

    ==================
    Module description
    ==================
    Contains the abstract class `SolverInterface` for defining solver interfaces,
    as well as a class `SolverStatus` that collects solver statistics,
    and the `ExitStatus` class that represents possible exist statuses.

    Each solver has its own class that inherits from `SolverInterface`.

"""
#
#==============================================================================
from enum import Enum
import time

#==============================================================================
class SolverInterface(object):
    """
        Abstract class for defining solver interfaces. All classes implementing
        the ``SolverInterface``
    """

    # REQUIRED functions:

    @staticmethod
    def supported():
        """
            Check for support in current system setup. Return True if the system
            has package installed or supports solver, else returns False.

        Returns:
            [bool]: Solver support by current system setup.
        """
        return False

    # REQUIRED functions to mimic `Model` interface:

    def __init__(self):
        self.cpm_status = SolverStatus("dummy") # status of solving this model

    def __add__(self):
        """
            Adds a constraint to the solver, eagerly (e.g. instantly passed to API)
        """
        raise NotImplementedError("Solver does not support eagerly adding constraints")

    def minimize(self, expr):
        """
            Minimize the given objective function

            `minimize()` can be called multiple times, only the last one is stored
        """
        raise NotImplementedError("Solver does not support objective functions")

    def maximize(self, expr):
        """
            Maximize the given objective function

            `maximize()` can be called multiple times, only the last one is stored
        """
        raise NotImplementedError("Solver does not support objective functions")

    def status(self):
        return self.cpm_status

    def solve(self, model, time_limit=None):
        """
            Build the CPMpy model into solver-supported model ready for solving
            and returns the answer (True/False/objective.value())

            Overwrites self.cpm_status

        :param model: CPMpy model to be parsed.
        :type model: Model

        :param time_limit: optional, time limit in seconds
        :type time_limit: int or float

        :return: Bool or Int
        """
        return False

    # OPTIONAL functions

    def solution_hint(self, cpm_vars, vals):
        """
        For warmstarting the solver with a variable assignment

        Typically implemented in SAT-based solvers

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
        raise NotImplementedError("Solver does not support solution hinting")

    def get_core(self):
        """
        For use with s.solve(assumptions=[...]). Only meaningful if the solver returned UNSAT.

        Typically implemented in SAT-based solvers
        
        Returns a small subset of assumption literals that are unsat together.
        (a literal is either a `_BoolVarImpl` or a `NegBoolView` in case of its negation, e.g. x or ~x)
        Setting these literals to True makes the model UNSAT, setting any to False makes it SAT
        """
        raise NotImplementedError("Solver does not support unsat core extraction")


    # shared helper functions

    def _solve_return(self, cpm_status, objective_value):
        """
            Take a CPMpy Model and SolverStatus object and return
            the proper answer (True/False/objective_value)

        :param cpm_status: status extracted from the solver
        :type cpm_status: SolverStatus

        :param objective_value: None or Int, as computed by solver

        :return: Bool or Int
        """
        # return computed value
        if objective_value is not None and \
            (cpm_status.exitstatus == ExitStatus.OPTIMAL or \
             cpm_status.exitstatus == ExitStatus.FEASIBLE):
            # optimisation problem
            return objective_value
        else:
            # satisfaction problem
            if cpm_status.exitstatus == ExitStatus.FEASIBLE or \
               cpm_status.exitstatus == ExitStatus.OPTIMAL:
                return True
        return False


#==============================================================================
class ExitStatus(Enum):
    """
    Exit status of the solver

    Attributes:

        `NOT_RUN`: Has not been run

        `OPTIMAL`: Optimal solution to an optimisation problem found

        `FEASIBLE`: Feasible solution to a satisfaction problem found,
                    or feasible (but not proven optimal) solution to an
                    optimisation problem found

        `UNSATISFIABLE`: No satisfying solution exists

        `ERROR`: Some error occured (solver should have thrown Exception)

        `UNKNOWN`: Outcome unknown, for example when timeout is reached
    """
    NOT_RUN = 1
    OPTIMAL = 2
    FEASIBLE = 3
    UNSATISFIABLE = 4
    ERROR = 5
    UNKNOWN = 6

#==============================================================================
class SolverStatus(object):
    """
        Status and statistics of a solver run
    """
    exitstatus: ExitStatus
    runtime: time

    def __init__(self, name):
        self.solver_name = name
        self.exitstatus = ExitStatus.NOT_RUN
        self.runtime = None

    def __repr__(self):
        return "{} ({} seconds)".format(self.exitstatus, self.runtime)
