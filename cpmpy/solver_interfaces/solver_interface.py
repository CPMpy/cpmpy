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
    def __init__(self):
        self.name = "dummy"

    def supported(self):
        """
            Check for support in current system setup. Return True if the system
            has package installed or supports solver, else returns False.

        Returns:
            [bool]: Solver support by current system setup.
        """
        return False

    def solve(self, model):
        """
            Build the CPMpy model into solver-supported model ready for solving
            and returns the solver statistics generated during model solving.

        :param model: CPMpy model to be parsed.
        :type model: Model

        :return: an object of :class:`SolverStatus`
        """
        return SolverStatus()

#
#==============================================================================
class ExitStatus(Enum):
    """
        Exit status of the solver

        Attributes:
            NOT_RUN: Has not been run
            OPTIMAL: Optimal solution to an optimisation problem found
            FEASIBLE: Feasible solution to a satisfaction problem found,
                      or feasible (but not proven optimal) solution to an
                      optimisation problem found
            UNSATISFIABLE: No satisfying solution exists
            ERROR: Some error occured (solver should have thrown Exception)
    """
    NOT_RUN = 1
    OPTIMAL = 2
    FEASIBLE = 3
    UNSATISFIABLE = 4
    ERROR = 5


#
#==============================================================================
class SolverStatus(object):
    """
        Status and statistics of a solver run
    """
    exitstatus: ExitStatus
    runtime: time

    def __init__(self):
        self.exitstatus = ExitStatus.NOT_RUN
        self.runtime = None
        self.solver_name = None

    def __repr__(self):
        return "{} ({} seconds)".format(self.exitstatus, self.runtime)
