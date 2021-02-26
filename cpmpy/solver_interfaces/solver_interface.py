"""
    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        SolverInterface
        SolverStats
        ExitStatus

    ==================
    Module description
    ==================
    bla blabla

    ==============
    Module details
    ==============

"""
#
#==============================================================================
from enum import Enum
import time
from abc import ABC, abstractmethod

#==============================================================================
class SolverInterface(ABC):
    """
        Abstract class for defining solver interfaces. All classes implementing
        the ``SolverInterface``
    """
    def __init__(self):
        self.name = "dummy"

    @abstractmethod
    def supported(self):
        """
            Check for support in current system setup. Return True if the system
            has package installed or supports solver else returns False.

        Returns:
            [bool]: Solver support by current system setup.
        """
        return True

    @abstractmethod
    def solve(self, model):
        """
            Build the CPMpy model into solver-supported model ready for solving
            and returns the solver statistics generated during model solving.

        :param model: CPMpy model to be parsed.
        :type model: Model

        :return: an object of :class:`SolverStats`
        """
        return SolverStats()

#
#==============================================================================
class ExitStatus(Enum):
    NOT_RUN = 1
    OPTIMAL = 2
    FEASIBLE = 3
    UNSATISFIABLE = 4
    ERROR = 5

#
#==============================================================================
class SolverStats(object):
    """
        Statistics on the solved model
    """
    status: ExitStatus
    runtime: time

    def __init__(self):
        self.status = ExitStatus.NOT_RUN
        self.runtime = None

    def __repr__(self):
        return "{} ({} seconds)".format(self.status, self.runtime)