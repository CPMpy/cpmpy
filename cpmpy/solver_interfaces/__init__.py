#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## __init__.py
##
"""
    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        SolverInterface
        SolverStats
        get_supported_solvers

    ==================
    Module description
    ==================

    ==============
    Module details
    ==============
"""
# solver interfaces, including status/stats
from enum import Enum
import time

class SolverInterface:
    """
        Abstract class for defining solver interfaces. All classes implementing
        the ``SolverInterface``
    """
    def __init__(self):
        self.name = "dummy"

    def supported(self):
        """
            Check for support in current system setup. Return True if the system
            has package installed or supports solver else returns False.

        Returns:
            [bool]: Solver support by current system setup.
        """
        return True

    def solve(self, model):
        """
            Build the CpMPy model into solver-supported model ready for solving
            and returns the solver statistics generated during model solving.

        :param model: CpMPy model to be parsed.
        :type model: Model

        :return: an object of :class:`SolverStats`
        """
        return SolverStats()

def get_supported_solvers():
    """
        Returns a list of currently supported solvers.

    :return: a list of SolverInterface sub-classes :list[SolverInterface]:
    """
    return [sv for sv in builtin_solvers if sv.supported()]

class ExitStatus(Enum):
    NOT_RUN = 1
    OPTIMAL = 2
    FEASIBLE = 3
    UNSATISFIABLE = 4
    ERROR = 5

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

# builtin solvers implementing SolverInterface
from .minizinc_text import *
from .minizinc_python import *
from .ortools_python import *

# the order matters: default will be first supported one
builtin_solvers=[MiniZincPython(),MiniZincText(),ORToolsPython()]

