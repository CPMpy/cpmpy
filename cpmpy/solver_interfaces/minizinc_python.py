#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## minizinc_python.py
##
"""
    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        MiniZincPython

    ==================
    Module description
    ==================

    ==============
    Module details
    ==============
"""

from .solver_interface import ExitStatus, SolverStatus
from .minizinc_text import MiniZincText
from ..model_tools.get_variables import get_variables

class MiniZincPython(MiniZincText):
    """
    Interface to the python 'minizinc' package

    Requires that the 'minizinc' python package is installed:

        $ pip install minizinc
    
    as well as the MiniZinc bundled binary packages, downloadable from:
    https://www.minizinc.org/software.html

    Creates the following attributes:
    mzn_inst: the minizinc.Instance created by _model()
    mzn_result: the minizinc.Result used in solve()
    """

    def __init__(self):
        self.name = "minizinc_python"

    def supported(self):
        try:
            import minizinc
            return True
        except ImportError as e:
            return False

    def _model(self, model, solvername=None):
        import minizinc
        if solvername is None:
            # default solver
            solvername = "gecode"

        # from superclass
        mzn_txt = self.convert(model)

        # minizinc-python API
        # Create a MiniZinc model
        mznmodel = minizinc.Model()
        mznmodel.add_string(mzn_txt)

        # Transform Model into a instance
        slv = minizinc.Solver.lookup(solvername)
        return minizinc.Instance(slv, mznmodel)

    def solve(self, model, solvername=None):
        if not self.supported():
            raise Exception("Install the python 'minizinc' package to use this '{}' solver interface".format(self.name))
        import minizinc

        # create self.mzn_inst
        self.mzn_inst = self._model(model, solvername=solvername)

        # Solve the instance
        self.mzn_result = self.mzn_inst.solve(**{'output-time':True})#all_solutions=True)

        # translate status
        my_status = SolverStatus()
        my_status.solver_name = self.name
        if self.mzn_result.status == minizinc.result.Status.SATISFIED:
            my_status.exitstatus = ExitStatus.FEASIBLE
        elif self.mzn_result.status == minizinc.result.Status.ALL_SOLUTIONS:
            my_status.exitstatus = ExitStatus.FEASIBLE
        elif self.mzn_result.status == minizinc.result.Status.OPTIMAL_SOLUTION:
            my_status.exitstatus = ExitStatus.OPTIMAL
        elif self.mzn_result.status == minizinc.result.Status.UNSATISFIABLE:
            my_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif self.mzn_result.status == minizinc.result.Status.ERROR:
            my_status.exitstatus = ExitStatus.ERROR
            raise Exception("MiniZinc solver returned with status 'Error'")
        elif self.mzn_result.status == minizinc.result.Status.UNKNOWN:
            # means, no solution was found (e.g. within timeout?)...
            my_status.exitstatus = ExitStatus.ERROR
        else:
            raise NotImplementedError # a new status type was introduced, please report on github


        # get runtime and solution
        if 'time' in self.mzn_result.statistics:
            my_status.runtime = self.mzn_result.statistics['time'] # --output-time

        if self.mzn_result.status.has_solution():
            # runtime
            mznsol = self.mzn_result.solution
            my_status.runtime = self.mzn_result.statistics['time'].total_seconds()

            # fill in variables
            modelvars = get_variables(model)
            for var in modelvars:
                varname = str(var)
                if hasattr(mznsol, varname):
                    var._value = getattr(mznsol, varname)
                else:
                    print("Warning, no value for ",varname)

        return my_status
