#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## minizinc.py
##
"""
    Interface to the python 'minizinc' package

    Requires that the 'minizinc' python package is installed:

        $ pip install minizinc
    
    as well as the MiniZinc bundled binary packages, downloadable from:
    https://www.minizinc.org/software.html

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_minizinc

    ==============
    Module details
    ==============
"""

from .solver_interface import ExitStatus, SolverStatus
from .minizinc_text import MiniZincText
from ..transformations.get_variables import get_variables_model

class CPM_minizinc(MiniZincText):
    """
    Creates the following attributes:

    mzn_inst: the minizinc.Instance created by _model()
    mzn_result: the minizinc.Result used in solve()
    """

    @staticmethod
    def supported():
        try:
            import minizinc
            return True
        except ImportError as e:
            return False

    def __init__(self, cpm_model=None, solvername=None):
        """
        Constructor of the solver object

        Requires a CPMpy model as input, and will create the corresponding
        minizinc model and solver object (mzn_model and mzn_solver)
        """
        if not self.supported():
            raise Exception("Install the python 'minizinc-python' package to use this '{}' solver interface".format(self.name))
        import minizinc

        super().__init__()
        self.name = "minizinc_python"

        self._model(cpm_model, solvername=solvername)


    def _model(self, model, solvername=None):
        import minizinc
        if solvername is None:
            # default solver
            solvername = "gecode"
        self.mzn_solver = minizinc.Solver.lookup(solvername)

        # minizinc-python API
        # Create a MiniZinc model
        self.mzn_model = minizinc.Model()
        if model is None:
            self.user_vars = []
        else:
            # store original vars and objective (before flattening)
            self.user_vars = get_variables_model(model)
            mzn_txt = self.convert(model)
            self.mzn_model.add_string(mzn_txt)

        self.mzn_model = minizinc.Model()
        self.mzn_model.add_string(mzn_txt)
        return


    def solve(self, **kwargs):
        """
            Call the (already created) solver

            keyword arguments can be any argument accepted by minizinc.Instance.solve()
            For example, set 'all_solutions=True' to have it enumerate all solutions
        """
        import minizinc

        # Transform Model into a instance
        self.mzn_inst = minizinc.Instance(self.mzn_solver, self.mzn_model)

        # Solve the instance
        kwargs['output-time'] = True # required for time getting
        self.mzn_result = self.mzn_inst.solve(**kwargs)#all_solutions=True)

        # translate status
        my_status = SolverStatus(self.name)
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
            for var in self.user_vars:
                varname = str(var).replace('[','_').replace(']','') # DANGER, hardcoded
                if hasattr(mznsol, varname):
                    var._value = getattr(mznsol, varname)
                else:
                    print("Warning, no value for ",varname)

        #TODO: return self._solve_return(self.cpm_status, objective_value)
        return my_status

    def __add__(self, cons):
        raise NotImplementedError("adding constraints iteratively not yet implemented for CPM_minzinc")
    def minimize(self, expr):
        """
            Minimize the given objective function

            `minimize()` can be called multiple times, only the last one is stored
        """
        raise NotImplementedError("not yet implemented for CPM_minzinc")
    def maximize(self, expr):
        """
            Maximize the given objective function

            `maximize()` can be called multiple times, only the last one is stored
        """
        raise NotImplementedError("not yet implemented for CPM_minzinc")
