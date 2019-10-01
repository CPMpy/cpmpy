# interface to https://pypi.org/project/minizinc/
# TODO

from . import *
from ..expressions import *
from ..variables import *
from .minizinc_text import *

class MiniZincPython(MiniZincText):
    # needs the python package 'minizinc' installed
    def __init__(self):
        self.name = "minizinc_python"

    def supported(self):
        return True # always possible

    def solve(self, model):
        # from superclass
        mzn_txt = self.convert(model)

        # minizinc-python API
        import minizinc # TODO, catch if not installed

        # Create a MiniZinc model
        mznmodel = minizinc.Model()
        mznmodel.add_string(mzn_txt)

        # Transform Model into a instance
        cbc = minizinc.Solver.lookup("gecode")
        inst = minizinc.Instance(cbc, mznmodel)
        #inst["a"] = 1
        # Solve the instance
        mznresult = inst.solve()#all_solutions=True)

        # translate status
        solstats = SolverStats()
        if mznresult.status == minizinc.result.Status.SATISFIED:
            solstats.status = ExitStatus.FEASIBLE
        elif mznresult.status == minizinc.result.Status.ALL_SOLUTIONS:
            solstats.status = ExitStatus.FEASIBLE
        elif mznresult.status == minizinc.result.Status.OPTIMAL_SOLUTION:
            solstats.status = ExitStatus.OPTIMAL
        elif mznresult.status == minizinc.result.Status.UNSATISFIABLE:
            solstats.status = ExitStatus.UNSATISFIABLE
        elif mznresult.status == minizinc.result.Status.ERROR:
            solstats.status = ExitStatus.ERROR
        elif mznresult.status == minizinc.result.Status.UNKNOWN:
            solstats.status = ExitStatus.ERROR

        # get solution (and runtime)
        if mznresult.status.has_solution():
            # runtime
            mznsol = mznresult._solutions[-1]
            solstats.runtime = mznsol.statistics['time'].total_seconds()
            #print(mznsol)
            
            # fill in variables
            modelvars = get_variables(model)
            for var in modelvars:
                varname = str(var)
                if varname in mznsol.assignments:
                    var._value = mznsol[varname]
                else:
                    print("Warning, no value for ",varname)

        return solstats

