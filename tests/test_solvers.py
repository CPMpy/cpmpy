import numpy as np
import unittest
#from cpmpy.solver_interfaces.minizinc_python import MiniZincPython
import cpmpy as cp

#supported_solvers= [MiniZincPython()]
class TestSolvers(unittest.TestCase):
    def test_installed_solvers(self):
        # basic model
        x = cp.IntVar(0,2, 3)

        constraints = [
            x[0] < x[1],
            x[1] < x[2]]
        model = cp.Model(constraints)
        #for solver in supported_solvers:
        for solver in cp.get_supported_solvers():
            model.solve(solver=solver)
            self.assertEqual([xi.value() for xi in x], [0, 1, 2])
