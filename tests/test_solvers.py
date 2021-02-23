import numpy as np
import unittest
import cpmpy as cp

class TestSolvers(unittest.TestCase):
    def test_installed_solvers(self):
        supported_solvers= [cp.MiniZincPython()]
        # basic model
        x = cp.IntVar(0,2, 3)

        constraints = [
            x[0] < x[1],
            x[1] < x[2]]
        model = cp.Model(constraints)
        for solver in supported_solvers:
            model.solve(solver=solver)
            self.assertEqual([xi.value() for xi in x], [0, 1, 2])
        # Checking all supported solvers
        # for solver in cp.get_supported_solvers():
        #     model.solve(solver=solver)

