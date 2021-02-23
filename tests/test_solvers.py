from cppy.solver_interfaces import get_supported_solvers
import numpy as np
import unittest
import cpmpy as cp

class TestSolvers(unittest.TestCase):
    def test_installed_solvers(self):
        # basic model
        x = cp.IntVar(0,2, 3)

        constraints = [
            x[0] < x[1],
            x[1] < x[2]]
        model = cp.Model(constraints)

        # Checking all supported solvers
        for solver in get_supported_solvers():
            _ = model.solve(solver=solver)
            self.assertEqual(vars.value(), [0, 1, 2])

