import unittest

import cpmpy as cp

class TestSolveAll(unittest.TestCase):


    def test_solveall_no_obj(self):

        x = cp.intvar(0,3, shape=3)
        m = cp.Model(cp.sum(x) <= 3)

        for name, solver in cp.SolverLookup.base_solvers():
            if not solver.supported():
                continue

            try:
                solver = cp.SolverLookup.get(name,model=m)
                self.assertEqual(20, solver.solveAll(solution_limit=1000))
            except NotImplementedError:
                pass # solver does not support constraint or solveAll


    def test_solveall_with_obj(self):

        x = cp.intvar(0, 3, shape=3)
        m = cp.Model(minimize=cp.sum(x))

        for name, solver in cp.SolverLookup.base_solvers():
            if not solver.supported():
                continue

            try:
                solver = cp.SolverLookup.get(name, model=m)
                self.assertEqual(1, solver.solveAll(solution_limit=1000))
            except (Exception, NotImplementedError):
                pass # solver does not support finding all optimal solutions
