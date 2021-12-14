import unittest

from cpmpy.solvers import CPM_pysat, CPM_ortools
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy import *

class TestInterface(unittest.TestCase):

    # Replace by your own solver class
    solver_class = CPM_ortools

    def test_empty_constructor(self):

        solver = TestInterface.solver_class()

        self.assertIsNotNone(solver.status())
        self.assertEqual(solver.status().exitstatus, ExitStatus.NOT_RUN)
        self.assertNotEquals(solver.status().solver_name, "dummy")

    def test_constructor(self):

        x,y = boolvar(shape=2)
        model = Model([x & y])

        solver = TestInterface.solver_class(model)

        self.assertIsNotNone(solver.status())
        self.assertEqual(solver.status().exitstatus, ExitStatus.NOT_RUN)
        self.assertNotEquals(solver.status().solver_name, "dummy")


    def test_add_constraint(self):

        solver = TestInterface.solver_class()

        x,y = boolvar(shape=(2))
        z = boolvar(shape=3)

        solver += [x & y]
        self.assertEqual(len(solver.user_vars), 2)

        solver += [sum(z) >= 2]
        self.assertEqual(len(solver.user_vars), 5)

    def test_solve(self):

        solver = TestInterface.solver_class()

        x,y,z = boolvar(shape=3)

        solver += x.implies(y & z)
        solver += y | z

        self.assertTrue(solver.solve())
        self.assertTrue(solver.status().exitstatus, ExitStatus.FEASIBLE)

        self.assertEquals((x,y,z), (0,1,0))


    def test_objective(self):

        x = intvar(1, 10)
        solver = TestInterface.solver_class()

        try:
            solver.minimize(x)
        except NotImplementedError:
            return

        self.assertFalse(hasattr(solver, "objective_value_"))
        self.assertTrue(solver.solve())
        self.assertEqual(solver.objective_value(), 1)

    def test_add_var(self):

        x = boolvar()

        solver = TestInterface.solver_class()

        solver.solver_var(x)

        self.assertEqual(len(solver.user_vars), 1)
        self.assertEqual(len(solver._varmap), 1)
