import unittest

import pytest

from cpmpy.expressions.core import Operator, Comparison
from cpmpy.solvers import CPM_pysat, CPM_ortools, CPM_minizinc, CPM_gurobi
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint

SOLVER_CLASS = CPM_ortools  # Replace by your own solver class


class TestInterface(unittest.TestCase):

    def setUp(self) -> None:
        self.solver = SOLVER_CLASS()

        self.bvar = boolvar(shape=3)
        self.x, self.y, self.z = self.bvar

        ivar = intvar(1, 10, shape=2)
        self.i, self.j = ivar

    def test_empty_constructor(self):

        self.assertTrue(hasattr(self, "solver"))

        self.assertIsNotNone(self.solver.status())
        self.assertEqual(self.solver.status().exitstatus, ExitStatus.NOT_RUN)
        self.assertNotEqual(self.solver.status().solver_name, "dummy")

    def test_constructor(self):

        m = Model([self.x & self.y])
        solver = SOLVER_CLASS(m)

        self.assertIsNotNone(solver.status())
        self.assertEqual(solver.status().exitstatus, ExitStatus.NOT_RUN)
        self.assertNotEqual(solver.status().solver_name, "dummy")

    def test_add_var(self):

        self.solver += self.x

        self.assertEqual(1, len(self.solver.user_vars))
        self.assertEqual(1, len(self.solver._varmap))

    def test_add_constraint(self):

        self.solver += [self.x & self.y]
        self.assertEqual(2, len(self.solver.user_vars))

        self.solver += [sum(self.bvar) == 2]
        self.assertEqual(3, len(self.solver.user_vars))
        self.assertGreaterEqual(3, len(self.solver._varmap))  # Possible that solver requires extra intermediate vars

    def test_solve(self):

        self.solver += self.x.implies(self.y & self.z)
        self.solver += self.y | self.z
        self.solver += ~ self.z

        self.assertTrue(self.solver.solve())
        self.assertTrue(self.solver.status().exitstatus == ExitStatus.FEASIBLE or self.solver.status().exitstatus == ExitStatus.OPTIMAL)

        self.assertListEqual([0, 1, 0], [self.x.value(), self.y.value(), self.z.value()])

    def test_solve_infeasible(self):

        self.solver += self.x.implies(self.y & self.z)
        self.solver += ~ self.z
        self.solver += self.x

        self.assertFalse(self.solver.solve())
        self.assertEqual(ExitStatus.UNSATISFIABLE, self.solver.status().exitstatus)

    def test_objective(self):

        try:
            self.solver.minimize(self.i)
        except NotImplementedError:
            # TODO: assert false or just ignore and return?
            return

        self.assertTrue(hasattr(self.solver, "objective_value_"))
        self.assertTrue(self.solver.solve())
        self.assertEqual(1, self.solver.objective_value())
        self.assertEqual(ExitStatus.OPTIMAL, self.solver.status().exitstatus)