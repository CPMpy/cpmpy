import unittest

import pytest

from cpmpy.expressions.core import Operator, Comparison
from cpmpy.solvers import CPM_pysat, CPM_ortools, CPM_minizinc, CPM_gurobi
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint

SOLVER_CLASS = CPM_gurobi  # Replace by your own solver class


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

        self.solver += [sum(self.bvar) >= 2]
        self.assertEqual(3, len(self.solver.user_vars))
        self.assertGreaterEqual(3, len(self.solver._varmap))  # Possible that solver requires extra intermediate vars

    def test_solve(self):

        self.solver += self.x.implies(self.y & self.z)
        self.solver += self.y | self.z
        self.solver += ~ self.z

        self.assertTrue(self.solver.solve())
        self.assertEqual(ExitStatus.FEASIBLE, self.solver.status().exitstatus)

        self.assertListEqual([0, 1, 0], [self.x.value(), self.y.value(), self.z.value()])

    def test_objective(self):

        try:
            self.solver.minimize(self.i)
        except NotImplementedError:
            # TODO: assert false or just ignore and return?
            return

        self.assertFalse(hasattr(self.solver, "objective_value_"))
        self.assertTrue(self.solver.solve())
        self.assertEqual(1, self.solver.objective_value())
        self.assertEqual(ExitStatus.OPTIMAL, self.solver.status().exitstatus)

    #########################
    #    Test operators     #
    #########################

    def check_xy(self):
        self.assertIn(self.x, self.solver.user_vars)
        self.assertIn(self.y, self.solver.user_vars)
        self.assertIn(self.x, self.solver._varmap)
        self.assertIn(self.y, self.solver._varmap)

    # Test boolean operators

    def test_eq(self):

        self.solver += self.x == self.y
        self.check_xy()

    def test_neq(self):

        self.solver += self.x != self.y
        self.check_xy()

    def test_lt(self):

        self.solver += self.x < self.y
        self.check_xy()

    def test_leq(self):

        self.solver += self.x <= self.y
        self.check_xy()

    def test_gt(self):

        self.solver += self.x > self.y
        self.check_xy()

    def test_geq(self):

        self.solver += self.x >= self.y
        self.check_xy()

    def test_and(self):

        self.solver += self.x & self.y
        self.check_xy()

    def test_or(self):

        self.solver += self.x | self.y
        self.check_xy()

    def test_xor(self):

        self.solver += self.x ^ self.y
        self.check_xy()

    def test_impl(self):

        self.solver += self.x.implies(self.y)
        self.check_xy()

    # Test non-boolean operators, checked by directly posting constraints
    def check_ij(self):
        self.assertIn(self.i, self.solver.user_vars)
        self.assertIn(self.j, self.solver.user_vars)
        self.assertIn(self.i, self.solver._varmap)
        self.assertIn(self.j, self.solver._varmap)

    def test_sum(self):

        self.solver += sum([self.i, self.j]) == 0
        self.check_ij()

    def test_sub(self):

        self.solver += (self.i - self.j) == 0
        self.check_ij()

    def test_mul(self):

        self.solver += (self.i * self.j) == 0
        self.check_ij()

    def test_div(self):

        self.solver += (self.i / self.j) == 0
        self.check_ij()

    def test_mod(self):

        self.solver += (self.i % self.j) == 0
        self.check_ij()

    def test_pow(self):

        self.solver += (self.i ** self.j) == 0
        self.check_ij()

    def test_min(self):

        self.solver += - self.i == 0
        self.assertIn(self.i, self.solver.user_vars)

    def test_abs(self):

        self.solver += abs(self.i) == 0
        self.assertIn(self.i, self.solver.user_vars)