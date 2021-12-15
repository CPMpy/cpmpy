import unittest

from cpmpy.solvers import CPM_pysat, CPM_ortools
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint


class TestInterface(unittest.TestCase):

    # Replace by your own solver class
    solver_class = CPM_ortools

    def setUp(self) -> None:
        self.solver = TestInterface.solver_class()

        self.bvar = boolvar(shape=3)
        self.x,self.y,self.z = self.bvar

        ivar = intvar(1,10, shape=2)
        self.i, self.j = ivar

    def test_empty_constructor(self):

        self.assertTrue(hasattr(self, "solver"))

        self.assertIsNotNone(self.solver.status())
        self.assertEqual(self.solver.status().exitstatus, ExitStatus.NOT_RUN)
        self.assertNotEquals(self.solver.status().solver_name, "dummy")

    def test_constructor(self):

        model = Model([self.x & self.y])
        solver = TestInterface.solver_class(model)

        self.assertIsNotNone(solver.status())
        self.assertEqual(solver.status().exitstatus, ExitStatus.NOT_RUN)
        self.assertNotEquals(solver.status().solver_name, "dummy")

    def test_add_var(self):

        self.solver += self.x

        self.assertEqual(1, len(self.solver.user_vars))
        self.assertEqual(1, len(self.solver._varmap))



    def test_add_constraint(self):

        self.solver += [self.x & self.y]
        self.assertEqual(2, len(self.solver.user_vars))

        self.solver += [sum(self.bvar) >= 2]
        self.assertEqual(3,len(self.solver.user_vars))
        self.assertGreaterEqual(3, len(self.solver._varmap)) # Possible that solver requires extra intermediate vars

    def test_solve(self):

        self.solver += self.x.implies(self.y & self.z)
        self.solver += self.y | self.z

        self.assertTrue(self.solver.solve())
        self.assertEqual(ExitStatus.FEASIBLE, self.solver.status().exitstatus)

        self.assertEquals((0,1,0), self.bvar)


    def test_objective(self):

        try:
            self.solver.minimize(self.i)
        except NotImplementedError:
            #TODO: assert false or just ignore and return?
            return

        self.assertFalse(hasattr(self.solver, "objective_value_"))
        self.assertTrue(self.solver.solve())
        self.assertEqual(1, self.solver.objective_value())
        self.assertEqual(ExitStatus.OPTIMAL, self.solver.status().exitstatus)

    def test_operators(self):
        """
        TODO: test every operator, but might be difficult, not all solvers support every operator...
        How should we check this? Just catch 'NotImplemented' errors? Or maybe make a different test for every operator?
        This way it is clear to the user which operators failed and if this is in line with his expectations...
        """
        pass

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


    # Test non-boolean operators, checked by directly posting constraint
    # TODO: Not all accepted operators are working --> because of transformations
    # Maybe we should add a "transform(cpm_expr)" function to the interface?

    def test_sum(self):

        try:
            self.solver._post_constraint(sum([self.x, self.y]))
        except NotImplementedError:
            return
        self.check_xy()


    def test_sub(self):
        #Todo: fix with tranformations
        try:
            self.solver._post_constraint(self.x - self.y)
        except NotImplementedError:
            return
        self.check_xy()


    def test_mul(self):

        try:
            self.solver._post_constraint(self.x * self.y)
        except NotImplementedError:
            return
        self.check_xy()

    def test_div(self):
        try:
            self.solver._post_constraint(self.x / self.y)
        except NotImplementedError:
            return
        self.check_xy()

    def test_mod(self):

        try:
            self.solver._post_constraint(self.x % self.y)
        except NotImplementedError:
            return
        self.check_xy()

    def test_pow(self):

        try:
            self.solver._post_constraint(self.x ** self.y)
        except NotImplementedError:
            return
        self.check_xy()

    def test_min(self):

        try:
            self.solver._post_constraint(- self.i)
        except NotImplementedError:
            return
        self.assertIn(self.i, self.solver.user_vars)

    def test_abs(self):
        #Todo fix with transformations
        try:
            self.solver._post_constraint(abs(self.i))
        except NotImplementedError:
            return
        self.assertIn(self.i, self.solver.user_vars)







