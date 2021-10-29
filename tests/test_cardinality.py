import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.solvers.pysat import CPM_pysat

class TestCardinality(unittest.TestCase):
    def setUp(self):
        self.bvs = boolvar(shape=3)

    def test_pysat_atmost(self):

        atmost = cp.Model(
            sum(self.bvs) < 2
        )
        ps = CPM_pysat(atmost)
        ps.solve()
        # all must be true

        self.assertLess(sum(self.bvs.value()), 2)

    def test_pysat_atleast(self):

        atmost = cp.Model(
            sum(self.bvs) > 2
        )
        ps = CPM_pysat(atmost)
        ps.solve()
        # all must be true
        self.assertEqual(sum(self.bvs.value()), 3)

    def test_pysat_equals(self):
        equals = cp.Model(
            sum(self.bvs) == 2
        )
        ps = CPM_pysat(equals)
        ps.solve()
        self.assertEqual(sum(self.bvs.value()), 2)

        equals2 = cp.Model(
            sum(self.bvs) >= 2,
            sum(self.bvs) <= 2,
        )
        ps2 = CPM_pysat(equals2)
        ps2.solve()
        self.assertEqual(sum(self.bvs.value()), 2)

        equals3 = cp.Model(
            sum(self.bvs) > 1,
            sum(self.bvs) < 3,
        )
        ps3 = CPM_pysat(equals3)
        ps3.solve()
        self.assertEqual(sum(self.bvs.value()), 2)

    def test_pysat_atmost_equals(self):
        atmost_equals = cp.Model(
            sum(self.bvs) <= 2,
        )
        ps = CPM_pysat(atmost_equals)
        ps.solve()
        self.assertLessEqual(sum(self.bvs.value()), 2)

    def test_pysat_atleast_equals(self):
        atleast_equals = cp.Model(
            sum(self.bvs) >= 2,
        )
        ps = CPM_pysat(atleast_equals)
        ps.solve()

        self.assertGreaterEqual(sum(self.bvs.value()), 2)

if __name__ == '__main__':
    unittest.main()
