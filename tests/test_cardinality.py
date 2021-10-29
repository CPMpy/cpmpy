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

    def test_pysat_equals(self):
        equals = cp.Model(
            sum(self.bvs) == 2
        )
        ps = CPM_pysat(equals)
        ps.solve()

    def test_pysat_atmost_equals(self):
        atmost_equals = cp.Model(
            sum(self.bvs) <= 2,
        )
        ps = CPM_pysat(atmost_equals)
        ps.solve()

    def test_pysat_atleast_equals(self):
        atleast_equals = cp.Model(
            sum(self.bvs) >= 2,
        )
        ps = CPM_pysat(atleast_equals)
        ps.solve()

if __name__ == '__main__':
    unittest.main()
