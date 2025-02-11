import unittest
import cpmpy as cp 
from cpmpy import *
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.to_cnf import to_cnf

class TestEncodePseudoBooleanConstraint(unittest.TestCase):
    def setUp(self):
        self.bv = boolvar(shape=3)

    def test_pysat_simple_atmost(self):

        atmost = cp.Model(
            ## <
            - 2 * self.bv[0] < 3,
            ## <=
            - 3 * self.bv[1] <= 3,
            ## >
            2 * self.bv[2] > 1,
            ## >=
            4 * self.bv[2] >= 3,
        )
        ps = CPM_pysat(atmost)
        ps.solve()


    def test_pysat_wsum_triv_sat(self):
        ls = cp.Model(
            2 * self.bv[0] + 3 * self.bv[1] <= 10,
        )
        ps = CPM_pysat(ls)
        solved = ps.solve()
        self.assertTrue(solved)

    def test_pysat_unsat(self):
        ls = cp.Model(
            2 * self.bv[0] + 3 * self.bv[1] <= 3,
            self.bv[0] == 1,
            self.bv[1] == 1
        )

        ps = CPM_pysat(ls)
        solved = ps.solve()
        self.assertFalse(solved)

    def test_encode_pb_expressions(self):
        expressions = [
            - self.bv[2] == -1,
            - 2 * self.bv[2] == -2,
            self.bv[0] - self.bv[2] > 0,
            -self.bv[0] + self.bv[2] > 0,
            2 * self.bv[0] + 3 * self.bv[2] > 0,
            2 * self.bv[0] - 3 * self.bv[2] + 2 * self.bv[1] > 0,
            self.bv[0] - 3 * self.bv[2] > 0,
            self.bv[0] - 3 * (self.bv[2] + 2 * self.bv[1])> 0,
            # now with var on RHS
            self.bv[0] - 3 * self.bv[1] > self.bv[2],
        ]

        ## check all types of linear constraints are handled
        for expression in expressions:
            Model(expression).solve("pysat")

    def test_encode_pb_oob(self):
        # test out of bounds (meaningless) thresholds
        expressions = [
            sum(self.bv*[2,2,2]) <= 10,  # true
            sum(self.bv*[2,2,2]) <= 6,   # true
            sum(self.bv*[2,2,2]) >= 10,  # false
            sum(self.bv*[2,2,2]) >= 6,   # undecided
            sum(self.bv*[2,-2,2]) <= 10,  # true
            sum(self.bv*[2,-2,2]) <= 4,   # true
            sum(self.bv*[2,-2,2]) >= 10,  # false
            sum(self.bv*[2,-2,2]) >= 4,   # undecided
        ]

        ## check all types of linear constraints are handled
        for expression in expressions:
            Model(expression).solve("pysat")

if __name__ == '__main__':
    unittest.main()

