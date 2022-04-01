import unittest
import cpmpy as cp 
from cpmpy import *
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.to_cnf import to_cnf

class TestEncodeLinearConstraint(unittest.TestCase):
    def setUp(self):
        self.bv = boolvar(shape=3)

    def test_pysat_simple_atmost(self):

        atmost = cp.Model(
            ## < This does not work
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

    def test_pysat_boolean_linear_sum(self):
        ls = cp.Model(
            2 * self.bv[0] + 3 * self.bv[1] <= 3,
            self.bv[1] == 1,
        )

        ps = CPM_pysat(ls)
        ps.solve()

    def test_pysat_complex_expressions(self):
        ls = cp.Model(
            2 * self.bv[0] + 3 * self.bv[1] <= 3,
            self.bv[1] == 1,
            self.bv[2] == 0
        )

        ps = CPM_pysat(ls)
        ps.solve()


    def test_pysat_unsat(self):
        ls = cp.Model(
            2 * self.bv[0] + 3 * self.bv[1] <= 3,
            self.bv[0] == 1,
            self.bv[1] == 1
        )

        ps = CPM_pysat(ls)
        solved = ps.solve()
        self.assertFalse(solved)

    def test_encode_linear_expressions(self):
        expressions = [
            - self.bv[2] == -1,
            - 2 * self.bv[2] == -2,
            self.bv[0] - self.bv[2] > 0,
            -self.bv[0] + self.bv[2] > 0,
            2 * self.bv[0] + 3 * self.bv[2] > 0,
            2 * self.bv[0] - 3 * self.bv[2] + 2 * self.bv[1] > 0,
            self.bv[0] - 3 * self.bv[2] > 0,
            self.bv[0] - 3 * (self.bv[2] + 2 * self.bv[1])> 0,
        ]

        ## check all types of linear constraints are handled
        for expression in expressions:
            ps = CPM_pysat(Model(
                expression
            ))
            ps.solve()

if __name__ == '__main__':
    unittest.main()

