import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.int2bool_onehot import int2bool_onehot
from cpmpy.transformations.flatten_model import flatten_constraint

class TestEncodeLinearConstraint(unittest.TestCase):
    def setUp(self):
        self.iv = intvar(lb=1, ub=4, shape=3)
        self.bv = boolvar(shape=3)

    def test_encode_linear_expressions(self):
        print(-(-self.iv[0]))
        expressions = [
            - self.iv[2] == -1,
            - self.bv[2] == -1,
            - 2 * self.iv[2] == -2,
            - 2 * self.bv[2] == -2,
            self.iv[0] - self.iv[2] > 1,
            self.bv[0] - self.bv[2] > 0,
            -self.bv[0] + self.bv[2] > 0,
            self.iv[0] - 3 * self.iv[2] >= 1,
            self.bv[0] - 3 * self.bv[2] > 0,
            2 * self.iv[0] + 3 * self.iv[0] + 3 * self.iv[2] < 9,
            2 * self.iv[0] - 3 * self.iv[2] < 9,
            2 * self.iv[0] + 3  * (self.iv[2] + self.iv[1]) < 9,
            2 * self.iv[0] + 3  * (self.iv[2] - self.iv[1]) < 9,
        ]

        for expression in expressions:
            print(f"{expression=}", "\n\n\t", flatten_constraint(expression), "\n")
            ps = CPM_pysat(cp.Model(
                expression
            ))
            ps.solve()

class TestLinearConstraint(unittest.TestCase):
    def setUp(self):
        self.bv = boolvar(shape=3)

    def test_pysat_simple_atmost(self):

        atmost = cp.Model(
            ## < This does not work
            2 * self.bv[0] < 3,
            ## <=
            3 * self.bv[1] <= 3,
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
        )
        ps = CPM_pysat(ls)
        ps.solve()

class TestIntVarLinearConstraint(unittest.TestCase):
    def setUp(self) -> None:
        self.iv = intvar(lb=4, ub=8, shape=2)
        return super().setUp()

    def test_linearsum_smaller(self):
        ls = cp.Model(
            self.iv[0] + self.iv[1] < 9
        )
        ps = CPM_pysat(ls)
        ps.solve()

    def test_linearsum_smaller_equal(self):
        ls = cp.Model(
            self.iv[0] + self.iv[1] <= 9
        )
        ps = CPM_pysat(ls)
        ps.solve()

    def test_linearsum_larger_equal(self):
        ls = cp.Model(
            self.iv[0] + self.iv[1] >= 9
        )
        ps = CPM_pysat(ls)
        ps.solve()
    
    def test_linearsum_larger(self):
        ls = cp.Model(
            self.iv[0] + self.iv[1] > 8
        )
        ps = CPM_pysat(ls)
        ps.solve()

    def test_linearsum_equality(self):
        ls = cp.Model(
            self.iv[0] + self.iv[1] == 15
        )
        ps = CPM_pysat(ls)
        ps.solve()

    def test_weighted_linearsum_smaller(self):
        ls = cp.Model(
            2 * self.iv[0] + 3 * self.iv[1] < 9
        )
        ps = CPM_pysat(ls)
        ps.solve()

    def test_weighted_linearsum_smaller_equal(self):
        ls = cp.Model(
            2 * self.iv[0] + 3 * self.iv[1] <= 9
        )
        ps = CPM_pysat(ls)
        ps.solve()

    def test_weighted_linearsum_larger_equal(self):
        ls = cp.Model(
            2 * self.iv[0] + 3 * self.iv[1] >= 9
        )
        ps = CPM_pysat(ls)
        ps.solve()
    
    def test_weighted_linearsum_larger(self):
        ls = cp.Model(
            2 * self.iv[0] + 3 * self.iv[1] > 8
        )
        ps = CPM_pysat(ls)
        ps.solve()

    def test_weighted_linearsum_equality(self):
        ls = cp.Model(
            2 * self.iv[0] + 3 * self.iv[1] == 15
        )
        ps = CPM_pysat(ls)
        ps.solve()
    

if __name__ == '__main__':
    unittest.main()

