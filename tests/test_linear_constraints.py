import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.int2bool_onehot import int2bool_onehot

class TestLinearConstraint(unittest.TestCase):
    def setUp(self):
        self.bv = boolvar(shape=5)


    def test_pysat_simple_atmost(self):

        atmost = cp.Model(
            ## < This does not work
            2 * self.bv[0] < 3,
            # self.bv[0] *2 < 3,
            ## <=
            3 * self.bv[1] <= 3,
            # self.bv[1] *3 <= 3,
            ## >
            2 * self.bv[2] > 3,
            # self.bv[2] * 2 > 3,
            ## >=
            4 * self.bv[2] >= 3,
            # self.bv[3] * 4 >= 3
        )
        print(int2bool_onehot(atmost))
        ps = CPM_pysat(atmost)
        ps.solve()
        print(self.bv.value())


    def test_pysat_boolean_linear_sum(self):
        ls = cp.Model(
            2 * self.bv[0] + 3 * self.bv[1] <= 3,
        )
        print(ls.constraints)
        ps = CPM_pysat(ls)


if __name__ == '__main__':
    unittest.main()

