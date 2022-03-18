import unittest
import cpmpy as cp 
from cpmpy import *
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.to_cnf import to_cnf

# class TestCardinality(unittest.TestCase):
#     def setUp(self):
#         self.bvs = boolvar(shape=5)

#     def test_unit_wsum(self):
#         v = self.bvs[0] + self.bvs[1] + self.bvs[2] < 2
#         print(to_cnf(v))
#         s = CPM_pysat(Model(v))
#         s.solve()
#         print(self.bvs.value())

#     def test_wsum(self):
#         formula = 2 * self.bvs[0] + self.bvs[1] + self.bvs[2] < 2
#         cnf_formula = to_cnf(formula)
#         print(formula, cnf_formula)
#         s = CPM_pysat(Model(cnf_formula))
#         s.solve()
#         print(self.bvs.value())

class TestEncodeLinearConstraint(unittest.TestCase):
    def setUp(self):
        self.bv = boolvar(shape=3)

    # def test_pysat_simple_atmost(self):

    #     atmost = cp.Model(
    #         ## < This does not work
    #         2 * self.bv[0] < 3,
    #         ## <=
    #         3 * self.bv[1] <= 3,
    #         ## >
    #         2 * self.bv[2] > 1,
    #         ## >=
    #         4 * self.bv[2] >= 3,
    #     )
    #     ps = CPM_pysat(atmost)
    #     ps.solve()

    # def test_pysat_boolean_linear_sum(self):
    #     ls = cp.Model(
    #         2 * self.bv[0] + 3 * self.bv[1] <= 3,
    #     )
    #     ps = CPM_pysat(ls)
    #     ps.solve()


    def test_encode_linear_expressions(self):
        expressions = [
            # - self.bv[2] == -1,
            # - 2 * self.bv[2] == -2,
            # self.bv[0] - self.bv[2] > 0,
            # -self.bv[0] + self.bv[2] > 0,
            self.bv[0] - 3 * self.bv[2] > 0,
        ]

        for expression in expressions:
            print(expression, to_cnf(expression))
            ps = CPM_pysat(Model(
                expression
            ))
            ps.solve()

if __name__ == '__main__':
    unittest.main()

