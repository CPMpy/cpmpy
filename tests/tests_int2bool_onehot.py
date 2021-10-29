import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.to_bool import extract_boolvar, intvar_to_boolvar, to_bool_constraint

class TestInt2BoolOneHot(unittest.TestCase):
    def test_to_bool_constraint(self):
        x1 = intvar(0, 4, name="x1")
        ivarmap, constraint = intvar_to_boolvar(x1)

        bool_constraint = to_bool_constraint(x1!=4, ivarmap)
        self.assertEqual(bool_constraint,  [~ivarmap[x1][4]])

        bool_constraint = to_bool_constraint(x1==4, ivarmap)
        self.assertEqual(bool_constraint,  [ivarmap[x1][4]])

        bool_constraint = to_bool_constraint(x1 < 4, ivarmap)
        self.assertEqual(bool_constraint,  [ivarmap[x1][4]])

if __name__ == '__main__':
    unittest.main()



# import builtins

# from cpmpy import *
# from cpmpy.solvers.pysat import CPM_pysat
# import numpy as np
# import math
# from cpmpy.transformations.to_cnf import to_cnf

# from cpmpy.transformations.to_bool import intvar_to_boolvar

# x1 = intvar(0, 4, name="x1")

# x = intvar(0, 5, shape=3, name="x")

# x2 = intvar(0, 2, name="x2")
# x3 = intvar(0, 3, name="x3")
# bv2 = boolvar()
# bv3 = boolvar()

# y = intvar(2, 4, shape=(3, 4))
# x = [x1, x2, x3]
# a = [3, 2, 5]
# a0 = 5

# m = Model(
#     # x1 != x2,
#     (x1 + x2) < 5,
#     # (x1 + x2 + x3) < 5,
#     # (3 * x1 + 2 * x2 + 5 * x3) < 15,
# )

# print("\n -----Make CNF ------")

# s = CPM_pysat(m)

# print("\n -----Solve ------")
# s.solve()