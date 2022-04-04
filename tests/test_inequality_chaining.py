import pdb
from cpmpy import *
# from cpmpy.solvers import CPM_pysat
import unittest

class TestInequalityChaining(unittest.TestCase):
    # def test_single_inequality(self):
    #     bv1, bv2 = boolvar(shape=2)
    #     print(bv1 < bv2)
    #     pass
    def test_chaining_lt(self):

        bv1, bv2, bv3 = boolvar(shape=3)
        # pdb.set_trace()
        c1  = (bv1 > bv2) and (bv2 < bv3)
        c2  = (bv1 < bv2 < bv3)
        self.assertEqual(str(c1), str(c2))

    def test_chaining_lte(self):
        pass
    def test_chaining_gt(self):
        pass
    def test_chaining_gte(self):
        pass
    def test_chaining_mix_inequalities(self):
        pass


if __name__ == '__main__':
    unittest.main()