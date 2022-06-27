from cpmpy import *
import unittest

class TestInequalityChaining(unittest.TestCase):
    def test_chaining_lt(self):

        bv1, bv2, bv3 = boolvar(shape=3)
        c1  = (bv1 < bv2) and (bv2 < bv3)
        c2  = (bv1 < bv2 < bv3)
        self.assertEqual(str(c1), str(c2))


if __name__ == '__main__':
    unittest.main()