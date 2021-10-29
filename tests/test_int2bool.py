import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.to_bool import extract_boolvar, intvar_to_boolvar

class TestInt2Bool(unittest.TestCase):
    def test_intvar_to_boolvar(self):
        iv = intvar(0, 5, shape=1, name="x")
        ivarmap, constraints = intvar_to_boolvar(iv)
        self.assertEqual(len(ivarmap[iv]), iv.ub - iv.lb + 1)
        self.assertEqual(len(extract_boolvar(ivarmap)), iv.ub - iv.lb + 1)

    def test_boolvar_to_boolvar(self):
        bv = boolvar()
        ivarmap, constraints = intvar_to_boolvar(bv)
        self.assertEqual(ivarmap[bv], bv)
        self.assertEqual(len(constraints), 0)
        print(ivarmap, constraints)

    def test_intvar_arr_to_boolvar_array(self):
        iv = intvar(0, 5, shape=10, name="x")
        ivarmap, constraints = intvar_to_boolvar(iv)
        self.assertEqual(len(ivarmap), iv.shape[0])

        # handle more complex matrix data structures
        iv_matrix = intvar(0, 5, shape=(10, 10), name="x")
        ivarmap, constraints = intvar_to_boolvar(iv_matrix)

        self.assertEqual(len(ivarmap), iv_matrix.shape[0] * iv_matrix.shape[1])
        for varmap in ivarmap.values():
            self.assertEqual(len(varmap), iv_matrix[0, 0].ub - iv_matrix[0, 0].lb + 1)


if __name__ == '__main__':
    unittest.main()


