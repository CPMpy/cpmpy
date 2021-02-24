import unittest
import cpmpy as cp


class TestSolvers(unittest.TestCase):
    def test_zero_boolvar(self):
        with self.assertRaises(cp.NullShapeError):
            bv = cp.BoolVar(0)

    def test_unit_boolvar(self):
        bv = cp.BoolVar(1)
        self.assertIsInstance(bv, cp.BoolVarImpl, "BoolVar of shape 1 should be base class BoolVarImpl")

    def test_boolvar(self):
        for i in range(2, 10):
            bv = cp.BoolVar(i)
            self.assertEqual(bv.shape, (i,), "Shape should be equal size")
            self.assertIsInstance(bv, cp.NDVarArray, f"Instance not {cp.NDVarArray} got {type(bv)}")

    def test_zero_IntVar(self):
        with self.assertRaises(cp.NullShapeError):
            iv = cp.IntVar(0, 1, 0)

    def test_unit_IntVar(self):
        iv = cp.IntVar(0, 1, 1)
        self.assertIsInstance(iv, cp.IntVarImpl, "BoolVar of shape 1 should be base class BoolVarImpl")

    def test_vector_IntVar(self):
        for i in range(2, 10):
            iv = cp.IntVar(0, i, i)
            self.assertEqual(iv.shape, (i,), f"Shape should be equal size: expected {(i, )} got {iv.shape}")
            self.assertIsInstance(iv, cp.NDVarArray, f"Instance not {cp.NDVarArray} got {type(iv)}")

    def test_array_IntVar(self):
        for i in range(2, 10):
            for j in range(2, 10):
                iv = cp.IntVar(0, i, shape=(i, j))
                self.assertEqual(iv.shape, (i,j), "Shape should be equal size")
                self.assertIsInstance(iv, cp.NDVarArray, f"Instance not {cp.NDVarArray} got {type(iv)}")
