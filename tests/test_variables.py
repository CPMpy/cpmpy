import unittest
import cpmpy as cp
from cpmpy.expressions.variables import NullShapeError, _IntVarImpl, _BoolVarImpl, NegBoolView, NDVarArray


class TestSolvers(unittest.TestCase):
    def test_zero_boolvar(self):
        with self.assertRaises(NullShapeError):
            bv = cp.boolvar(0)

    def test_unit_boolvar(self):
        bv = cp.boolvar(1)
        self.assertIsInstance(bv, _BoolVarImpl, "boolvar of shape 1 should be base class _BoolVarImpl")
        self.assertIsInstance(~bv, NegBoolView)
        self.assertIsInstance(~(~bv), _BoolVarImpl)

    def test_boolvar(self):
        for i in range(2, 10):
            bv = cp.boolvar(i)
            self.assertEqual(bv.shape, (i,), "Shape should be equal size")
            self.assertIsInstance(bv, NDVarArray, f"Instance not {NDVarArray} got {type(bv)}")

    def test_zero_intvar(self):
        with self.assertRaises(NullShapeError):
            iv = cp.intvar(0, 1, 0)

    def test_unit_intvar(self):
        iv = cp.intvar(0, 1, 1)
        self.assertIsInstance(iv, _IntVarImpl, "boolvar of shape 1 should be base class _BoolVarImpl")

    def test_vector_intvar(self):
        for i in range(2, 10):
            iv = cp.intvar(0, i, i)
            self.assertEqual(iv.shape, (i,), f"Shape should be equal size: expected {(i, )} got {iv.shape}")
            self.assertIsInstance(iv, NDVarArray, f"Instance not {NDVarArray} got {type(iv)}")

    def test_array_intvar(self):
        for i in range(2, 10):
            for j in range(2, 10):
                iv = cp.intvar(0, i, shape=(i, j))
                self.assertEqual(iv.shape, (i,j), "Shape should be equal size")
                self.assertIsInstance(iv, NDVarArray, f"Instance not {NDVarArray} got {type(iv)}")

    def test_namevar(self):
        a = cp.boolvar(name="a")
        self.assertEqual(str(a), "a")

        b = cp.boolvar(shape=(3,), name="b")
        self.assertEqual(str(b), "[b[0] b[1] b[2]]")

        c = cp.boolvar(shape=(2,3), name="c")
        self.assertEqual(str(c), "[[c[0,0] c[0,1] c[0,2]]\n [c[1,0] c[1,1] c[1,2]]]")
