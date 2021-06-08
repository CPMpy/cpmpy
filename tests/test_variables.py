import unittest
import cpmpy as cp


class TestSolvers(unittest.TestCase):
    def test_zero_boolvar(self):
        with self.assertRaises(cp.NullShapeError):
            bv = cp.BoolVar(0)

    def test_unit_boolvar(self):
        bv = cp.BoolVar(1)
        self.assertIsInstance(bv, cp.BoolVarImpl, "BoolVar of shape 1 should be base class BoolVarImpl")
        self.assertIsInstance(~bv, cp.NegBoolView)
        self.assertIsInstance(~(~bv), cp.BoolVarImpl)

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

    def test_namevar(self):
        a = cp.BoolVar(name="a")
        self.assertEqual(str(a), "a")

        b = cp.BoolVar(shape=(3,), name="b")
        self.assertEqual(str(b), "[b[0] b[1] b[2]]")

        c = cp.BoolVar(shape=(2,3), name="c")
        self.assertEqual(str(c), "[[c[0,0] c[0,1] c[0,2]]\n [c[1,0] c[1,1] c[1,2]]]")
