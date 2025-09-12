import unittest
import cpmpy as cp
import numpy as np
from cpmpy.expressions.variables import NullShapeError, _IntVarImpl, _BoolVarImpl, NegBoolView, NDVarArray, _gen_var_names


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

    def test_invalid_bv(self):

        self.assertRaises(ValueError, lambda: cp.boolvar(name="BV123"))
        self.assertRaises(ValueError, lambda: cp.boolvar(name="BV123", shape=3))
        self.assertRaises(ValueError, lambda: cp.boolvar(name=("BV0", "x", "y"), shape=3))
        self.assertRaises(ValueError, lambda: cp.boolvar(name=("x", "BV1", "y"), shape=3))
        self.assertRaises(ValueError, lambda: cp.boolvar(name=[["x","y","z"],["a", "BV1", "b"]], shape=(2,3)))


        # this seems fine but it is not!! can still clash
        self.assertRaises(ValueError, lambda: cp.boolvar(name="IV123"))
        self.assertRaises(ValueError, lambda: cp.boolvar(name="IV123", shape=3))
        self.assertRaises(ValueError, lambda: cp.boolvar(name=("IV0", "x", "y"), shape=3))
        self.assertRaises(ValueError, lambda: cp.boolvar(name=("x", "IV1", "y"), shape=3))
        self.assertRaises(ValueError, lambda: cp.boolvar(name=[["x","y","z"],["a", "IV1", "b"]], shape=(2,3)))


    def test_invalid_iv(self):

        self.assertRaises(ValueError, lambda: cp.intvar(0, 10, name="IV123"))
        self.assertRaises(ValueError, lambda: cp.intvar(0, 10, name="IV123", shape=3))
        self.assertRaises(ValueError, lambda: cp.intvar(0, 10, name=("IV0", "x", "y"), shape=3))
        self.assertRaises(ValueError, lambda: cp.intvar(0, 10, name=("x", "IV1", "y"), shape=3))
        self.assertRaises(ValueError, lambda: cp.intvar(0,10, name=[["x","y","z"],["a", "IV0", "b"]], shape=(2,3)))

        # this seems fine but it is not!! can still clash
        self.assertRaises(ValueError, lambda: cp.intvar(0, 10, name="BV123"))
        self.assertRaises(ValueError, lambda: cp.intvar(0, 10, name="BV123", shape=3))
        self.assertRaises(ValueError, lambda: cp.intvar(0, 10, name=("BV0", "x", "y"), shape=3))
        self.assertRaises(ValueError, lambda: cp.intvar(0, 10, name=("x", "BV1", "y"), shape=3))
        self.assertRaises(ValueError, lambda: cp.intvar(0,10, name=[["x","y","z"],["a", "BV0", "b"]], shape=(2,3)))

    def test_clear(self):
        def n_none(v):
            return sum(v.value() == None)

        iv = cp.intvar(1,9, shape=9)
        m = cp.Model(cp.AllDifferent(iv))
        self.assertEqual(n_none(iv), 9)
        m.solve()
        self.assertEqual(n_none(iv), 0)
        iv.clear()
        self.assertEqual(n_none(iv), 9)

        bv = cp.boolvar(9)
        m = cp.Model(sum(bv) > 3)
        self.assertEqual(n_none(bv), 9)
        m.solve()
        self.assertEqual(n_none(bv), 0)
        bv.clear()
        self.assertEqual(n_none(bv), 9)


class TestGenVarNames(unittest.TestCase):

    def test_gen_var_names_basic_string(self):
        self.assertEqual(_gen_var_names('x', (2, 2)), ['x[0,0]', 'x[0,1]', 'x[1,0]', 'x[1,1]'])
        self.assertEqual(_gen_var_names('y', (1, 3)), ['y[0,0]', 'y[0,1]', 'y[0,2]'])
        self.assertEqual(_gen_var_names('z', 4), ['z[0]', 'z[1]', 'z[2]', 'z[3]'])

    def test_gen_var_names_none_name(self):
        self.assertEqual(_gen_var_names(None, (2, 2)), [None, None, None, None])
        self.assertEqual(_gen_var_names(None, (1, 3)), [None, None, None])
        self.assertEqual(_gen_var_names(None, 4), [None, None, None, None])

    def test_gen_var_names_invalid_name_type(self):
        with self.assertRaises(TypeError):
            _gen_var_names(123, (2, 2))
        with self.assertRaises(TypeError):
            _gen_var_names(45.6, (1, 3))

    def test_gen_var_names_enumerable_name_matching_shape(self):
        self.assertEqual(_gen_var_names(list("abcd"), (4)), ['a', 'b', 'c', 'd'])
        self.assertEqual(_gen_var_names(np.array([list("wx"), list("yz")]), (2, 2)), ['w', 'x', 'y', 'z'])

    def test_gen_var_names_shape_mismatch(self):
        with self.assertRaises(ValueError):
            _gen_var_names(list("abc"), (2,2))
        with self.assertRaises(ValueError):
            _gen_var_names(np.array(list("abc")), (2,2))

    def test_gen_var_names_duplicated_names(self):
        with self.assertRaises(ValueError):
            _gen_var_names(list("aabd"), (4))
        with self.assertRaises(ValueError):
            _gen_var_names(np.array(list("xxyz")), (4))


if __name__ == "__main__":
    unittest.main()
