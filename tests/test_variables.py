import pytest

import cpmpy as cp
import numpy as np
from cpmpy.expressions.variables import NullShapeError, _IntVarImpl, _BoolVarImpl, NegBoolView, NDVarArray, _gen_var_names


class TestSolvers:
    def test_zero_boolvar(self):
        with pytest.raises(NullShapeError):
            bv = cp.boolvar(0)

    def test_unit_boolvar(self):
        bv = cp.boolvar(1)
        assert isinstance(bv, _BoolVarImpl), "boolvar of shape 1 should be base class _BoolVarImpl"
        assert isinstance(~bv, NegBoolView)
        assert isinstance(~(~bv), _BoolVarImpl)

    def test_boolvar(self):
        for i in range(2, 10):
            bv = cp.boolvar(i)
            assert bv.shape == (i,), "Shape should be equal size"
            assert isinstance(bv, NDVarArray), f"Instance not {NDVarArray} got {type(bv)}"

    def test_zero_intvar(self):
        with pytest.raises(NullShapeError):
            iv = cp.intvar(0, 1, 0)

    def test_unit_intvar(self):
        iv = cp.intvar(0, 1, 1)
        assert isinstance(iv, _IntVarImpl), "boolvar of shape 1 should be base class _BoolVarImpl"

    def test_vector_intvar(self):
        for i in range(2, 10):
            iv = cp.intvar(0, i, i)
            assert iv.shape == (i,), f"Shape should be equal size: expected {(i, )} got {iv.shape}"
            assert isinstance(iv, NDVarArray), f"Instance not {NDVarArray} got {type(iv)}"

    def test_array_intvar(self):
        for i in range(2, 10):
            for j in range(2, 10):
                iv = cp.intvar(0, i, shape=(i, j))
                assert iv.shape == (i,j), "Shape should be equal size"
                assert isinstance(iv, NDVarArray), f"Instance not {NDVarArray} got {type(iv)}"

    def test_namevar(self):
        a = cp.boolvar(name="a")
        assert str(a) == "a"

        b = cp.boolvar(shape=(3,), name="b")
        assert str(b) == "[b[0] b[1] b[2]]"

        c = cp.boolvar(shape=(2,3), name="c")
        assert str(c) == "[[c[0,0] c[0,1] c[0,2]]\n [c[1,0] c[1,1] c[1,2]]]"

    def test_invalid_bv(self):

        pytest.raises(ValueError, lambda: cp.boolvar(name="BV123"))
        pytest.raises(ValueError, lambda: cp.boolvar(name="BV123", shape=3))
        pytest.raises(ValueError, lambda: cp.boolvar(name=("BV0", "x", "y"), shape=3))
        pytest.raises(ValueError, lambda: cp.boolvar(name=("x", "BV1", "y"), shape=3))
        pytest.raises(ValueError, lambda: cp.boolvar(name=[["x","y","z"],["a", "BV1", "b"]], shape=(2,3)))


        # this seems fine but it is not!! can still clash
        pytest.raises(ValueError, lambda: cp.boolvar(name="IV123"))
        pytest.raises(ValueError, lambda: cp.boolvar(name="IV123", shape=3))
        pytest.raises(ValueError, lambda: cp.boolvar(name=("IV0", "x", "y"), shape=3))
        pytest.raises(ValueError, lambda: cp.boolvar(name=("x", "IV1", "y"), shape=3))
        pytest.raises(ValueError, lambda: cp.boolvar(name=[["x","y","z"],["a", "IV1", "b"]], shape=(2,3)))


    def test_invalid_iv(self):

        pytest.raises(ValueError, lambda: cp.intvar(0, 10, name="IV123"))
        pytest.raises(ValueError, lambda: cp.intvar(0, 10, name="IV123", shape=3))
        pytest.raises(ValueError, lambda: cp.intvar(0, 10, name=("IV0", "x", "y"), shape=3))
        pytest.raises(ValueError, lambda: cp.intvar(0, 10, name=("x", "IV1", "y"), shape=3))
        pytest.raises(ValueError, lambda: cp.intvar(0,10, name=[["x","y","z"],["a", "IV0", "b"]], shape=(2,3)))

        # this seems fine but it is not!! can still clash
        pytest.raises(ValueError, lambda: cp.intvar(0, 10, name="BV123"))
        pytest.raises(ValueError, lambda: cp.intvar(0, 10, name="BV123", shape=3))
        pytest.raises(ValueError, lambda: cp.intvar(0, 10, name=("BV0", "x", "y"), shape=3))
        pytest.raises(ValueError, lambda: cp.intvar(0, 10, name=("x", "BV1", "y"), shape=3))
        pytest.raises(ValueError, lambda: cp.intvar(0,10, name=[["x","y","z"],["a", "BV0", "b"]], shape=(2,3)))

    def test_clear(self):
        def n_none(v):
            return sum(v.value() == None)

        iv = cp.intvar(1,9, shape=9)
        m = cp.Model(cp.AllDifferent(iv))
        assert n_none(iv) == 9
        m.solve()
        assert n_none(iv) == 0
        iv.clear()
        assert n_none(iv) == 9

        bv = cp.boolvar(9)
        m = cp.Model(sum(bv) > 3)
        assert n_none(bv) == 9
        m.solve()
        assert n_none(bv) == 0
        bv.clear()
        assert n_none(bv) == 9


class TestGenVarNames:

    def test_gen_var_names_basic_string(self):
        assert _gen_var_names('x', (2, 2)) == ['x[0,0]', 'x[0,1]', 'x[1,0]', 'x[1,1]']
        assert _gen_var_names('y', (1, 3)) == ['y[0,0]', 'y[0,1]', 'y[0,2]']
        assert _gen_var_names('z', 4) == ['z[0]', 'z[1]', 'z[2]', 'z[3]']

    def test_gen_var_names_none_name(self):
        assert _gen_var_names(None, (2, 2)) == [None, None, None, None]
        assert _gen_var_names(None, (1, 3)) == [None, None, None]
        assert _gen_var_names(None, 4) == [None, None, None, None]

    def test_gen_var_names_invalid_name_type(self):
        with pytest.raises(TypeError):
            _gen_var_names(123, (2, 2))
        with pytest.raises(TypeError):
            _gen_var_names(45.6, (1, 3))

    def test_gen_var_names_enumerable_name_matching_shape(self):
        assert _gen_var_names(list("abcd"), (4)) == ['a', 'b', 'c', 'd']
        assert _gen_var_names(np.array([list("wx"), list("yz")]), (2, 2)) == ['w', 'x', 'y', 'z']

    def test_gen_var_names_shape_mismatch(self):
        with pytest.raises(ValueError):
            _gen_var_names(list("abc"), (2,2))
        with pytest.raises(ValueError):
            _gen_var_names(np.array(list("abc")), (2,2))

    def test_gen_var_names_duplicated_names(self):
        with pytest.raises(ValueError):
            _gen_var_names(list("aabd"), (4))
        with pytest.raises(ValueError):
            _gen_var_names(np.array(list("xxyz")), (4))
