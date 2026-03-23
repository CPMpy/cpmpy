import pytest
import tempfile
import os
from os.path import join

from numpy import logaddexp
import cpmpy as cp
from cpmpy.expressions.utils import flatlist
from cpmpy.expressions.variables import NullShapeError, _IntVarImpl, _BoolVarImpl, NegBoolView, NDVarArray


class TestModel:
    
    def setup_method(self) -> None:
        self.tempdir = tempfile.mkdtemp()
        print(self.tempdir)
    
    def teardown_method(self) -> None:
        os.rmdir(self.tempdir)

    def test_ndarray(self):
        iv = cp.intvar(1,9, shape=3)
        m = cp.Model( iv > 3 )
        m += (iv[0] == 5)
        assert m.solve()

    def test_empty(self):
        m = cp.Model()
        m += [] # should do nothing
        assert(len(m.constraints) == 0)

    def test_io_nempty(self):
        fname = join(self.tempdir, "model")
        iv = cp.intvar(1,9, shape=3)
        m = cp.Model( iv > 3 )
        m += (iv[0] == 5)
        m.to_file(fname)

        with pytest.warns(UserWarning):
            loaded = cp.Model.from_file(fname)
            assert loaded.solve()
        os.remove(fname)

    def test_io_counters(self):
        _BoolVarImpl.counter = 0  # don't try this at home
        _IntVarImpl.counter = 0  # don't try this at home
        fname = join(self.tempdir, "model")
        iv = cp.intvar(1,9, shape=3)
        bv = cp.boolvar()
        m = cp.Model( iv > 3, ~bv )
        m += (iv[0] == 5)
        m.to_file(fname)

        assert _BoolVarImpl.counter == 1
        assert _IntVarImpl.counter == 3
        _BoolVarImpl.counter = 0  # don't try this at home
        _IntVarImpl.counter = 0  # don't try this at home
        loaded = cp.Model.from_file(fname)
        assert _BoolVarImpl.counter == 1
        assert _IntVarImpl.counter == 3
        os.remove(fname)

    def test_copy(self):
        x,y,z = [cp.boolvar(name=n) for n in "xyz"]

        cons1 = x > y
        cons2 = x + y == 1
        m = cp.Model(cons1, cons2)

        memodict = dict()
        m_dcopy = m.copy()
        print(memodict)
        m_dcopy.solve()

        assert cons1.value()
        assert cons2.value()

        m.solve()

        m2 = m.copy()

        assert m2.constraints[0].value()
        assert m2.constraints[1].value()


    def test_deepcopy(self):
        import copy
        x,y,z = [cp.boolvar(name=n) for n in "xyz"]

        cons1 = x > y
        cons2 = x + y == 1
        cons3 = z > y
        m = cp.Model([cons1, cons2], [cons3])

        memodict = dict()
        m_dcopy = copy.deepcopy(m, memodict)
        m_dcopy.solve()

        assert cons1.value() is None
        assert cons2.value() is None
        assert cons3.value() is None

        m.solve()

        m2 = copy.deepcopy(m)

        for cons in flatlist(m2.constraints):
            assert cons.value()


    def test_unknown_solver(self):

        model = cp.Model(cp.any(cp.boolvar(shape=3)))

        pytest.raises(ValueError, lambda : model.solve(solver="notasolver"))
