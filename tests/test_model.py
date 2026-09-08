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


class TestModelDescription:

    def test_single_constraint(self):
        x = cp.boolvar(shape=5)
        m = cp.Model()
        cons = cp.sum(x) <= 3
        with m.description("at most three items"):
            m += cons
        assert len(m.constraints) == 1
        assert m.constraints[0] is cons
        assert cons.name != "and"
        assert str(cons) == "at most three items"
        assert cons._description is not None

    def test_multiple_adds(self):
        x = cp.boolvar(shape=5)
        m = cp.Model()
        c1 = x[0].implies(x[1])
        c2 = x[0].implies(x[2])
        with m.description("first item implies the next two"):
            m += c1
            m += c2
        assert len(m.constraints) == 1
        wrapped = m.constraints[0]
        assert wrapped.name == "and"
        assert str(wrapped) == "first item implies the next two"
        assert c1._description is None
        assert c2._description is None
        assert c1 in wrapped.args
        assert c2 in wrapped.args

    def test_list_of_constraints(self):
        a, b = cp.boolvar(name="a"), cp.boolvar(name="b")
        m = cp.Model()
        with m.description("both"):
            m += [a, b]
        assert len(m.constraints) == 1
        wrapped = m.constraints[0]
        assert wrapped.name == "and"
        assert str(wrapped) == "both"
        assert a._description is None
        assert b._description is None

    def test_empty(self):
        m = cp.Model()
        with m.description("nothing"):
            pass
        assert len(m.constraints) == 0

    def test_exception_drops_buffer(self):
        m = cp.Model()
        a = cp.boolvar(name="a")
        try:
            with m.description("will fail"):
                m += a == 1
                raise ValueError("boom")
        except ValueError:
            pass
        assert len(m.constraints) == 0

    def test_outside_and_other_model_unaffected(self):
        m = cp.Model()
        other = cp.Model()
        a, b = cp.boolvar(name="a"), cp.boolvar(name="b")
        m += a
        other += b
        with m.description("described"):
            m += a | b
        m += b
        other += a
        assert len(m.constraints) == 3
        assert m.constraints[0] is a
        assert str(m.constraints[1]) == "described"
        assert m.constraints[2] is b
        assert len(other.constraints) == 2
        assert other.constraints[0] is b
        assert other.constraints[1] is a

    def test_boolvar_keeps_name(self):
        b = cp.boolvar(name="flag")
        m = cp.Model()
        with m.description("must be true"):
            m += b
        assert b.name == "flag"
        assert repr(b) == "flag"
        assert str(b) == "must be true"
        assert b._description is not None

    def test_nested_same_model(self):
        x = cp.boolvar(shape=5)
        m = cp.Model()
        with m.description("packing rules"):
            m += cp.sum(x) <= 3
            with m.description("first item implies the next two"):
                m += x[0].implies(x[1])
                m += x[0].implies(x[2])
        assert len(m.constraints) == 1
        outer = m.constraints[0]
        assert outer.name == "and"
        assert str(outer) == "packing rules"
        assert len(outer.args) == 2
        inner = outer.args[1]
        assert inner.name == "and"
        assert str(inner) == "first item implies the next two"
        assert len(inner.args) == 2

    def test_nested_different_models(self):
        m1, m2 = cp.Model(), cp.Model()
        a, b = cp.boolvar(name="a"), cp.boolvar(name="b")
        with m1.description("on m1"):
            with m2.description("on m2"):
                m1 += a
                m2 += b
        assert len(m1.constraints) == 1
        assert str(m1.constraints[0]) == "on m1"
        assert a.name == "a"
        assert len(m2.constraints) == 1
        assert str(m2.constraints[0]) == "on m2"
        assert b.name == "b"

    def test_nested_inner_exception(self):
        m = cp.Model()
        a, b = cp.boolvar(name="a"), cp.boolvar(name="b")
        with m.description("outer"):
            m += a == 1
            try:
                with m.description("inner"):
                    m += b == 1
                    raise ValueError("boom")
            except ValueError:
                pass
        assert len(m.constraints) == 1
        assert str(m.constraints[0]) == "outer"
        assert a.name == "a"
        assert b.name == "b"

