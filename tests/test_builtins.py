import pytest

import cpmpy as cp
from cpmpy.expressions.python_builtins import all as cpm_all, any as cpm_any
from cpmpy.exceptions import CPMpyException

iv = cp.intvar(-8, 8, shape=5)


class TestBuiltin:

    def test_max(self):
        constraints = [cp.max(iv) + 9 <= 8]
        model = cp.Model(constraints)
        assert model.solve()
        assert cp.max(iv.value()) <= -1

        _max, define = cp.max(iv).decompose()
        model = cp.Model(_max != 4, define)

        assert model.solve()
        assert max(iv.value()) != 4
        assert cp.max(iv).value() != 4

    def test_min(self):
        constraints = [cp.min(iv) + 9 == 8]
        model = cp.Model(constraints)
        assert model.solve()
        assert str(cp.min(iv.value())) == '-1'

        _min, define = cp.max(iv).decompose()
        model = cp.Model(_min != 4, define)

        assert model.solve()
        assert min(iv.value()) != 4
        assert cp.min(iv).value() != 4


    def test_abs(self):
        constraints = [cp.abs(iv[0]) + 9 <= 8]
        model = cp.Model(constraints)
        assert not model.solve()

        #with list
        constraints = [cp.abs(iv+2) <= 8, iv < 0]
        model = cp.Model(constraints)
        assert model.solve()

        constraints = [cp.abs([iv[0], iv[2], iv[1], -8]) <= 8, iv < 0]
        model = cp.Model(constraints)
        assert model.solve()

        _abs, define = cp.abs(iv[0]).decompose()
        model = cp.Model(_abs != 4, define)

        assert model.solve()
        assert abs(iv[0].value()) != 4
        assert cp.abs(iv[0]).value() != 4

    # Boolean builtins
    def test_all(self):
        # edge-cases
        # Only CPMpy expressions
        x = [cp.boolvar(), cp.BoolVal(False), cp.boolvar()]
        assert str(cpm_all(x)) == "boolval(False)"
        x = [cp.BoolVal(True)]
        assert str(cpm_all(x)) == "boolval(True)"
        x = [cp.BoolVal(False)]
        assert str(cpm_all(x)) == "boolval(False)"

        # mix of Python and CPMpy expressions
        x = [cp.boolvar(), False, cp.boolvar()]
        assert str(cpm_all(x)) == "boolval(False)"
        x = [False, cp.BoolVal(False)]
        assert str(cpm_all(x)) == "boolval(False)"
        x = [False, cp.BoolVal(True)]
        assert str(cpm_all(x)) == "boolval(False)"
        x = [cp.BoolVal(False), False]
        assert str(cpm_all(x)) == "boolval(False)"
        x = [cp.BoolVal(True), False]
        assert str(cpm_all(x)) == "boolval(False)"

        # only Python constants, should override default
        x = [False, True]
        assert str(cpm_all(x)) == "False"
        x = []
        assert str(cpm_all(x)) == "True"

        # should also work with overloaded operators
        expr = cp.BoolVal(False) & cp.BoolVal(True)
        assert str(expr) == "boolval(False)"
        expr = False & cp.BoolVal(True)
        assert str(expr) == "boolval(False)"
        expr = cp.BoolVal(False) & True
        assert str(expr) == "boolval(False)"

        # 1 and 0 are not Boolean
        pytest.raises(ValueError, lambda : cp.BoolVal(False) & 1)
        pytest.raises(ValueError, lambda : cp.BoolVal(False) & 0)

    def test_any(self):
        # edge-cases

        # Only CPMpy expressions
        x = [cp.boolvar(), cp.BoolVal(True), cp.boolvar()]
        assert str(cpm_any(x)) == "boolval(True)"
        x = [cp.BoolVal(True)]
        assert str(cpm_any(x)) == "boolval(True)"
        x = [cp.BoolVal(False)]
        assert str(cpm_any(x)) == "boolval(False)"
        

        # mix of Python and CPMpy expressions
        x = [cp.boolvar(), True, cp.boolvar()]
        assert str(cpm_any(x)) == "boolval(True)"
        x = [True, cp.BoolVal(True)]
        assert str(cpm_any(x)) == "boolval(True)"
        x = [False, cp.BoolVal(False)]
        assert str(cpm_any(x)) == "boolval(False)"
        x = [cp.BoolVal(True), True]
        assert str(cpm_any(x)) == "boolval(True)"
        
        # only Python constants, should override default
        x = [False, True]
        assert str(cpm_any(x)) == "True"
        x = []
        assert str(cpm_any(x)) == "False"
        
        # should also work with overloaded operators
        expr = cp.BoolVal(False) | cp.BoolVal(True)
        assert str(expr) == "boolval(True)"
        expr = False | cp.BoolVal(True)
        assert str(expr) == "boolval(True)"
        expr = cp.BoolVal(False) | True
        assert str(expr) == "boolval(True)"

        # 1 and 0 are not Boolean
        pytest.raises(ValueError, lambda : cp.BoolVal(False) | 1)
        pytest.raises(ValueError, lambda : cp.BoolVal(False) | 0)
