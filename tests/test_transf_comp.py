import pytest
import numpy as np
from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl # to reset counters

def _generate_inputs():
    ivs = intvar(1,9, shape=3, name="x")
    for case in [
            (min(ivs) == 3, "[min(x[0],x[1],x[2]) == 3]"),
            (min(ivs) > 3, "[(min(x[0],x[1],x[2])) == (IV0), IV0 > 3]"),
            (min(ivs) <= 3, "[(min(x[0],x[1],x[2])) == (IV0), IV0 <= 3]"),
            (3 != max(ivs), "[(max(x[0],x[1],x[2])) == (IV0), IV0 != 3]"),
            (3 > max(ivs), "[(max(x[0],x[1],x[2])) == (IV0), IV0 < 3]"),
            (3 <= max(ivs), "[(max(x[0],x[1],x[2])) == (IV0), IV0 >= 3]"),
            ]:
        yield case

@pytest.fixture()
def setup():
    _IntVarImpl.counter = 0
    _BoolVarImpl.counter = 0
    yield

class TestTransform:
    @pytest.mark.parametrize(("constraint", "expected"),_generate_inputs(), ids=str)
    def test_transforms(self, constraint, expected, setup):
        flat=only_numexpr_equality(flatten_constraint(constraint))
        assert str(flat) == expected 
        assert Model(constraint).solve()


