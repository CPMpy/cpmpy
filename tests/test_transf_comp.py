import pytest
from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl

# three integer variables
xs = intvar(1, 9, shape=3, name="x")

# these expected in- and output constraints for `only_numexpr_equality(flatten_constraint)`
TEST_CASES = [
    (min(xs) == 3, "[min(x[0],x[1],x[2]) == 3]"),
    (min(xs) > 3, "[(min(x[0],x[1],x[2])) == (IV0), IV0 > 3]"),
    (min(xs) <= 3, "[(min(x[0],x[1],x[2])) == (IV0), IV0 <= 3]"),
    (3 != max(xs), "[(max(x[0],x[1],x[2])) == (IV0), IV0 != 3]"),
    (3 > max(xs), "[(max(x[0],x[1],x[2])) == (IV0), IV0 < 3]"),
    (3 <= max(xs), "[(max(x[0],x[1],x[2])) == (IV0), IV0 >= 3]"),
]


@pytest.fixture()
def setup():
    # reset counters *before* parametrization
    _IntVarImpl.counter = 0
    _BoolVarImpl.counter = 0
    yield


class TestTransform:
    @pytest.mark.parametrize("case", TEST_CASES)
    def test_flatten_constraint(self, case, setup):
        constraint, expected = case
        flat = only_numexpr_equality(flatten_constraint(constraint))
        assert str(flat) == expected
        assert Model(constraint).solve()
