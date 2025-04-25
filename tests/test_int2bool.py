import pytest

import cpmpy as cp
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.expressions.core import Comparison, Operator, BoolVal

from cpmpy.transformations.int2bool import int2bool
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl  # to reset counters


x = cp.intvar(0, 2, name="x")
y = cp.intvar(0, 2, name="y")
z = cp.intvar(0, 2, name="z")

p = cp.boolvar(name="p")
q = cp.boolvar(name="p")

c = cp.intvar(2, 2, name="c")

CONSTRAINTS = [
    BoolVal(True),
    BoolVal(False),
    p.implies(q),
] + [
    con if antecedent is True else antecedent.implies(con)
    for cmp in ("==", "!=", ">=", "<=", ">", "<")
    for con in (
        Comparison(cmp, c, 1),
        Comparison(cmp, c, 2),
        Comparison(cmp, x, 1),
        Comparison(cmp, x, 5),
        Comparison(cmp, Operator("sum", [[x, y, z]]), 3),
        Comparison(cmp, Operator("wsum", [[2, 3, 5], [x, y, z]]), 12),
        Comparison(cmp, Operator("wsum", [[2, 3, 5], [x, y, z]]), 100),
        Comparison(cmp, Operator("wsum", [[2, 3, 5, 4], [x, y, z, c]]), 16),
    )
    for antecedent in (True, p, ~p)
]


@pytest.fixture()
def setup():
    _IntVarImpl.counter = 0
    _BoolVarImpl.counter = 0
    yield


class TestTransInt2Bool:

    @pytest.mark.parametrize("constraint", CONSTRAINTS, ids=str)
    def test_transforms(self, constraint, setup):
        flat = int2bool(flatten_constraint(constraint))
        num_sols_cons = cp.Model(constraint).solveAll(solver="ortools")  # trusted model
        num_sols_flat = cp.Model(flat).solveAll(solver="pysat")
        assert (
            num_sols_cons == num_sols_flat
        ), f"Constraint and its transformations have different number of solutions, meaning:\n{constraint} != {flat}"
