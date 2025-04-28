import pytest

from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.core import Comparison, Operator, BoolVal
from cpmpy.expressions.utils import argvals
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.model import Model

from cpmpy.transformations.int2bool import int2bool
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl, intvar, boolvar


# x = intvar(0, 2, name="x")
# y = intvar(0, 2, name="y")
# z = intvar(0, 2, name="z")

x = intvar(1, 3, name="x")
y = intvar(1, 3, name="y")
z = intvar(1, 3, name="z")


p = boolvar(name="p")
q = boolvar(name="q")

c = intvar(2, 2, name="c")

CONSTRAINTS = [
    # BoolVal(True),  # TODO or tools problem
    BoolVal(False),
    p,
    ~p,
    p.implies(q),
] + [
    con if antecedent is True else antecedent.implies(con)
    for cmp in (
        ">=",
        "<=",
        "==",
        # "!=", ">", "<", # not produced by linearize
    )
    for con in (
        Comparison(cmp, c, 1),
        Comparison(cmp, c, 2),
        Comparison(cmp, x, 2),
        Comparison(cmp, x, 5),
        Comparison(cmp, Operator("sum", [[x, y, z]]), 4),
        # Comparison(cmp, Operator("wsum", [[2, 3, 5], [x, y, z]]), 11), TODO or tools gives too moany sols
        # Comparison(cmp, Operator("wsum", [[2, 3, 5, 4], [x, y, z, c]]), 16),  # TODO same
        Comparison(cmp, Operator("wsum", [[2, 3, 5], [x, y, z]]), -10),
        Comparison(cmp, Operator("wsum", [[2, 3, 5], [x, y, z]]), 100),
        Comparison(
            cmp,
            Operator(
                "wsum",
                [
                    [2, 3, 4],
                    [
                        intvar(-2, 2, name="x"),
                        intvar(2, 4, name="y"),
                        ~boolvar(name="b"),
                    ],
                ],
            ),
            15,
        ),  # non-zero lbs
        Comparison(cmp, Operator("wsum", [[2, 3], [x, p]]), 5),
    )
    for antecedent in (True, p, ~p)
]


ENCODINGS = [
    "direct",
    "order",
    "binary",
]


@pytest.fixture()
def setup():
    _IntVarImpl.counter = 0
    _BoolVarImpl.counter = 0
    yield


class TestTransInt2Bool:

    import importlib
    import itertools

    @pytest.mark.parametrize(
        ("constraint", "encoding"), itertools.product(CONSTRAINTS, ENCODINGS), ids=str
    )
    @pytest.mark.skipif(
        not (CPM_pysat.supported() and importlib.util.find_spec("pypblib")),
        reason="PySAT+pblib not supported",
    )
    def test_transforms(self, constraint, encoding, setup):
        user_vars = set(get_variables(constraint))
        ivarmap = dict()
        flat = int2bool(
            flatten_constraint(constraint), ivarmap=ivarmap, encoding=encoding
        )

        cons_sols = []
        flat_sols = []

        Model(constraint).solveAll(
            solver="ortools",
            display=lambda: cons_sols.append(tuple(argvals(user_vars))),
        )
        cons_sols = sorted(cons_sols)
        pysat = CPM_pysat(encoding=encoding)
        for c in flat:
            pysat.add(c)

        pysat.user_vars |= user_vars  #

        # pysat.user_vars = set(get_variables(flat))
        pysat.ivarmap = ivarmap
        pysat.solveAll(display=lambda: flat_sols.append(tuple(argvals(user_vars))))
        flat_sols = sorted(flat_sols)

        def show_int_var(x):
            bnd = x.get_bounds()
            enc = pysat.ivarmap.get(x.name, None)
            return f"{x} in {bnd[0]}..{bnd[1]} = {enc if enc is None else enc._xs}"

        assert (
            cons_sols == flat_sols
        ), f"""Incorrect transformation:
         U_VARS: {", ".join(show_int_var(x) for x in user_vars)}
          INPUT: {constraint} (#sols={len(cons_sols)})
         OUTPUT: {flat} (#sols={len(flat_sols)})
         SOL_IN: {cons_sols}
         SOL_OU: {flat_sols}
        """
