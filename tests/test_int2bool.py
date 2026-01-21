import pytest

from cpmpy import SolverLookup
from cpmpy.expressions.core import BoolVal, Comparison, Operator
from cpmpy.expressions.utils import argvals
from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl, boolvar, intvar
from cpmpy.model import Model
from utils import skip_on_missing_pblib
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.int2bool import int2bool

# add some small but non-trivial integer variables (i.e. non-zero lower bounds, domain size not a power of two)
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
        "!=",
        # ">", "<", # not produced by linearize
    )
    for con in (
        Comparison(cmp, c, 1),
        Comparison(cmp, c, 2),
        Comparison(cmp, x, 2),
        Comparison(cmp, x, 5),
        Comparison(cmp, Operator("sum", [[x, y, z]]), 4),
        Comparison(cmp, Operator("wsum", [[2, 3, 5], [x, y, z]]), 12),
        Comparison(cmp, Operator("wsum", [[2, 3, 5], [x, y, z]]), -10),
        Comparison(cmp, Operator("wsum", [[2, 3, 5], [x, y, z]]), 100),  # where ub(lhs)<rhs
        Comparison(cmp, Operator("wsum", [[2, 3], [x, p]]), 5),  # mix int and bool terms
        Comparison(cmp, Operator("wsum", [[3], [q]]), 2),
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
        ),  # another extra weird constraint
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

    def idfn(val):
        if isinstance(val, tuple):
            # solver name, class tuple
            return val[0]
        else:
            return f"{val}"

    @pytest.mark.requires_solver("pindakaas", "pysat")
    @skip_on_missing_pblib(skip_on_exception_only=True)
    @pytest.mark.parametrize(
        ("constraint", "encoding"),
        itertools.product(CONSTRAINTS, ENCODINGS),
        ids=idfn,
    )
    def test_transforms(self, solver, constraint, encoding, setup):
        user_vars = set(get_variables(constraint))
        ivarmap = dict()
        flat = int2bool(flatten_constraint(constraint), ivarmap=ivarmap, encoding=encoding)

        cons_sols = []
        flat_sols = []

        # "Trusted" solver (not using int2bool)
        Model(constraint).solveAll(
            solver="ortools",
            display=lambda: cons_sols.append(tuple(argvals(user_vars))),
        )
        cons_sols = sorted(cons_sols)
        solver = SolverLookup().get(solver)
        solver.encoding = encoding
        for c in flat:
            solver.add(c)

        # unfortunately, some tricky edge cases where trivial constraints remove their variables by using the above `add` method
        # this only happens in this test set-up
        # to fix this, we add user variables which may have been removed!
        solver.user_vars |= user_vars

        solver.ivarmap = ivarmap
        solver.solveAll(display=lambda: flat_sols.append(tuple(argvals(user_vars))))
        flat_sols = sorted(flat_sols)

        def show_int_var(x):
            bnd = x.get_bounds()
            enc = solver.ivarmap.get(x.name, None)
            return f"{x} in {bnd[0]}..{bnd[1]} = {enc if enc is None else enc._xs}"

        assert cons_sols == flat_sols, f"""Incorrect transformation:
         U_VARS: {", ".join(show_int_var(x) for x in user_vars)}
          INPUT: {constraint} (#sols={len(cons_sols)})
         OUTPUT: {flat} (#sols={len(flat_sols)})
         SOL_IN: {cons_sols}
         SOL_OU: {flat_sols}
        """
