import pytest

import cpmpy as cp
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.core import Comparison, Operator, BoolVal
from cpmpy.expressions.utils import argvals
from cpmpy.model import Model
from cpmpy import SolverLookup

from cpmpy.transformations.int2bool import int2bool
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl, intvar, boolvar

# add some small but non-trivial integer variables (i.e. non-zero lower bounds, domain size not a power of two)
x = intvar(1, 3, name="x")
y = intvar(1, 3, name="y")
z = intvar(1, 3, name="z")

p = boolvar(name="p")
q = boolvar(name="q")

c = intvar(2, 2, name="c")

SOLVERS = [
    "pindakaas",
    "pysat",
]
SOLVERS = [
    (name, solver) for name, solver in SolverLookup.base_solvers() if name in SOLVERS and solver.supported()
]

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

    @pytest.mark.parametrize(
        ("solver", "constraint", "encoding"),
        itertools.product(SOLVERS, CONSTRAINTS, ENCODINGS),
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
        name, solver_class = solver
        solver = solver_class()
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

    def test_int2bool_cse_one_var(self):
        x = cp.intvar(0, 2, name="x")
        slv = cp.solvers.CPM_pindakaas()
        slv.encoding = "direct"
        # assert str(slv.transform((x == 0) )) == "[(EncDir(x)[0]) + (EncDir (x)[1]) == 1, EncDir(x)[0], ~EncDir(x)[1]]"
        assert str(slv.transform((x == 0) | (x == 2))) == "[(⟦x == 0⟧) or (⟦x == 2⟧), sum([⟦x == 0⟧, ⟦x == 1⟧, ⟦x == 2⟧]) == 1]"

    def test_int2bool_cse_two_vars(self):
        slv = cp.solvers.CPM_pindakaas()
        slv.encoding = "direct"
        x = cp.intvar(0, 2, name="x")
        y = cp.intvar(0, 2, name="y")
        assert str(slv.transform((x == 0) | (y == 2))) == "[(⟦x == 0⟧) or (⟦y == 2⟧), sum([⟦x == 0⟧, ⟦x == 1⟧, ⟦x == 2⟧]) == 1, sum([⟦y == 0⟧, ⟦y == 1⟧, ⟦y == 2⟧]) == 1]"
