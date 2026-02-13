import pytest
import itertools

import cpmpy as cp

from cpmpy import SolverLookup
from cpmpy.expressions.core import BoolVal, Comparison, Operator
from cpmpy.expressions.utils import argvals
from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl, boolvar, intvar
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.int2bool import int2bool, IntVarEnc
from utils import skip_on_missing_pblib


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

    def idfn(val):
        if isinstance(val, tuple):
            # solver name, class tuple
            return val[0]
        else:
            return f"{val}"

    @pytest.mark.requires_solver("pindakaas", "pysat")
    @pytest.mark.parametrize(
        ("constraint", "encoding"),
        itertools.product(CONSTRAINTS, ENCODINGS),
        ids=idfn,
    )
    @skip_on_missing_pblib()
    def test_transforms(self, solver, constraint, encoding, setup):
        IntVarEnc.NAMED = True
        user_vars = tuple(get_variables(constraint))
        ivarmap = dict()
        csemap = dict()
        flat = int2bool(flatten_constraint(constraint), ivarmap=ivarmap, encoding=encoding, csemap=csemap)

        cons_sols = []
        flat_sols = []

        # "Trusted" solver (not using int2bool)
        cp.Model(constraint).solveAll(
            solver="ortools",
            display=lambda: cons_sols.append(tuple(argvals(user_vars))),
        )
        cons_sols = sorted(cons_sols)
        solver = SolverLookup().get(solver)
        solver.encoding = encoding
        solver._csemap = csemap
        solver.ivarmap = ivarmap
        for c in flat:
            solver.add(c)

        # ensure all user vars are known to the CNF solver
        for x in user_vars:
            solver.add(x == x)

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

    def test_int2bool_confirm_cse(self):
        """
        This example show how we save some variables for this case, although not yet constraints.
        ```
        [(BV1) or (BV2), sum([BV[x == 0], BV[x == 1], BV[x == 2]]) == 1, (BV1) -> (BV[x == 0]), (~BV1) -> (sum([0, 1, 2, -3] * (BV[x == 0], BV[x == 1], BV[x == 2], BV3)) <= -1), (~BV1) -> (sum([0, 1, 2, -1] * (BV[x == 0], BV[x == 1], BV[x == 2], BV3)) >= 0), sum([1, -1] * (BV1, ~BV3)) <= 0, (BV2) -> (BV[x == 2]), (~BV2) -> (sum([0, 1, 2, -1] * (BV[x == 0], BV[x == 1], BV[x == 2], BV4)) <= 1), (~BV2) -> (sum([0, 1, 2, -3] * (BV[x == 0], BV[x == 1], BV[x == 2], BV4)) >= 0), sum([1, -1] * (BV2, ~BV4)) <= 0]
        [(BV[x == 0]) or (BV[x == 2]), sum([BV[x == 0], BV[x == 1], BV[x == 2]]) == 1, (BV[x == 0]) -> (BV[x == 0]), (~BV[x == 0]) -> (sum([0, 1, 2, -3] * (BV[x == 0], BV[x == 1], BV[x == 2], BV2)) <= -1), (~BV[x == 0]) -> (sum([0, 1, 2, -1] * (BV[x == 0], BV[x == 1], BV[x == 2], BV2)) >= 0), sum([1, -1] * (BV[x == 0], ~BV2)) <= 0, (BV[x == 2]) -> (BV[x == 2]), (~BV[x == 2]) -> (sum([0, 1, 2, -1] * (BV[x == 0], BV[x == 1], BV[x == 2], BV3)) <= 1), (~BV[x == 2]) -> (sum([0, 1, 2, -3] * (BV[x == 0], BV[x == 1], BV[x == 2], BV3)) >= 0), sum([1, -1] * (BV[x == 2], ~BV3)) <= 0]
        ```
        """

        x = cp.intvar(0, 2, name="x")
        slv = cp.solvers.CPM_pindakaas()
        slv.encoding = "direct"

        slv._csemap = None

        con = [(x == 0) | (x == 2)]
        without_cse = slv.transform(con)
        _BoolVarImpl.counter = 0
        _IntVarImpl.counter = 0
        with_cse = cp.solvers.CPM_pindakaas().transform(con)

        print(without_cse)
        print(with_cse)

        assert len(get_variables(with_cse)) < len(get_variables(without_cse))

@pytest.mark.skip(
    "Currently, the expected output is too complex too evaluate, since we are not encoding variables up-front as in the original approach. Once int2bool is better integrated with the other transformations, the output should look like this."
)
class Int2BoolCSE:
    def test_int2bool_cse_one_var(self):
        x = cp.intvar(0, 2, name="x")
        slv = cp.solvers.CPM_pindakaas()
        slv.encoding = "direct"
        assert str(slv.transform((x == 0) | (x == 2))) == "[(⟦x == 0⟧) or (⟦x == 2⟧), sum([⟦x == 0⟧, ⟦x == 1⟧, ⟦x == 2⟧]) == 1]"

    def test_int2bool_cse_one_var_order(self):
        x = cp.intvar(0, 2, name="x")
        slv = cp.solvers.CPM_pindakaas()
        slv.encoding = "order"
        assert str(slv.transform((x >= 1) | (x >= 2))) == "[(⟦x >= 1⟧) or (⟦x >= 2⟧), sum([1, -1] * (⟦x >= 2⟧, ⟦x >= 1⟧)) <= 0]"
        # TODO this could be a CSE improvement?
        # assert str(slv.transform((x >= 1) | (x < 2))) == "[(⟦x == 0⟧) or (⟦x == 2⟧), sum([⟦x == 0⟧, ⟦x == 1⟧, ⟦x == 2⟧]) == 1]"

    def test_int2bool_cse_two_vars(self):
        slv = cp.solvers.CPM_pindakaas()
        x = cp.intvar(0, 2, name="x")
        y = cp.intvar(0, 2, name="y")
        slv.encoding = "direct"
        assert (
            str(slv.transform((x == 0) | (y == 2)))
            == "[(⟦x == 0⟧) or (⟦y == 2⟧), sum([⟦x == 0⟧, ⟦x == 1⟧, ⟦x == 2⟧]) == 1, sum([⟦y == 0⟧, ⟦y == 1⟧, ⟦y == 2⟧]) == 1]"
        )
