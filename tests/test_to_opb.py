"""
Solution-equivalence tests for the ``to_opb`` transformation.

For each case we enumerate all solutions of the original
constraint and of its ``to_opb`` translation (decoding the integer-variable encodings via
``ivarmap``) and assert the two solution sets are identical, so the transformation is
verified to be semantics-preserving.

``to_opb`` normalizes everything to pseudo-Boolean ``wsum(...) >= const`` comparisons over
Boolean literals; integer variables are boolean encoded and recorded in ``ivarmap``.
"""

import cpmpy as cp

from cpmpy.expressions.core import Comparison
from cpmpy.transformations.to_opb import to_opb
from cpmpy.transformations.cse import CSEMap
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.utils import argvals

import pytest

a, b, c = cp.boolvar(shape=3, name=["a", "b", "c"])
x = cp.intvar(1, 2, name="x")
y, z = cp.intvar(0, 1, shape=2, name=["y", "z"])

cases = [
    a,
    a | b,
    a & b,
    a != b,
    a == b,
    a.implies(b),
    a.implies(b | c),
    a.implies(b & c),
    a.implies(b != c),
    a.implies(b == c),
    a.implies(b.implies(c)),
    (b | c).implies(a),
    (b & c).implies(a),
    (b != c).implies(a),
    (b == c).implies(a),
    (b.implies(c)).implies(a),
    cp.Xor([a, b]),
    cp.sum([2 * x + 3 * y]) <= 4,
    cp.sum([2 * x + 3 * y + 5 * z]) <= 6,
    cp.sum([2 * x + 3 * cp.intvar(0, 1)]) <= 4,
    (a + b + c) == 1,
    (a + b + c) != 1,
    a + b + c > 2,
    a + b + c <= 2,
    # NOTE: cp.sum(intvar(2,3,shape=3)) <= 3 is omitted here: min sum is 6, so it is
    # trivially UNSAT and to_opb raises (covered by TestToOpbBehavior).
    (~a & ~b) | (a & b),  # https://github.com/cpmpy/cpmpy/issues/823
    c | (a & b),          # above minimized
]


class TestToOpb:
    def idfn(val):
        if isinstance(val, tuple):
            # solver name, class tuple
            return val
        else:
            return f"{val}"

    @pytest.mark.parametrize(
        "case",
        cases,
        ids=idfn,
    )
    def test_toopb(self, case):
        # test for equivalent solutions with/without to_opb
        vs = cp.cpm_array(get_variables(case))
        s1 = self.allsols([case], vs)

        csemap, ivarmap = CSEMap(), dict()
        opb = to_opb(case, csemap, ivarmap)

        # to_opb normal form: weighted-sum ">=" comparisons over Boolean literals
        assert all(isinstance(con, Comparison) and con.name == ">=" for con in opb), \
            f"Not in OPB normal form: {opb}"

        s2 = self.allsols(opb, vs, ivarmap=ivarmap)
        assert s1 == s2, f"The equivalence check failed for translation from:\n\n{case}\n\nto:\n\n{opb}"

    def allsols(self, cons, vs, ivarmap=None):
        m = cp.Model(cons)
        sols = set()

        def display():
            if ivarmap:
                for x_enc in ivarmap.values():
                    x_enc._x._value = x_enc.decode()
            sols.add(tuple(argvals(vs)))

        m.solveAll(solver="ortools", display=display, solution_limit=100)
        assert len(sols) < 100, sols
        return sols


class TestToOpbBehavior:
    """Degenerate cases that ``to_opb`` handles specially (correct, not bugs)."""

    def test_trivially_false_raises(self):
        # 3*y with y in {0,1} maxes at 3, so >= 20 is unsatisfiable; to_opb cannot
        # express a trivially-false constraint as an OPB constraint.
        with pytest.raises(NotImplementedError):
            to_opb(cp.sum([3 * y]) >= 20, CSEMap(), dict())

    def test_trivially_true_returns_empty(self):
        # 3*y <= 4 always holds for y in {0,1}, so to_opb drops it entirely.
        assert to_opb(cp.sum([3 * y]) <= 4, CSEMap(), dict()) == []
