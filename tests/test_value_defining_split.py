"""
Tests that transformations put auxiliary/defining constraints on the defining
stream (and the rewritten input on the value stream), not the other way around.
"""
import cpmpy as cp
from cpmpy.expressions.globalconstraints import GlobalConstraint
from cpmpy.expressions.utils import flatlist
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl
from cpmpy.transformations.cse import CSEMap
from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.flatten_model import flatten_constraint, apply_transform
from cpmpy.transformations.int2bool import int2bool
from cpmpy.transformations.linearize import linearize_constraint
from cpmpy.transformations.reification import only_bv_reifies, only_implies, reify_rewrite


class TestValueDefiningSplit:
    def setup_method(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0

    def test_flatten_cse_defining(self):
        x = cp.intvar(-10, 10, name="x")
        y = cp.intvar(-10, 10, name="y")
        csemap = CSEMap()

        v0, d0 = flatten_constraint(cp.abs(x) + y <= 15, csemap=csemap)
        assert len(v0) == 1 and len(d0) == 1
        assert "abs" in str(d0[0])
        assert "abs" not in str(v0[0])

        v1, d1 = flatten_constraint(cp.abs(x) + y >= 11, csemap=csemap)
        assert len(v1) == 1 and d1 == []  # CSE reuses defining

    def test_only_bv_reifies(self):
        x = cp.intvar(-10, 10, name="x")
        b = cp.boolvar(name="b")
        # BE -> BV rewrite flattens nested abs into a defining equality
        v, d = only_bv_reifies([((cp.abs(x) + x) >= 3).implies(b)], csemap=CSEMap())
        assert str(v) == "[(~b) -> ((IV0) + (x) < 3)]"
        assert str(d) == "[(abs(x)) == (IV0)]"

    def test_only_implies(self):
        x = cp.intvar(-10, 10, name="x")
        b = cp.boolvar(name="b")
        v, d = only_implies([(b) == ((cp.abs(x) + x) >= 3)], csemap=CSEMap())
        assert str(v) == "[(b) -> ((IV0) + (x) >= 3), (~b) -> ((IV0) + (x) < 3)]"
        assert str(d) == "[(abs(x)) == (IV0)]"

    def test_reify_rewrite(self):
        ivs = cp.intvar(1, 9, shape=3, name="ivs")
        rv = cp.boolvar(name="rv")
        flat_v, flat_d = flatten_constraint(rv == (cp.max(ivs) > 5), csemap=CSEMap())
        v, d = apply_transform(reify_rewrite, flat_v, flat_d)
        # value is the reified comparison over an aux; defining defines that aux
        assert len(v) == 1 and "rv" in str(v[0])
        assert any("max" in str(c) for c in d)
        assert all("max" not in str(c) for c in v)

    def test_only_numexpr_equality(self):
        x, y, z = cp.intvar(0, 10, shape=3, name=tuple("xyz"))
        v, d = only_numexpr_equality([cp.max(x, y, z) < 5], supported=frozenset(), csemap=CSEMap())
        assert len(v) == 1 and len(d) == 1
        assert "max" in str(d[0]) and "==" in str(d[0])
        assert "max" not in str(v[0])

    def test_decompose_in_tree(self):
        class MyGlobal(GlobalConstraint):
            def __init__(self, arr):
                super().__init__("mycustomglobal", tuple(flatlist(arr)))

            def decompose(self):
                return [cp.sum(self.args) == 1], [self.args[0] == 1]

            def decompose_positive(self):
                return [cp.sum(self.args) >= 1], [self.args[0] == 1]

        a, b, c = cp.intvar(0, 10, shape=3, name=("a", "b", "c"))
        bv = cp.boolvar(name="bv")
        cons = MyGlobal([a, b, c])

        v, d = decompose_in_tree([cons])
        assert str(v) == "[sum(a, b, c) >= 1]"
        assert str(d) == "[a == 1]"

        v, d = decompose_in_tree([bv.implies(cons)])
        assert str(v) == "[(bv) -> (sum(a, b, c) >= 1)]"
        assert str(d) == "[a == 1]"

        # nested global function: max defining stays defining
        x, y, z = cp.intvar(0, 10, shape=3, name=tuple("xyz"))
        q = cp.intvar(0, 2, name="q")
        v, d = decompose_in_tree([bv == ((cp.max([x, y, z]) + q) <= 10)], csemap=CSEMap())
        assert str(v) == "[(bv) == ((IV0) + (q) <= 10)]"
        assert len(d) == 4
        assert all("IV0" in str(c) for c in d)

    def test_linearize(self):
        x, y, z = cp.intvar(0, 10, shape=3, name=tuple("xyz"))
        v, d = linearize_constraint([cp.max(x, y) < z], supported={"max"}, csemap=CSEMap())
        assert str(v) == "[(max(x,y)) <= (IV0)]"
        assert str(d) == "[sum([1, -1] * [z, IV0]) == 1]"

    def test_int2bool_domain_defining(self):
        x = cp.intvar(0, 2, name="x")
        v, d = int2bool([cp.sum([x]) >= 1], ivarmap={}, encoding="direct", csemap=CSEMap())
        assert len(v) == 1
        assert len(d) == 1
        # value is the rewritten PB constraint; defining is the domain/exactly-one
        assert "sum" in str(d[0]) and "== 1" in str(d[0])
        assert "sum" not in str(v[0]) or str(v[0]).startswith("~")

    def test_apply_transform_preserves_existing_defining(self):
        x = cp.intvar(-10, 10, name="x")
        y = cp.intvar(-10, 10, name="y")
        csemap = CSEMap()
        v, d = flatten_constraint(cp.abs(x) + y <= 15, csemap=csemap)
        assert len(d) == 1
        # a no-op-ish transform on both streams must keep defining as defining
        v2, d2 = apply_transform(only_bv_reifies, v, d, csemap=csemap)
        assert any("abs" in str(c) for c in d2)
        assert all("abs" not in str(c) for c in v2)
