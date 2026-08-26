"""
Tests that transformations put auxiliary/defining constraints on the defining
stream (and the rewritten input on the value stream), not the other way around.
"""
import cpmpy as cp
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
        assert str(v0) == "[(IV0) + (y) <= 15]"
        assert str(d0) == "[(abs(x)) == (IV0)]"

        v1, d1 = flatten_constraint(cp.abs(x) + y >= 11, csemap=csemap)
        assert str(v1) == "[(IV0) + (y) >= 11]"
        assert str(d1) == "[]"  # CSE reuses defining

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
        assert str(v) == "[(IV0 > 5) == (rv)]"
        assert str(d) == "[(max(ivs[0],ivs[1],ivs[2])) == (IV0)]"

    def test_only_numexpr_equality(self):
        x, y, z = cp.intvar(0, 10, shape=3, name=tuple("xyz"))
        v, d = only_numexpr_equality([cp.max(x, y, z) < 5], supported=frozenset(), csemap=CSEMap())
        assert str(v) == "[IV0 < 5]"
        assert str(d) == "[(max(x,y,z)) == (IV0)]"

    def test_decompose_in_tree(self):
        # Globals/functions already return (value, defining) from decompose();
        # decompose_in_tree must keep that second list on the defining stream.
        x = cp.intvar(0, 5, shape=3, name="x")
        b = cp.boolvar(name="b")

        # Table.decompose_positive: value = any(row selectors), defining = row implications
        v, d = decompose_in_tree([cp.Table(x, [[0, 1, 2], [1, 2, 3]])])
        assert str(v) == "[(BV0) or (BV1)]"
        assert str(d) == ("[(BV0) -> (and(x[0] == 0, x[1] == 1, x[2] == 2)), "
                          "(BV1) -> (and(x[0] == 1, x[1] == 2, x[2] == 3))]")

        # nested Maximum: value keeps the comparison; defining totally defines the aux
        v, d = decompose_in_tree([b == (cp.max(x) <= 3)], csemap=CSEMap())
        assert str(v) == "[(b) == (IV0 <= 3)]"
        assert str(d) == ("[(IV0) >= (x[0]), (IV0) >= (x[1]), (IV0) >= (x[2]), "
                          "or((IV0) <= (x[0]), (IV0) <= (x[1]), (IV0) <= (x[2]))]")

        # Element: value is the comparison on the aux; defining channels index -> arr
        arr = cp.intvar(0, 5, shape=4, name="a")
        i = cp.intvar(0, 3, name="i")
        v, d = decompose_in_tree([arr[i] == 2], csemap=CSEMap())
        assert str(v) == "[IV1 == 2]"
        assert str(d) == ("[(i == 0) -> ((IV1) == (a[0])), (i == 1) -> ((IV1) == (a[1])), "
                          "(i == 2) -> ((IV1) == (a[2])), (i == 3) -> ((IV1) == (a[3]))]")

    def test_linearize(self):
        x, y, z = cp.intvar(0, 10, shape=3, name=tuple("xyz"))
        v, d = linearize_constraint([cp.max(x, y) < z], supported={"max"}, csemap=CSEMap())
        assert str(v) == "[(max(x,y)) <= (IV0)]"
        assert str(d) == "[sum([1, -1] * [z, IV0]) == 1]"

    def test_int2bool_domain_defining(self):
        x = cp.intvar(0, 2, name="x")
        v, d = int2bool([cp.sum([x]) >= 1], ivarmap={}, encoding="direct", csemap=CSEMap())
        assert str(v) == "[~BV[x == 0]]"
        assert str(d) == "[sum(BV[x == 0], BV[x == 1], BV[x == 2]) == 1]"

    def test_apply_transform_preserves_existing_defining(self):
        x = cp.intvar(-10, 10, name="x")
        y = cp.intvar(-10, 10, name="y")
        csemap = CSEMap()
        v, d = flatten_constraint(cp.abs(x) + y <= 15, csemap=csemap)
        v2, d2 = apply_transform(only_bv_reifies, v, d, csemap=csemap)
        assert str(v2) == "[(IV0) + (y) <= 15]"
        assert str(d2) == "[(abs(x)) == (IV0)]"
