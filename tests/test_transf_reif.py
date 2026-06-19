import pytest

import numpy as np
import cpmpy as cp
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl # to reset counters

class TestTransfReif:
    def setup_method(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0

    def test_only_implies(self):
        a,b,c = [cp.boolvar(name=n) for n in "abc"]

        cases = [((a).implies(b), "[(a) -> (b)]"),
                 ((~a).implies(b), "[(~a) -> (b)]"),
                 ((a).implies(b|c), "[(a) -> ((b) or (c))]"),
                 ((a).implies(b&c), "[(a) -> ((b) and (c))]"),
                 ((b|c).implies(a), "[(~a) -> (~b), (~a) -> (~c)]"),
                 ((b&c).implies(a), "[(~a) -> ((~b) or (~c))]"),
                 ((a)==(b), "[(a) -> (b), (b) -> (a)]"),
                 ((~a)==(b), "[(~a) -> (b), (b) -> (~a)]"),
                 ((b|c)==(a), "[(a) -> ((b) or (c)), (~a) -> (~b), (~a) -> (~c)]"),
                 ((b&c)==(a), "[(a) -> (b), (a) -> (c), (~a) -> ((~b) or (~c))]"),
                ]

        # test transformation
        for (expr, strexpr) in cases:
            assert  str(only_implies(only_bv_reifies((expr,)))) == strexpr
            assert cp.Model(expr).solve()

    def test_reif_element(self):
        bvs = cp.boolvar(shape=5, name="bvs")
        iv = cp.intvar(1,10, name="iv")
        rv = cp.boolvar(name="rv")

        # have to be careful with Element, if an Element over
        # Bools is treated as a BoolExpr then it would be treated as a
        # reification... which is unwanted.
        # so for now, it remains an IntExpr, but if that changes, the following
        # will break
        e1 = (bvs[iv] == rv)
        e2 = (cp.cpm_array([1,0,1,1])[iv] == rv)
        for e in [e1,e2]:
            assert cp.Model(e).solve()


        # Another case to be careful with:
        # in reified context, the index variable can have a larger domain
        # than the array range, needs a reified equality decomposition.
        arr = cp.cpm_array([0,1,2])

        cases = [(-1,3,5), # idx.lb, idx.ub, cnt
                 (-1,2,4),
                 (-1,1,3),
                 (-1,0,2),
                 (0,3,4),
                 (0,2,3),
                 (0,1,2),
                 (1,2,2),
                 (1,3,3),
                 (2,3,2),
                ]

        for (lb,ub,cnt) in cases:
            idx = cp.intvar(lb,ub, name="idx")
            e = (rv == (arr[idx] != 1))
            assert cp.Model(e).solveAll() == cnt

        # Another case, with a more specific check... if the element-wise decomp is empty
        e = bvs[0].implies(cp.Element([1,2,3], iv) < 1)
        assert not cp.Model(e, bvs[0]==True).solve()


    def test_reif_rewrite(self):
        bvs = cp.boolvar(shape=4, name="bvs")
        ivs = cp.intvar(1,9, shape=3, name="ivs")
        rv = cp.boolvar(name="rv")
        arr = cp.cpm_array([0,1,2])

        f = lambda expr : str(reify_rewrite(flatten_constraint(expr)))
        fd = lambda expr : str(reify_rewrite(flatten_constraint(decompose_in_tree(expr))))


        # various reify_rewrite cases:
        assert f(rv == bvs[0]) == "[(rv) == (bvs[0])]"
        assert f(rv == cp.all(bvs)) == "[(and(bvs[0], bvs[1], bvs[2], bvs[3])) == (rv)]"
        assert f(rv.implies(cp.any(bvs))) == "[(rv) -> (or(bvs[0], bvs[1], bvs[2], bvs[3]))]"
        assert f((bvs[0].implies(bvs[1])).implies(rv)) == "[(~rv) -> (bvs[0]), (~rv) -> (~bvs[1])]"
        pytest.raises(ValueError, lambda : f(rv == cp.AllDifferent(ivs)))
        assert fd([rv.implies(cp.AllDifferent(ivs))]) == "[(rv) -> ((ivs[0]) != (ivs[1])), (rv) -> ((ivs[0]) != (ivs[2])), (rv) -> ((ivs[1]) != (ivs[2]))]"
        assert f(rv == (arr[cp.intvar(0, 2)] != 1)) == "[([0 1 2][IV0]) == (IV1), (IV1 != 1) == (rv)]"
        assert f(rv == (cp.max(ivs) > 5)) == "[(max(ivs[0],ivs[1],ivs[2])) == (IV2), (IV2 > 5) == (rv)]"
        assert f(rv.implies(cp.min(ivs) != 0)) == "[(min(ivs[0],ivs[1],ivs[2])) == (IV3), (rv) -> (IV3 != 0)]"
        assert f((cp.min(ivs) != 0).implies(rv)) == "[(min(ivs[0],ivs[1],ivs[2])) == (IV4), (IV4 != 0) -> (rv)]"
