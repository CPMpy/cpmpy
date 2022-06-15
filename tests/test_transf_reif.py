import unittest
import numpy as np
from cpmpy import *
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.reification import only_bv_implies, reify_rewrite
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl # to reset counters

class TestTransfReif(unittest.TestCase):
    def setUp(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0

    def test_only_bv_implies(self):
        a,b,c = [boolvar(name=n) for n in "abc"]
        
        cases = [((a).implies(b), "[(a) -> (b)]"),
                 ((~a).implies(b), "[(~a) -> (b)]"),
                 ((a).implies(b|c), "[(a) -> ((b) or (c))]"),
                 ((a).implies(b&c), "[(a) -> ((b) and (c))]"),
                 ((b|c).implies(a), "[(~a) -> ((~b) and (~c))]"),
                 ((b&c).implies(a), "[(~a) -> ((~b) or (~c))]"),
                 ((a)==(b), "[(a) -> (b), (b) -> (a)]"),
                 ((~a)==(b), "[(~a) -> (b), (b) -> (~a)]"),
                 ((b|c)==(a), "[(~a) -> ((~b) and (~c)), (a) -> ((b) or (c))]"),
                 ((b&c)==(a), "[(~a) -> ((~b) or (~c)), (a) -> ((b) and (c))]"),
                ]

        # test transformation
        for (expr, strexpr) in cases:
            self.assertEqual( str(only_bv_implies(expr)), strexpr )
            self.assertTrue(Model(expr).solve())

    def test_reif_element(self):
        bvs = boolvar(shape=5, name="bvs")
        iv = intvar(1,10, name="iv")
        rv = boolvar(name="rv")

        # have to be careful with Element, if an Element over
        # Bools is treated as a BoolExpr then it would be treated as a
        # reification... which is unwanted.
        # so for now, it remains an IntExpr, but if that changes, the following
        # will break
        e1 = (bvs[iv] == rv)
        e2 = (cpm_array([1,0,1,1])[iv] == rv)
        for e in [e1,e2]:
            self.assertTrue(Model(e).solve())


        # Another case to be careful with:
        # in reified context, the index variable can have a larger domain
        # than the array range, needs a reified equality decomposition.
        arr = cpm_array([0,1,2])

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
            idx = intvar(lb,ub, name="idx")
            e = (rv == (arr[idx] != 1))
            self.assertEqual(Model(e).solveAll(), cnt)


    def test_reif_rewrite(self):
        bvs = boolvar(shape=4, name="bvs")
        ivs = intvar(1,9, shape=3, name="ivs")
        rv = boolvar(name="rv")
        arr = cpm_array([0,1,2])
        
        # various reify_rewrite cases:
        cases = [(rv == bvs[0], "[(rv) == (bvs[0])]"),
                 (rv == all(bvs), "[(and([bvs[0], bvs[1], bvs[2], bvs[3]])) == (rv)]"),
                 (rv.implies(any(bvs)), "[(rv) -> (or([bvs[0], bvs[1], bvs[2], bvs[3]]))]"),
                 ((bvs[0].implies(bvs[1])).implies(rv), "[((~bvs[0]) or (bvs[1])) -> (rv)]"),
                 (rv == AllDifferent(ivs), "[(and([BV0, BV1, BV2])) == (rv), ((ivs[0]) != (ivs[1])) == (BV0), ((ivs[0]) != (ivs[2])) == (BV1), ((ivs[1]) != (ivs[2])) == (BV2)]"),
                 (rv.implies(AllDifferent(ivs)), "[(rv) -> (and([BV6, BV7, BV8])), ((ivs[0]) != (ivs[1])) == (BV6), ((ivs[0]) != (ivs[2])) == (BV7), ((ivs[1]) != (ivs[2])) == (BV8)]"),
                 (rv == (arr[intvar(-1,3)] != 1), "[((BV12) or (BV13)) == (rv), (IV0 == 0) == (BV12), (IV0 == 2) == (BV13)]"),
                 (rv == (arr[intvar(0,2)] != 1), "[([0 1 2][IV1]) == (IV2), (IV2 != 1) == (rv)]"),
                 (rv == (max(ivs) > 5), "[(max(ivs[0],ivs[1],ivs[2])) == (IV4), (IV4 > 5) == (rv)]"),
                 (rv.implies(min(ivs) != 0), "[(min(ivs[0],ivs[1],ivs[2])) == (IV6), (rv) -> (IV6 != 0)]"),
                 ((min(ivs) != 0).implies(rv), "[(min(ivs[0],ivs[1],ivs[2])) == (IV8), (IV8 != 0) -> (rv)]"),
                ]

        # test transformation
        for (expr, strexpr) in cases:
            self.assertEqual( str(reify_rewrite(flatten_constraint(expr))), strexpr )
            self.assertTrue(Model(expr).solve())
