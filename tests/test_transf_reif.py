import unittest
import numpy as np
from cpmpy import *
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.reification import only_bv_implies, reify_rewrite

class TestTransfReif(unittest.TestCase):
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

    def test_reif_rewrite(self):
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

        idx = intvar(-1,3, name="idx")
        arr = cpm_array([0,1,2])

        e = (rv == (arr[idx] != 1))
        self.assertEqual(Model(e).solveAll(), 5)

        
        cases = [(rv == (arr[idx] != 1), "[((BV2) or (BV3)) == (rv), (idx == 0) == (BV2), (idx == 2) == (BV3)]"),
                ]

        # test transformation
        for (expr, strexpr) in cases:
            self.assertEqual( str(reify_rewrite(flatten_constraint(expr))), strexpr )
            self.assertTrue(Model(expr).solve())
