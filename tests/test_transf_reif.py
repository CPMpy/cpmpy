import unittest
import numpy as np
from cpmpy import *
from cpmpy.transformations.reification import only_bv_implies
from cpmpy.transformations.get_variables import get_variables

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

