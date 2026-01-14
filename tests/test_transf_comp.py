import unittest
import numpy as np

import cpmpy as cp
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl # to reset counters

class TestTransfComp(unittest.TestCase):
    def setUp(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0

    def test_only_numexpr_eq(self):
        ivs = cp.intvar(1,9, shape=3, name="ivs")
        
        cases = [(cp.min(ivs) == 3, ["min(ivs[0],ivs[1],ivs[2]) == 3"]),
                 (cp.min(ivs) > 3, ["(min(ivs[0],ivs[1],ivs[2])) == (IV0)", "IV0 > 3"]),
                 (cp.min(ivs) <= 3, ["(min(ivs[0],ivs[1],ivs[2])) == (IV2)", "IV2 <= 3"]),
                 (3 != cp.max(ivs), ["(max(ivs[0],ivs[1],ivs[2])) == (IV4)", "IV4 != 3"]),
                 (3 > cp.max(ivs), ["(max(ivs[0],ivs[1],ivs[2])) == (IV6)", "IV6 < 3"]),
                 (3 <= cp.max(ivs), ["(max(ivs[0],ivs[1],ivs[2])) == (IV8)", "IV8 >= 3"]),
                ]

        # test transformation
        transform = lambda expr: only_numexpr_equality(flatten_constraint(expr))

        for (expr, strexpr) in cases:
            self.assertSetEqual( set([str(c) for c in transform(expr)]), set(strexpr) )
            self.assertTrue(cp.Model(expr).solve())


