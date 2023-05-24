import unittest
import numpy as np
from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.decompose_global import decompose_global
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl  # to reset counters


class TestTransfDecomp(unittest.TestCase):

    def setUp(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0

    def test_decompose_bool(self):
        ivs = [intvar(1, 9, name=n) for n in "xyz"]
        bv = boolvar(name="bv")

        cons = AllDifferent(ivs)
        self.assertEqual(str(decompose_global(cons)), "[(x) != (y), (x) != (z), (y) != (z)]")
        self.assertEqual(str(decompose_global(cons, supported={"alldifferent"})), str([cons]))

        # reified
        cons = bv.implies(AllDifferent(ivs))
        self.assertEqual(str(decompose_global(cons)),
                         "[(bv) -> ((x) != (y)), (bv) -> ((x) != (z)), (bv) -> ((y) != (z))]")
        self.assertEqual(str(decompose_global(cons, supported={"alldifferent"})),
                         "[(bv) -> ((x) != (y)), (bv) -> ((x) != (z)), (bv) -> ((y) != (z))]")
        self.assertEqual(str(decompose_global(cons, supported={"alldifferent"}, supported_reif={"alldifferent"})),str([cons]))

        cons = AllDifferent(ivs).implies(bv)
        self.assertEqual(str(decompose_global(cons)),
                         "[(and([BV0, BV1, BV2])) -> (bv), ((x) != (y)) == (BV0), ((x) != (z)) == (BV1), ((y) != (z)) == (BV2)]")
        self.assertEqual(str(decompose_global(cons, supported={"alldifferent"})),
                         "[(and([BV3, BV4, BV5])) -> (bv), ((x) != (y)) == (BV3), ((x) != (z)) == (BV4), ((y) != (z)) == (BV5)]")
        self.assertEqual(str(decompose_global(cons, supported={"alldifferent"}, supported_reif={"alldifferent"})),
                         str([cons]))

        cons = AllDifferent(ivs) == (bv)
        self.assertEqual(str(decompose_global(cons)),
                         "[(and([BV6, BV7, BV8])) == (bv), ((x) != (y)) == (BV6), ((x) != (z)) == (BV7), ((y) != (z)) == (BV8)]")
        self.assertEqual(str(decompose_global(cons, supported={"alldifferent"})),
                         "[(and([BV9, BV10, BV11])) == (bv), ((x) != (y)) == (BV9), ((x) != (z)) == (BV10), ((y) != (z)) == (BV11)]")
        self.assertEqual(str(decompose_global(cons, supported={"alldifferent"}, supported_reif={"alldifferent"})),
                         str([cons]))

        # tricky one
        cons = AllDifferent(ivs) < (bv)
        self.assertEqual(str(decompose_global(cons)),
                         "[(BV15) < (bv), (and([BV12, BV13, BV14])) == (BV15), ((x) != (y)) == (BV12), ((x) != (z)) == (BV13), ((y) != (z)) == (BV14)]")

    def test_decompose_num(self):

        ivs = [intvar(1, 9, name=n) for n in "xy"]
        bv = boolvar(name="bv")

        cons = min(ivs) <= 1
        self.assertEqual(str(decompose_global(cons)),
                         "[(BV0) or (BV1), ((x) <= (IV0)) == (BV0), ((y) <= (IV0)) == (BV1), (x) >= (IV0), (y) >= (IV0), IV0 <= 1]")
        # reified
        cons = bv.implies(min(ivs) <= 1)
        self.assertEqual(str(decompose_global(cons)),
                         "[(bv) -> ((BV2) or (BV3)), ((x) <= (IV1)) == (BV2), ((y) <= (IV1)) == (BV3), (bv) -> ((x) >= (IV1)), (bv) -> ((y) >= (IV1)), (bv) -> (IV1 <= 1)]")
        self.assertEqual(str(decompose_global(cons, supported={"min"})),str([cons]))

        cons = (min(ivs) <= 1).implies(bv)
        self.assertEqual(str(decompose_global(cons)),
                         "[(and([BV6, BV7, BV8, BV9])) -> (bv), ((BV4) or (BV5)) == (BV6), ((x) <= (IV2)) == (BV4), ((y) <= (IV2)) == (BV5), ((x) >= (IV2)) == (BV7), ((y) >= (IV2)) == (BV8), (IV2 <= 1) == (BV9)]")
        self.assertEqual(str(decompose_global(cons, supported={"min"})), str([cons]))

        cons = (min(ivs) <= 1) == (bv)
        self.assertEqual(str(decompose_global(cons)),
                         "[(and([BV12, BV13, BV14, BV15])) == (bv), ((BV10) or (BV11)) == (BV12), ((x) <= (IV3)) == (BV10), ((y) <= (IV3)) == (BV11), ((x) >= (IV3)) == (BV13), ((y) >= (IV3)) == (BV14), (IV3 <= 1) == (BV15)]")
        self.assertEqual(str(decompose_global(cons, supported={"min"})), str([cons]))

    def test_decompose_partial_func(self):

        arr = intvar(1, 9, shape=2)
        idx = intvar(-1,5,name="idx")
        bv = boolvar(name ="bv")

        cons = arr[idx] <= 10
        self.assertEqual(str(decompose_global(cons)), "[(idx == 0) -> (BV0), (IV0 <= 10) == (BV0), (idx == 1) -> (BV1), (IV1 <= 10) == (BV1), idx >= 0, idx < 2]")
        self.assertEqual(str(decompose_global(cons, supported={"element"})), str([cons]))

        ivc, bvc = _IntVarImpl.counter, _BoolVarImpl.counter
        cons =(arr[idx] <= 10) == bv
        decomp = cons.args[0].args[0].decompose_comparison("<=",10)
        reified_decomp = flatten_constraint(all(decomp) == bv)

        _IntVarImpl.counter, _BoolVarImpl.counter = ivc, bvc
        self.assertEqual(str(decompose_global(cons)),str(reified_decomp))
        _IntVarImpl.counter, _BoolVarImpl.counter = ivc, bvc
        self.assertEqual(str(decompose_global(cons, supported={"element"})), str(reified_decomp))
        self.assertEqual(str(decompose_global(cons, supported={"element"}, supported_reif={"element"})), str([cons]))




