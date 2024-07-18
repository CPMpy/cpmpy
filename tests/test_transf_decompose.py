import unittest
import numpy as np
from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl  # to reset counters


class TestTransfDecomp(unittest.TestCase):

    def setUp(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0

    def test_decompose_bool(self):
        ivs = [intvar(1, 9, name=n) for n in "xyz"]
        bv = boolvar(name="bv")

        cons = [AllDifferent(ivs)]
        self.assertEqual(str(decompose_in_tree(cons)), "[(x) != (y), (x) != (z), (y) != (z)]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"alldifferent"})), str(cons))

        # reified
        cons = [bv.implies(AllDifferent(ivs))]
        self.assertEqual(str(decompose_in_tree(cons)),
                         "[(bv) -> (and([(x) != (y), (x) != (z), (y) != (z)]))]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"alldifferent"})),
                         "[(bv) -> (and([(x) != (y), (x) != (z), (y) != (z)]))]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"alldifferent"}, supported_reified={"alldifferent"})),str(cons))

        cons = [AllDifferent(ivs).implies(bv)]
        self.assertEqual(str(decompose_in_tree(cons)),
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) -> (bv)]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"alldifferent"})),
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) -> (bv)]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"alldifferent"}, supported_reified={"alldifferent"})),
                         str(cons))

        cons = [AllDifferent(ivs) == (bv)]
        self.assertEqual(str(decompose_in_tree(cons)),
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) == (bv)]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"alldifferent"})),
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) == (bv)]")
        self.assertEqual(str(decompose_in_tree(cons, supported_reified={"alldifferent"})),
                         str(cons))

        # tricky one
        cons = [AllDifferent(ivs) < (bv)]
        self.assertEqual(str(decompose_in_tree(cons)),
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) < (bv)]")

    def test_decompose_num(self):

        ivs = [intvar(1, 9, name=n) for n in "xy"]
        bv = boolvar(name="bv")

        cons = [min(ivs) <= 1]
        self.assertEqual(str(decompose_in_tree(cons)),
                         "[IV0 <= 1, ((x) <= (IV0)) or ((y) <= (IV0)), (x) >= (IV0), (y) >= (IV0)]")
        # reified
        cons = [bv.implies(min(ivs) <= 1)]
        self.assertEqual(str(decompose_in_tree(cons)),
                         "[(bv) -> (IV1 <= 1), ((x) <= (IV1)) or ((y) <= (IV1)), (x) >= (IV1), (y) >= (IV1)]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})),str(cons))

        cons = [(min(ivs) <= 1).implies(bv)]
        self.assertEqual(str(decompose_in_tree(cons)),
                         "[(IV2 <= 1) -> (bv), ((x) <= (IV2)) or ((y) <= (IV2)), (x) >= (IV2), (y) >= (IV2)]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})), str(cons))

        cons = [(min(ivs) <= 1) == (bv)]
        self.assertEqual(str(decompose_in_tree(cons)),
                         "[(IV3 <= 1) == (bv), ((x) <= (IV3)) or ((y) <= (IV3)), (x) >= (IV3), (y) >= (IV3)]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})), str(cons))


    def test_decompose_nested(self):

        ivs = [intvar(1,9,name=n) for n in "xyz"]

        cons = [AllDifferent(ivs) == 0]
        self.assertEqual(str(decompose_in_tree(cons)), "[not([and([(x) != (y), (x) != (z), (y) != (z)])])]")

        cons = [0 == AllDifferent(ivs)]
        self.assertEqual(str(decompose_in_tree(cons)), "[not([and([(x) != (y), (x) != (z), (y) != (z)])])]")

        cons = [AllDifferent(ivs) == AllEqual(ivs[:-1])]
        self.assertEqual(str(decompose_in_tree(cons)), "[(and([(x) != (y), (x) != (z), (y) != (z)])) == ((x) == (y))]")

        cons = [min(ivs) == max(ivs)]
        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})),
                         '[(IV0) == (min(x,y,z)), or([(x) >= (IV0), (y) >= (IV0), (z) >= (IV0)]), (x) <= (IV0), (y) <= (IV0), (z) <= (IV0)]')

        self.assertEqual(str(decompose_in_tree(cons, supported={"max"})),
                         "[(IV1) == (max(x,y,z)), or([(x) <= (IV1), (y) <= (IV1), (z) <= (IV1)]), (x) >= (IV1), (y) >= (IV1), (z) >= (IV1)]")

        # numerical in non-comparison context
        cons = [AllEqual([min(ivs[:-1]),ivs[-1]])]
        self.assertEqual(str(decompose_in_tree(cons, supported={"allequal"})),
                         "[allequal(IV2,z), (IV3) == (IV2), ((x) <= (IV3)) or ((y) <= (IV3)), (x) >= (IV3), (y) >= (IV3)]")

        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})),
                         "[(min(x,y)) == (z)]")

