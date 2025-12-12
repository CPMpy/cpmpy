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
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))),
                            {"IV0 <= 1", "((IV0) >= (x)) or ((IV0) >= (y))", "(IV0) <= (x)", "(IV0) <= (y)"})
        # reified
        cons = [bv.implies(min(ivs) <= 1)]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))),
                            {"(bv) -> (IV1 <= 1)", "((IV1) >= (x)) or ((IV1) >= (y))", "(IV1) <= (x)", "(IV1) <= (y)"})
        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})),str(cons))

        cons = [(min(ivs) <= 1).implies(bv)]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))),
                            {"(IV2 <= 1) -> (bv)", "((IV2) >= (x)) or ((IV2) >= (y))", "(IV2) <= (x)", "(IV2) <= (y)"})
        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})), str(cons))

        cons = [(min(ivs) <= 1) == (bv)]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))),
                            {"(IV3 <= 1) == (bv)",  "((IV3) >= (x)) or ((IV3) >= (y))", "(IV3) <= (x)", "(IV3) <= (y)"})
        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})), str(cons))


    def test_decompose_nested(self):

        ivs = [intvar(1,9,name=n) for n in "xyz"]

        cons = [AllDifferent(ivs) == 0]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))), {"not([and([(x) != (y), (x) != (z), (y) != (z)])])"})

        cons = [0 == AllDifferent(ivs)]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))), {"not([and([(x) != (y), (x) != (z), (y) != (z)])])"})

        cons = [AllDifferent(ivs) == AllEqual(ivs[:-1])]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))), {"(and([(x) != (y), (x) != (z), (y) != (z)])) == ((x) == (y))"})

        cons = [min(ivs) == max(ivs)]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons, supported={"min"}))),
                            {"(min(x,y,z)) == (IV0)", "or([(IV0) <= (x), (IV0) <= (y), (IV0) <= (z)])", "(IV0) >= (x)", "(IV0) >= (y)", "(IV0) >= (z)"})

        self.assertEqual(set(map(str,decompose_in_tree(cons, supported={"max"}))),
                         {"(IV1) == (max(x,y,z))", "or([(IV1) >= (x), (IV1) >= (y), (IV1) >= (z)])", "(IV1) <= (x)", "(IV1) <= (y)", "(IV1) <= (z)"})

        # numerical in non-comparison context
        cons = [AllEqual([min(ivs[:-1]),ivs[-1]])]
        self.assertEqual(set(map(str,decompose_in_tree(cons, supported={"allequal"}))),
                         {"allequal(IV2,z)", "((IV2) >= (x)) or ((IV2) >= (y))", "(IV2) <= (x)", "(IV2) <= (y)"})

        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})),
                         "[(min(x,y)) == (z)]")

