import unittest
import cpmpy as cp
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl  # to reset counters


class TestTransfDecomp(unittest.TestCase):

    def setUp(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0

    def test_decompose_bool(self):
        ivs = [cp.intvar(1, 9, name=n) for n in "xyz"]
        bv = cp.boolvar(name="bv")

        cons = [cp.AllDifferent(ivs)]
        self.assertEqual(str(decompose_in_tree(cons)), "[and([(x) != (y), (x) != (z), (y) != (z)])]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"alldifferent"})), str(cons))

        # reified
        cons = [bv.implies(cp.AllDifferent(ivs))]
        self.assertEqual(str(decompose_in_tree(cons)),
                         "[(bv) -> (and([(x) != (y), (x) != (z), (y) != (z)]))]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"alldifferent"})),
                         "[(bv) -> (and([(x) != (y), (x) != (z), (y) != (z)]))]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"alldifferent"}, supported_reified={"alldifferent"})),str(cons))

        cons = [cp.AllDifferent(ivs).implies(bv)]
        self.assertEqual(str(decompose_in_tree(cons)),
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) -> (bv)]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"alldifferent"})),
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) -> (bv)]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"alldifferent"}, supported_reified={"alldifferent"})),
                         str(cons))

        cons = [cp.AllDifferent(ivs) == (bv)]
        self.assertEqual(str(decompose_in_tree(cons)),
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) == (bv)]")
        self.assertEqual(str(decompose_in_tree(cons, supported={"alldifferent"})),
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) == (bv)]")
        self.assertEqual(str(decompose_in_tree(cons, supported_reified={"alldifferent"})),
                         str(cons))

        # tricky one
        cons = [cp.AllDifferent(ivs) < (bv)]
        self.assertEqual(str(decompose_in_tree(cons)),
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) < (bv)]")

    def test_decompose_num(self):

        ivs = [cp.intvar(1, 9, name=n) for n in "xy"]
        bv = cp.boolvar(name="bv")

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

        ivs = [cp.intvar(1,9,name=n) for n in "xyz"]

        cons = [cp.AllDifferent(ivs) == 0]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))), {"not([and([(x) != (y), (x) != (z), (y) != (z)])])"})

        cons = [0 == cp.AllDifferent(ivs)]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))), {"not([and([(x) != (y), (x) != (z), (y) != (z)])])"})

        cons = [cp.AllDifferent(ivs) == cp.AllEqual(ivs[:-1])]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))), {"(and([(x) != (y), (x) != (z), (y) != (z)])) == ((x) == (y))"})

        cons = [min(ivs) == max(ivs)]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons, supported={"min"}))),
                            {"(min(x,y,z)) == (IV0)", "or([(IV0) <= (x), (IV0) <= (y), (IV0) <= (z)])", "(IV0) >= (x)", "(IV0) >= (y)", "(IV0) >= (z)"})

        self.assertEqual(set(map(str,decompose_in_tree(cons, supported={"max"}))),
                         {"(IV1) == (max(x,y,z))", "or([(IV1) >= (x), (IV1) >= (y), (IV1) >= (z)])", "(IV1) <= (x)", "(IV1) <= (y)", "(IV1) <= (z)"})

        # numerical in non-comparison context
        cons = [cp.AllEqual([min(ivs[:-1]),ivs[-1]])]
        self.assertEqual(set(map(str,decompose_in_tree(cons, supported={"allequal"}))),
                         {"allequal(IV2,z)", "((IV2) >= (x)) or ((IV2) >= (y))", "(IV2) <= (x)", "(IV2) <= (y)"})

        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})),
                         "[(min(x,y)) == (z)]")

