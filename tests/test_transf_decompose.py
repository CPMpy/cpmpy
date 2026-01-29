import unittest
import cpmpy as cp
from cpmpy.expressions.globalconstraints import GlobalConstraint
from cpmpy.expressions.globalfunctions import GlobalFunction
from cpmpy.expressions.utils import flatlist
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

        cons = [cp.min(ivs) <= 1]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))),
                            {"IV0 <= 1", "((IV0) >= (x)) or ((IV0) >= (y))", "(IV0) <= (x)", "(IV0) <= (y)"})
        # reified
        cons = [bv.implies(cp.min(ivs) <= 1)]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))),
                            {"(bv) -> (IV1 <= 1)", "((IV1) >= (x)) or ((IV1) >= (y))", "(IV1) <= (x)", "(IV1) <= (y)"})
        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})),str(cons))

        cons = [(cp.min(ivs) <= 1).implies(bv)]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons))),
                            {"(IV2 <= 1) -> (bv)", "((IV2) >= (x)) or ((IV2) >= (y))", "(IV2) <= (x)", "(IV2) <= (y)"})
        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})), str(cons))

        cons = [(cp.min(ivs) <= 1) == (bv)]
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

        cons = [cp.min(ivs) == cp.max(ivs)]
        self.assertSetEqual(set(map(str,decompose_in_tree(cons, supported={"min"}))),
                            {"(min(x,y,z)) == (IV0)", "or([(IV0) <= (x), (IV0) <= (y), (IV0) <= (z)])", "(IV0) >= (x)", "(IV0) >= (y)", "(IV0) >= (z)"})

        self.assertEqual(set(map(str,decompose_in_tree(cons, supported={"max"}))),
                         {"(IV1) == (max(x,y,z))", "or([(IV1) >= (x), (IV1) >= (y), (IV1) >= (z)])", "(IV1) <= (x)", "(IV1) <= (y)", "(IV1) <= (z)"})

        # numerical in non-comparison context
        cons = [cp.AllEqual([cp.min(ivs[:-1]),ivs[-1]])]
        self.assertEqual(set(map(str,decompose_in_tree(cons, supported={"allequal"}))),
                         {"allequal(IV2,z)", "((IV2) >= (x)) or ((IV2) >= (y))", "(IV2) <= (x)", "(IV2) <= (y)"})

        self.assertEqual(str(decompose_in_tree(cons, supported={"min"})),
                         "[(min(x,y)) == (z)]")


    def test_globals_in_decomp(self):

        class MyGlobal1(GlobalConstraint):

            def __init__(self, arr):
                super().__init__("myglobal1", flatlist(arr))

            def decompose(self):
                return ([MyGlobalFunc(self.args)+5 <= 0, cp.max(self.args) == 1],
                        [MyGlobal2(self.args)])

        class MyGlobalFunc(GlobalFunction):

            def __init__(self, arr):
                super().__init__("myglobalfunc", flatlist(arr))

            def decompose(self):
                return cp.sum(self.args), [self.args[0] != 0]

        class MyGlobal2(GlobalConstraint):

            def __init__(self, arr):
                super().__init__("myglobal2", flatlist(arr))
            def decompose(self):
                return [cp.sum(self.args) >= 3], []


        # non-nested case
        x = cp.intvar(0,10,shape=2, name="x")

        cons = MyGlobal1([x])
        self.assertSetEqual(set(map(str,decompose_in_tree([cons], supported={"myglobalfunc","max"}))),
                            {'((myglobalfunc(x[0],x[1])) + 5 <= 0) and (max(x[0],x[1]) == 1)',
                             '(x[0]) + (x[1]) >= 3'})

        # decompose all
        self.assertSetEqual(set(map(str, decompose_in_tree([cons], supported={"max"}))),
                            {'(((x[0]) + (x[1])) + 5 <= 0) and (max(x[0],x[1]) == 1)',
                             '(x[0]) + (x[1]) >= 3','x[0] != 0'})

        # nested case
        bv = cp.boolvar(name="bv")

        cons = bv == MyGlobal1([x])
        self.assertSetEqual(set(map(str, decompose_in_tree([cons], supported={"myglobalfunc", "max"}))),
                            {'(bv) == (((myglobalfunc(x[0],x[1])) + 5 <= 0) and (max(x[0],x[1]) == 1))',
                             '(x[0]) + (x[1]) >= 3'})

        self.assertSetEqual(set(map(str, decompose_in_tree([cons], supported={"max"}))),
                            {'(bv) == ((((x[0]) + (x[1])) + 5 <= 0) and (max(x[0],x[1]) == 1))',
                             '(x[0]) + (x[1]) >= 3', 'x[0] != 0'})


