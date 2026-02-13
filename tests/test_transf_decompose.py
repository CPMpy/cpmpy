import unittest
import cpmpy as cp
from cpmpy.expressions.globalconstraints import GlobalConstraint
from cpmpy.expressions.globalfunctions import GlobalFunction
from cpmpy.expressions.utils import flatlist
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl  # to reset counters
from cpmpy.transformations.linearize import decompose_linear


class TestTransfDecomp:

    def setup_method(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0

    def test_decompose_bool(self):
        ivs = [cp.intvar(1, 9, name=n) for n in "xyz"]
        bv = cp.boolvar(name="bv")

        cons = [cp.AllDifferent(ivs)]
        assert str(decompose_in_tree(cons)) == "[and([(x) != (y), (x) != (z), (y) != (z)])]"
        assert str(decompose_in_tree(cons, supported={"alldifferent"})) == str(cons)

        # reified
        cons = [bv.implies(cp.AllDifferent(ivs))]
        assert str(decompose_in_tree(cons)) == \
                         "[(bv) -> (and([(x) != (y), (x) != (z), (y) != (z)]))]"
        assert str(decompose_in_tree(cons, supported={"alldifferent"})) == \
                         "[(bv) -> (and([(x) != (y), (x) != (z), (y) != (z)]))]"
        assert str(decompose_in_tree(cons, supported={"alldifferent"}, supported_reified={"alldifferent"})) ==str(cons)

        cons = [cp.AllDifferent(ivs).implies(bv)]
        assert str(decompose_in_tree(cons)) == \
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) -> (bv)]"
        assert str(decompose_in_tree(cons, supported={"alldifferent"})) == \
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) -> (bv)]"
        assert str(decompose_in_tree(cons, supported={"alldifferent"}, supported_reified={"alldifferent"})) == \
                         str(cons)

        cons = [cp.AllDifferent(ivs) == (bv)]
        assert str(decompose_in_tree(cons)) == \
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) == (bv)]"
        assert str(decompose_in_tree(cons, supported={"alldifferent"})) == \
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) == (bv)]"
        assert str(decompose_in_tree(cons, supported_reified={"alldifferent"})) == \
                         str(cons)

        # tricky one
        cons = [cp.AllDifferent(ivs) < (bv)]
        assert str(decompose_in_tree(cons)) == \
                         "[(and([(x) != (y), (x) != (z), (y) != (z)])) < (bv)]"

    def test_decompose_num(self):

        ivs = [cp.intvar(1, 9, name=n) for n in "xy"]
        bv = cp.boolvar(name="bv")

        cons = [cp.min(ivs) <= 1]
        assert set(map(str,decompose_in_tree(cons))) == \
                            {"IV0 <= 1", "((IV0) >= (x)) or ((IV0) >= (y))", "(IV0) <= (x)", "(IV0) <= (y)"}
        # reified
        cons = [bv.implies(cp.min(ivs) <= 1)]
        assert set(map(str,decompose_in_tree(cons))) == \
                            {"(bv) -> (IV1 <= 1)", "((IV1) >= (x)) or ((IV1) >= (y))", "(IV1) <= (x)", "(IV1) <= (y)"}
        assert str(decompose_in_tree(cons, supported={"min"})) ==str(cons)

        cons = [(cp.min(ivs) <= 1).implies(bv)]
        assert set(map(str,decompose_in_tree(cons))) == \
                            {"(IV2 <= 1) -> (bv)", "((IV2) >= (x)) or ((IV2) >= (y))", "(IV2) <= (x)", "(IV2) <= (y)"}
        assert str(decompose_in_tree(cons, supported={"min"})) == str(cons)

        cons = [(cp.min(ivs) <= 1) == (bv)]
        assert set(map(str,decompose_in_tree(cons))) == \
                            {"(IV3 <= 1) == (bv)",  "((IV3) >= (x)) or ((IV3) >= (y))", "(IV3) <= (x)", "(IV3) <= (y)"}
        assert str(decompose_in_tree(cons, supported={"min"})) == str(cons)


    def test_decompose_nested(self):

        ivs = [cp.intvar(1,9,name=n) for n in "xyz"]

        cons = [cp.AllDifferent(ivs) == 0]
        assert set(map(str,decompose_in_tree(cons))) == {"not([and([(x) != (y), (x) != (z), (y) != (z)])])"}

        cons = [0 == cp.AllDifferent(ivs)]
        assert set(map(str,decompose_in_tree(cons))) == {"not([and([(x) != (y), (x) != (z), (y) != (z)])])"}

        cons = [cp.AllDifferent(ivs) == cp.AllEqual(ivs[:-1])]
        assert set(map(str,decompose_in_tree(cons))) == {"(and([(x) != (y), (x) != (z), (y) != (z)])) == ((x) == (y))"}

        cons = [cp.min(ivs) == cp.max(ivs)]
        assert set(map(str,decompose_in_tree(cons, supported={"min"}))) == \
                            {"(min(x,y,z)) == (IV0)", "or([(IV0) <= (x), (IV0) <= (y), (IV0) <= (z)])", "(IV0) >= (x)", "(IV0) >= (y)", "(IV0) >= (z)"}

        assert set(map(str,decompose_in_tree(cons, supported={"max"}))) == \
                         {"(IV1) == (max(x,y,z))", "or([(IV1) >= (x), (IV1) >= (y), (IV1) >= (z)])", "(IV1) <= (x)", "(IV1) <= (y)", "(IV1) <= (z)"}

        # numerical in non-comparison context
        cons = [cp.AllEqual([cp.min(ivs[:-1]),ivs[-1]])]
        assert set(map(str,decompose_in_tree(cons, supported={"allequal"}))) == \
                         {"allequal(IV2,z)", "((IV2) >= (x)) or ((IV2) >= (y))", "(IV2) <= (x)", "(IV2) <= (y)"}

        assert str(decompose_in_tree(cons, supported={"min"})) == \
                         "[(min(x,y)) == (z)]"


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
        assert set(map(str,decompose_in_tree([cons], supported={"myglobalfunc","max"}))) == \
                            {'((myglobalfunc(x[0],x[1])) + 5 <= 0) and (max(x[0],x[1]) == 1)',
                             '(x[0]) + (x[1]) >= 3'}

        # decompose all
        assert set(map(str, decompose_in_tree([cons], supported={"max"}))) == \
                            {'(((x[0]) + (x[1])) + 5 <= 0) and (max(x[0],x[1]) == 1)',
                             '(x[0]) + (x[1]) >= 3','x[0] != 0'}

        # nested case
        bv = cp.boolvar(name="bv")

        cons = bv == MyGlobal1([x])
        assert set(map(str, decompose_in_tree([cons], supported={"myglobalfunc", "max"}))) == \
                            {'(bv) == (((myglobalfunc(x[0],x[1])) + 5 <= 0) and (max(x[0],x[1]) == 1))',
                             '(x[0]) + (x[1]) >= 3'}

        assert set(map(str, decompose_in_tree([cons], supported={"max"}))) == \
                            {'(bv) == ((((x[0]) + (x[1])) + 5 <= 0) and (max(x[0],x[1]) == 1))',
                             '(x[0]) + (x[1]) >= 3', 'x[0] != 0'}


    def test_decompose_linear(self):

        x = cp.intvar(1,3, shape=2, name=("a","b"))
        bv = cp.boolvar(name="bv")

        cons = cp.AllDifferent(x)
        assert set(map(str, decompose_linear([cons]))) == \
                            {"and([(a == 1) + (b == 1) <= 1, (a == 2) + (b == 2) <= 1, (a == 3) + (b == 3) <= 1])"}
        # second call gives same result (no ivarmap state)
        assert set(map(str, decompose_linear([cons]))) == \
                            {"and([(a == 1) + (b == 1) <= 1, (a == 2) + (b == 2) <= 1, (a == 3) + (b == 3) <= 1])"}

        # nested
        cons = bv == cp.AllDifferent(x)
        assert set(map(str, decompose_linear([cons]))) == \
                            {"(bv) == (and([(a == 1) + (b == 1) <= 1, (a == 2) + (b == 2) <= 1, (a == 3) + (b == 3) <= 1]))"}

        # test nvalue
        cons = cp.NValue(x) == 8
        assert set(map(str, decompose_linear([cons]))) == \
                            {"sum([(a == 1) or (b == 1), (a == 2) or (b == 2), (a == 3) or (b == 3)]) == 8"}

        # test element
        cons = cp.cpm_array([10,20,30,40])[x[0]] == 8
        assert set(map(str, decompose_linear([cons]))) == \
                            {"sum([20, 30, 40] * [a == 1, a == 2, a == 3]) == 8"}  # a == 0 is False (a in 1..3) 

        # test table
        cons = cp.Table(x, [[1,1], [2,3]])
        assert set(map(str, decompose_linear([cons]))) == \
                            {'((a == 1) and (b == 1)) or ((a == 2) and (b == 3))'}

        # test count
        cons = cp.Count(x, 2) >= 1
        assert set(map(str, decompose_linear([cons]))) == \
                            {'(a == 2) + (b == 2) >= 1'}

    def test_issue_546(self):
        # https://github.com/CPMpy/cpmpy/issues/546
        x = cp.intvar(1,3,shape=2, name=tuple("ab"))
        arr = x.tolist() + [2]

        cons = cp.AllDifferent(arr)
        assert set(map(str, decompose_linear([cons]))) == \
                            {'and([sum([a == 1, b == 1, False]) <= 1, '
                             'sum([a == 2, b == 2, True]) <= 1, '
                             'sum([a == 3, b == 3, False]) <= 1])'}

        # also test full transformation stack
        if "gurobi" in cp.SolverLookup.solvernames():  # otherwise, not supported
            model = cp.Model(cons)
            model.solve(solver="gurobi")

        if "exact" in cp.SolverLookup.solvernames():  # otherwise, not supported
            model = cp.Model(cons)
            model.solve(solver="exact")