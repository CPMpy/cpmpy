import copy

import numpy as np
import pytest

import cpmpy as cp
from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl
from cpmpy.expressions.globalfunctions import GlobalFunction
from cpmpy.exceptions import TypeError, NotSupportedError, IncompleteFunctionError
from cpmpy.expressions.utils import STAR, argvals, argval
from cpmpy.solvers import CPM_minizinc
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.safening import no_partial_functions

from utils import skip_on_missing_pblib, inclusive_range, lambda_assert, full_test_constraint

""""
Tests for global constraints.
For each global constraint, we test the following:
 * test if a feasible constraint is indeed feasible
 * test if an infeasible constraint is indeed infeasible
 * test the negation of the constraint
 * test the decomposition of the constraint
 * test whether constants can be used instead of variables in the constraint
"""

@pytest.mark.usefixtures("solver")
@skip_on_missing_pblib(skip_on_exception_only=True)
class TestGlobalConstraints:

    def setup_method(self):
        _BoolVarImpl.counter = 0
        _IntVarImpl.counter = 0

    def test_alldifferent(self, solver):
        x = cp.intvar(1,3, shape=3)
        y = cp.intvar(1,2, shape=3)
        full_test_constraint(cp.AllDifferent(x), solver)
        full_test_constraint(cp.AllDifferent(x[0], x[1], 3), solver)
        full_test_constraint(cp.AllDifferent(y), solver, satisfiable=False)
        # edge case, 1 argument
        assert cp.Model(cp.AllDifferent([x[0]])).solve(solver=solver) is True

    def test_alldifferent_except0(self, solver):
        x = cp.intvar(0,3, shape=4)
        y = cp.intvar(1,3, shape=4) # cannot assign 0 to y

        full_test_constraint(cp.AllDifferentExcept0(x), solver)
        full_test_constraint(cp.AllDifferentExcept0(x[0], x[1], 3), solver)
        full_test_constraint(cp.AllDifferentExcept0(y), solver, satisfiable=False)
        # edge case, 1 argument
        assert cp.Model(cp.AllDifferentExcept0([x[0]])).solve(solver=solver) is True
        
    def test_alldifferent_except_n(self, solver):
        x = cp.intvar(1,3, shape=4)
        y = cp.intvar(1,3, shape=4) # cannot assign 0 to y

        full_test_constraint(cp.AllDifferentExceptN(x, 1), solver)
        full_test_constraint(cp.AllDifferentExceptN(x, [0, 1]), solver)
        full_test_constraint(cp.AllDifferentExceptN(x, 0), solver, satisfiable=False)
        # edge case, 1 argument
        assert cp.Model(cp.AllDifferentExceptN([x[0]], 1)).solve(solver=solver) is True

    def test_circuit(self, solver):
        x = cp.intvar(0,4, shape=4) # lb=-1 crashes Choco -- TODO: check if Choco or pychoco bug
        y = cp.intvar(1,5, shape=4)
        y_neg = cp.intvar(-2,2, shape=4)
        full_test_constraint(cp.Circuit(x), solver)
        full_test_constraint(cp.Circuit([x[0],x[1],0]), solver)
        full_test_constraint(cp.Circuit(y), solver, satisfiable=False)
        full_test_constraint(cp.Circuit(y_neg), solver, satisfiable=False)
        # edge case with 1 argument is undefined, constructor raises ValueError

    def test_inverse(self, solver):
        fwd = cp.intvar(-1,4, shape=3)
        rev = cp.intvar(-1,4, shape=3)
        full_test_constraint(cp.Inverse(fwd, rev), solver)
        full_test_constraint(cp.Inverse([fwd[0],fwd[1],2], [rev[0],rev[1],2]), solver)
        full_test_constraint(cp.Inverse(cp.intvar(1,3,shape=3), cp.intvar(1,3,shape=3)), solver, satisfiable=False)
        # edge case, 1 argument
        full_test_constraint(cp.Inverse([fwd[0]], [rev[0]]), solver)

    def test_in_domain(self, solver):
        x = cp.intvar(-2,2)
        y = cp.intvar(-2,2,shape=3)

        full_test_constraint(cp.InDomain(x,[-2,2]), solver)
        full_test_constraint(cp.InDomain(x,[]), solver, satisfiable=False)
        full_test_constraint(cp.InDomain(x,[5]), solver, satisfiable=False)

        full_test_constraint(cp.InDomain(cp.min(y), [-1,2]), solver)
        full_test_constraint(cp.InDomain(x, [-1]), solver)

    def test_lex_lesseq(self, solver):
        x = cp.intvar(0,2, shape=3)
        y = cp.intvar(0,2, shape=3)
        z = cp.intvar(4,5, shape=3)

        full_test_constraint(cp.LexLessEq(x, y), solver)
        full_test_constraint(cp.LexLessEq(x, [1,1,1]), solver)
        full_test_constraint(cp.LexLessEq(x, [-1,1,1]), solver, satisfiable=False)
        full_test_constraint(cp.LexLessEq(z,x), solver, satisfiable=False)
        # edge case with 1 argument
        full_test_constraint(cp.LexLessEq([x[0]], [y[0]]), solver)

    def test_lex_less(self, solver):
        x = cp.intvar(0,2, shape=3)
        y = cp.intvar(0,2, shape=3)
        z = cp.intvar(2,4, shape=3)

        full_test_constraint(cp.LexLess(x, y), solver)
        full_test_constraint(cp.LexLess(x, [1,1,1]), solver)
        full_test_constraint(cp.LexLess(x, [0,0,0]), solver, satisfiable=False)
        full_test_constraint(cp.LexLess(z,x), solver, satisfiable=False)
        # edge case with 1 argument
        full_test_constraint(cp.LexLess([x[0]], [y[0]]), solver)

    def test_lex_chain(self, solver):
        from cpmpy import BoolVal
        X = cp.intvar(0, 3, shape=10)
        c1 = X[:-1] == 1
        Y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        c = cp.LexChainLess([X, Y])
        c2 = c != (BoolVal(True))
        m = cp.Model([c1, c2])
        assert m.solve(solver=solver)
        assert c2.value()
        assert not c.value()

        Y = cp.intvar(0, 0, shape=10)
        c = cp.LexChainLessEq([X, Y])
        m = cp.Model(c)
        assert m.solve(solver=solver)
        from cpmpy.expressions.utils import argval
        assert sum(argval(X)) == 0

        Z = cp.intvar(0, 1, shape=(4,2))
        c = cp.LexChainLess(Z)
        m = cp.Model(c)
        assert m.solve(solver=solver)
        assert sum(argval(Z[0])) == 0
        assert sum(argval(Z[1])) == 1
        assert argval(Z[1,0]) == 0
        assert sum(argval(Z[2])) == 1
        assert argval(Z[2,1]) == 0
        assert sum(argval(Z[3])) >= 1
    
    def test_lex_chain_lesseq(self, solver):
        X = cp.intvar(0, 2, shape=(3, 2))
        # LexChainLessEq: each row <=_lex next row
        full_test_constraint(cp.LexChainLessEq(X), solver)
        full_test_constraint(cp.LexChainLessEq([X[0], X[1], [0, 0]]), solver)
        # unsat: force first row > lex second
        y = cp.intvar(0, 2, shape=(2, 2))
        full_test_constraint(cp.LexChainLessEq([y[0], y[1]]), solver)
        m = cp.Model([cp.LexChainLessEq([y[0], y[1]]), y[0, 0] == 2, y[0, 1] == 0, y[1, 0] == 0, y[1, 1] == 0])
        assert m.solve(solver=solver) is False

    def test_table(self, solver):
        x = cp.intvar(-2,2, shape=3)

        full_test_constraint(cp.Table(x, [[1,1,1],[0,0,0]]), solver)
        full_test_constraint(cp.Table(x, [[1,2,3],[3,2,1]]), solver, satisfiable=False)
        # edge case with 1 argument
        full_test_constraint(cp.Table([x[0]], [[1]]), solver)

    def test_negative_table(self, solver):
        x = cp.intvar(-2,2, shape=3)

        full_test_constraint(cp.NegativeTable(x, [[1,1,1],[0,0,0]]), solver)
        full_test_constraint(cp.NegativeTable(x, [[1,2,3],[0,0,0]]), solver)
        # cannot check unsat case, would wipe all possible assignments...
        # edge case with 1 argument
        full_test_constraint(cp.NegativeTable([x[0]], [[1]]), solver)

    def test_shorttable(self, solver):
        x = cp.intvar(-2,2, shape=3)

        full_test_constraint(cp.ShortTable(x, [[1,1,1],[0,0,0]]), solver)
        full_test_constraint(cp.ShortTable(x, [[1,2,3],[3,2,1]]), solver, satisfiable=False)
        full_test_constraint(cp.ShortTable(x, [[1,2,3],["*", "*", 1]]), solver)
        full_test_constraint(cp.ShortTable(x, [[1,2,3],["*", "*", 4]]), solver, satisfiable=False)
        # edge case with 1 argument
        full_test_constraint(cp.ShortTable([x[0]], [[1]]), solver)
        # cannot check case with 1 star arg... would simply be True

    def test_table_accepts_ndarray(self, solver):
        """Table, NegativeTable, ShortTable accept np.ndarray as table; value() works (stored as list)."""
        iv = cp.intvar(0, 10, shape=2)
        tab = np.array([[1, 2], [3, 4]], dtype=int)
        t = cp.Table(iv, tab)
        for var, val in zip(iv, [1, 2]):
            var._value = val
        assert t.value()
        for var, val in zip(iv, [3, 4]):
            var._value = val
        assert t.value()
        for var, val in zip(iv, [0, 0]):
            var._value = val
        assert not t.value()

        nt = cp.NegativeTable(iv, tab)
        for var, val in zip(iv, [1, 2]):
            var._value = val
        assert not nt.value()
        for var, val in zip(iv, [0, 0]):
            var._value = val
        assert nt.value()

        tab_star = np.array([[1, 2], [STAR, 4]], dtype=object)
        st = cp.ShortTable(iv, tab_star)
        for var, val in zip(iv, [1, 2]):
            var._value = val
        assert st.value()
        for var, val in zip(iv, [99, 4]):
            var._value = val
        assert st.value()

    def test_regular(self, solver):
        # test based on the example from XCSP3 specifications https://arxiv.org/pdf/1611.03398
        x = cp.intvar(0, 1, shape=7)

        transitions = [("a", 0, "a"), ("a", 1, "b"), ("b", 1, "c"), ("c", 0, "d"), ("d", 0, "d"), ("d", 1, "e"),
                       ("e", 0, "e")]
        start = "a"
        ends = ["e"]

        full_test_constraint(cp.Regular(x, transitions, "a", ["e"]), solver)

    def test_xor(self, solver):
        bv = cp.boolvar(shape=3)
        
        full_test_constraint(cp.Xor(bv), solver)
        # edge case with 1 argument
        full_test_constraint(cp.Xor([bv[0]]), solver)

        # test with constants
        full_test_constraint(cp.Xor(bv.tolist() + [True]), solver)
        full_test_constraint(cp.Xor(bv.tolist() + [True, True]), solver)
        full_test_constraint(cp.Xor(bv.tolist() + [True, True, True]), solver)
        full_test_constraint(cp.Xor(bv.tolist() + [False]), solver)
        full_test_constraint(cp.Xor(bv.tolist() + [False, True]), solver)
       

    def test_ite(self, solver):
        x,y,z = cp.intvar(0,2, shape=3, name=("x","y","z"))

        full_test_constraint(cp.IfThenElse(x >= 1, y == 0, z == 0), solver)
        full_test_constraint(cp.IfThenElse(x < 0, y == 0, z == 0), solver)
        full_test_constraint(cp.IfThenElse(x == 1, y == 5, z == 5), solver, satisfiable=False)

        # test with constants
        full_test_constraint(cp.IfThenElse(True, y == 0, z == 0), solver)
        full_test_constraint(cp.IfThenElse(False, y == 0, z == 0), solver)


    def test_cumulative(self, solver):
        
        start = cp.intvar(0,3, shape=3)
        dur = cp.intvar(1,2,shape=3)
        end = cp.intvar(1,5, shape=3)
        demand = cp.intvar(1,2,shape=3)

        # test with variable demand
        full_test_constraint(cp.Cumulative(start, dur, end, demand, 2), solver)
        full_test_constraint(cp.Cumulative(start, dur, None, demand, 2), solver)
        # test unsat case
        full_test_constraint(cp.Cumulative(start, [1,2,3], end,1, 1), solver, satisfiable=False)

        # test with single task
        full_test_constraint(cp.Cumulative([start[0]], [4], [end[0]], [demand[0]], 2), solver)

        # test decomposition with numpy capacity -- ensure fix of #435
        assert cp.Model(cp.Cumulative(start, dur, None, demand, np.int64(2)).decompose()).solve(solver= solver) is True

        # test with negative durations
        dur = cp.intvar(-1,1, shape=3)
        neg_dur = cp.intvar(-2,-1, shape=3)
        assert cp.Model(cp.Cumulative(start, dur, end, demand, 2)).solve(solver=solver) is True
        assert cp.Model(cp.Cumulative(start, neg_dur, end, demand, 2)).solve(solver=solver) is False
        
        # test with negative demand
        demand = cp.intvar(-1,1, shape=3)
        neg_demand = cp.intvar(-2,-1, shape=3)
        assert cp.Model(cp.Cumulative(start, dur, end, demand, 2)).solve(solver=solver) is True
        assert cp.Model(cp.Cumulative(start, dur, end, neg_demand, 2)).solve(solver=solver) is False

        # test reified with start + dur != end
        bv = cp.boolvar()
        assert cp.Model(bv.implies(cp.Cumulative(start, [1,1,1], [10,10,10], demand, 2))).solve(solver=solver) is True

    def test_no_overlap(self, solver):
        start = cp.intvar(0,3, shape=3)
        dur = cp.intvar(1,2, shape=3)
        end = cp.intvar(1,5, shape=3)

        full_test_constraint(cp.NoOverlap(start, dur, end), solver)
        full_test_constraint(cp.NoOverlap(start, dur, None), solver)
        full_test_constraint(cp.NoOverlap(start, [1,2,3], end), solver, satisfiable=False)

        # test with single task
        full_test_constraint(cp.NoOverlap([start[0]], [4], [end[0]]), solver)

        # test with negative durations
        dur = cp.intvar(-1,1, shape=3)
        neg_dur = cp.intvar(-2,-1, shape=3)
        assert cp.Model(cp.NoOverlap(start, dur, end)).solve(solver=solver) is True
        assert cp.Model(cp.NoOverlap(start, neg_dur, end)).solve(solver=solver) is False

    def test_precedence(self, solver):
        x = cp.intvar(1,3, shape=4, name="x")

        full_test_constraint(cp.Precedence(x, [0,1,2]), solver)
        full_test_constraint(cp.Precedence([x[0], x[1], 4], [0, 1, 2]), solver)
        # edge case with 1 argument
        full_test_constraint(cp.Precedence([x[0]], [0, 1, 2]), solver)

        # Check bug fix pull request #742
        # - ensure first constraint from paper is satisfied
        cons = cp.Precedence(x, [0, 1, 2])
        assert not cp.Model([cons, (x[0] == 1) | (x[0] == 2)]).solve(solver=solver)
    
    def test_global_cardinality_count(self, solver):
        iv = cp.intvar(-2, 2, shape=3)
        val = [1,2,3]
        cnt = cp.intvar(0, len(iv), shape=len(val))

        full_test_constraint(cp.GlobalCardinalityCount(iv, val, cnt), solver)
        full_test_constraint(cp.GlobalCardinalityCount(iv, val, cnt, closed=True), solver)
        full_test_constraint(cp.GlobalCardinalityCount([iv[0], iv[1], 3], val, cnt), solver)
        # edge case with 1 argument
        full_test_constraint(cp.GlobalCardinalityCount([iv[0]], val, cnt), solver)

    def test_allequal(self, solver):
        x = cp.intvar(-2, 2, shape=3)

        full_test_constraint(cp.AllEqual(x), solver)
        full_test_constraint(cp.AllEqual(x[0], x[1], 0), solver)
        full_test_constraint(cp.AllEqual([x[0], x[1], 3]), solver, satisfiable=False)
        # edge case with 1 argument (trivially true, just check if no errors occur)
        assert cp.Model(cp.AllEqual([x[0]])).solve(solver=solver) is True

    def test_allequal_except_n(self, solver):
        x = cp.intvar(-2, 2, shape=4)
        # except 0: all non-zero must be equal
        full_test_constraint(cp.AllEqualExceptN(x, 0), solver)
        full_test_constraint(cp.AllEqualExceptN(x, [0, 1]), solver)
        full_test_constraint(cp.AllEqualExceptN([x[0], x[1], 2], 0), solver)
        full_test_constraint(cp.AllEqualExceptN([x[0], x[1], 3], 3), solver)
        full_test_constraint(cp.AllEqualExceptN([x[0], x[1], cp.intvar(4,4)], 0), solver)
        full_test_constraint(cp.AllEqualExceptN([x[0], x[1], cp.intvar(4,4)], 5), solver, satisfiable=False)
        # edge case with 1 argument (trivially true, just check if no errors occur)
        assert cp.Model(cp.AllEqualExceptN([x[0]], 0)).solve(solver=solver) is True

    def test_increasing(self, solver):
        x = cp.intvar(-2, 2, shape=3)
        y = cp.intvar(0, 1, shape=3)

        full_test_constraint(cp.Increasing(x), solver)
        full_test_constraint(cp.Increasing(x[0], x[1], 2), solver)
        full_test_constraint(cp.Increasing([3, x[0], x[1]]), solver, satisfiable=False)
        # edge case with 1 argument (trivially true, just check if no errors occur)
        assert cp.Model(cp.Increasing([x[0]])).solve(solver=solver) is True

    def test_decreasing(self, solver):
        x = cp.intvar(-2, 2, shape=3)

        full_test_constraint(cp.Decreasing(x), solver)
        full_test_constraint(cp.Decreasing(x[0], x[1], 2), solver)
        full_test_constraint(cp.Decreasing([x[0], x[1], 3]), solver, satisfiable=False)
        # edge case with 1 argument (trivially true, just check if no errors occur)
        assert cp.Model(cp.Decreasing([x[0]])).solve(solver=solver) is True

    def test_increasing_strict(self, solver):
        x = cp.intvar(-2, 2, shape=3)
        y = cp.intvar(0, 1, shape=3)

        full_test_constraint(cp.IncreasingStrict(x), solver)
        full_test_constraint(cp.IncreasingStrict(x[0], x[1], 2), solver)
        full_test_constraint(cp.IncreasingStrict([2, x[0], x[1]]), solver, satisfiable=False)
        # edge case with 1 argument (trivially true, just check if no errors occur)
        assert cp.Model(cp.IncreasingStrict([x[0]])).solve(solver=solver) is True

    def test_decreasing_strict(self, solver):
        x = cp.intvar(-2, 2, shape=3)

        full_test_constraint(cp.DecreasingStrict(x), solver)
        full_test_constraint(cp.DecreasingStrict(x[0], x[1], -2), solver)
        full_test_constraint(cp.DecreasingStrict([x[0], x[1], 1]), solver, satisfiable=False)
        # edge case with 1 argument (trivially true, just check if no errors occur)
        assert cp.Model(cp.DecreasingStrict([x[0]])).solve(solver=solver) is True

@pytest.mark.usefixtures("solver")
@skip_on_missing_pblib(skip_on_exception_only=True)
class TestGlobalFunctions:

    def setup_method(self):
        _BoolVarImpl.counter = 0
        _IntVarImpl.counter = 0

    def test_minimum(self, solver):
        x = cp.intvar(-2,2,shape=3)
        
        full_test_constraint(cp.Minimum(x) == 0, solver)
        full_test_constraint(cp.Minimum(x) >= 3, solver, satisfiable=False)

        # edge case with 1 argument
        full_test_constraint(cp.Minimum([x[0]]) == 0, solver)

    def test_maximum(self, solver):
        x = cp.intvar(-2,2,shape=3)
        
        full_test_constraint(cp.Maximum(x) == 0, solver)
        full_test_constraint(cp.Maximum(x) >= 3, solver, satisfiable=False)

        # edge case with 1 argument
        full_test_constraint(cp.Maximum([x[0]]) == 0, solver)


    def test_abs(self, solver):
        x = cp.intvar(-2,2)
        full_test_constraint(cp.Abs(x) == 0, solver)
        full_test_constraint(cp.Abs(x) < 0, solver, satisfiable=False)

        pos = cp.intvar(0,2)
        full_test_constraint(cp.Abs(pos) == 2, solver)

        neg = cp.intvar(0,2)
        full_test_constraint(cp.Abs(neg) == 2, solver)

    def test_element(self, solver):
        # test 1-D
        arr = cp.intvar(-2, 2, 3, name="arr")
        idx = cp.intvar(-2, 2, name="idx")
        oob_idx = cp.intvar(4,5, name="oob_idx")

        # test directly the constraint
        full_test_constraint(cp.Element(arr, idx) == 0, solver)
        full_test_constraint(cp.Element(arr, oob_idx) == 1, solver, satisfiable=False)
        # test through __get_item__
        full_test_constraint(arr[idx] == 0, solver)
        full_test_constraint(arr[oob_idx] == 1, solver, satisfiable=False)
        # test one arg
        full_test_constraint(cp.Element([arr[0]], idx) == 0, solver)
        full_test_constraint(arr[:1][idx] == 0, solver)
        
        # test 2-D element
        mtrx = cp.intvar(0, 2, shape=(2,2), name="iv")
        a,b = cp.intvar(0, 1, shape=2) # idx is always safe
        full_test_constraint(mtrx[a,b] == 0, solver)
        # TODO: 2D element is translated to mtrx.flatten()[mtrx.shape[0]*a + b]
        #   but this means that if a=0, and b = 2, the constraint **should** be unsafe, but it is not!
        #   we will simply return mtrx[1,0] in this case, which is incorrect!!
        
        # test optimization where 1 dim is index
        cons = mtrx[1, idx] == 0
        assert str(cons) == "[iv[1,0] iv[1,1]][idx] == 0"
        cons = mtrx[idx, 1] == 0
        assert str(cons) == "[iv[0,1] iv[1,1]][idx] == 0"

        # test multi-dimensional elemnent with 1 expression index
        x = cp.intvar(1,5, shape=(3,4,5),name="x")
        a,b = cp.intvar(0,2, shape=2, name=tuple("ab")) # idx is always safe
        assert str(x[a,1,3]) == "[x[0,1,3] x[1,1,3] x[2,1,3]][a]"
        assert str(x[1,a,3]) == "[x[1,0,3] x[1,1,3] x[1,2,3] x[1,3,3]][a]"
        assert str(x[1,2,a]) == "[x[1,2,0] x[1,2,1] x[1,2,2] x[1,2,3] x[1,2,4]][a]"

    def test_element_index_dom_mismatched(self, solver):
        """
            Check transform of `[0,1,2][x in -1..1] == y in 1..5`
            Note the index variable has a lower bound *outside* the indexable range, and an upper bound inside AND lower than the indexable range upper bound
        """
        elem = cp.Element([0, 1, 2], cp.intvar(-1, 1, name="x"))
        constraint = elem <= cp.intvar(1, 5, name="y")
        decomposed = decompose_in_tree(no_partial_functions([constraint], safen_toplevel={"element"}))

        expected = {
            # safening constraints
            "(BV0) == ((x >= 0) and (x <= 2))",
            "(BV0) -> ((IV0) == (x))",
            "(~BV0) -> (IV0 == 0)",
            "BV0",
            # actual decomposition
            '(IV0 == 0) -> (IV1 == 0)',
            '(IV0 == 1) -> (IV1 == 1)',
            '(IV0 == 2) -> (IV1 == 2)',
            '(IV1) <= (y)'
        }
        assert set(map(str, decomposed)) == expected

        # should raise a warning if we don't safen first
        with pytest.warns(UserWarning, match=".*unsafe.*"):
            val, decomp = elem.decompose()
            expected = {
                # actual decomposition
                '(x == 0) -> (IV2 == 0)',
                '(x == 1) -> (IV2 == 1)',
                'x >= 0', 'x < 3'
            }
            assert set(map(str, decomp)) == expected

        # also for linear decomp
        with pytest.warns(UserWarning, match=".*unsafe.*"):
            val, decomp = elem.decompose_linear()
            expected = {
                'x >= 0', 'x < 3'
            }
            assert set(map(str, decomp)) == expected
            assert str(val) == "sum([0, 1] * [x == 0, x == 1])"

    def test_modulo(self, solver):
        x,y,z = cp.intvar(-2,2, shape=3, name=tuple("xyz"))
        full_test_constraint(x % y == z, solver)

    def test_div(self, solver):
        x,y,z = cp.intvar(-2,2, shape=3, name=tuple("xyz"))
        full_test_constraint(x // y == z, solver)
    
    def test_multiplication(self, solver):
        x,y = cp.intvar(-2,2,shape=2,name=("x","y"))
        full_test_constraint(x * y >= 1, solver)
        full_test_constraint(x * y == 3, solver, satisfiable=False)

    def test_power(self, solver):
        x = cp.intvar(-2, 2)
        y = cp.intvar(0, 10)
        # x ** 2 == y (constant exponent)
        full_test_constraint(x**2 == y, solver)
        full_test_constraint(cp.intvar(0, 1)**2 == cp.intvar(5, 5), solver, satisfiable=False)

    def test_count(self, solver):
        x = cp.intvar(-2,2, shape=3)

        full_test_constraint(cp.Count(x, 0) == 1, solver)
        full_test_constraint(cp.Count(x, 0) > 3, solver, satisfiable=False)
        full_test_constraint(cp.Count([x[0],x[1],2], 0) == 1, solver)
        # edge case with 1 argument
        full_test_constraint(cp.Count([x[0]], 0) == 1, solver)
        
    def test_nvalue(self, solver):
        x = cp.intvar(-2,2, shape=3)

        full_test_constraint(cp.NValue(x) == 1, solver)
        full_test_constraint(cp.NValue(x) == 0, solver, satisfiable=False)
        full_test_constraint(cp.NValue([x[0],x[1],2]) == 1, solver)
        full_test_constraint(cp.NValue([x[0],x[1],2,3]) == 1, solver, satisfiable=False)
        # edge case with 1 argument (trivially true, just check if no errors occur)
        assert cp.Model(cp.NValue([x[0]]) >= 1).solve(solver=solver) is True

        # test not contiguous -- TODO, not sure what this test is doing here...
        iv = cp.intvar(0, 10, shape=(3, 3))
        assert cp.Model([cp.NValue(i) == 3 for i in iv.T]).solve(solver=solver)
        
    def test_nvalue_except(self, solver):

        x = cp.intvar(-2,2, shape=3)

        full_test_constraint(cp.NValueExcept(x, 0) == 1, solver)
        full_test_constraint(cp.NValueExcept(x, 0) == 0, solver)
        full_test_constraint(cp.NValueExcept(x, 4) == 0, solver, satisfiable=False)
        full_test_constraint(cp.NValueExcept([x[0],x[1],2], 0) == 1, solver)
        full_test_constraint(cp.NValueExcept([x[0],x[1],2,3], 0) == 1, solver, satisfiable=False)
        # edge case with 1 argument
        full_test_constraint(cp.NValueExcept([x[0]], 0) == 1, solver)

        # test not contiguous -- TODO, not sure what this test is doing here...
        iv = cp.intvar(0, 10, shape=(3, 3))
        assert cp.Model([cp.NValueExcept(i, 0) == 3 for i in iv.T]).solve(solver=solver)

    def test_among(self, solver):
        x = cp.intvar(-2, 2, shape=3)

        full_test_constraint(cp.Among(x, [0, 1]) <= 2, solver)
        full_test_constraint(cp.Among([x[0], x[1], 0], [0, 1]) == 1, solver)
        full_test_constraint(cp.Among(x, [3,4]) > 0, solver, satisfiable=False)
        # edge case with 1 argument
        full_test_constraint(cp.Among([x[0]], [0, 1]) == 1, solver)


@pytest.mark.usefixtures("solver")
class TestBounds:
    def test_bounds_minimum(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        z = cp.intvar(1, 9)
        expr = cp.Minimum([x,y,z])
        lb,ub = expr.get_bounds()
        assert lb ==-8
        assert ub ==-1
        assert not cp.Model(expr<lb).solve(solver=solver)
        assert not cp.Model(expr>ub).solve(solver=solver)


    def test_bounds_maximum(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        z = cp.intvar(1, 9)
        expr = cp.Maximum([x,y,z])
        lb,ub = expr.get_bounds()
        assert lb ==1
        assert ub ==9
        assert not cp.Model(expr<lb).solve(solver=solver)
        assert not cp.Model(expr>ub).solve(solver=solver)

    def test_bounds_abs(self, solver):
        x = cp.intvar(-8, 5)
        y = cp.intvar(-7, -2)
        z = cp.intvar(1, 9)
        for var,test_lb,test_ub in [(x,0,8),(y,2,7),(z,1,9)]:
            lb, ub = cp.Abs(var).get_bounds()
            assert test_lb ==lb
            assert test_ub ==ub

    def test_bounds_div(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7,-1)
        z = cp.intvar(3,9)
        op1 = cp.Division(x,y)
        lb1,ub1 = op1.get_bounds()
        assert lb1 ==-8
        assert ub1 ==8
        op2 = cp.Division(x,z)
        lb2,ub2 = op2.get_bounds()
        assert lb2 ==-2
        assert ub2 ==2
        for lhs in inclusive_range(*x.get_bounds()):
            for rhs in inclusive_range(*y.get_bounds()):
                val = cp.Division(lhs,rhs).value()
                assert val >=lb1
                assert val <=ub1
            for rhs in inclusive_range(*z.get_bounds()):
                val = cp.Division(lhs, rhs).value()
                assert val >=lb2
                assert val <=ub2

    def test_bounds_mod(self, solver):
        x = cp.intvar(-8, 8)
        xneg = cp.intvar(-8, 0)
        xpos = cp.intvar(0, 8)
        y = cp.intvar(-5, -1)
        z = cp.intvar(1, 4)
        op1 = cp.Modulo(xneg,y)
        lb1, ub1 = op1.get_bounds()
        assert lb1 ==-4
        assert ub1 ==0
        op2 = cp.Modulo(xpos,z)
        lb2, ub2 = op2.get_bounds()
        assert lb2 ==0
        assert ub2 ==3
        op3 = cp.Modulo(xneg,z)
        lb3, ub3 = op3.get_bounds()
        assert lb3 ==-3
        assert ub3 ==0
        op4 = cp.Modulo(xpos,y)
        lb4, ub4 = op4.get_bounds()
        assert lb4 ==0
        assert ub4 ==4
        op5 = cp.Modulo(x,y)
        lb5, ub5 = op5.get_bounds()
        assert lb5 ==-4
        assert ub5 ==4
        op6 = cp.Modulo(x,z)
        lb6, ub6 = op6.get_bounds()
        assert lb6 ==-3
        assert ub6 ==3
        for lhs in inclusive_range(*x.get_bounds()):
            for rhs in inclusive_range(*y.get_bounds()):
                val = cp.Modulo(lhs,rhs).value()
                assert val >=lb5
                assert val <=ub5
            for rhs in inclusive_range(*z.get_bounds()):
                val = cp.Modulo(lhs, rhs).value()
                assert val >=lb6
                assert val <=ub6

    def test_bounds_pow(self, solver):
        x = cp.intvar(-8, 5)
        op = cp.Power(x,3)
        lb, ub = op.get_bounds()
        assert lb ==-8 ** 3
        assert ub ==5 ** 3

        op = cp.Power(x, 4)
        lb, ub = op.get_bounds()
        assert lb == 5 ** 4
        assert ub == 8 ** 4

    def test_bounds_element(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        z = cp.intvar(1, 9)
        expr = cp.Element([x, y, z],z)
        lb, ub = expr.get_bounds()
        assert lb ==-8
        assert ub ==9
        assert not cp.Model(expr < lb).solve(solver=solver)
        assert not cp.Model(expr > ub).solve(solver=solver)

    def test_incomplete_func(self, solver):
        # element constraint
        arr = cp.cpm_array([1,2,3])
        i = cp.intvar(0,5,name="i")
        p = cp.boolvar()

        cons = (arr[i] == 1).implies(p)
        m = cp.Model([cons, i == 5])
        assert m.solve(solver=solver)
        assert cons.value()

        # div constraint
        a,b = cp.intvar(1,2,shape=2)
        cons = (42 // (a - b)) >= 3
        m = cp.Model([p.implies(cons), a == b])
        assert m.solve(solver=solver)  # ortools does not support divisor spanning 0
        pytest.raises(IncompleteFunctionError, cons.value)
        assert not argval(cons)

        # mayhem
        cons = (arr[10 // (a - b)] == 1).implies(p)
        m = cp.Model([cons, a == b])
        assert m.solve(solver=solver)
        assert cons.value()

    def test_bounds_count(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        z = cp.intvar(1, 9)
        a = cp.intvar(1, 9)
        expr = cp.Count([x, y, z], a)
        lb, ub = expr.get_bounds()
        assert lb ==0
        assert ub ==3
        assert not cp.Model(expr < lb).solve(solver=solver)
        assert not cp.Model(expr > ub).solve(solver=solver)

    def test_bounds_xor(self, solver):
        # just one case of a Boolean global constraint
        expr = cp.Xor(cp.boolvar(3))
        assert expr.get_bounds() ==(0,1)

@skip_on_missing_pblib(skip_on_exception_only=True)
@pytest.mark.usefixtures("solver")
class TestTypeChecks:
    def test_all_diff(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.AllDifferent(x,y)]).solve(solver=solver)
        assert cp.Model([cp.AllDifferent(a,b)]).solve(solver=solver)
        assert cp.Model([cp.AllDifferent(x,y,b)]).solve(solver=solver)

    def test_all_diff_ex0(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.AllDifferentExcept0(x,y)]).solve(solver=solver)
        assert cp.Model([cp.AllDifferentExcept0(a,b)]).solve(solver=solver)
        #self.assertTrue(cp.Model([cp.AllDifferentExcept0(x,y,b)]).solve())

    def test_all_equal(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.AllEqual(x,y,-1)]).solve(solver=solver)
        assert cp.Model([cp.AllEqual(a,b,False, a | b)]).solve(solver=solver)
        assert not cp.Model([cp.AllEqual(x,y,b)]).solve(solver=solver)

    def test_all_equal_exceptn(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.AllEqualExceptN([x,y,-1],211)]).solve(solver=solver)
        assert cp.Model([cp.AllEqualExceptN([x,y,-1,4],4)]).solve(solver=solver)
        assert cp.Model([cp.AllEqualExceptN([x,y,-1,4],-1)]).solve(solver=solver)
        assert cp.Model([cp.AllEqualExceptN([a,b,False, a | b], 4)]).solve(solver=solver)
        assert cp.Model([cp.AllEqualExceptN([a,b,False, a | b], 0)]).solve(solver=solver)
        assert cp.Model([cp.AllEqualExceptN([a,b,False, a | b, y], -1)]).solve(solver=solver)

        # test with list of n
        iv = cp.intvar(0, 4, shape=7)
        assert not cp.Model([cp.AllEqualExceptN([iv], [7,8]), iv[0] != iv[1]]).solve(solver=solver)
        assert cp.Model([cp.AllEqualExceptN([iv], [4, 1]), iv[0] != iv[1]]).solve(solver=solver)

    def test_not_all_equal_exceptn(self, solver):
        x = cp.intvar(lb=0, ub=3, shape=3)
        n = 2
        constr = cp.AllEqualExceptN(x,n)

        model = cp.Model([~constr, x == [1, 2, 1]])
        assert not model.solve(solver=solver)

        model = cp.Model([~constr])
        assert model.solve(solver=solver)
        assert not constr.value()

        assert not cp.Model([constr, ~constr]).solve(solver=solver)

        all_sols = set()
        not_all_sols = set()

        circuit_models = cp.Model(constr).solveAll(solver=solver, display=lambda: all_sols.add(tuple(x.value())))
        not_circuit_models = cp.Model(~constr).solveAll(solver=solver, display=lambda: not_all_sols.add(tuple(x.value())))

        total = cp.Model(x == x).solveAll(solver=solver)

        for sol in all_sols:
            for var, val in zip(x, sol):
                var._value = val
            assert constr.value()

        for sol in not_all_sols:
            for var, val in zip(x, sol):
                var._value = val
            assert not constr.value()

        assert total == len(all_sols) + len(not_all_sols)


    def test_increasing(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Increasing(x,y)]).solve(solver=solver)
        assert cp.Model([cp.Increasing(a,b)]).solve(solver=solver)
        assert cp.Model([cp.Increasing(x,y,b)]).solve(solver=solver)
        z = cp.intvar(2,5)
        assert not cp.Model([cp.Increasing(z,b)]).solve(solver=solver)

    def test_decreasing(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Decreasing(x,y)]).solve(solver=solver)
        assert cp.Model([cp.Decreasing(a,b)]).solve(solver=solver)
        assert not cp.Model([cp.Decreasing(x,y,b)]).solve(solver=solver)
        z = cp.intvar(2,5)
        assert cp.Model([cp.Decreasing(z,b)]).solve(solver=solver)

    def test_increasing_strict(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.IncreasingStrict(x,y)]).solve(solver=solver)
        assert cp.Model([cp.IncreasingStrict(a,b)]).solve(solver=solver)
        assert cp.Model([cp.IncreasingStrict(x,y,b)]).solve(solver=solver)
        z = cp.intvar(1,5)
        assert not cp.Model([cp.IncreasingStrict(z,b)]).solve(solver=solver)

    def test_decreasing_strict(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, 0)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.DecreasingStrict(x,y)]).solve(solver=solver)
        assert cp.Model([cp.DecreasingStrict(a,b)]).solve(solver=solver)
        assert not cp.Model([cp.DecreasingStrict(x,y,b)]).solve(solver=solver)
        z = cp.intvar(1,5)
        assert cp.Model([cp.DecreasingStrict(z,b)]).solve(solver=solver)

    def test_circuit(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Circuit(x+2,2,0)]).solve(solver=solver)
        assert cp.Model([cp.Circuit(a,b)]).solve(solver=solver)

    def test_multicicruit(self, solver):
        c1 = cp.Circuit(cp.intvar(0,4, shape=5))
        c2 = cp.Circuit(cp.intvar(0,2, shape=3))
        assert cp.Model(c1 & c2).solve(solver=solver)

    def test_inverse(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert not cp.Model([cp.Inverse([x,y,x],[x,y,x])]).solve(solver=solver)
        assert cp.Model([cp.Inverse([a,b],[a,b])]).solve(solver=solver) # identity function

    def test_ite(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.IfThenElse(b,b&a,False)]).solve(solver=solver)
        pytest.raises(TypeError, cp.IfThenElse,a,b,0)
        pytest.raises(TypeError, cp.IfThenElse,1,x,y)

    def test_min(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Minimum([x,y]) == x]).solve(solver=solver)
        assert cp.Model([cp.Minimum([a,b | a]) == b]).solve(solver=solver)
        assert cp.Model([cp.Minimum([x,y,b]) == -2]).solve(solver=solver)

    def test_max(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Maximum([x,y]) == x]).solve(solver=solver)
        assert cp.Model([cp.Maximum([a,b | a]) == b]).solve(solver=solver)
        assert cp.Model([cp.Maximum([x,y,b]) == 2 ]).solve(solver=solver)

    def test_element(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Element([x,y],x) == x]).solve(solver=solver)
        assert cp.Model([cp.Element([a,b | a],x) == b]).solve(solver=solver)
        pytest.raises(TypeError,cp.Element,[x,y],b)
        assert cp.Model([cp.Element([y,a],x) == False]).solve(solver=solver)

    def test_xor(self, solver):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Xor([a,b,b])]).solve(solver=solver)
        pytest.raises(TypeError, cp.Xor, (x, b))
        pytest.raises(TypeError, cp.Xor, (x, y))

    def test_gcc(self, solver):
        x = cp.intvar(0, 1)
        z = cp.intvar(-8, 8)
        q = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        h = cp.intvar(-7, 7)
        v = cp.intvar(-7, 7)
        b = cp.boolvar()
        a = cp.boolvar()

        # type checks
        pytest.raises(TypeError, cp.GlobalCardinalityCount, [x,y], [x,False], [h,v])
        pytest.raises(TypeError, cp.GlobalCardinalityCount, [x,y], [z,b], [h,v])
        pytest.raises(TypeError, cp.GlobalCardinalityCount, [b,a], [a,b], [h,v])
        pytest.raises(TypeError, cp.GlobalCardinalityCount, [x, y], [h, v], [z, b])
        pytest.raises(TypeError, cp.GlobalCardinalityCount, [x, y], [x, h], [True, v])
        pytest.raises(TypeError, cp.GlobalCardinalityCount, [x, y], [x, h], [v, a])

        iv = cp.intvar(0,10, shape=3)
        SOLVERNAMES = [name for name, solver_cls in cp.SolverLookup.base_solvers() if solver_cls.supported()]
        for name in SOLVERNAMES:
            if name == "pysdd": continue
            try:
                assert cp.Model([cp.GlobalCardinalityCount(iv, [1,4], [1,1])]).solve(solver=name)
                # test closed version
                assert not cp.Model(cp.GlobalCardinalityCount(iv, [1,4], [0,0], closed=True)).solve(solver=name)
            except (NotImplementedError, NotSupportedError):
                pass

    def test_count(self, solver):
        x = cp.intvar(0, 1)
        z = cp.intvar(-8, 8)
        q = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()

        assert cp.Model([cp.Count([x,y],z) == 1]).solve(solver=solver)
        pytest.raises(TypeError, cp.Count, [x,y],[x,False])

    def test_among(self, solver):

        iv = cp.intvar(0,10, shape=3, name="x")

        for name, cls in cp.SolverLookup.base_solvers():
            if cls.supported() is False:
                print("Solver not supported: ", name)
                continue
            try:
                assert cp.Model([cp.Among(iv, [1,2]) == 3]).solve(solver=name)
                assert all(x.value() in [1,2] for x in iv)
                assert cp.Model([cp.Among(iv, [1,100]) > 2]).solve(solver=name)
                assert all(x.value() == 1 for x in iv)
            except (NotSupportedError, NotImplementedError):
                print("Solver not supported: ", name)
                continue


    def test_table(self, solver):
        iv = cp.intvar(-8,8,3)

        constraints = [cp.Table([iv[0], [iv[1], iv[2]]], [ (5, 2, 2)])] # not flatlist, should work
        model = cp.Model(constraints)
        assert model.solve(solver=solver)

        pytest.raises(TypeError, cp.Table, [iv[0], iv[1], iv[2], 5], [(5, 2, 2)])
        pytest.raises(TypeError, cp.Table, [iv[0], iv[1], iv[2], [5]], [(5, 2, 2)])
        pytest.raises(TypeError, cp.Table, [iv[0], iv[1], iv[2], ['a']], [(5, 2, 2)])

    def test_issue627(self, solver):
        for s, cls in cp.SolverLookup.base_solvers():
            if cls.supported():
                try:
                    # constant look-up
                    assert cp.Model([cp.boolvar() == cp.Element([0], 0)]).solve(solver=s)
                    # constant out-of-bounds look-up
                    assert not cp.Model([cp.boolvar() == cp.Element([0], 1)]).solve(solver=s)
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_issue_699(self, solver):
        x,y = cp.intvar(0,10, shape=2, name=tuple("xy"))
        assert cp.Model(cp.AllDifferentExcept0([x,0,y,0]).decompose()).solve(solver=solver)
        assert cp.Model(cp.AllDifferentExceptN([x,3,y,0], 3).decompose()).solve(solver=solver)
        assert cp.Model(cp.AllDifferentExceptN([x,3,y,0], [3,0]).decompose()).solve(solver=solver)



@pytest.mark.usefixtures("solver")
def test_issue801_expr_in_cumulative(solver):

    if solver in ("pysat", "pysdd", "pindakaas", "rc2"):
        pytest.skip(f"{solver} does not support integer variables")
    if solver == "cplex":
        pytest.skip(f"waiting for PR #769 to be merged.")

    start = cp.intvar(0,5,shape=3,name="start")
    dur = [1,2,3]
    end = cp.intvar(0,10,shape=3,name="end")
    bv = cp.boolvar(shape=3,name="bv")

    assert cp.Model(cp.NoOverlap(bv * start, dur, end)).solve(solver=solver)
    if solver != "pumpkin": # Pumpkin does not support variables as duration
        assert cp.Model(cp.NoOverlap(bv * start,bv * dur, end)).solve(solver=solver)
    assert cp.Model(cp.NoOverlap(bv * start, dur, bv * end)).solve(solver=solver)

    # also for cumulative
    assert cp.Model(cp.Cumulative(bv * start, dur, end,1, 3)).solve(solver=solver)
    assert cp.Model(cp.Cumulative(bv * start, dur, bv * end, 1, 3)).solve(solver=solver)
    if solver != "pumpkin": # Pumpkin does not support variables as duration, demand or capacity
        assert cp.Model(cp.Cumulative(bv * start,bv * dur, end, 1, 3)).solve(solver=solver)
        assert cp.Model(cp.Cumulative(bv * start, dur, end, bv * [2, 3, 4], 3 * bv[0])).solve(solver=solver)
        assert cp.Model(cp.Cumulative(bv * start, dur, end, 1, 3 * bv[0])).solve(solver=solver)
