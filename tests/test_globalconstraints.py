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

from utils import skip_on_missing_pblib, inclusive_range, lambda_assert


@skip_on_missing_pblib(skip_on_exception_only=True)
class TestGlobal:

    def setup_method(self):
        _BoolVarImpl.counter = 0
        _IntVarImpl.counter = 0

    def test_alldifferent(self):
        """Test all different constraint with a set of
        unit cases.
        """
        lb = 1
        start = 2
        nTests = 10
        for i in range(start, start + nTests):
            # construct the model vars = lb..i
            vars = cp.intvar(lb, i, i)

            # CONSTRAINTS
            constraint = [ cp.AllDifferent(vars) ]

            # MODEL Transformation to default solver specification
            model = cp.Model(constraint)

            # SOLVE
            if True:
                _ = model.solve()
                vals = [x.value() for x in vars]

                # ensure all different values
                assert len(vals) ==len(set(vals)), f"solver does provide solution validating given constraints."
    
    def test_alldifferent2(self):
        # test known input/outputs
        tuples = [
                  ((1,2,3), True),
                  ((1,2,1), False),
                  ((0,1,2), True),
                  ((2,0,3), True),
                  ((2,0,2), False),
                  ((0,0,2), False),
                 ]
        iv = cp.intvar(0,4, shape=3)
        c = cp.AllDifferent(iv)
        for (vals, oracle) in tuples:
            ret = cp.Model(c, iv == vals).solve()
            assert (ret == oracle), f"Mismatch solve for {vals,oracle}"
            # don't try this at home, forcibly overwrite variable values (so even when ret=false)
            for (var,val) in zip(iv,vals):
                var._value = val
            assert (c.value() == oracle), f"Wrong value function for {vals,oracle}"

    def test_not_alldifferent(self):
        # from fuzztester of Ruben Kindt, #143
        pos = cp.intvar(lb=0, ub=5, shape=3, name="positions")
        m = cp.Model()
        m += ~cp.AllDifferent(pos)
        assert m.solve("ortools")
        assert not cp.AllDifferent(pos).value()

    def test_alldifferent_except0(self):
        # test known input/outputs
        tuples = [
                  ((1,2,3), True),
                  ((1,2,1), False),
                  ((0,1,2), True),
                  ((2,0,3), True),
                  ((2,0,2), False),
                  ((0,0,2), True),
                 ]
        iv = cp.intvar(0,4, shape=3)
        c = cp.AllDifferentExcept0(iv)
        for (vals, oracle) in tuples:
            ret = cp.Model(c, iv == vals).solve()
            assert (ret == oracle), f"Mismatch solve for {vals,oracle}"
            # don't try this at home, forcibly overwrite variable values (so even when ret=false)
            for (var,val) in zip(iv,vals):
                var._value = val
            assert (c.value() == oracle), f"Wrong value function for {vals,oracle}"

        # and some more
        iv = cp.intvar(-8, 8, shape=3)
        assert cp.Model([cp.AllDifferentExcept0(iv)]).solve()
        assert cp.AllDifferentExcept0(iv).value()
        assert cp.Model([cp.AllDifferentExcept0(iv), iv == [0,0,1]]).solve()
        assert cp.AllDifferentExcept0(iv).value()

        #test with mixed types
        bv = cp.boolvar()
        assert cp.Model([cp.AllDifferentExcept0(iv[0], bv)]).solve()

    def test_alldifferent_except_n(self):
        # test known input/outputs
        tuples = [
            ((1, 2, 3), True),
            ((1, 2, 1), False),
            ((0, 1, 2), True),
            ((2, 0, 3), True),
            ((2, 0, 2), True),
            ((0, 0, 2), False),
        ]
        iv = cp.intvar(0, 4, shape=3)
        c = cp.AllDifferentExceptN(iv, 2)
        for (vals, oracle) in tuples:
            ret = cp.Model(c, iv == vals).solve()
            assert (ret == oracle), f"Mismatch solve for {vals, oracle}"
            # don't try this at home, forcibly overwrite variable values (so even when ret=false)
            for (var, val) in zip(iv, vals):
                var._value = val
            assert (c.value() == oracle), f"Wrong value function for {vals, oracle}"

        # and some more
        iv = cp.intvar(-8, 8, shape=3)
        assert cp.Model([cp.AllDifferentExceptN(iv,2)]).solve()
        assert cp.AllDifferentExceptN(iv,2).value()
        assert cp.Model([cp.AllDifferentExceptN(iv,7), iv == [7, 7, 1]]).solve()
        assert cp.AllDifferentExceptN(iv,7).value()

        # test with mixed types
        bv = cp.boolvar()
        assert cp.Model([cp.AllDifferentExceptN([iv[0], bv],4)]).solve()

        # test with list of n
        iv = cp.intvar(0, 4, shape=7)
        assert not cp.Model([cp.AllDifferentExceptN([iv], [7,8])]).solve()
        assert cp.Model([cp.AllDifferentExceptN([iv], [4, 1])]).solve()

    def test_not_alldifferentexcept0(self):
        iv = cp.intvar(-8, 8, shape=3)
        assert cp.Model([~cp.AllDifferentExcept0(iv)]).solve()
        assert not cp.AllDifferentExcept0(iv).value()
        assert not cp.Model([~cp.AllDifferentExcept0(iv), iv == [0, 0, 1]]).solve()

    def test_alldifferent_onearg(self):
        iv = cp.intvar(0,10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    assert cp.Model(cp.AllDifferent([iv])).solve(solver=s)
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_circuit(self):
        """
        Circuit constraint unit tests the hamiltonian circuit on a
        successor array. For example, if

            arr = [3, 0, 5, 4, 2, 1]

        then

            arr[0] = 3

        means that there is a directed edge from 0 -> 3.
        """
        # Test with domain (0,5)
        x = cp.intvar(0, 5, 6)
        constraints = [cp.Circuit(x)]
        model = cp.Model(constraints)
        assert model.solve()
        assert cp.Circuit(x).value()

        constraints = [cp.Circuit(x).decompose()]
        model = cp.Model(constraints)
        assert model.solve()
        assert cp.Circuit(x).value()

        # Test with domain (-2,7)
        x = cp.intvar(-2, 7, 6)
        circuit = cp.Circuit(x)
        model = cp.Model([circuit])
        assert model.solve()
        assert circuit.value()

        model = cp.Model([~circuit])
        assert model.solve()
        assert not circuit.value()

        # Test decomposition with domain (-2,7)
        constraints = [cp.Circuit(x).decompose()]
        model = cp.Model(constraints)
        assert model.solve()
        assert cp.Circuit(x).value()

        # Test with smaller domain (1,5)
        x = cp.intvar(1, 5, 5)
        circuit = cp.Circuit(x)
        model = cp.Model([circuit])
        assert not model.solve()
        assert not circuit.value()

        model = cp.Model([~circuit])
        assert model.solve()
        assert not circuit.value()

        # Test decomposition with domain (1,5)
        constraints = [cp.Circuit(x).decompose()]
        model = cp.Model(constraints)
        assert not model.solve()
        assert not cp.Circuit(x).value()


    def test_not_circuit(self):
        x = cp.intvar(lb=-1, ub=5, shape=4)
        circuit = cp.Circuit(x)
        model = cp.Model([~circuit, x == [1,2,3,0]])
        assert not model.solve()

        model = cp.Model([~circuit])
        assert model.solve()
        assert not circuit.value()
        assert not cp.Model([circuit, ~circuit]).solve()

        all_sols = set()
        not_all_sols = set()

        circuit_models = cp.Model(circuit).solveAll(display=lambda : all_sols.add(tuple(x.value())))
        not_circuit_models = cp.Model(~circuit).solveAll(display=lambda : not_all_sols.add(tuple(x.value())))

        total = cp.Model(x == x).solveAll()

        for sol in all_sols:
            for var,val in zip(x, sol):
                var._value = val
            assert circuit.value()

        for sol in not_all_sols:
            for var,val in zip(x, sol):
                var._value = val
            assert not circuit.value()

        assert total == len(all_sols) + len(not_all_sols)


    def test_inverse(self):
        # Arrays
        fwd = cp.intvar(0, 9, shape=10)
        rev = cp.intvar(0, 9, shape=10)

        # Constraints
        inv = cp.Inverse(fwd, rev)
        fix_fwd = (fwd == [9, 4, 7, 2, 1, 3, 8, 6, 0, 5])

        # Inverse of the above
        expected_inverse = [8, 4, 3, 5, 1, 9, 7, 2, 6, 0]

        # Test decomposed model:
        model = cp.Model(inv.decompose(), fix_fwd)
        assert model.solve()
        assert list(rev.value()) == expected_inverse

        # Not decomposed:
        model = cp.Model(inv, fix_fwd)
        assert model.solve()
        assert list(rev.value()) == expected_inverse

        # constraint can be used as value
        assert inv.value()

    def test_inverse_onearg(self):
        iv = cp.intvar(0,10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    assert cp.Model(cp.Inverse([iv], [0])).solve(solver=s)
                except (NotImplementedError, NotSupportedError):
                    pass


    def test_in_domain(self):
        iv = cp.intvar(-8, 8)
        iv_arr = cp.intvar(-8, 8, shape=5)

        # Test InDomain with constant list
        vals = [1, 5, 8, -4]
        model = cp.Model([cp.InDomain(iv, vals)])
        assert model.solve()
        assert iv.value() in vals

        # Test InDomain with empty list (should be unsat)
        model = cp.Model([cp.InDomain(iv, [])])
        assert not model.solve()

        # Test InDomain with singleton list
        model = cp.Model([cp.InDomain(iv, [1])])
        assert model.solve()
        assert iv.value() == 1

        # Test InDomain using minimum of array
        model = cp.Model([cp.InDomain(cp.min(iv_arr), vals)])
        assert model.solve()

        # Test InDomain with boolean var and constants
        bv = cp.boolvar()
        vals3 = [1, 5, 8, -4]
        model = cp.Model([cp.InDomain(bv, vals3)])
        assert model.solve()
        assert bv.value() in set(vals3)

    def test_lex_lesseq(self):
        from cpmpy import BoolVal
        X = cp.intvar(0, 3, shape=10)
        c1 = X[:-1] == 1
        Y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        c = cp.LexLessEq(X, Y)
        c2 = c != (BoolVal(True))
        m = cp.Model([c1, c2])
        assert m.solve()
        assert c2.value()
        assert not c.value()

        Y = cp.intvar(0, 0, shape=10)
        c = cp.LexLessEq(X, Y)
        m = cp.Model(c)
        assert m.solve("ortools")
        from cpmpy.expressions.utils import argval
        assert sum(argval(X)) == 0

    def test_lex_less(self):
        from cpmpy import BoolVal
        X = cp.intvar(0, 3, shape=10)
        c1 = X[:-1] == 1
        Y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        c = cp.LexLess(X, Y)
        c2 = c != (BoolVal(True))
        m = cp.Model([c1, c2])
        assert m.solve()
        assert c2.value()
        assert not c.value()

        Y = cp.intvar(0, 0, shape=10)
        c = cp.LexLess(X, Y)
        m = cp.Model(c)
        assert not m.solve("ortools")

        Z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        c = cp.LexLess(X, Z)
        m = cp.Model(c)
        assert m.solve("ortools")
        from cpmpy.expressions.utils import argval
        assert sum(argval(X)) == 0


    def test_lex_chain(self):
        from cpmpy import BoolVal
        X = cp.intvar(0, 3, shape=10)
        c1 = X[:-1] == 1
        Y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        c = cp.LexChainLess([X, Y])
        c2 = c != (BoolVal(True))
        m = cp.Model([c1, c2])
        assert m.solve()
        assert c2.value()
        assert not c.value()

        Y = cp.intvar(0, 0, shape=10)
        c = cp.LexChainLessEq([X, Y])
        m = cp.Model(c)
        assert m.solve("ortools")
        from cpmpy.expressions.utils import argval
        assert sum(argval(X)) == 0

        Z = cp.intvar(0, 1, shape=(4,2))
        c = cp.LexChainLess(Z)
        m = cp.Model(c)
        assert m.solve()
        assert sum(argval(Z[0])) == 0
        assert sum(argval(Z[1])) == 1
        assert argval(Z[1,0]) == 0
        assert sum(argval(Z[2])) == 1
        assert argval(Z[2,1]) == 0
        assert sum(argval(Z[3])) >= 1


    def test_indomain_onearg(self):

        iv = cp.intvar(0, 10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    assert cp.Model(cp.InDomain(iv, [2])).solve(solver=s)
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_table(self):
        iv = cp.intvar(-8,8,3)

        constraints = [cp.Table([iv[0], iv[1], iv[2]], [ (5, 2, 2)])]
        model = cp.Model(constraints)
        assert model.solve()

        model = cp.Model(constraints[0].decompose())
        assert model.solve()

        constraints = [cp.Table(iv, [[10, 8, 2], [5, 2, 2]])]
        model = cp.Model(constraints)
        assert model.solve()

        model = cp.Model(constraints[0].decompose())
        assert model.solve()

        assert cp.Table(iv, [[10, 8, 2], [5, 2, 2]]).value()
        assert not cp.Table(iv, [[10, 8, 2], [5, 3, 2]]).value()

        constraints = [cp.Table(iv, [[10, 8, 2], [5, 9, 2]])]
        model = cp.Model(constraints)
        assert not model.solve()

        constraints = [cp.Table(iv, [[10, 8, 2], [5, 9, 2]])]
        model = cp.Model(constraints[0].decompose())
        assert not model.solve()

    def test_table_value(self):
        """Test Table.value() with known assignments (and unassigned -> None)."""
        iv = cp.intvar(0, 10, shape=3)
        table = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        c = cp.Table(iv, table)
        # (assignment, expected value())
        cases = [
            ([1, 2, 3], True),
            ([4, 5, 6], True),
            ([7, 8, 9], True),
            ([1, 2, 4], False),
            ([0, 0, 0], False),
            ([1, None, 3], None),
            ([None, 2, 3], None),
        ]
        for vals, oracle in cases:
            for var, val in zip(iv, vals):
                var._value = val
            assert c.value() == oracle, f"Table.value() for {vals}"

    def test_negative_table(self):
        iv = cp.intvar(-8,8,3)

        constraints = [cp.NegativeTable([iv[0], iv[1], iv[2]], [ (5, 2, 2)])]
        model = cp.Model(constraints)
        assert model.solve()

        model = cp.Model(constraints[0].decompose())
        assert model.solve()

        constraints = [cp.NegativeTable(iv, [[10, 8, 2], [5, 2, 2]])]
        model = cp.Model(constraints)
        assert model.solve()

        model = cp.Model(constraints[0].decompose())
        assert model.solve()

        assert cp.NegativeTable(iv, [[10, 8, 2], [5, 2, 2]]).value()

        constraints = [~cp.NegativeTable(iv, [[10, 8, 2], [5, 2, 2]])]
        model = cp.Model(constraints)
        assert model.solve()
        assert not cp.NegativeTable(iv, [[10, 8, 2], [5, 2, 2]]).value()
        assert cp.Table(iv, [[10, 8, 2], [5, 2, 2]]).value()

        constraints = [cp.NegativeTable(iv, [[10, 8, 2], [5, 9, 2]])]
        model = cp.Model(constraints)
        assert model.solve()

        constraints = [cp.NegativeTable(iv, [[10, 8, 2], [5, 9, 2]])]
        model = cp.Model(constraints[0].decompose())
        assert model.solve()

        constraints = [cp.NegativeTable(iv, [[10, 8, 2], [5, 9, 2]]), cp.Table(iv, [[10, 8, 2], [5, 9, 2]])]
        model = cp.Model(constraints)
        assert not model.solve()

        constraints = [cp.NegativeTable(iv, [[10, 8, 2], [5, 9, 2]]), cp.Table(iv, [[10, 8, 2], [5, 9, 2]])]
        model = cp.Model(constraints[0].decompose())
        model += constraints[1].decompose()
        assert not model.solve()

    def test_negative_table_value(self):
        """Test NegativeTable.value() with known assignments (and unassigned -> None)."""
        iv = cp.intvar(0, 10, shape=3)
        table = [[1, 2, 3], [4, 5, 6]]
        c = cp.NegativeTable(iv, table)
        # (assignment, expected value(): True = row NOT in table)
        cases = [
            ([7, 8, 9], True),
            ([0, 0, 0], True),
            ([1, 2, 3], False),
            ([4, 5, 6], False),
            ([1, None, 3], None),
        ]
        for vals, oracle in cases:
            for var, val in zip(iv, vals):
                var._value = val
            assert c.value() == oracle, f"NegativeTable.value() for {vals}"

    def test_shorttable(self):
        iv = cp.intvar(-8,8,shape=3, name="x")

        solver = "choco" if cp.SolverLookup.lookup("choco").supported() else "ortools"

        cons = cp.ShortTable([iv[0], iv[1], iv[2]], [ (5, 2, 2)])
        model = cp.Model(cons)
        assert model.solve()

        model = cp.Model(cons.decompose())
        assert model.solve()

        short_cons = cp.ShortTable(iv, [[10, 8, 2], ['*', '*', 2]])
        model = cp.Model(short_cons)
        assert model.solve(solver=solver)

        model = cp.Model(short_cons.decompose())
        assert model.solve()

        assert short_cons.value()
        assert iv[-1].value() == 2
        assert not cp.ShortTable(iv, [[10, 8, 2], [STAR, STAR, 3]]).value()

        short_cons = cp.ShortTable(iv, [[10, 8, STAR], [STAR, 9, 2]])
        model = cp.Model(short_cons)
        assert not model.solve(solver=solver)

        short_cons = cp.ShortTable(iv, [[10, 8, STAR], [5, 9, STAR]])
        model = cp.Model(short_cons.decompose())
        assert not model.solve()

        # unconstrained
        true_cons = cp.ShortTable(iv, [[1,2,3],[STAR, STAR, STAR]])
        assert cp.Model(true_cons).solve(solver=solver)
        assert cp.Model(true_cons).solveAll(solver=solver) == 17 ** 3
        constraining, defining = true_cons.decompose() # should be True, []
        assert constraining[0]

    def test_shorttable_value(self):
        """Test ShortTable.value() with known assignments and STAR; unassigned -> None."""
        iv = cp.intvar(0, 10, shape=3)
        # table rows: [1,*,3] and [*,5,6]; so [1,x,3] and [y,5,6] match
        c = cp.ShortTable(iv, [[1, STAR, 3], [STAR, 5, 6]])
        cases = [
            ([1, 0, 3], True),
            ([1, 99, 3], True),
            ([0, 5, 6], True),
            ([99, 5, 6], True),
            ([2, 5, 6], True),   # matches [STAR, 5, 6]
            ([1, 5, 7], False),
            ([2, 4, 6], False),
            ([1, None, 3], None),
        ]
        for vals, oracle in cases:
            for var, val in zip(iv, vals):
                var._value = val
            assert c.value() == oracle, f"ShortTable.value() for {vals}"

    def test_table_accepts_ndarray(self):
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

    def test_table_onearg(self):

        iv = cp.intvar(0, 10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    assert cp.Model(cp.Table([iv], [[0]])).solve(solver=s)
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_regular(self):
        # test based on the example from XCSP3 specifications https://arxiv.org/pdf/1611.03398
        x = cp.intvar(0, 1, shape=7)

        transitions = [("a", 0, "a"), ("a", 1, "b"), ("b", 1, "c"), ("c", 0, "d"), ("d", 0, "d"), ("d", 1, "e"),
                       ("e", 0, "e")]
        start = "a"
        ends = ["e"]

        true_sols = set()
        false_sols = set()

        solutions = [(0,0,0,1,1,0,1), (0,0,1,1,0,0,1), (0,0,1,1,0,1,0), (0,1,1,0,0,0,1), (0,1,1,0,0,1,0),
                     (0,1,1,0,1,0,0), (1,1,0,0,0,0,1), (1,1,0,0,0,1,0), (1,1,0,0,1,0,0), (1,1,0,1,0,0,0)]

        true_model = cp.Model(cp.Regular(x, transitions, start, ends))
        false_model = cp.Model(~cp.Regular(x, transitions, start, ends))

        num_true = true_model.solveAll(display=lambda : true_sols.add(tuple(argvals(x))))
        num_false = false_model.solveAll(display=lambda : false_sols.add(tuple(argvals(x))))

        assert num_true == len(solutions)
        assert true_sols == set(solutions)

        assert num_true + num_false == 2**7
        assert len(true_sols & false_sols) == 0# no solutions can be in both


    def test_minimum(self):
        iv = cp.intvar(-8, 8, 3)
        constraints = [cp.Minimum(iv) + 9 == 8]
        model = cp.Model(constraints)
        assert model.solve()
        assert str(min(iv.value())) == '-1'

        _min, define = cp.Minimum(iv).decompose()
        model = cp.Model(_min == 4, define)

        assert model.solve()
        assert min(iv.value()) == 4
        assert cp.Minimum(iv).value() == 4


    def test_minimum_onearg(self):

        iv = cp.intvar(0, 10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    assert cp.Model(cp.min([iv]) == 0).solve(solver=s)
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_maximum(self):
        iv = cp.intvar(-8, 8, 3)
        constraints = [cp.Maximum(iv) + 9 <= 8]
        model = cp.Model(constraints)
        assert model.solve()
        assert max(iv.value()) <= -1

        _max, define = cp.Maximum(iv).decompose()
        model = cp.Model(_max == 4, define)

        assert model.solve()
        assert max(iv.value()) == 4
        assert cp.Maximum(iv).value() == 4

    def test_maximum_onearg(self):

        iv = cp.intvar(0, 10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    assert cp.Model(cp.max([iv]) == 0).solve(solver=s)
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_abs(self):
        from cpmpy.transformations.decompose_global import decompose_in_tree
        iv = cp.intvar(-8, 8, name="x")
        constraints = [cp.Abs(iv) + 9 <= 8]
        model = cp.Model(constraints)
        assert not model.solve()

        constraints = [cp.Abs(iv - 4) + 1 > 12]
        model = cp.Model(constraints)
        assert model.solve()
        assert cp.Model(decompose_in_tree(constraints)).solve()#test with decomposition

        _abs, define = cp.Abs(iv).decompose()
        model = cp.Model(_abs == 4, define)

        assert model.solve()
        assert iv.value() in [-4,4]
        assert _abs.value() == 4
        assert cp.Abs(iv).value() == 4
        assert model.solveAll() == 2

        pos = cp.intvar(0,8, name="x")
        constraints = [cp.Abs(pos) != 4]
        assert cp.Model(decompose_in_tree(constraints)).solveAll() == 8

        neg = cp.intvar(-8,0, name="x")
        constraints = [cp.Abs(neg) != 4]
        assert cp.Model(decompose_in_tree(constraints)).solveAll() == 8


    def test_element(self):
        # test 1-D
        iv = cp.intvar(-8, 8, 3, name="iv")
        idx = cp.intvar(-8, 8, name="idx")
        # test directly the constraint
        cons = cp.Element(iv,idx) == 8
        model = cp.Model(cons)
        assert model.solve()
        assert cons.value()
        assert iv.value()[idx.value()] == 8
        # test through __get_item__
        cons = iv[idx] == 8
        model = cp.Model(cons)
        assert model.solve()
        assert cons.value()
        assert iv.value()[idx.value()] == 8
        # test 2-D
        iv = cp.intvar(-8, 8, shape=(3, 3), name="iv")
        a,b = cp.intvar(0, 2, shape=2)
        cons = iv[a,b] == 8
        model = cp.Model(cons)
        assert model.solve()
        assert cons.value()
        assert iv.value()[a.value(), b.value()] == 8
        arr = cp.cpm_array([[1, 2, 3], [4, 5, 6]])
        cons = arr[a,b] == 1
        model = cp.Model(cons)
        assert model.solve()
        assert cons.value()
        assert arr[a.value(), b.value()] == 1
        # test optimization where 1 dim is index
        cons = iv[2, idx] == 8
        assert str(cons) == "[iv[2,0] iv[2,1] iv[2,2]][idx] == 8"
        cons = iv[idx, 2] == 8
        assert str(cons) == "[iv[0,2] iv[1,2] iv[2,2]][idx] == 8"

    def test_multid_1expr(self):

        x = cp.intvar(1,5, shape=(3,4,5),name="x")
        a,b = cp.intvar(0,2, shape=2, name=tuple("ab")) # idx is always safe

        expr = x[a,1,3]
        assert str(expr) == "[x[0,1,3] x[1,1,3] x[2,1,3]][a]"

        expr = x[1,a,3]
        assert str(expr) == "[x[1,0,3] x[1,1,3] x[1,2,3] x[1,3,3]][a]"

        expr = x[1,2,a]
        assert str(expr) == "[x[1,2,0] x[1,2,1] x[1,2,2] x[1,2,3] x[1,2,4]][a]"

    def test_element_onearg(self):

        iv = cp.intvar(0, 10)
        idx = cp.intvar(0,0)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    assert cp.Model(cp.Element([iv],idx) == 0).solve(solver=s)
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_element_index_dom_mismatched(self):
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

    def test_modulo(self):

        x, z = cp.intvar(-2,2, shape=2, name=["x","z"])
        y = cp.intvar(1,5, name="y")
        vars = [x,y,z]

        constraint = [x % y  == z]
        decomp = decompose_in_tree(constraint)

        all_sols = set()
        lin_all_sols = set()
        count = cp.Model(constraint).solveAll(solver="ortools", display=lambda: all_sols.add(tuple(argvals(vars))))
        decomp_count = cp.Model(decomp).solveAll(solver="ortools", display=lambda: lin_all_sols.add(tuple(argvals(vars))))

        assert all_sols == lin_all_sols# same on decision vars
        assert count ==decomp_count# same on all vars

    def test_div(self):
        x, z = cp.intvar(-2, 2, shape=2, name=["x", "z"])
        y = cp.intvar(1, 5, name="y")
        vars = [x, y, z]

        constraint = [x // y == z]
        decomp = decompose_in_tree(constraint)

        all_sols = set()
        decomp_sols = set()
        count = cp.Model(constraint).solveAll(solver="ortools", display=lambda: all_sols.add(tuple(argvals(vars))))
        decomp_count = cp.Model(decomp).solveAll(solver="ortools",
                                                display=lambda: decomp_sols.add(tuple(argvals(vars))))

        assert all_sols == decomp_sols# same on decision vars
        assert count == decomp_count# same on all vars

    def test_xor(self):
        bv = cp.boolvar(5)
        assert cp.Model(cp.Xor(bv)).solve()
        assert cp.Xor(bv).value()

    def test_xor_with_constants(self):

        bvs = cp.boolvar(shape=3)

        cases =[bvs.tolist() + [True],
                bvs.tolist() + [True, True],
                bvs.tolist() + [True, True, True],
                bvs.tolist() + [False],
                bvs.tolist() + [False, True],
                [True]]

        for args in cases:
            expr = cp.Xor(args)
            model = cp.Model(expr)

            assert model.solve()
            assert expr.value()

            # also check with decomposition
            model = cp.Model(expr.decompose())
            assert model.solve()
            assert expr.value()

        # edge case with False constants
        assert not cp.Model(cp.Xor([False, False])).solve()
        assert not cp.Model(cp.Xor([False, False, False])).solve()

    def test_ite_with_constants(self):
        x,y,z = cp.boolvar(shape=3)
        expr = cp.IfThenElse(True, y, z)
        assert cp.Model(expr).solve()
        assert expr.value()
        expr = cp.IfThenElse(False, y, z)
        assert cp.Model(expr).solve()

        expr = cp.IfThenElse(x, y, z)
        assert cp.Model(~expr).solve()
        assert not expr.value()
        x,y, z = x.value(), y.value(), z.value()
        assert (x and z) or (not x and y)



    def test_not_xor(self):
        bv = cp.boolvar(shape=5, name=tuple("abcde"))
        assert cp.Model(~cp.Xor(bv)).solve()
        assert not cp.Xor(bv).value()
        nbNotModels = cp.Model(~cp.Xor(bv)).solveAll(display=lambda_assert(lambda: not cp.Xor(bv).value()))
        nbModels = cp.Model(cp.Xor(bv)).solveAll(display=lambda_assert(lambda: cp.Xor(bv).value()))
        nbDecompModels = cp.Model(cp.Xor(bv).decompose()).solveAll(display=lambda_assert(lambda: cp.Xor(bv).value()))
        assert nbDecompModels == nbModels
        total = cp.Model(bv == bv).solveAll()
        assert str(total) == str(nbModels + nbNotModels)

    def test_minimax_python(self):
        from cpmpy import min,max
        iv = cp.intvar(1,9, 10)
        assert isinstance(min(iv), GlobalFunction)
        assert isinstance(max(iv), GlobalFunction)

    def test_minimax_cpm(self):
        iv = cp.intvar(1,9, 10)
        mi = cp.min(iv)
        ma = cp.max(iv)
        assert isinstance(mi, GlobalFunction)
        assert isinstance(ma, GlobalFunction)
        
        def solve_return(model):
            model.solve()
            return model.objective_value()
        assert  solve_return(cp.Model([], minimize=mi)) == 1
        assert  solve_return(cp.Model([], minimize=ma)) == 1
        assert  solve_return(cp.Model([], maximize=mi)) == 9
        assert  solve_return(cp.Model([], maximize=ma)) == 9

    def test_cumulative_deepcopy(self):
        import numpy
        m = cp.Model()
        start = cp.intvar(0, 10, 4, "start")
        duration = numpy.array([1, 2, 2, 1])
        end = start + duration
        demand = numpy.array([1, 1, 1, 1])
        capacity = 2
        m += cp.AllDifferent(start)
        m += cp.Cumulative(start, duration, end, demand, capacity)
        m2 = copy.deepcopy(m)  # should not throw an exception
        assert repr(m) == repr(m2)# should be True

    def test_cumulative_single_demand(self):
        import numpy
        m = cp.Model()
        start = cp.intvar(0, 10, 4, "start")
        duration = numpy.array([1, 2, 2, 1])
        end = start + duration
        demand = 1
        capacity = 1
        m += cp.Cumulative(start, duration, end, demand, capacity)
        assert m.solve()

    def test_cumulative_decomposition_capacity(self):
        import numpy as np

        # before merging #435 there was an issue with capacity constraint
        start = cp.intvar(0, 10, 4, "start")
        duration = [1, 2, 2, 1]
        end = cp.intvar(0, 10, shape=4, name="end")
        demand = 10 # tasks cannot be scheduled
        capacity = np.int64(5) # bug only happened with numpy ints
        cons = cp.Cumulative(start, duration, end, demand, capacity)
        assert not cp.Model(cons).solve()# this worked fine
        # also test decomposition
        assert not cp.Model(cons.decompose()).solve()# capacity was not taken into account and this failed

    def test_cumulative_nested_expressions(self):
        import numpy as np

        # before merging #435 there was an issue with capacity constraint
        start = cp.intvar(0, 10, 4, "start")
        duration = [1, 2, 2, 1]
        end = start + duration
        demand = 10 # tasks cannot be scheduled
        capacity = np.int64(5) # bug only happened with numpy ints
        cons = cp.Cumulative(start, duration, end, demand, capacity)
        assert not cp.Model(cons).solve()# this worked fine
        # also test decomposition
        assert not cp.Model(cons.decompose()).solve()# capacity was not taken into account and this failed

    @pytest.mark.skipif(not CPM_minizinc.supported(),
                        reason="Minizinc not installed")
    def test_cumulative_single_task_mzn(self):
        start = cp.intvar(0, 10, name="start")
        dur = 5
        end = cp.intvar(0, 10, name="end")
        demand = 2
        capacity = 10

        m = cp.Model()
        m += cp.Cumulative([start],[dur], [end],[demand], capacity)

        assert m.solve(solver="ortools")
        assert m.solve(solver="minizinc")

    @pytest.mark.skipif(not CPM_minizinc.supported(),
                        reason="Minizinc not installed")
    def test_cumulative_nested(self):
        start = cp.intvar(0, 10, name="start", shape=3)
        dur = [5,5,5]
        end = cp.intvar(0, 10, name="end", shape=3)
        demand = [5,5,9]
        capacity = 10
        bv = cp.boolvar()

        cons = cp.Cumulative(start, dur, end, demand, capacity)

        m = cp.Model(bv.implies(cons), start + dur != end)

        assert m.solve(solver="ortools")
        assert m.solve(solver="minizinc")

    @pytest.mark.skipif(not CPM_minizinc.supported(),
                        reason="Minizinc not installed")
    def test_negative_table_minizinc(self):
        """Test negative_table constraint with minizinc solver"""
        iv = cp.intvar(-8, 8, 3)

        # Test basic negative_table
        constraints = [cp.NegativeTable([iv[0], iv[1], iv[2]], [(5, 2, 2)])]
        model = cp.Model(constraints)
        assert model.solve(solver="minizinc")
        # Verify the solution doesn't match the forbidden tuple
        assert (iv[0].value(), iv[1].value(), iv[2].value()) != (5, 2, 2)

        # Test with multiple forbidden tuples
        constraints = [cp.NegativeTable(iv, [[10, 8, 2], [5, 2, 2]])]
        model = cp.Model(constraints)
        assert model.solve(solver="minizinc")
        sol = tuple(iv.value())
        assert sol not in [(10, 8, 2), (5, 2, 2)]

        # Test that negative_table and table are contradictory when they have the same tuples
        constraints = [cp.NegativeTable(iv, [[10, 8, 2], [5, 9, 2]]), 
                       cp.Table(iv, [[10, 8, 2], [5, 9, 2]])]
        model = cp.Model(constraints)
        assert not model.solve(solver="minizinc")



    def test_cumulative_no_np(self):
        start = cp.intvar(0, 10, 4, "start")
        duration = (1, 2, 2, 1) # smt weird such as a tuple
        end = [cp.intvar(0,20, name=f"end[{i}]") for i in range(4)] # force smt weird
        demand = 1
        capacity = 1
        cons = cp.Cumulative(start, duration, end, demand, capacity)
        assert cp.Model(cons).solve()
        assert cons.value()
        # also test decomposition
        assert cp.Model(cons.decompose()).solve()
        assert cons.value()

    def test_cumulative_no_np2(self):
        start = cp.intvar(0, 10, 4, "start")
        duration = (1, 2, 2, 1) # smt weird such as a tuple
        end = [cp.intvar(0,20, name=f"end[{i}]") for i in range(4)] # force smt weird
        demand = [1,1,1,1]
        capacity = 1
        cons = cp.Cumulative(start, duration, end, demand, capacity)
        assert cp.Model(cons).solve()
        assert cons.value()
        # also test decomposition
        assert cp.Model(cons.decompose()).solve()
        assert cons.value()

    def test_cumulative_negative_dur(self):
        start = cp.intvar(0,10,shape=3, name="start")
        dur = cp.intvar(-5,-1, shape=3, name="dur")
        end = cp.intvar(-5,10, shape=3, name="end")
        bv = cp.boolvar()

        expr = cp.Cumulative(start, dur, end, 1, 5)
        assert not cp.Model(expr).solve()
        assert cp.Model(bv == expr).solve()
        assert not bv.value()



    def test_ite(self):
        x = cp.intvar(0, 5, shape=3, name="x")
        iter = cp.IfThenElse(x[0] > 2, x[1] > x[2], x[1] == x[2])
        constraints = [iter]
        assert cp.Model(constraints).solve()

        constraints = [iter, x == [0, 4, 4]]
        assert cp.Model(constraints).solve()

        constraints = [iter, x == [4, 4, 3]]
        assert cp.Model(constraints).solve()

        constraints = [iter, x == [4, 4, 4]]
        assert not cp.Model(constraints).solve()

        constraints = [iter, x == [1, 3, 2]]
        assert not cp.Model(constraints).solve()

    def test_global_cardinality_count(self):
        iv = cp.intvar(-8, 8, shape=5)
        occ = cp.intvar(0, len(iv), shape=3)
        val = [1, 4, 5]
        assert cp.Model([cp.GlobalCardinalityCount(iv, val, occ), cp.AllDifferent(val)]).solve()
        assert cp.GlobalCardinalityCount(iv, val, occ).value()
        assert all(cp.Count(iv, val[i]).value() == occ[i].value() for i in range(len(val)))
        val = [1, 4, 5]
        assert cp.Model([cp.GlobalCardinalityCount(iv, val, occ)]).solve()
        assert cp.GlobalCardinalityCount(iv, val, occ).value()
        assert all(cp.Count(iv, val[i]).value() == occ[i].value() for i in range(len(val)))
        occ = [2, 3, 0]
        assert cp.Model([cp.GlobalCardinalityCount(iv, val, occ)]).solve()
        assert cp.GlobalCardinalityCount(iv, val, occ).value()
        assert all(cp.Count(iv, val[i]).value() == occ[i] for i in range(len(val)))
        assert cp.GlobalCardinalityCount([iv[0],iv[2],iv[1],iv[4],iv[3]], val, occ).value()

    def test_not_global_cardinality_count(self):
        iv = cp.intvar(-8, 8, shape=5)
        val = [0,1,2]
        occ = cp.intvar(0, len(iv), shape=3)
        assert cp.Model([~cp.GlobalCardinalityCount(iv, val, occ), cp.AllDifferent(val)]).solve()
        assert ~cp.GlobalCardinalityCount(iv, val, occ).value()
        assert not all(cp.Count(iv, val[i]).value() == occ[i].value() for i in range(len(val)))
        val = [1, 4, 5]
        assert cp.Model([~cp.GlobalCardinalityCount(iv, val, occ)]).solve()
        assert ~cp.GlobalCardinalityCount(iv, val, occ).value()
        assert not all(cp.Count(iv, val[i]).value() == occ[i].value() for i in range(len(val)))
        occ = [2, 3, 0]
        assert cp.Model([~cp.GlobalCardinalityCount(iv, val, occ)]).solve()
        assert ~cp.GlobalCardinalityCount(iv, val, occ).value()
        assert not all(cp.Count(iv, val[i]).value() == occ[i] for i in range(len(val)))
        assert ~cp.GlobalCardinalityCount([iv[0],iv[2],iv[1],iv[4],iv[3]], val, occ).value()

    def test_gcc_onearg(self):
        iv = cp.intvar(0, 10)
        for s, cls in cp.SolverLookup.base_solvers():
            if cls.supported():
                try:
                    assert cp.Model(cp.GlobalCardinalityCount([iv], [3],[1])).solve(solver=s)
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_count(self):
        iv = cp.intvar(-8, 8, shape=3)
        assert cp.Model([iv[0] == 0, iv[1] != 1, iv[2] != 2, cp.Count(iv, 0) == 3]).solve()
        assert str(iv.value()) =='[0 0 0]'
        x = cp.intvar(-8,8)
        y = cp.intvar(0,5)
        assert cp.Model(cp.Count(iv, x) == y).solve()
        assert str(cp.Count(iv, x).value()) == str(y.value())

        assert cp.Model(cp.Count(iv, x) != y).solve()
        assert cp.Model(cp.Count(iv, x) >= y).solve()
        assert cp.Model(cp.Count(iv, x) <= y).solve()
        assert cp.Model(cp.Count(iv, x) < y).solve()
        assert cp.Model(cp.Count(iv, x) > y).solve()

        assert cp.Model(cp.Count([iv[0],iv[2],iv[1]], x) > y).solve()

    def test_count_onearg(self):

        iv = cp.intvar(0, 10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    assert cp.Model(cp.Count([iv], 1) == 0).solve(solver=s)
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_nvalue(self):

        iv = cp.intvar(-8, 8, shape=3)
        cnt = cp.intvar(0,10)

        assert not cp.Model(cp.all(iv == 1), cp.NValue(iv) > 1).solve()
        assert cp.Model(cp.all(iv == 1), cp.NValue(iv) > cnt).solve()
        assert len(set(iv.value())) > cnt.value()

        assert cp.Model(cp.NValue(iv) != cnt).solve()
        assert cp.Model(cp.NValue(iv) >= cnt).solve()
        assert cp.Model(cp.NValue(iv) <= cnt).solve()
        assert cp.Model(cp.NValue(iv) < cnt).solve()
        assert cp.Model(cp.NValue(iv) > cnt).solve()

        # test nested
        bv = cp.boolvar()
        cons = bv == (cp.NValue(iv) <= 2)

        def check_true():
            assert cons.value()
        cp.Model(cons).solveAll(display=check_true)

        # test not contiguous
        iv = cp.intvar(0, 10, shape=(3, 3))
        assert cp.Model([cp.NValue(i) == 3 for i in iv.T]).solve()
        
    def test_nvalue_except(self):

        iv = cp.intvar(-8, 8, shape=3)
        cnt = cp.intvar(0, 10)


        assert not cp.Model(cp.all(iv == 1), cp.NValueExcept(iv, 6) > 1).solve()
        assert cp.Model(cp.NValueExcept(iv, 10) > 1).solve()
        assert cp.Model(cp.all(iv == 1), cp.NValueExcept(iv, 1) == 0).solve()
        assert cp.Model(cp.all(iv == 1), cp.NValueExcept(iv, 6) > cnt).solve()
        assert len(set(iv.value())) > cnt.value()

        val = 6
        assert cp.Model(cp.NValueExcept(iv, val) != cnt).solve()
        assert cp.Model(cp.NValueExcept(iv, val) >= cnt).solve()
        assert cp.Model(cp.NValueExcept(iv, val) <= cnt).solve()
        assert cp.Model(cp.NValueExcept(iv, val) < cnt).solve()
        assert cp.Model(cp.NValueExcept(iv, val) > cnt).solve()

        # test nested
        bv = cp.boolvar()
        cons = bv == (cp.NValueExcept(iv, val) <= 2)

        def check_true():
            assert cons.value()

        cp.Model(cons).solveAll(display=check_true)

        # test not contiguous
        iv = cp.intvar(0, 10, shape=(3, 3))
        assert cp.Model([cp.NValueExcept(i, val) == 3 for i in iv.T]).solve()


    @pytest.mark.skipif(not CPM_minizinc.supported(),
                        reason="Minizinc not installed")
    def test_nvalue_minizinc(self):
        iv = cp.intvar(-8, 8, shape=3)
        cnt = cp.intvar(0, 10)

        assert not cp.Model(cp.all(iv == 1), cp.NValue(iv) > 1).solve('minizinc')
        assert cp.Model(cp.all(iv == 1), cp.NValue(iv) > cnt).solve('minizinc')
        assert len(set(iv.value())) > cnt.value()

        assert cp.Model(cp.NValue(iv) != cnt).solve('minizinc')
        assert cp.Model(cp.NValue(iv) >= cnt).solve('minizinc')
        assert cp.Model(cp.NValue(iv) <= cnt).solve('minizinc')
        assert cp.Model(cp.NValue(iv) < cnt).solve('minizinc')
        assert cp.Model(cp.NValue(iv) > cnt).solve('minizinc')

        # test nested
        bv = cp.boolvar()
        cons = bv == (cp.NValue(iv) <= 2)

        def check_true():
            assert cons.value()

        cp.Model(cons).solveAll(solver='minizinc')


    def test_precedence(self):
        iv = cp.intvar(0,5, shape=6, name="x")

        cons = cp.Precedence(iv, [0,2,1])
        assert cp.Model([cons, iv == [5,0,2,0,0,1]]).solve()
        assert cons.value()
        assert cp.Model([cons, iv == [0,0,0,0,0,0]]).solve()
        assert cons.value()
        assert not cp.Model([cons, iv == [0,1,2,0,0,0]]).solve()

        cons = cp.Precedence([iv[0], iv[1], 4], [0, 1, 2]) # python list in stead of cpm_array
        assert cp.Model([cons]).solve()

        # Check bug fix pull request #742
        # - ensure first constraint from paper is satisfied
        cons = cp.Precedence(iv, [0, 1, 2])
        assert not cp.Model([cons, (iv[0] == 1) | (iv[0] == 2)]).solve()


    def test_no_overlap(self):
        start = cp.intvar(0,5, shape=3)
        end = cp.intvar(0,5, shape=3)
        cons = cp.NoOverlap(start, [2,1,1], end)
        assert cp.Model(cons).solve()
        assert cons.value()
        assert cp.Model(cons.decompose()).solve()
        assert cons.value()

        def check_val():
            assert cons.value() is False

        cp.Model(~cons).solveAll(display=check_val)

class TestBounds:
    def test_bounds_minimum(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        z = cp.intvar(1, 9)
        expr = cp.Minimum([x,y,z])
        lb,ub = expr.get_bounds()
        assert lb ==-8
        assert ub ==-1
        assert not cp.Model(expr<lb).solve()
        assert not cp.Model(expr>ub).solve()


    def test_bounds_maximum(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        z = cp.intvar(1, 9)
        expr = cp.Maximum([x,y,z])
        lb,ub = expr.get_bounds()
        assert lb ==1
        assert ub ==9
        assert not cp.Model(expr<lb).solve()
        assert not cp.Model(expr>ub).solve()

    def test_bounds_abs(self):
        x = cp.intvar(-8, 5)
        y = cp.intvar(-7, -2)
        z = cp.intvar(1, 9)
        for var,test_lb,test_ub in [(x,0,8),(y,2,7),(z,1,9)]:
            lb, ub = cp.Abs(var).get_bounds()
            assert test_lb ==lb
            assert test_ub ==ub

    def test_bounds_div(self):
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

    def test_bounds_mod(self):
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

    def test_bounds_pow(self):
        x = cp.intvar(-8, 5)
        op = cp.Power(x,3)
        lb, ub = op.get_bounds()
        assert lb ==-8 ** 3
        assert ub ==5 ** 3

        op = cp.Power(x, 4)
        lb, ub = op.get_bounds()
        assert lb == 5 ** 4
        assert ub == 8 ** 4

    def test_bounds_element(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        z = cp.intvar(1, 9)
        expr = cp.Element([x, y, z],z)
        lb, ub = expr.get_bounds()
        assert lb ==-8
        assert ub ==9
        assert not cp.Model(expr < lb).solve()
        assert not cp.Model(expr > ub).solve()

    def test_incomplete_func(self):
        # element constraint
        arr = cp.cpm_array([1,2,3])
        i = cp.intvar(0,5,name="i")
        p = cp.boolvar()

        cons = (arr[i] == 1).implies(p)
        m = cp.Model([cons, i == 5])
        assert m.solve()
        assert cons.value()

        # div constraint
        a,b = cp.intvar(1,2,shape=2)
        cons = (42 // (a - b)) >= 3
        m = cp.Model([p.implies(cons), a == b])
        if cp.SolverLookup.lookup("z3").supported():
            assert m.solve(solver="z3")# ortools does not support divisor spanning 0 work here
            pytest.raises(IncompleteFunctionError, cons.value)
            assert not argval(cons)

        # mayhem
        cons = (arr[10 // (a - b)] == 1).implies(p)
        m = cp.Model([cons, a == b])
        if cp.SolverLookup.lookup("z3").supported():
            assert m.solve(solver="z3")
            assert cons.value()

    def test_bounds_count(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        z = cp.intvar(1, 9)
        a = cp.intvar(1, 9)
        expr = cp.Count([x, y, z], a)
        lb, ub = expr.get_bounds()
        assert lb ==0
        assert ub ==3
        assert not cp.Model(expr < lb).solve()
        assert not cp.Model(expr > ub).solve()

    def test_bounds_xor(self):
        # just one case of a Boolean global constraint
        expr = cp.Xor(cp.boolvar(3))
        assert expr.get_bounds() ==(0,1)

@skip_on_missing_pblib(skip_on_exception_only=True)
class TestTypeChecks:
    def test_all_diff(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.AllDifferent(x,y)]).solve()
        assert cp.Model([cp.AllDifferent(a,b)]).solve()
        assert cp.Model([cp.AllDifferent(x,y,b)]).solve()

    def test_all_diff_ex0(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.AllDifferentExcept0(x,y)]).solve()
        assert cp.Model([cp.AllDifferentExcept0(a,b)]).solve()
        #self.assertTrue(cp.Model([cp.AllDifferentExcept0(x,y,b)]).solve())

    def test_all_equal(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.AllEqual(x,y,-1)]).solve()
        assert cp.Model([cp.AllEqual(a,b,False, a | b)]).solve()
        assert not cp.Model([cp.AllEqual(x,y,b)]).solve()

    def test_all_equal_exceptn(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.AllEqualExceptN([x,y,-1],211)]).solve()
        assert cp.Model([cp.AllEqualExceptN([x,y,-1,4],4)]).solve()
        assert cp.Model([cp.AllEqualExceptN([x,y,-1,4],-1)]).solve()
        assert cp.Model([cp.AllEqualExceptN([a,b,False, a | b], 4)]).solve()
        assert cp.Model([cp.AllEqualExceptN([a,b,False, a | b], 0)]).solve()
        assert cp.Model([cp.AllEqualExceptN([a,b,False, a | b, y], -1)]).solve()

        # test with list of n
        iv = cp.intvar(0, 4, shape=7)
        assert not cp.Model([cp.AllEqualExceptN([iv], [7,8]), iv[0] != iv[1]]).solve()
        assert cp.Model([cp.AllEqualExceptN([iv], [4, 1]), iv[0] != iv[1]]).solve()

    def test_not_all_equal_exceptn(self):
        x = cp.intvar(lb=0, ub=3, shape=3)
        n = 2
        constr = cp.AllEqualExceptN(x,n)

        model = cp.Model([~constr, x == [1, 2, 1]])
        assert not model.solve()

        model = cp.Model([~constr])
        assert model.solve()
        assert not constr.value()

        assert not cp.Model([constr, ~constr]).solve()

        all_sols = set()
        not_all_sols = set()

        circuit_models = cp.Model(constr).solveAll(display=lambda: all_sols.add(tuple(x.value())))
        not_circuit_models = cp.Model(~constr).solveAll(display=lambda: not_all_sols.add(tuple(x.value())))

        total = cp.Model(x == x).solveAll()

        for sol in all_sols:
            for var, val in zip(x, sol):
                var._value = val
            assert constr.value()

        for sol in not_all_sols:
            for var, val in zip(x, sol):
                var._value = val
            assert not constr.value()

        assert total == len(all_sols) + len(not_all_sols)


    def test_increasing(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Increasing(x,y)]).solve()
        assert cp.Model([cp.Increasing(a,b)]).solve()
        assert cp.Model([cp.Increasing(x,y,b)]).solve()
        z = cp.intvar(2,5)
        assert not cp.Model([cp.Increasing(z,b)]).solve()

    def test_decreasing(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Decreasing(x,y)]).solve()
        assert cp.Model([cp.Decreasing(a,b)]).solve()
        assert not cp.Model([cp.Decreasing(x,y,b)]).solve()
        z = cp.intvar(2,5)
        assert cp.Model([cp.Decreasing(z,b)]).solve()

    def test_increasing_strict(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.IncreasingStrict(x,y)]).solve()
        assert cp.Model([cp.IncreasingStrict(a,b)]).solve()
        assert cp.Model([cp.IncreasingStrict(x,y,b)]).solve()
        z = cp.intvar(1,5)
        assert not cp.Model([cp.IncreasingStrict(z,b)]).solve()

    def test_decreasing_strict(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, 0)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.DecreasingStrict(x,y)]).solve()
        assert cp.Model([cp.DecreasingStrict(a,b)]).solve()
        assert not cp.Model([cp.DecreasingStrict(x,y,b)]).solve()
        z = cp.intvar(1,5)
        assert cp.Model([cp.DecreasingStrict(z,b)]).solve()

    def test_circuit(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Circuit(x+2,2,0)]).solve()
        assert cp.Model([cp.Circuit(a,b)]).solve()

    def test_multicicruit(self):
        c1 = cp.Circuit(cp.intvar(0,4, shape=5))
        c2 = cp.Circuit(cp.intvar(0,2, shape=3))
        assert cp.Model(c1 & c2).solve()


    def test_inverse(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert not cp.Model([cp.Inverse([x,y,x],[x,y,x])]).solve()
        assert cp.Model([cp.Inverse([a,b],[a,b])]).solve() # identity function

    def test_ite(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.IfThenElse(b,b&a,False)]).solve()
        pytest.raises(TypeError, cp.IfThenElse,a,b,0)
        pytest.raises(TypeError, cp.IfThenElse,1,x,y)

    def test_min(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Minimum([x,y]) == x]).solve()
        assert cp.Model([cp.Minimum([a,b | a]) == b]).solve()
        assert cp.Model([cp.Minimum([x,y,b]) == -2]).solve()

    def test_max(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Maximum([x,y]) == x]).solve()
        assert cp.Model([cp.Maximum([a,b | a]) == b]).solve()
        assert cp.Model([cp.Maximum([x,y,b]) == 2 ]).solve()

    def test_element(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Element([x,y],x) == x]).solve()
        assert cp.Model([cp.Element([a,b | a],x) == b]).solve()
        pytest.raises(TypeError,cp.Element,[x,y],b)
        assert cp.Model([cp.Element([y,a],x) == False]).solve()

    def test_xor(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        assert cp.Model([cp.Xor([a,b,b])]).solve()
        pytest.raises(TypeError, cp.Xor, (x, b))
        pytest.raises(TypeError, cp.Xor, (x, y))

    def test_gcc(self):
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
        SOLVERNAMES = [name for name, solver in cp.SolverLookup.base_solvers() if solver.supported()]
        for name in SOLVERNAMES:
            if name == "pysdd": continue
            try:
                assert cp.Model([cp.GlobalCardinalityCount(iv, [1,4], [1,1])]).solve(solver=name)
                # test closed version
                assert not cp.Model(cp.GlobalCardinalityCount(iv, [1,4], [0,0], closed=True)).solve(solver=name)
            except (NotImplementedError, NotSupportedError):
                pass

    def test_count(self):
        x = cp.intvar(0, 1)
        z = cp.intvar(-8, 8)
        q = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()

        assert cp.Model([cp.Count([x,y],z) == 1]).solve()
        pytest.raises(TypeError, cp.Count, [x,y],[x,False])

    def test_among(self):

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


    def test_table(self):
        iv = cp.intvar(-8,8,3)

        constraints = [cp.Table([iv[0], [iv[1], iv[2]]], [ (5, 2, 2)])] # not flatlist, should work
        model = cp.Model(constraints)
        assert model.solve()

        pytest.raises(TypeError, cp.Table, [iv[0], iv[1], iv[2], 5], [(5, 2, 2)])
        pytest.raises(TypeError, cp.Table, [iv[0], iv[1], iv[2], [5]], [(5, 2, 2)])
        pytest.raises(TypeError, cp.Table, [iv[0], iv[1], iv[2], ['a']], [(5, 2, 2)])

    def test_issue627(self):
        for s, cls in cp.SolverLookup.base_solvers():
            if cls.supported():
                try:
                    # constant look-up
                    assert cp.Model([cp.boolvar() == cp.Element([0], 0)]).solve(solver=s)
                    # constant out-of-bounds look-up
                    assert not cp.Model([cp.boolvar() == cp.Element([0], 1)]).solve(solver=s)
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_issue_699(self):
        x,y = cp.intvar(0,10, shape=2, name=tuple("xy"))
        assert cp.Model(cp.AllDifferentExcept0([x,0,y,0]).decompose()).solve()
        assert cp.Model(cp.AllDifferentExceptN([x,3,y,0], 3).decompose()).solve()
        assert cp.Model(cp.AllDifferentExceptN([x,3,y,0], [3,0]).decompose()).solve()



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
