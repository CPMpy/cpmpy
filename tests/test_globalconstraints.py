import copy
import unittest

import pytest

import cpmpy as cp
from cpmpy.expressions.globalfunctions import GlobalFunction
from cpmpy.exceptions import TypeError, NotSupportedError
from cpmpy.solvers import CPM_minizinc


class TestGlobal(unittest.TestCase):
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
                self.assertEqual(len(vals),len(set(vals)), msg=f"solver does provide solution validating given constraints.")
    
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
        self.assertTrue(m.solve("ortools"))
        self.assertFalse(cp.AllDifferent(pos).value())

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
        self.assertTrue(cp.Model([cp.AllDifferentExcept0(iv)]).solve())
        self.assertTrue(cp.AllDifferentExcept0(iv).value())
        self.assertTrue(cp.Model([cp.AllDifferentExcept0(iv), iv == [0,0,1]]).solve())
        self.assertTrue(cp.AllDifferentExcept0(iv).value())

        #test with mixed types
        bv = cp.boolvar()
        self.assertTrue(cp.Model([cp.AllDifferentExcept0(iv[0], bv)]).solve())

    def test_not_alldifferentexcept0(self):
        iv = cp.intvar(-8, 8, shape=3)
        self.assertTrue(cp.Model([~cp.AllDifferentExcept0(iv)]).solve())
        self.assertFalse(cp.AllDifferentExcept0(iv).value())
        self.assertFalse(cp.Model([~cp.AllDifferentExcept0(iv), iv == [0, 0, 1]]).solve())

    def test_alldifferent_onearg(self):
        iv = cp.intvar(0,10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    self.assertTrue(cp.Model(cp.AllDifferent([iv])).solve(solver=s))
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
        x = cp.intvar(0, 5, 6)
        constraints = [cp.Circuit(x)]
        model = cp.Model(constraints)

        self.assertTrue(model.solve())
        self.assertTrue(cp.Circuit(x).value())

        constraints = [cp.Circuit(x).decompose()]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())
        self.assertTrue(cp.Circuit(x).value())


    def test_not_circuit(self):
        x = cp.intvar(lb=0, ub=2, shape=3)
        circuit = cp.Circuit(x)

        model = cp.Model([~circuit, x == [1,2,0]])
        self.assertFalse(model.solve())

        model = cp.Model([~circuit])
        self.assertTrue(model.solve())
        self.assertFalse(circuit.value())

        self.assertFalse(cp.Model([circuit, ~circuit]).solve())

        circuit_sols = set()
        not_circuit_sols = set()

        circuit_models = cp.Model(circuit).solveAll(display=lambda : circuit_sols.add(tuple(x.value())))
        not_circuit_models = cp.Model(~circuit).solveAll(display=lambda : not_circuit_sols.add(tuple(x.value())))

        total = cp.Model(x == x).solveAll()

        for sol in circuit_sols:
            for var,val in zip(x, sol):
                var._value = val
            self.assertTrue(circuit.value())

        for sol in not_circuit_sols:
            for var,val in zip(x, sol):
                var._value = val
            self.assertFalse(circuit.value())

        self.assertEqual(total, len(circuit_sols) + len(not_circuit_sols))


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
        self.assertTrue(model.solve())
        self.assertEqual(list(rev.value()), expected_inverse)

        # Not decomposed:
        model = cp.Model(inv, fix_fwd)
        self.assertTrue(model.solve())
        self.assertEqual(list(rev.value()), expected_inverse)

        # constraint can be used as value
        self.assertTrue(inv.value())

    def test_inverse_onearg(self):
        iv = cp.intvar(0,10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    self.assertTrue(cp.Model(cp.Inverse([iv], [0])).solve(solver=s))
                except (NotImplementedError, NotSupportedError):
                    pass


    def test_InDomain(self):
        iv = cp.intvar(-8, 8)
        iv_arr = cp.intvar(-8, 8, shape=5)
        cons = [cp.InDomain(iv, iv_arr)]
        model = cp.Model(cons)
        self.assertTrue(model.solve())
        self.assertIn(iv.value(), iv_arr.value())
        vals = [1, 5, 8, -4]
        cons = [cp.InDomain(iv, vals)]
        model = cp.Model(cons)
        self.assertTrue(model.solve())
        self.assertIn(iv.value(), vals)
        cons = [cp.InDomain(iv, [])]
        model = cp.Model(cons)
        self.assertFalse(model.solve())
        cons = [cp.InDomain(iv, [1])]
        model = cp.Model(cons)
        self.assertTrue(model.solve())
        self.assertEqual(iv.value(),1)
        cons = cp.InDomain(min(iv_arr), vals)
        model = cp.Model(cons)
        self.assertTrue(model.solve())
        iv2 = cp.intvar(-8, 8)
        vals = [1, 5, 8, -4, iv2]
        cons = [cp.InDomain(iv, vals)]
        model = cp.Model(cons)
        self.assertTrue(model.solve())
        self.assertIn(iv.value(), vals)

    def test_indomain_onearg(self):

        iv = cp.intvar(0, 10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    self.assertTrue(cp.Model(cp.InDomain(iv, [2])).solve(solver=s))
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_table(self):
        iv = cp.intvar(-8,8,3)

        constraints = [cp.Table([iv[0], iv[1], iv[2]], [ (5, 2, 2)])]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())

        model = cp.Model(constraints[0].decompose())
        self.assertTrue(model.solve())

        constraints = [cp.Table(iv, [[10, 8, 2], [5, 2, 2]])]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())

        model = cp.Model(constraints[0].decompose())
        self.assertTrue(model.solve())

        self.assertTrue(cp.Table(iv, [[10, 8, 2], [5, 2, 2]]).value())
        self.assertFalse(cp.Table(iv, [[10, 8, 2], [5, 3, 2]]).value())

        constraints = [cp.Table(iv, [[10, 8, 2], [5, 9, 2]])]
        model = cp.Model(constraints)
        self.assertFalse(model.solve())

        constraints = [cp.Table(iv, [[10, 8, 2], [5, 9, 2]])]
        model = cp.Model(constraints[0].decompose())
        self.assertFalse(model.solve())

    def test_table_onearg(self):

        iv = cp.intvar(0, 10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    self.assertTrue(cp.Model(cp.Table([iv], [[0]])).solve(solver=s))
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_minimum(self):
        iv = cp.intvar(-8, 8, 3)
        constraints = [cp.Minimum(iv) + 9 == 8]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())
        self.assertEqual(str(min(iv.value())), '-1')

        model = cp.Model(cp.Minimum(iv).decompose_comparison('==', 4))
        self.assertTrue(model.solve())
        self.assertEqual(str(min(iv.value())), '4')

    def test_minimum_onearg(self):

        iv = cp.intvar(0, 10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    self.assertTrue(cp.Model(cp.min([iv]) == 0).solve(solver=s))
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_maximum(self):
        iv = cp.intvar(-8, 8, 3)
        constraints = [cp.Maximum(iv) + 9 <= 8]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())
        self.assertTrue(max(iv.value()) <= -1)

        model = cp.Model(cp.Maximum(iv).decompose_comparison('!=', 4))
        self.assertTrue(model.solve())
        self.assertNotEqual(str(max(iv.value())), '4')

    def test_maximum_onearg(self):

        iv = cp.intvar(0, 10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    self.assertTrue(cp.Model(cp.max([iv]) == 0).solve(solver=s))
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_abs(self):
        from cpmpy.transformations.decompose_global import decompose_in_tree
        iv = cp.intvar(-8, 8)
        constraints = [cp.Abs(iv) + 9 <= 8]
        model = cp.Model(constraints)
        self.assertFalse(model.solve())

        constraints = [cp.Abs(iv - 4) + 1 > 12]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())
        self.assertTrue(cp.Model(decompose_in_tree(constraints)).solve()) #test with decomposition

        model = cp.Model(cp.Abs(iv).decompose_comparison('!=', 4))
        self.assertTrue(model.solve())
        self.assertNotEqual(str(abs(iv.value())), '4')

    def test_element(self):
        # test 1-D
        iv = cp.intvar(-8, 8, 3)
        idx = cp.intvar(-8, 8)
        # test directly the constraint
        cons = cp.Element(iv,idx) == 8
        model = cp.Model(cons)
        self.assertTrue(model.solve())
        self.assertTrue(cons.value())
        self.assertEqual(iv.value()[idx.value()], 8)
        # test through __get_item__
        cons = iv[idx] == 8
        model = cp.Model(cons)
        self.assertTrue(model.solve())
        self.assertTrue(cons.value())
        self.assertEqual(iv.value()[idx.value()], 8)
        # test 2-D
        iv = cp.intvar(-8, 8, shape=(3, 3))
        a,b = cp.intvar(0, 2, shape=2)
        cons = iv[a,b] == 8
        model = cp.Model(cons)
        self.assertTrue(model.solve())
        self.assertTrue(cons.value())
        self.assertEqual(iv.value()[a.value(), b.value()], 8)
        arr = cp.cpm_array([[1, 2, 3], [4, 5, 6]])
        cons = arr[a,b] == 1
        model = cp.Model(cons)
        self.assertTrue(model.solve())
        self.assertTrue(cons.value())
        self.assertEqual(arr[a.value(), b.value()], 1)

    def test_element_onearg(self):

        iv = cp.intvar(0, 10)
        idx = cp.intvar(0,0)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    self.assertTrue(cp.Model(cp.Element([iv],idx) == 0).solve(solver=s))
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_xor(self):
        bv = cp.boolvar(5)
        self.assertTrue(cp.Model(cp.Xor(bv)).solve())
        self.assertTrue(cp.Xor(bv).value())

    def test_not_xor(self):
        bv = cp.boolvar(5)
        self.assertTrue(cp.Model(~cp.Xor(bv)).solve())
        self.assertFalse(cp.Xor(bv).value())
        nbNotModels = cp.Model(~cp.Xor(bv)).solveAll(display=lambda: self.assertFalse(cp.Xor(bv).value()))
        nbModels = cp.Model(cp.Xor(bv)).solveAll(display=lambda: self.assertTrue(cp.Xor(bv).value()))
        nbDecompModels = cp.Model(cp.Xor(bv).decompose()).solveAll(display=lambda: self.assertTrue(cp.Xor(bv).value()))
        self.assertEqual(nbDecompModels,nbModels)
        total = cp.Model(bv == bv).solveAll()
        self.assertEqual(str(total), str(nbModels + nbNotModels))

    def test_minimax_python(self):
        from cpmpy import min,max
        iv = cp.intvar(1,9, 10)
        self.assertIsInstance(min(iv), GlobalFunction)
        self.assertIsInstance(max(iv), GlobalFunction)

    def test_minimax_cpm(self):
        iv = cp.intvar(1,9, 10)
        mi = cp.min(iv)
        ma = cp.max(iv)
        self.assertIsInstance(mi, GlobalFunction)
        self.assertIsInstance(ma, GlobalFunction)
        
        def solve_return(model):
            model.solve()
            return model.objective_value()
        self.assertEqual( solve_return(cp.Model([], minimize=mi)), 1)
        self.assertEqual( solve_return(cp.Model([], minimize=ma)), 1)
        self.assertEqual( solve_return(cp.Model([], maximize=mi)), 9)
        self.assertEqual( solve_return(cp.Model([], maximize=ma)), 9)

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
        self.assertEqual(repr(m), repr(m2))  # should be True

    def test_cumulative_single_demand(self):
        import numpy
        m = cp.Model()
        start = cp.intvar(0, 10, 4, "start")
        duration = numpy.array([1, 2, 2, 1])
        end = start + duration
        demand = 1
        capacity = 1
        m += cp.Cumulative(start, duration, end, demand, capacity)
        self.assertTrue(m.solve())

    def test_cumulative_decomposition_capacity(self):
        import numpy as np

        # before merging #435 there was an issue with capacity constraint
        start = cp.intvar(0, 10, 4, "start")
        duration = [1, 2, 2, 1]
        end = cp.intvar(0, 10, shape=4, name="end")
        demand = 10 # tasks cannot be scheduled
        capacity = np.int64(5) # bug only happened with numpy ints
        cons = cp.Cumulative(start, duration, end, demand, capacity)
        self.assertFalse(cp.Model(cons).solve()) # this worked fine
        # also test decomposition
        self.assertFalse(cp.Model(cons.decompose()).solve()) # capacity was not taken into account and this failed

    @pytest.mark.skipif(not CPM_minizinc.supported(),
                        reason="Minizinc not installed")
    def test_cumulative_single_demand(self):
        start = cp.intvar(0, 10, name="start")
        dur = 5
        end = cp.intvar(0, 10, name="end")
        demand = 2
        capacity = 10

        m = cp.Model()
        m += cp.Cumulative([start], [dur], [end], [demand], capacity)

        self.assertTrue(m.solve(solver="ortools"))
        self.assertTrue(m.solve(solver="minizinc"))

    @pytest.mark.skipif(not CPM_minizinc.supported(),
                        reason="Minizinc not installed")
    def test_cumulative_nested(self):
        start = cp.intvar(0, 10, name="start", shape=3)
        dur = [5,5,5]
        end = cp.intvar(0, 10, name="end", shape=3)
        demand = [5,5,9]
        capacity = 10
        bv = cp.boolvar()

        cons = cp.Cumulative([start], [dur], [end], [demand], capacity)

        m = cp.Model(bv.implies(cons), start + dur != end)

        self.assertTrue(m.solve(solver="ortools"))
        self.assertTrue(m.solve(solver="minizinc"))



    def test_cumulative_no_np(self):
        start = cp.intvar(0, 10, 4, "start")
        duration = (1, 2, 2, 1) # smt weird such as a tuple
        end = [cp.intvar(0,20, name=f"end[{i}]") for i in range(4)] # force smt weird
        demand = 1
        capacity = 1
        cons = cp.Cumulative(start, duration, end, demand, capacity)
        self.assertTrue(cp.Model(cons).solve())
        self.assertTrue(cons.value())
        # also test decomposition
        self.assertTrue(cp.Model(cons.decompose()).solve())
        self.assertTrue(cons.value())

    def test_cumulative_no_np2(self):
        start = cp.intvar(0, 10, 4, "start")
        duration = (1, 2, 2, 1) # smt weird such as a tuple
        end = [cp.intvar(0,20, name=f"end[{i}]") for i in range(4)] # force smt weird
        demand = [1,1,1,1]
        capacity = 1
        cons = cp.Cumulative(start, duration, end, demand, capacity)
        self.assertTrue(cp.Model(cons).solve())
        self.assertTrue(cons.value())
        # also test decomposition
        self.assertTrue(cp.Model(cons.decompose()).solve())
        self.assertTrue(cons.value())

    def test_ite(self):
        x = cp.intvar(0, 5, shape=3, name="x")
        iter = cp.IfThenElse(x[0] > 2, x[1] > x[2], x[1] == x[2])
        constraints = [iter]
        self.assertTrue(cp.Model(constraints).solve())

        constraints = [iter, x == [0, 4, 4]]
        self.assertTrue(cp.Model(constraints).solve())

        constraints = [iter, x == [4, 4, 3]]
        self.assertTrue(cp.Model(constraints).solve())

        constraints = [iter, x == [4, 4, 4]]
        self.assertFalse(cp.Model(constraints).solve())

        constraints = [iter, x == [1, 3, 2]]
        self.assertFalse(cp.Model(constraints).solve())

    def test_global_cardinality_count(self):
        iv = cp.intvar(-8, 8, shape=5)
        val = cp.intvar(-3, 3, shape=3)
        occ = cp.intvar(0, len(iv), shape=3)
        self.assertTrue(cp.Model([cp.GlobalCardinalityCount(iv, val, occ), cp.AllDifferent(val)]).solve())
        self.assertTrue(cp.GlobalCardinalityCount(iv, val, occ).value())
        self.assertTrue(all(cp.Count(iv, val[i]).value() == occ[i].value() for i in range(len(val))))
        val = [1, 4, 5]
        self.assertTrue(cp.Model([cp.GlobalCardinalityCount(iv, val, occ)]).solve())
        self.assertTrue(cp.GlobalCardinalityCount(iv, val, occ).value())
        self.assertTrue(all(cp.Count(iv, val[i]).value() == occ[i].value() for i in range(len(val))))
        occ = [2, 3, 0]
        self.assertTrue(cp.Model([cp.GlobalCardinalityCount(iv, val, occ)]).solve())
        self.assertTrue(cp.GlobalCardinalityCount(iv, val, occ).value())
        self.assertTrue(all(cp.Count(iv, val[i]).value() == occ[i] for i in range(len(val))))
        self.assertTrue(cp.GlobalCardinalityCount([iv[0],iv[2],iv[1],iv[4],iv[3]], val, occ).value())

    def test_not_global_cardinality_count(self):
        iv = cp.intvar(-8, 8, shape=5)
        val = cp.intvar(-3, 3, shape=3)
        occ = cp.intvar(0, len(iv), shape=3)
        self.assertTrue(cp.Model([~cp.GlobalCardinalityCount(iv, val, occ), cp.AllDifferent(val)]).solve())
        self.assertTrue(~cp.GlobalCardinalityCount(iv, val, occ).value())
        self.assertFalse(all(cp.Count(iv, val[i]).value() == occ[i].value() for i in range(len(val))))
        val = [1, 4, 5]
        self.assertTrue(cp.Model([~cp.GlobalCardinalityCount(iv, val, occ)]).solve())
        self.assertTrue(~cp.GlobalCardinalityCount(iv, val, occ).value())
        self.assertFalse(all(cp.Count(iv, val[i]).value() == occ[i].value() for i in range(len(val))))
        occ = [2, 3, 0]
        self.assertTrue(cp.Model([~cp.GlobalCardinalityCount(iv, val, occ)]).solve())
        self.assertTrue(~cp.GlobalCardinalityCount(iv, val, occ).value())
        self.assertFalse(all(cp.Count(iv, val[i]).value() == occ[i] for i in range(len(val))))
        self.assertTrue(~cp.GlobalCardinalityCount([iv[0],iv[2],iv[1],iv[4],iv[3]], val, occ).value())

    def test_gcc_onearg(self):

        iv = cp.intvar(0, 10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    self.assertTrue(cp.Model(cp.GlobalCardinalityCount([iv], [3],[1])).solve(solver=s))
                except (NotImplementedError, NotSupportedError):
                    pass

    def test_count(self):
        iv = cp.intvar(-8, 8, shape=3)
        self.assertTrue(cp.Model([iv[0] == 0, iv[1] != 1, iv[2] != 2, cp.Count(iv, 0) == 3]).solve())
        self.assertEqual(str(iv.value()),'[0 0 0]')
        x = cp.intvar(-8,8)
        y = cp.intvar(0,5)
        self.assertTrue(cp.Model(cp.Count(iv, x) == y).solve())
        self.assertEqual(str(cp.Count(iv, x).value()), str(y.value()))

        self.assertTrue(cp.Model(cp.Count(iv, x) != y).solve())
        self.assertTrue(cp.Model(cp.Count(iv, x) >= y).solve())
        self.assertTrue(cp.Model(cp.Count(iv, x) <= y).solve())
        self.assertTrue(cp.Model(cp.Count(iv, x) < y).solve())
        self.assertTrue(cp.Model(cp.Count(iv, x) > y).solve())

        self.assertTrue(cp.Model(cp.Count([iv[0],iv[2],iv[1]], x) > y).solve())

    def test_count_onearg(self):

        iv = cp.intvar(0, 10)
        for s, cls in cp.SolverLookup.base_solvers():
            print(s)
            if cls.supported():
                try:
                    self.assertTrue(cp.Model(cp.Count([iv], 1) == 0).solve(solver=s))
                except (NotImplementedError, NotSupportedError):
                    pass


    def test_nvalue(self):

        iv = cp.intvar(-8, 8, shape=3)
        cnt = cp.intvar(0,10)

        self.assertFalse(cp.Model(cp.all(iv == 1), cp.NValue(iv) > 1).solve())
        self.assertTrue(cp.Model(cp.all(iv == 1), cp.NValue(iv) > cnt).solve())
        self.assertGreater(len(set(iv.value())), cnt.value())

        self.assertTrue(cp.Model(cp.NValue(iv) != cnt).solve())
        self.assertTrue(cp.Model(cp.NValue(iv) >= cnt).solve())
        self.assertTrue(cp.Model(cp.NValue(iv) <= cnt).solve())
        self.assertTrue(cp.Model(cp.NValue(iv) < cnt).solve())
        self.assertTrue(cp.Model(cp.NValue(iv) > cnt).solve())

        # test nested
        bv = cp.boolvar()
        cons = bv == (cp.NValue(iv) <= 2)
        def check_true():
            self.assertTrue(cons.value())
        cp.Model(cons).solveAll(display=check_true)

class TestBounds(unittest.TestCase):
    def test_bounds_minimum(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        z = cp.intvar(1, 9)
        expr = cp.Minimum([x,y,z])
        lb,ub = expr.get_bounds()
        self.assertEqual(lb,-8)
        self.assertEqual(ub,-1)
        self.assertFalse(cp.Model(expr<lb).solve())
        self.assertFalse(cp.Model(expr>ub).solve())


    def test_bounds_maximum(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        z = cp.intvar(1, 9)
        expr = cp.Maximum([x,y,z])
        lb,ub = expr.get_bounds()
        self.assertEqual(lb,1)
        self.assertEqual(ub,9)
        self.assertFalse(cp.Model(expr<lb).solve())
        self.assertFalse(cp.Model(expr>ub).solve())

    def test_bounds_abs(self):
        x = cp.intvar(-8, 5)
        y = cp.intvar(-7, -2)
        z = cp.intvar(1, 9)
        for var,test_lb,test_ub in [(x,0,8),(y,2,7),(z,1,9)]:
            lb, ub = cp.Abs(var).get_bounds()
            self.assertEqual(test_lb,lb)
            self.assertEqual(test_ub,ub)

    def test_bounds_element(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        z = cp.intvar(1, 9)
        expr = cp.Element([x, y, z],z)
        lb, ub = expr.get_bounds()
        self.assertEqual(lb,-8)
        self.assertEqual(ub,9)
        self.assertFalse(cp.Model(expr < lb).solve())
        self.assertFalse(cp.Model(expr > ub).solve())

    def test_bounds_count(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        z = cp.intvar(1, 9)
        a = cp.intvar(1, 9)
        expr = cp.Count([x, y, z], a)
        lb, ub = expr.get_bounds()
        self.assertEqual(lb,0)
        self.assertEqual(ub,3)
        self.assertFalse(cp.Model(expr < lb).solve())
        self.assertFalse(cp.Model(expr > ub).solve())

    def test_bounds_xor(self):
        # just one case of a Boolean global constraint
        expr = cp.Xor(cp.boolvar(3))
        self.assertEqual(expr.get_bounds(),(0,1))


class TestTypeChecks(unittest.TestCase):
    def test_AllDiff(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        self.assertTrue(cp.Model([cp.AllDifferent(x,y)]).solve())
        self.assertTrue(cp.Model([cp.AllDifferent(a,b)]).solve())
        self.assertTrue(cp.Model([cp.AllDifferent(x,y,b)]).solve())

    def test_allDiffEx0(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        self.assertTrue(cp.Model([cp.AllDifferentExcept0(x,y)]).solve())
        self.assertTrue(cp.Model([cp.AllDifferentExcept0(a,b)]).solve())
        #self.assertTrue(cp.Model([cp.AllDifferentExcept0(x,y,b)]).solve())

    def test_allEqual(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        self.assertTrue(cp.Model([cp.AllEqual(x,y,-1)]).solve())
        self.assertTrue(cp.Model([cp.AllEqual(a,b,False, a | b)]).solve())
        #self.assertTrue(cp.Model([cp.AllEqual(x,y,b)]).solve())

    def test_circuit(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        self.assertTrue(cp.Model([cp.Circuit(x+2,2,0)]).solve())
        self.assertRaises(TypeError,cp.Circuit,(a,b))
        self.assertRaises(TypeError,cp.Circuit,(x,y,b))

    def test_multicicruit(self):
        c1 = cp.Circuit(cp.intvar(0,4, shape=5))
        c2 = cp.Circuit(cp.intvar(0,2, shape=3))
        self.assertTrue(cp.Model(c1 & c2).solve())


    def test_inverse(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        self.assertFalse(cp.Model([cp.Inverse([x,y,x],[x,y,x])]).solve())
        self.assertRaises(TypeError,cp.Inverse,[a,b],[x,y])
        self.assertRaises(TypeError,cp.Inverse,[a,b],[b,False])

    def test_ITE(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        self.assertTrue(cp.Model([cp.IfThenElse(b,b&a,False)]).solve())
        self.assertRaises(TypeError, cp.IfThenElse,a,b,0)
        self.assertRaises(TypeError, cp.IfThenElse,1,x,y)

    def test_min(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        self.assertTrue(cp.Model([cp.Minimum([x,y]) == x]).solve())
        self.assertTrue(cp.Model([cp.Minimum([a,b | a]) == b]).solve())
        self.assertTrue(cp.Model([cp.Minimum([x,y,b]) == -2]).solve())

    def test_max(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        self.assertTrue(cp.Model([cp.Maximum([x,y]) == x]).solve())
        self.assertTrue(cp.Model([cp.Maximum([a,b | a]) == b]).solve())
        self.assertTrue(cp.Model([cp.Maximum([x,y,b]) == 2 ]).solve())

    def test_element(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        self.assertTrue(cp.Model([cp.Element([x,y],x) == x]).solve())
        self.assertTrue(cp.Model([cp.Element([a,b | a],x) == b]).solve())
        self.assertRaises(TypeError,cp.Element,[x,y],b)
        self.assertTrue(cp.Model([cp.Element([y,a],x) == False]).solve())

    def test_xor(self):
        x = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()
        self.assertTrue(cp.Model([cp.Xor([a,b,b])]).solve())
        self.assertRaises(TypeError, cp.Xor, (x, b))
        self.assertRaises(TypeError, cp.Xor, (x, y))

    def test_cumulative(self):
        x = cp.intvar(0, 8)
        z = cp.intvar(-8, 8)
        q = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()

        self.assertTrue(cp.Model([cp.Cumulative([x,y],[x,2],[z,q],1,x)]).solve(solver="minizinc:com.google.ortools.sat"))
        self.assertRaises(TypeError, cp.Cumulative, [x,y],[x,y],[a,y],1,x)
        self.assertRaises(TypeError, cp.Cumulative, [x,y],[x,y],[x,y],1,x)
        self.assertRaises(TypeError, cp.Cumulative, [x,y],[x,y],[x,y],x,False)

    def test_gcc(self):
        x = cp.intvar(0, 1)
        z = cp.intvar(-8, 8)
        q = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        h = cp.intvar(-7, 7)
        v = cp.intvar(-7, 7)
        b = cp.boolvar()
        a = cp.boolvar()

        self.assertTrue(cp.Model([cp.GlobalCardinalityCount([x,y], [z,q], [h,v])]).solve())
        self.assertRaises(TypeError, cp.GlobalCardinalityCount, [x,y], [x,False], [h,v])
        self.assertRaises(TypeError, cp.GlobalCardinalityCount, [x,y], [z,b], [h,v])
        self.assertRaises(TypeError, cp.GlobalCardinalityCount, [b,a], [a,b], [h,v])
        self.assertRaises(TypeError, cp.GlobalCardinalityCount, [x, y], [h, v], [z, b])
        self.assertRaises(TypeError, cp.GlobalCardinalityCount, [x, y], [x, h], [True, v])
        self.assertRaises(TypeError, cp.GlobalCardinalityCount, [x, y], [x, h], [v, a])


    def test_count(self):
        x = cp.intvar(0, 1)
        z = cp.intvar(-8, 8)
        q = cp.intvar(-8, 8)
        y = cp.intvar(-7, -1)
        b = cp.boolvar()
        a = cp.boolvar()

        self.assertTrue(cp.Model([cp.Count([x,y],z) == 1]).solve())
        self.assertRaises(TypeError, cp.Count, [x,y],[x,False])

    def test_table(self):
        iv = cp.intvar(-8,8,3)

        constraints = [cp.Table([iv[0], [iv[1], iv[2]]], [ (5, 2, 2)])] # not flatlist, should work
        model = cp.Model(constraints)
        self.assertTrue(model.solve())

        self.assertRaises(TypeError, cp.Table, [iv[0], iv[1], iv[2], 5], [(5, 2, 2)])
        self.assertRaises(TypeError, cp.Table, [iv[0], iv[1], iv[2], [5]], [(5, 2, 2)])
        self.assertRaises(TypeError, cp.Table, [iv[0], iv[1], iv[2], ['a']], [(5, 2, 2)])
