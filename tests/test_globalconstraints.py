import unittest

from numpy import ndarray

import cpmpy as cp
from cpmpy.expressions.globalconstraints import GlobalConstraint, GlobalCardinalityCount, Count, AllDifferentExcept0

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

    def test_not_alldifferent(self):
        # from fuzztester of Ruben Kindt, #143
        pos = cp.intvar(lb=0, ub=5, shape=3, name="positions")
        m = cp.Model()
        m += ~cp.AllDifferent(pos)
        self.assertTrue(m.solve("ortools"))
        self.assertFalse(cp.AllDifferent(pos).value())

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


    def test_table(self):
        iv = cp.intvar(-8,8,3)
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

    def test_minimum(self):
        iv = cp.intvar(-8, 8, 3)
        constraints = [cp.Minimum(iv) + 9 == 8]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())
        self.assertEqual(str(min(iv.value())), '-1')

        model = cp.Model(cp.Minimum(iv).decompose_comparison('==', 4))
        self.assertTrue(model.solve())
        self.assertEqual(str(min(iv.value())), '4')


    def test_maximum(self):
        iv = cp.intvar(-8, 8, 3)
        constraints = [cp.Maximum(iv) + 9 <= 8]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())
        self.assertTrue(max(iv.value()) <= -1)

        model = cp.Model(cp.Maximum(iv).decompose_comparison('!=', 4))
        self.assertTrue(model.solve())
        self.assertNotEqual(str(max(iv.value())), '4')

    def test_element(self):
        iv = cp.intvar(-8, 8, 3)
        idx = cp.intvar(-8, 8)
        constraints = [cp.Element(iv,idx) == 8]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())
        self.assertTrue(iv.value()[idx.value()] == 8)
        self.assertTrue(cp.Element(iv,idx).value() == 8)


    def test_Xor(self):
        bv = cp.boolvar(5)
        self.assertTrue(cp.Model(cp.Xor(bv)).solve())
        self.assertTrue(cp.Xor(bv).value())

    def test_minimax_python(self):
        from cpmpy import min,max
        iv = cp.intvar(1,9, 10)
        self.assertIsInstance(min(iv), GlobalConstraint) 
        self.assertIsInstance(max(iv), GlobalConstraint) 

    def test_minimax_cpm(self):
        iv = cp.intvar(1,9, 10)
        mi = cp.min(iv)
        ma = cp.max(iv)
        self.assertIsInstance(mi, GlobalConstraint) 
        self.assertIsInstance(ma, GlobalConstraint) 
        
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
        m2 = m.deepcopy()  # should not throw an exception
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

    def test_global_cardinality_count(self):
        iv = cp.intvar(-8, 8, shape=3)
        gcc = cp.intvar(0, 10, shape=iv[0].ub + 1)
        self.assertTrue(cp.Model([GlobalCardinalityCount(iv, gcc), iv == [5,5,4]]).solve())
        self.assertEqual( str(gcc.value()), '[0 0 0 0 1 2 0 0 0]')
        self.assertTrue(GlobalCardinalityCount(iv, gcc).value())

        self.assertTrue(cp.Model([GlobalCardinalityCount(iv, gcc).decompose(), iv == [5, 5, 4]]).solve())
        self.assertEqual(str(gcc.value()), '[0 0 0 0 1 2 0 0 0]')
        self.assertTrue(GlobalCardinalityCount(iv, gcc).value())
    def test_not_global_cardinality_count(self):
        iv = cp.intvar(-8, 8, shape=3)
        gcc = cp.intvar(0, 10, shape=iv[0].ub + 1)
        self.assertTrue(cp.Model([~GlobalCardinalityCount(iv, gcc), iv == [5, 5, 4]]).solve())
        self.assertNotEqual(str(gcc.value()), '[0 0 0 0 1 2 0 0 0]')
        self.assertFalse(GlobalCardinalityCount(iv, gcc).value())

        self.assertFalse(cp.Model([~GlobalCardinalityCount(iv, gcc), iv == [5, 5, 4],
                                   gcc == [0, 0, 0, 0, 1, 2, 0, 0, 0]]).solve())

    def test_count(self):
        iv = cp.intvar(-8, 8, shape=3)
        self.assertTrue(cp.Model([iv[0] == 0, iv[1] != 1, iv[2] != 2, Count(iv, 0) == 3]).solve())
        self.assertEqual(str(iv.value()),'[0 0 0]')
        x = cp.intvar(-8,8)
        y = cp.intvar(0,5)
        self.assertTrue(cp.Model(Count(iv, x) == y).solve())
        self.assertEqual(str(Count(iv, x).value()), str(y.value()))

        self.assertTrue(cp.Model(Count(iv, x) != y).solve())
        self.assertTrue(cp.Model(Count(iv, x) >= y).solve())
        self.assertTrue(cp.Model(Count(iv, x) <= y).solve())
        self.assertTrue(cp.Model(Count(iv, x) < y).solve())
        self.assertTrue(cp.Model(Count(iv, x) > y).solve())
    def test_alldifferentexcept0(self):
        iv = cp.intvar(-8, 8, shape=3)
        self.assertTrue(cp.Model([AllDifferentExcept0(iv)]).solve())
        self.assertTrue(AllDifferentExcept0(iv).value())
        self.assertTrue(cp.Model([AllDifferentExcept0(iv), iv == [0,0,1]]).solve())
        self.assertTrue(AllDifferentExcept0(iv).value())

    def test_not_alldifferentexcept0(self):
        iv = cp.intvar(-8, 8, shape=3)
        self.assertTrue(cp.Model([~AllDifferentExcept0(iv)]).solve())
        self.assertFalse(AllDifferentExcept0(iv).value())
        self.assertFalse(cp.Model([~AllDifferentExcept0(iv), iv == [0, 0, 1]]).solve())
