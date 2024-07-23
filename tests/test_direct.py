import unittest

import numpy as np
import pytest

from cpmpy import *
from cpmpy.solvers import CPM_gurobi, CPM_pysat, CPM_minizinc, CPM_pysdd, CPM_z3, CPM_exact, CPM_choco


class TestDirectORTools(unittest.TestCase):

    def test_direct_automaton(self):
        trans_vars = boolvar(shape=4, name="trans")
        trans_tabl = [ # corresponds to regex 0* 1+ 0+
            (0, 0, 0),
            (0, 1, 1),
            (1, 1, 1),
            (1, 0, 2),
            (2, 0, 2)
        ]

        model = SolverLookup.get("ortools")

        model += DirectConstraint("AddAutomaton", (trans_vars, 0, [2], trans_tabl),
                                  novar=[1, 2, 3])  # optional, what not to scan for vars

        self.assertEqual(model.solveAll(), 6)

    def test_direct_no_overlap(self):

        interval1_args = intvar(3,10, shape=3)
        interval2_args = intvar(2,10, shape=3)

        interval1 = directvar("NewIntervalVar", interval1_args, name="ITV1", insert_name_at_index=3)
        interval2 = directvar("NewIntervalVar", interval2_args, name="ITV2", insert_name_at_index=3)

        solver = SolverLookup.get("ortools")

        solver += DirectConstraint(name="AddNoOverlap",
                                   arguments=([interval1, interval2]))

        assert solver.solve()

        print("Interval1: start:{}, size:{}, end:{}".format(*interval1_args.value()))
        print("Interval2: start:{}, size:{}, end:{}".format(*interval2_args.value()))


    def test_direct_optional_interval(self):

        start = intvar(0,10, shape=3)
        end = intvar(0,10,shape=3)
        dur = [3,4,5]
        bvars = boolvar(shape=3, name="bv")

        intervals = directvar("NewOptionalIntervalVar",
                              arguments=(start, np.array(dur), end, bvars),
                              shape=3,
                              insert_name_at_index=4)

        solver = SolverLookup.get("ortools")
        solver += DirectConstraint(name="AddNoOverlap", arguments=intervals)
        self.assertTrue(solver.solve())
        self.assertFalse(solver.solve(assumptions=bvars))

@pytest.mark.skipif(not CPM_exact.supported(), reason="Exact not installed")
class TestDirectExact(unittest.TestCase):

    def test_direct_left_reif(self):
        x,y = boolvar(2)

        model = SolverLookup.get("exact")
        print(model)
        # add x -> y>=1
        model += DirectConstraint("addRightReification", (x, [1], [y], 1), novar=[1,3])
        print(model)
        self.assertEqual(model.solveAll(), 3)


@pytest.mark.skipif(not CPM_pysat.supported(),
                    reason="PySAT not installed")
class TestDirectPySAT(unittest.TestCase):

    def test_direct_clause(self):
        x,y = boolvar(2)

        model = SolverLookup.get("pysat")

        model += DirectConstraint("add_clause", [x, y])

        self.assertTrue(model.solve())
        self.assertTrue(x.value() or y.value())

@pytest.mark.skipif(not CPM_pysdd.supported(),
                    reason="PySDD not installed")
class TestDirectPySDD(unittest.TestCase):

    def test_direct_clause(self):
        x,y = boolvar(2)

        model = SolverLookup.get("pysdd")

        model += DirectConstraint("conjoin", (x, y))

        self.assertTrue(model.solve())
        self.assertTrue(x.value() or y.value())

@pytest.mark.skipif(not CPM_z3.supported(),
                    reason="Z3py not installed")
class TestDirectZ3(unittest.TestCase):

    def test_direct_distinct(self):
        iv = intvar(1,9, shape=3)

        model = SolverLookup.get("z3")

        model += DirectConstraint("Distinct", iv)

        self.assertTrue(model.solve())
        self.assertTrue(AllDifferent(iv).value())

    # def test_string_cons(self):
    #
    #     str1 = directvar("String", arguments=("string 1",), novar=[0])
    #     str2 = directvar("String", arguments=("string 2",), novar=[0])
    #
    #     solver = SolverLookup.get("z3")
    #     length1 = directvar("Length", (str1,))
    #     z3_3 = directvar("IntVal", arguments=(3,))
    #     eqcons = DirectConstraint("eq", (length1, z3_3))
    #     solver += eqcons
    #
    #     assert solver.solve()



@pytest.mark.skipif(not CPM_minizinc.supported(),
                    reason="MinZinc not installed")
class TestDirectMiniZinc(unittest.TestCase):

    def test_direct_clause(self):
        iv = intvar(1,9, shape=3)

        model = SolverLookup.get("minizinc")

        # MiniZinc is oddly different for DirectConstraint, because it is a text-based language
        # so, as DirectConstraint name, you need to give the name of a text-based constraint,
        # NOT a name of a function of the mzn_model class...

        # this just to demonstrate, there are no 0's in the domains...
        model += DirectConstraint("alldifferent_except_0", iv)

        self.assertTrue(model.solve())
        self.assertTrue(AllDifferent(iv).value())

    # TODO: how would we do directVar for minizinc??


@pytest.mark.skipif(not CPM_gurobi.supported(),
                    reason="Gurobi not installed")
class TestDirectGurobi(unittest.TestCase):

    def test_direct_poly(self):

        x = intvar(0,10,name="x")
        y = intvar(0,100,name="y")
        p = np.arange(3)

        model = SolverLookup.get("gurobi")

        # y = 2 x^3 + 1.5 x^2 + 1
        p = [2, 1.5, 0, 1]
        model += DirectConstraint("addGenConstrPoly", (x, y, p),
                                  novar=[2])  # optional, what not to scan for vars

        self.assertTrue(model.solve())

        x_val = x.value()
        x_terms = [x_val**3, x_val**2, x_val**1, x_val**0]
        poly_val = sum(np.array(p)*x_terms)

        self.assertEqual(y.value(), poly_val)

@pytest.mark.skipif(not CPM_choco.supported(),
                    reason="pychoco not installed")
class TestDirectChoco(unittest.TestCase):

    def test_direct_global(self):
        iv = intvar(1,9, shape=3)

        model = SolverLookup.get("choco")

        model += DirectConstraint("increasing", iv)
        model += iv[1] < iv[0]

        self.assertFalse(model.solve())

    def test_set_var(self):

        set1 = directvar("setvar", ([1,2,3],), keyword_name="name", name="set1")
        set2 = directvar("setvar", ([3,4,5],), keyword_name="name", name="set2")
        # variable that can take subsets of {0,1,2,3,4,5,6,7,8,9}
        set3 = directvar("setvar", (list(), list(range(10))), keyword_name="name", name="set3")
        solver = SolverLookup.get("choco")
        solver += DirectConstraint("set_union", ([set1, set2], set3))

        self.assertTrue(solver.solve())
        chc_setvar = solver.solver_var(set3)
        self.assertSetEqual({1,2,3,4,5},solver.chc_sol.get_set_val(chc_setvar))