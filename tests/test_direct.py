import unittest

import numpy as np
import pytest

from cpmpy import *
from cpmpy.solvers import CPM_gurobi, CPM_pysat, CPM_minizinc, CPM_pysdd, CPM_z3, CPM_exact, CPM_choco, CPM_hexaly

@pytest.mark.requires_solver("ortools")
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

@pytest.mark.requires_solver("exact")
class TestDirectExact(unittest.TestCase):

    def test_direct_left_reif(self):
        x,y = boolvar(2)

        model = SolverLookup.get("exact")
        print(model)
        # add x -> y>=1
        model += DirectConstraint("addRightReification", (x, 1, [(1, y)], 1), novar=[1,3])
        print(model)
        self.assertEqual(model.solveAll(), 3)

@pytest.mark.requires_solver("pysat")
class TestDirectPySAT(unittest.TestCase):

    def test_direct_clause(self):
        x,y = boolvar(2)

        model = SolverLookup.get("pysat")

        model += DirectConstraint("add_clause", [x, y])

        self.assertTrue(model.solve())
        self.assertTrue(x.value() or y.value())

@pytest.mark.requires_solver("pysdd")
class TestDirectPySDD(unittest.TestCase):

    def test_direct_clause(self):
        x,y = boolvar(2)

        model = SolverLookup.get("pysdd")

        model += DirectConstraint("conjoin", (x, y))

        self.assertTrue(model.solve())
        self.assertTrue(x.value() or y.value())

@pytest.mark.requires_solver("z3")
class TestDirectZ3(unittest.TestCase):

    def test_direct_clause(self):
        iv = intvar(1,9, shape=3)

        model = SolverLookup.get("z3")

        model += DirectConstraint("Distinct", iv)

        self.assertTrue(model.solve())
        self.assertTrue(AllDifferent(iv).value())

@pytest.mark.requires_solver("minizinc")
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

@pytest.mark.requires_solver("gurobi")
class TestDirectGurobi(unittest.TestCase):

    def test_direct_poly(self):

        x = intvar(0,10,name="x")
        y = intvar(0,100,name="y")

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

@pytest.mark.requires_solver("choco")
class TestDirectChoco(unittest.TestCase):

    def test_direct_global(self):
        iv = intvar(1,9, shape=3)

        model = SolverLookup.get("choco")

        model += DirectConstraint("increasing", iv)
        model += iv[1] < iv[0]

        self.assertFalse(model.solve())


@pytest.mark.requires_solver("hexaly")
class TestDirectHexaly(unittest.TestCase):

    def test_direct_distance(self):

        a,b = intvar(0,10,shape=2, name=tuple("ab"))

        model = SolverLookup.get("hexaly")

        model += DirectConstraint("dist",(a,b)) >= 3 # model distance between two variables
        assert model.solve()

        self.assertGreaterEqual(abs(a.value() - b.value()),3)

