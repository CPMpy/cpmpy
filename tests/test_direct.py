import unittest

import numpy as np
import pytest

import cpmpy as cp
from utils import TestCase

@pytest.mark.usefixtures("solver")    
@pytest.mark.requires_solver("ortools")
class TestDirectORTools(TestCase):

    def test_direct_automaton(self):
        trans_vars = cp.boolvar(shape=4, name="trans")
        trans_tabl = [ # corresponds to regex 0* 1+ 0+
            (0, 0, 0),
            (0, 1, 1),
            (1, 1, 1),
            (1, 0, 2),
            (2, 0, 2)
        ]

        model = cp.SolverLookup.get("ortools")

        model += cp.DirectConstraint("AddAutomaton", (trans_vars, 0, [2], trans_tabl),
                                  novar=[1, 2, 3])  # optional, what not to scan for vars

        self.assertEqual(model.solveAll(), 6)

@pytest.mark.usefixtures("solver")
@pytest.mark.requires_solver("exact")
class TestDirectExact(TestCase):

    def test_direct_left_reif(self):
        x,y = cp.boolvar(2)

        model = cp.SolverLookup.get("exact")
        # add x -> y>=1
        model += cp.DirectConstraint("addRightReification", (x, 1, [(1, y)], 1), novar=[1,3])
        self.assertEqual(model.solveAll(), 3)

@pytest.mark.usefixtures("solver")
@pytest.mark.requires_solver("pysat")
class TestDirectPySAT(TestCase):

    def test_direct_clause(self):
        x,y = cp.boolvar(2)

        model = cp.SolverLookup.get("pysat")

        model += cp.DirectConstraint("add_clause", [x, y])

        self.assertTrue(model.solve())
        self.assertTrue(x.value() or y.value())

@pytest.mark.usefixtures("solver")
@pytest.mark.requires_solver("pysdd")
class TestDirectPySDD(TestCase):

    def test_direct_clause(self):
        x,y = cp.boolvar(2)

        model = cp.SolverLookup.get("pysdd")

        model += cp.DirectConstraint("conjoin", (x, y))

        self.assertTrue(model.solve())
        self.assertTrue(x.value() or y.value())

@pytest.mark.usefixtures("solver")
@pytest.mark.requires_solver("z3")
class TestDirectZ3(TestCase):

    def test_direct_distinct(self):
        iv = cp.intvar(1,9, shape=3)

        model = cp.SolverLookup.get("z3")

        model += cp.DirectConstraint("Distinct", iv)

        self.assertTrue(model.solve())
        self.assertTrue(cp.AllDifferent(iv).value())

@pytest.mark.usefixtures("solver")
@pytest.mark.requires_solver("minizinc")
class TestDirectMiniZinc(TestCase):

    def test_direct_alldiff_except_0(self):
        iv = cp.intvar(0,9, shape=3)

        model = cp.SolverLookup.get("minizinc")

        # MiniZinc is oddly different for cp.DirectConstraint, because it is a text-based language
        # so, as cp.DirectConstraint name, you need to give the name of a text-based constraint,
        # NOT a name of a function of the mzn_model class...

        # this just to demonstrate, there are no 0's in the domains...
        model += cp.DirectConstraint("alldifferent_except_0", iv)
        model += cp.Count(iv, 0) >= 2

        self.assertTrue(model.solve())
        self.assertTrue(cp.AllDifferentExcept0(iv).value())


@pytest.mark.usefixtures("solver")
@pytest.mark.requires_solver("gurobi")
class TestDirectGurobi(TestCase):

    def test_direct_poly(self):

        x = cp.intvar(0,10,name="x")
        y = cp.intvar(0,100,name="y")

        model = cp.SolverLookup.get("gurobi")

        # y = 2 x^3 + 1.5 x^2 + 1
        p = [2, 1.5, 0, 1]
        model += cp.DirectConstraint("addGenConstrPoly", (x, y, p),
                                  novar=[2])  # optional, what not to scan for vars

        self.assertTrue(model.solve())

        x_val = x.value()
        x_terms = [x_val**3, x_val**2, x_val**1, x_val**0]
        poly_val = sum(np.array(p)*x_terms)

        self.assertEqual(y.value(), poly_val)

@pytest.mark.usefixtures("solver")
@pytest.mark.requires_solver("choco")
class TestDirectChoco(TestCase):

    def test_direct_global(self):
        iv = cp.intvar(1,9, shape=3)

        model = cp.SolverLookup.get("choco")

        model += cp.DirectConstraint("increasing", iv)
        model += iv[1] < iv[0]

        self.assertFalse(model.solve())


@pytest.mark.usefixtures("solver")
@pytest.mark.requires_solver("hexaly")
class TestDirectHexaly(TestCase):

    def test_direct_distance(self):

        a,b = cp.intvar(0,10,shape=2, name=tuple("ab"))

        model = cp.SolverLookup.get("hexaly")

        model += cp.DirectConstraint("dist",(a,b)) >= 3 # model distance between two variables
        assert model.solve()

        self.assertGreaterEqual(abs(a.value() - b.value()),3)

