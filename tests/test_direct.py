import unittest

import numpy as np
import pytest

from cpmpy import *
from cpmpy.solvers import CPM_gurobi, CPM_pysat


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

@pytest.mark.skipif(not CPM_pysat.supported(),
                    reason="PySAT not installed")
class TestDirectPySAT(unittest.TestCase):

    def test_direct_clause(self):
        x,y = boolvar(2)

        model = SolverLookup.get("pysat")

        model += DirectConstraint("add_clause", [x, y])

        self.assertTrue(model.solve())
        self.assertTrue(x.value() or y.value())


@pytest.mark.skipif(not CPM_gurobi.supported(),
                    reason="Gurobi not installed")
class TestDirectGurobi(unittest.TestCase):

    def test_direct_poly(self):

        x = intvar(0,10,name="x")
        y = intvar(0,100,name="y")
        p = np.arange(3)

        model = SolverLookup.get("gurobi")

        model += DirectConstraint("addGenConstraintPoly", (x, y, p),
                                  novar=[2])  # optional, what not to scan for vars

        self.assertTrue(model.solve())

        x_val, y_val = x.value(), y.value()
        cons_val = p[0] * x_val ** 3 + p[1] * x_val ** 2 + p[2] * x_val ** 3

        self.assertEqual(y_val, cons_val)