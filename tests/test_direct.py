import unittest

import numpy as np
import pytest

from cpmpy import *
from cpmpy.solvers import CPM_gurobi


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

        model += DirectConstraint(name="AddAutomaton",
                                  arguments=(trans_vars, 0, [2], trans_tabl), novar=[1, 2, 3])

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

@pytest.mark.skipif(not CPM_gurobi.supported(),
                    reason="Gurobi not installed")
class TestDirectGurobi(unittest.TestCase):

    def test_direct_poly(self):

        x = intvar(0,10,name="x")
        y = intvar(0,100,name="y")
        p = np.arange(3)

        model = SolverLookup.get("gurobi")

        model += DirectConstraint(name="addGenConstraintPoly",
                                  arguments=(x, y, p), novar=[2])

        self.assertTrue(model.solve())

        x_val, y_val = x.value(), y.value()
        cons_val = p[0] * x_val ** 3 + p[1] * x_val ** 2 + p[2] * x_val ** 3

        self.assertEqual(y_val, cons_val)
