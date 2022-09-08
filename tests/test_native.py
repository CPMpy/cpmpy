import unittest

import numpy as np
from cpmpy import *


class TestNativeORTools(unittest.TestCase):

    def test_native_automaton(self):
        trans_vars = boolvar(shape=4, name="trans")
        trans_func = [ # corresponds to regex 0* 1+ 0+
            (0, 0, 0),
            (0, 1, 1),
            (1, 1, 1),
            (1, 0, 2),
            (2, 0, 2)
        ]

        model = Model()

        model += NativeConstraint(
            name="AddAutomaton",
            arg_list=[trans_vars,0,[2],trans_func],
            arg_novar=[1,2,3]
        )

        num_sols = model.solveAll(solver="ortools", display=trans_vars)

        self.assertEqual(num_sols, 6)

    def test_native_no_overlap(self):

        interval1_args = intvar(3,10, shape=3)
        interval2_args = intvar(2,10, shape=3)

        interval1 = DirectVar("NewIntervalVar", interval1_args)
        interval2 = DirectVar("NewIntervalVar", interval2_args)

        solver = SolverLookup.get("ortools")

        solver += NativeConstraint(name="AddNoOverlap",
                                   arg_list=[[interval1, interval2]]
                                   )

        assert solver.solve()

        print("Interval1: start:{}, size:{}, end:{}".format(*interval1_args.value()))
        print("Interval2: start:{}, size:{}, end:{}".format(*interval2_args.value()))

class TestNativeGurobi(unittest.TestCase):


    def test_native_poly(self):

        x = intvar(0,10,name="x")
        y = intvar(0,100,name="y")
        p = np.arange(3)

        model = Model()

        model += NativeConstraint(
            name="addGenConstraintPoly",
            arg_list=[x,y,p],
            arg_novar=[2]
        )

        self.assertTrue(model.solve(solver="gurobi"))

        x_val, y_val = x.value(), y.value()

        cons_val = p[0] * x_val ** 3 + p[1] * x_val ** 2 + p[2] * x_val ** 3

        self.assertEqual(y_val, cons_val)
