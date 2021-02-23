import unittest
import cpmpy as cp
import numpy as np

class TestExamples(unittest.TestCase):

    def test_send_more_money(self):
        # Construct the model.
        s,e,n,d,m,o,r,y = cp.IntVar(0,9, 8)

        constraint = []
        constraint += [ cp.alldifferent([s,e,n,d,m,o,r,y]) ]
        constraint += [    sum(   [s,e,n,d] * np.flip(10**np.arange(4)) )
                         + sum(   [m,o,r,e] * np.flip(10**np.arange(4)) )
                        == sum( [m,o,n,e,y] * np.flip(10**np.arange(5)) ) ]
        constraint += [ s > 0, m > 0 ]

        model = cp.Model(constraint)
        # for solver in cp.get_supported_solvers():
        #     _ = model.solve(solver=solver)
        #     self.assertEqual([x.value() for x in [s,e,n,d]], [9, 5, 6, 7])
        #     self.assertEqual([x.value() for x in [m,o,r,e]], [1, 0, 8, 5])
        #     self.assertEqual([x.value() for x in [m,o,n,e,y]], [1, 0, 6, 5, 2])

    def test_bus_schedule(self):
        demands = [8, 10, 7, 12, 4, 4]
        slots = len(demands)

        # variables
        x = cp.IntVar(0,sum(demands), slots)

        constraint  = [x[i] + x[i+1] >= demands[i] for i in range(0,slots-1)]
        constraint += [x[-1] + x[0] == demands[-1]] # 'around the clock' constraint

        objective = sum(x) # number of buses
        model = cp.Model(constraint, minimize=objective)
        # for solver in cp.get_supported_solvers():
        #     _ = model.solve(solver=solver)
        #     self.assertEqual(x.value(), [4, 4, 6, 1, 11, 0], f"Expected schedule:\n\t[4, 4, 6, 1, 11, 0] got {x.value()}")
        #     self.assertEqual(sum(x.value()), 26, f"Expected value is 26, got {sum(x.value())}")
