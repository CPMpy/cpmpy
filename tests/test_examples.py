import unittest
import cpmpy as cp
import numpy as np

class TestExamples(unittest.TestCase):

    def test_send_more_money(self):
        supported_solvers= [cp.MiniZincPython()]
        # Construct the model.
        s,e,n,d,m,o,r,y = cp.IntVar(0,9, 8)

        constraint = []
        constraint += [ cp.alldifferent([s,e,n,d,m,o,r,y]) ]
        constraint += [    sum(   [s,e,n,d] * np.flip(10**np.arange(4)) )
                         + sum(   [m,o,r,e] * np.flip(10**np.arange(4)) )
                        == sum( [m,o,n,e,y] * np.flip(10**np.arange(5)) ) ]
        constraint += [ s > 0, m > 0 ]

        model = cp.Model(constraint)
        # TODO: remove supported solvers and use cpmpy provided solver support
        for solver in supported_solvers:
        # for solver in cp.get_supported_solvers():
            _ = model.solve(solver=solver)
            self.assertEqual([x.value() for x in [s,e,n,d]], [9, 5, 6, 7])
            self.assertEqual([x.value() for x in [m,o,r,e]], [1, 0, 8, 5])
            self.assertEqual([x.value() for x in [m,o,n,e,y]], [1, 0, 6, 5, 2])

    def test_bus_schedule(self):
        supported_solvers= [cp.MiniZincPython()]
        demands = [8, 10, 7, 12, 4, 4]
        slots = len(demands)

        # variables
        x = cp.IntVar(0,sum(demands), slots)

        constraint  = [x[i] + x[i+1] >= demands[i] for i in range(0,slots-1)]
        constraint += [x[-1] + x[0] == demands[-1]] # 'around the clock' constraint

        objective = sum(x) # number of buses
        model = cp.Model(constraint, minimize=objective)
        # TODO: remove supported solvers and use cpmpy provided solver support
        # for solver in cp.get_supported_solvers():
        for solver in supported_solvers:
            _ = model.solve(solver=solver)
            self.assertEqual([xi.value() for xi in x], [4, 4, 6, 1, 11, 0], f"Expected schedule:\n\t[4, 4, 6, 1, 11, 0] got {x.value()}")
            self.assertEqual(sum(x.value()), 26, f"Expected value is 26, got {sum(x.value())}")

    def test_knapsack(self):
        # Problem data
        n = 10
        np.random.seed(1)
        values = np.random.randint(0,10, n)
        weights = np.random.randint(1,5, n)
        capacity = np.random.randint(sum(weights)*.2, sum(weights)*.5)

        # Construct the model.
        x = cp.BoolVar(n)

        constraint = [ sum(x*weights) <= capacity ]
        objective  = sum(x*values)

        model = cp.Model(constraint, maximize=objective)

        # Statistics are returned after solving.
        stats = model.solve()

        self.assertEqual(
            objective.value(),
            sum(xi.value() * vi for xi, vi in zip(x, values)), 
            f"Objective value:\n\texpected {sum(xi.value() * vi for xi, vi in zip(x, values))} got {objective.value()}"
        )
        
        # Variables can be asked for their value in the found solution
        print("Value:", objective.value())
        print("Solution:", x.value())
        print("In items: ", [i+1 for i,val in enumerate(x.value()) if val])