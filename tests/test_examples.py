from enum import Enum
import unittest
import cpmpy as cp
import numpy as np


supported_solvers= [cp.MiniZincPython()]

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
        # TODO: remove supported solvers and use cpmpy provided solver support
        for solver in supported_solvers:
        # for solver in cp.get_supported_solvers():
            _ = model.solve(solver=solver)
            self.assertEqual([x.value() for x in [s,e,n,d]], [9, 5, 6, 7])
            self.assertEqual([x.value() for x in [m,o,r,e]], [1, 0, 8, 5])
            self.assertEqual([x.value() for x in [m,o,n,e,y]], [1, 0, 6, 5, 2])

    def test_bus_schedule(self):
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
        for solver in supported_solvers:
            _ = model.solve(solver=solver)

            self.assertEqual(
                objective.value(),
                sum(xi.value() * vi for xi, vi in zip(x, values)), 
                f"Objective value:\n\texpected {sum(xi.value() * vi for xi, vi in zip(x, values))} got {objective.value()}"
            )

            self.assertEqual(
                [xi.value() for xi in x],
                [False, True, True, False, False, False, False, True, True, True],
                "Expected items: ")

    def test_who_killed_agatha(self):
        # Agatha, the butler, and Charles live in Dreadsbury Mansion, and 
        # are the only ones to live there. 
        n = 3
        (agatha, butler, charles) = range(n) # enum constants

        # Who killed agatha?
        victim = agatha
        killer = cp.IntVar(0,2)

        constraint = []
        # A killer always hates, and is no richer than his victim. 
        hates = cp.BoolVar((n,n))
        constraint += [ hates[killer, victim] == 1 ]

        richer = cp.BoolVar((n,n))
        constraint += [ richer[killer, victim] == 0 ]

        # implied richness: no one richer than himself, and anti-reflexive
        constraint += [ richer[i,i] == 0 for i in range(n) ]
        constraint += [ (richer[i,j] == 1) == (richer[j,i] == 0) for i in range(n) for j in range(n) if i != j ]

        # Charles hates noone that Agatha hates. 
        constraint += [ cp.implies(hates[agatha,i] == 1, hates[charles,i] == 0) for i in range(n) ]

        # Agatha hates everybody except the butler. 
        #cons_aga = (hates[agatha,(agatha,charles,butler] == [1,1,0])
        constraint += [ hates[agatha,agatha]  == 1,
                        hates[agatha,charles] == 1,
                        hates[agatha,butler]  == 0 ]

        # The butler hates everyone not richer than Aunt Agatha. 
        constraint += [ cp.implies(richer[i,agatha] == 0, hates[butler,i] == 1) for i in range(n) ]

        # The butler hates everyone whom Agatha hates. 
        constraint += [ cp.implies(hates[agatha,i] == 1, hates[butler,i] == 1) for i in range(n) ]

        # Noone hates everyone. 
        constraint += [ sum([hates[i,j] for j in range(n)]) <= 2 for i in range(n) ]

        model = cp.Model(constraint)
        for solver in supported_solvers:
            _ = model.solve(solver=solver)

            self.assertEqual(killer.value(), 0)

    def test_sudoku(self):
        x = 0 # cells whose value we seek
        n = 9 # matrix size
        given = np.array([
            [x, x, x,  2, x, 5,  x, x, x],
            [x, 9, x,  x, x, x,  7, 3, x],
            [x, x, 2,  x, x, 9,  x, 6, x],

            [2, x, x,  x, x, x,  4, x, 9],
            [x, x, x,  x, 7, x,  x, x, x],
            [6, x, 9,  x, x, x,  x, x, 1],

            [x, 8, x,  4, x, x,  1, x, x],
            [x, 6, 3,  x, x, x,  x, 8, x],
            [x, x, x,  6, x, 8,  x, x, x]])

        solution = [
            [3, 7, 8, 2, 6, 5, 9, 1, 4]
            [5, 9, 6, 8, 1, 4, 7, 3, 2]
            [1, 4, 2, 7, 3, 9, 5, 6, 8]
            [2, 1, 7, 3, 8, 6, 4, 5, 9]
            [8, 5, 4, 9, 7, 1, 6, 2, 3]
            [6, 3, 9, 5, 4, 2, 8, 7, 1]
            [7, 8, 5, 4, 2, 3, 1, 9, 6]
            [4, 6, 3, 1, 9, 7, 2, 8, 5]
            [9, 2, 1, 6, 5, 8, 3, 4, 7]
        ]

        # Variables
        puzzle = cp.IntVar(1, n, shape=given.shape)

        constraint = []
        # constraints on rows and columns
        constraint += [ cp.alldifferent(row) for row in puzzle ]
        constraint += [ cp.alldifferent(col) for col in puzzle.T ]

        # constraint on blocks
        for i in range(0,n,3):
            for j in range(0,n,3):
                constraint += [ cp.alldifferent(puzzle[i:i+3, j:j+3]) ]

        # constraints on values
        constraint += [ puzzle[given>0] == given[given>0] ]

        model = cp.Model(constraint)
        for solver in supported_solvers:
            _ = model.solve(solver=solver)
            for i in range(9):
                for j in range(9):
                    self.assertEqual(puzzle[i,j].value(), solution[j][i])
    
    def test_mario(self):
        data = { # a dictionary, json style
            'nbHouses': 15,
            'MarioHouse': 1,
            'LuigiHouse': 2,
            'fuelMax': 2000,
            'goldTotalAmount': 1500,
            'conso': [[0,221,274,808,13,677,670,921,943,969,13,18,217,86,322],[0,0,702,83,813,679,906,246,335,529,719,528,451,242,712],[274,702,0,127,110,72,835,5,161,430,686,664,799,523,73],[808,83,127,0,717,80,31,71,683,668,248,826,916,467,753],[13,813,110,717,0,951,593,579,706,579,101,551,280,414,294],[677,679,72,80,951,0,262,12,138,222,146,571,907,225,938],[670,906,835,31,593,262,0,189,558,27,287,977,226,454,501],[921,246,5,71,579,12,189,0,504,221,483,226,38,314,118],[943,335,161,683,706,138,558,504,0,949,393,721,267,167,420],[969,529,430,668,579,222,27,221,949,0,757,747,980,589,528],[13,719,686,248,101,146,287,483,393,757,0,633,334,492,859],[18,528,664,826,551,571,977,226,721,747,633,0,33,981,375],[217,451,799,916,280,907,226,38,267,980,334,33,0,824,491],[86,242,523,467,414,225,454,314,167,589,492,981,824,0,143],[322,712,73,753,294,938,501,118,420,528,859,375,491,143,0]],
            'goldInHouse': [0,0,40,67,89,50,6,19,47,68,94,86,34,14,14],
        }

        # Python is offset 0, MiniZinc (source of the data) is offset 1
        marioHouse, luigiHouse = data['MarioHouse']-1, data['LuigiHouse']-1 
        fuelLimit = data['fuelMax']
        nHouses = data['nbHouses']
        arc_fuel = data['conso'] # arc_fuel[a,b] = fuel from a to b

        # s[i] is the house succeeding to the ith house (s[i]=i if not part of the route)
        s = cp.IntVar(0,nHouses-1, shape=nHouses)

        cons = []
        # s should be a path, mimic (sub)circuit by connecting end-point back to start
        cons += [ s[luigiHouse] == marioHouse ]
        cons += [ cp.circuit(s) ] # should be subcircuit?

        # consumption, knowing that always conso[i,i]=0 
        # node_fuel[i] = arc_fuel[i, successor-of-i]
        arc_fuel = cp.cparray(arc_fuel) # needed to do arc_fuel[var1] == var2
        node_fuel = cp.IntVar(0,fuelLimit, shape=nHouses)
        cons += [ arc_fuel[i, s[i]] == node_fuel[i] for i in range(nHouses) ]
        cons += [ sum(node_fuel) < fuelLimit ]
        # BETTER, not possible untill I create an Element expr
        #node_fuel = [arc_fuel[i, s[i]] for i in range(nHouses)]
        #cons += [ sum(node_fuel) < fuelLimit ]

        # amount of gold earned, only for stops visited, s[i] != i
        gold = sum( (s != range(nHouses))*data['goldInHouse'] )

        model = cp.Model(cons, maximize=gold)

        expected_solution = [4, 0, 9, 1, 10, 8, 3, 5, 13, 6, 12, 7, 11, 14, 2]

        # Statistics are returned after solving.
        for solver in supported_solvers:
            _ = model.solve(solver=solver)
            self.assertEqual([si.value() for si in s], expected_solution)