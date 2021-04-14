import numpy as np
import unittest
import cpmpy as cp

class TestSolvers(unittest.TestCase):
    def test_installed_solvers(self):
        # basic model
        x = cp.IntVar(0,2, 3)

        constraints = [
            x[0] < x[1],
            x[1] < x[2]]
        model = cp.Model(constraints)
        for solver in cp.get_supported_solvers():
            model.solve()
            self.assertEqual([xi.value() for xi in x], [0, 1, 2])
    
    # should move this test elsewhere later
    def test_tsp(self):
        N = 6
        np.random.seed(0)
        b = np.random.randint(1,100, size=(N,N))
        distance_matrix= ((b + b.T)/2).astype(int)
        x = cp.IntVar(0, 1, shape=distance_matrix.shape) 
        constraint  = []
        constraint  += [sum(x[i,:])==1 for i in range(N)]
        constraint  += [sum(x[:,i])==1 for i in range(N)]
        constraint += [sum(x[i,i] for i in range(N))==0]

        objective = sum(x*distance_matrix)

        model = cp.Model(constraint, minimize=objective)
        objval = model.solve()
        self.assertEqual(objval, 214)
        self.assertEqual(x.value().tolist(),
        [[0, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0, 0]])

    def test_ortools(self):
        b = cp.BoolVar()
        x = cp.IntVar(1,13, shape=3)
        # reifiability (automatic handling in case of !=)
        self.assertTrue( cp.Model(b.implies((x[0]*x[1]) == x[2])).solve() )
        self.assertTrue( cp.Model(b.implies((x[0]*x[1]) != x[2])).solve() )
        self.assertTrue( cp.Model(((x[0]*x[1]) == x[2]).implies(b)).solve() )
        self.assertTrue( cp.Model(((x[0]*x[1]) != x[2]).implies(b)).solve() )
        self.assertTrue( cp.Model(((x[0]*x[1]) == x[2]) == b).solve() )
        self.assertTrue( cp.Model(((x[0]*x[1]) != x[2]) == b).solve() )
