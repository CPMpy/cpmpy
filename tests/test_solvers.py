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
    
    # should move elsewhere later
    def test_tsp(self):
        N = 6
        b = np.random.randint(1,100, size=(N,N))
        distance_matrix= (b + b.T)/2
        x = cp.IntVar(0, 1, shape=distance_matrix.shape) 
        constraint  = []
        constraint  += [sum(x[i,:])==1 for i in range(N)]
        constraint  += [sum(x[:,i])==1 for i in range(N)]
        constraint += [sum(x[i,i] for i in range(N))==0]

        objective =0 
        for i in range(N):
            for j in range(N):
                objective += x[i,j]*distance_matrix[i,j] 
        ## this is not working
        # objective = sum(x*distance_matrix)

        model = cp.Model(constraint, minimize=objective)
        stats = model.solve()
        solution = x.value()
