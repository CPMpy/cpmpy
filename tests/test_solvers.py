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
        
        # table
        t = cp.Table([x[0],x[1]], [[2,6],[7,3]])
        self.assertEqual( cp.Model(t, minimize=x[0]).solve(), 2 )
        self.assertEqual( cp.Model(t, maximize=x[0]).solve(), 7 )

        # modulo
        self.assertTrue( cp.Model([ x[0] == x[1] % x[2] ]).solve() )

    def test_ortools_direct_solver(self):
        """
        Test direct solver access.

        If any of these tests break, update docs/advanced_solver_features.md accordingly
        """
        from cpmpy.solver_interfaces.ortools import CPMpyORTools
        from ortools.sat.python import cp_model as ort

        # standard use
        x = cp.IntVar(0,3, shape=2)
        m = cp.Model([x[0] > x[1]])
        self.assertTrue(m.solve())
        self.assertEqual(x[0].value(), 3)
        self.assertEqual(x[1].value(), 0)


        # advanced solver params
        x = cp.IntVar(0,3, shape=2)
        m = cp.Model([x[0] > x[1]])
        s = CPMpyORTools(m)
        s.ort_solver.parameters.linearization_level = 2 # more linearisation heuristics
        s.ort_solver.parameters.num_search_workers = 8 # nr of concurrent threads
        self.assertTrue(s.solve())
        self.assertEqual(x[0].value(), 3)
        self.assertEqual(x[1].value(), 0)


        # all solution counting
        class ORT_solcount(ort.CpSolverSolutionCallback):
            def __init__(self):
                super().__init__()
                self.solcount = 0

            def on_solution_callback(self):
                self.solcount += 1
        cb = ORT_solcount()

        x = cp.IntVar(0,3, shape=2)
        m = cp.Model([x[0] > x[1]])
        s = CPMpyORTools(m)
        ort_status = s.ort_solver.SearchForAllSolutions(s.ort_model, cb)
        self.assertTrue(s._after_solve(ort_status)) # post-process after solve() call...
        self.assertEqual(x[0].value(), 3)
        self.assertEqual(x[1].value(), 0)
        self.assertEqual(cb.solcount, 6)


        # all solution counting with printing
        # (not actually testing the printing)
        class ORT_myprint(ort.CpSolverSolutionCallback):
            def __init__(self, varmap, x):
                super().__init__()
                self.solcount = 0
                self.varmap = varmap
                self.x = x

            def on_solution_callback(self):
                # populate values before printing
                for cpm_var in self.x:
                    cpm_var._value = self.Value(self.varmap[cpm_var])
        
                self.solcount += 1
                print("x:",self.x.value())
        cb = ORT_myprint(s.varmap, x)

        x = cp.IntVar(0,3, shape=2)
        m = cp.Model([x[0] > x[1]])
        s = CPMpyORTools(m)
        ort_status = s.ort_solver.SearchForAllSolutions(s.ort_model, cb)
        self.assertTrue(s._after_solve(ort_status)) # post-process after solve() call...
        self.assertEqual(x[0].value(), 3)
        self.assertEqual(x[1].value(), 0)
        self.assertEqual(cb.solcount, 6)


        # intermediate solutions
        m_opt = cp.Model([x[0] > x[1]], maximize=sum(x))
        s = CPMpyORTools(m_opt)
        ort_status = s.ort_solver.SolveWithSolutionCallback(s.ort_model, cb)
        self.assertEqual(s._after_solve(ort_status), 5.0) # post-process after solve() call...
        self.assertEqual(x[0].value(), 3)
        self.assertEqual(x[1].value(), 2)
        self.assertEqual(cb.solcount, 7)


        # manually enumerating solutions
        x = cp.IntVar(0,3, shape=2)
        m = cp.Model([x[0] > x[1]])
        s = CPMpyORTools(m)
        solcount = 0
        while(s.solve()):
            solcount += 1
            # add blocking clause, to CPMpy solver directly
            s += [ cp.any(x != x.value()) ]
        self.assertEqual(solcount, 6)


        # assumptions
        bv = cp.BoolVar(shape=3)
        iv = cp.IntVar(0,9, shape=3)
        # circular 'bigger then', UNSAT
        m = cp.Model([
            bv[0].implies(iv[0] > iv[1]),
            bv[1].implies(iv[1] > iv[2]),
            bv[2].implies(iv[2] > iv[0])
        ])
        s = CPMpyORTools(m)
        self.assertFalse(s.solve(assumptions=bv))
        self.assertTrue(len(s.get_core()) > 0)
        self.assertTrue(any(bv.value() == False))


