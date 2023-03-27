import unittest
import pytest
import numpy as np
import cpmpy as cp

from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.solvers.z3 import CPM_z3
from cpmpy.solvers.minizinc import CPM_minizinc

from cpmpy.exceptions import MinizincNameException

class TestSolvers(unittest.TestCase):
    def test_installed_solvers(self):
        # basic model
        v = cp.boolvar(3)
        x,y,z = v

        model = cp.Model(
                    x.implies(y & z),
                    y | z,
                    ~ z
                )

        for solvern,s in cp.SolverLookup.base_solvers():
            if s.supported(): # only supported solvers in test suite
                model.solve(solver=solvern)
                self.assertEqual([int(a) for a in v.value()], [0, 1, 0])

        for solvern in cp.SolverLookup.solvernames():
            s = cp.SolverLookup.get(solvern)
            if s.supported(): # only supported solvers in test suite
                s2 = cp.SolverLookup.get(solvern, model)
                try: 
                    s2.solve()
                    self.assertEqual([int(a) for a in v.value()], [0, 1, 0])
                except:
                    # its OK I guess... MiniZinc error
                    pass

    def test_installed_solvers_solveAll(self):
        # basic model
        v = cp.boolvar(3)
        x,y,z = v

        model = cp.Model(
                    x.implies(y & z),
                    y | z
                )

        for solvern,s in cp.SolverLookup.base_solvers():
            if s.supported(): # only supported solvers in test suite
                if solvern == "pysdd":
                    self.assertEqual(model.solveAll(solver=solvern), 4)
                else:
                    # some solvers do not support searching for all solutions...
                    # TODO: remove solution limit and replace with time limit (atm pysat does not support time limit and gurobi needs any(solution_limit, time_limit)...
                    self.assertEqual(model.solveAll(solver=solvern, solution_limit=4), 4)

    # should move this test elsewhere later
    def test_tsp(self):
        N = 6
        np.random.seed(0)
        b = np.random.randint(1,100, size=(N,N))
        distance_matrix= ((b + b.T)/2).astype(int)
        x = cp.intvar(0, 1, shape=distance_matrix.shape) 
        constraint  = []
        constraint  += [sum(x[i,:])==1 for i in range(N)]
        constraint  += [sum(x[:,i])==1 for i in range(N)]
        constraint += [sum(x[i,i] for i in range(N))==0]

        # sum over all elements in 2D matrix
        objective = (x*distance_matrix).sum()

        model = cp.Model(constraint, minimize=objective)
        self.assertTrue(model.solve())
        self.assertEqual(model.objective_value(), 214)
        self.assertEqual(x.value().tolist(),
        [[0, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0, 0]])

    def test_ortools(self):
        b = cp.boolvar()
        x = cp.intvar(1,13, shape=3)

        # reifiability (automatic handling in case of !=)
        # TODO, side-effect that his work...
        #self.assertTrue( cp.Model(b.implies((x[0]*x[1]) == x[2])).solve() )
        #self.assertTrue( cp.Model(b.implies((x[0]*x[1]) != x[2])).solve() )
        #self.assertTrue( cp.Model(((x[0]*x[1]) == x[2]).implies(b)).solve() )
        #self.assertTrue( cp.Model(((x[0]*x[1]) != x[2]).implies(b)).solve() )
        #self.assertTrue( cp.Model(((x[0]*x[1]) == x[2]) == b).solve() )
        #self.assertTrue( cp.Model(((x[0]*x[1]) != x[2]) == b).solve() )
        
        # table
        t = cp.Table([x[0],x[1]], [[2,6],[7,3]])

        m = cp.Model(t, minimize=x[0])
        self.assertTrue(m.solve())
        self.assertEqual( m.objective_value(), 2 )

        m = cp.Model(t, maximize=x[0])
        self.assertTrue(m.solve())
        self.assertEqual( m.objective_value(), 7 )

        # modulo
        self.assertTrue( cp.Model([ x[0] == x[1] % x[2] ]).solve() )

    def test_ortools_inverse(self):
        from cpmpy.solvers.ortools import CPM_ortools

        fwd = cp.intvar(0, 9, shape=10)
        rev = cp.intvar(0, 9, shape=10)

        # Fixed value for `fwd`
        fixed_fwd = [9, 4, 7, 2, 1, 3, 8, 6, 0, 5]
        # Inverse of the above
        expected_inverse = [8, 4, 3, 5, 1, 9, 7, 2, 6, 0]

        model = cp.Model(cp.Inverse(fwd, rev), fwd == fixed_fwd)

        solver = CPM_ortools(model)
        self.assertTrue(solver.solve())
        self.assertEqual(list(rev.value()), expected_inverse)


    def test_ortools_direct_solver(self):
        """
        Test direct solver access.

        If any of these tests break, update docs/advanced_solver_features.md accordingly
        """
        from cpmpy.solvers.ortools import CPM_ortools
        from ortools.sat.python import cp_model as ort

        # standard use
        x = cp.intvar(0,3, shape=2)
        m = cp.Model([x[0] > x[1]])
        self.assertTrue(m.solve())
        self.assertGreater(*x.value())


        # direct use
        o = CPM_ortools()
        o += x[0] > x[1]
        self.assertTrue(o.solve())
        o.minimize(x[0])
        o.solve()
        self.assertEqual(x[0].value(), 1)
        o.maximize(x[1])
        o.solve()
        self.assertEqual(x[1].value(), 2)


        # TODO: these tests our outdated, there are more
        # direct ways of setting params/sol enum now
        # advanced solver params
        x = cp.intvar(0,3, shape=2)
        m = cp.Model([x[0] > x[1]])
        s = CPM_ortools(m)
        s.ort_solver.parameters.linearization_level = 2 # more linearisation heuristics
        s.ort_solver.parameters.num_search_workers = 8 # nr of concurrent threads
        self.assertTrue(s.solve())
        self.assertGreater(*x.value())


        # all solution counting
        class ORT_solcount(ort.CpSolverSolutionCallback):
            def __init__(self):
                super().__init__()
                self.solcount = 0

            def on_solution_callback(self):
                self.solcount += 1
        cb = ORT_solcount()

        x = cp.intvar(0,3, shape=2)
        m = cp.Model([x[0] > x[1]])
        s = CPM_ortools(m)
        s.ort_solver.parameters.enumerate_all_solutions=True
        cpm_status = s.solve(solution_callback=cb)
        self.assertGreater(x[0], x[1])
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
        cb = ORT_myprint(s._varmap, x)

        x = cp.intvar(0,3, shape=2)
        m = cp.Model([x[0] > x[1]])
        s = CPM_ortools(m)
        s.ort_solver.parameters.enumerate_all_solutions=True
        cpm_status = s.solve(solution_callback=cb)
        self.assertGreater(x[0], x[1])
        self.assertEqual(cb.solcount, 6)


        # intermediate solutions
        m_opt = cp.Model([x[0] > x[1]], maximize=sum(x))
        s = CPM_ortools(m_opt)
        cpm_status = s.solve(solution_callback=cb)
        self.assertEqual(s.objective_value(), 5.0)

        self.assertGreater(x[0], x[1])
        self.assertEqual(cb.solcount, 7)


        # manually enumerating solutions
        x = cp.intvar(0,3, shape=2)
        m = cp.Model([x[0] > x[1]])
        s = CPM_ortools(m)
        solcount = 0
        while(s.solve()):
            solcount += 1
            # add blocking clause, to CPMpy solver directly
            s += [ cp.any(x != x.value()) ]
        self.assertEqual(solcount, 6)

        # native all solutions
        s = CPM_ortools(m)
        n = s.solveAll()
        self.assertEqual(n, 6)

        n = s.solveAll(display=x)
        self.assertEqual(n, 6)

        n = s.solveAll(cp_model_probing_level=0)
        self.assertEqual(n, 6)

        # assumptions
        bv = cp.boolvar(shape=3)
        iv = cp.intvar(0,9, shape=3)
        # circular 'bigger then', UNSAT
        m = cp.Model([
            bv[0].implies(iv[0] > iv[1]),
            bv[1].implies(iv[1] > iv[2]),
            bv[2].implies(iv[2] > iv[0])
        ])
        s = CPM_ortools(m)
        self.assertFalse(s.solve(assumptions=bv))
        self.assertTrue(len(s.get_core()) > 0)


    @pytest.mark.skipif(not CPM_pysat.supported(),
                        reason="PySAT not installed")
    def test_pysat(self):

        # Construct the model.
        (mayo, ketchup, curry, andalouse, samurai) = cp.boolvar(5)

        Nora = mayo | ketchup
        Leander = ~samurai | mayo
        Benjamin = ~andalouse | ~curry | ~samurai
        Behrouz = ketchup | curry | andalouse
        Guy = ~ketchup | curry | andalouse
        Daan = ~ketchup | ~curry | andalouse
        Celine = ~samurai
        Anton = mayo | ~curry | ~andalouse
        Danny = ~mayo | ketchup | andalouse | samurai
        Luc = ~mayo | samurai

        allwishes = [Nora, Leander, Benjamin, Behrouz, Guy, Daan, Celine, Anton, Danny, Luc]

        model = cp.Model(allwishes)

        # any solver
        self.assertTrue(model.solve())
        
        # direct solver
        ps = CPM_pysat(model)
        self.assertTrue(ps.solve())
        self.assertEqual([False, True, False, True, False], [v.value() for v in [mayo, ketchup, curry, andalouse, samurai]])

        indmodel = cp.Model()
        inds = cp.boolvar(shape=len(model.constraints))
        for i,c in enumerate(model.constraints):
            indmodel += [c | ~inds[i]] # implication
        ps2 = CPM_pysat(indmodel)

        # check get core, simple
        self.assertFalse(ps2.solve(assumptions=[mayo,~mayo]))
        self.assertEqual(ps2.get_core(), [mayo,~mayo])

        # check get core, more realistic
        self.assertFalse(ps2.solve(assumptions=[mayo]+[v for v in inds]))
        self.assertEqual(ps2.get_core(), [mayo,inds[6],inds[9]])


    @pytest.mark.skipif(not CPM_minizinc.supported(),
                        reason="MiniZinc not installed")
    def test_minizinc(self):
        # (from or-tools)
        b = cp.boolvar()
        x = cp.intvar(1,13, shape=3)

        # reifiability (automatic handling in case of !=)
        self.assertTrue( cp.Model(b.implies((x[0]*x[1]) == x[2])).solve(solver="minizinc") )
        self.assertTrue( cp.Model(b.implies((x[0]*x[1]) != x[2])).solve(solver="minizinc") )
        self.assertTrue( cp.Model(((x[0]*x[1]) == x[2]).implies(b)).solve(solver="minizinc") )
        self.assertTrue( cp.Model(((x[0]*x[1]) != x[2]).implies(b)).solve(solver="minizinc") )
        self.assertTrue( cp.Model(((x[0]*x[1]) == x[2]) == b).solve(solver="minizinc") )
        self.assertTrue( cp.Model(((x[0]*x[1]) != x[2]) == b).solve(solver="minizinc") )
        
        # table
        t = cp.Table([x[0],x[1]], [[2,6],[7,3]])

        m = cp.Model(t, minimize=x[0])
        self.assertTrue(m.solve(solver="minizinc"))
        self.assertEqual( m.objective_value(), 2 )

        m = cp.Model(t, maximize=x[0])
        self.assertTrue(m.solve(solver="minizinc"))
        self.assertEqual( m.objective_value(), 7 )

        # modulo
        self.assertTrue( cp.Model([ x[0] == x[1] % x[2] ]).solve(solver="minizinc") )


    @pytest.mark.skipif(not CPM_minizinc.supported(),
                        reason="MiniZinc not installed")
    def test_minizinc_names(self):
        a = cp.boolvar(name='5var')#has to start with alphabetic character
        b = cp.boolvar(name='va+r')#no special characters
        c = cp.boolvar(name='solve')#no keywords
        with self.assertRaises(MinizincNameException):
            cp.Model(a == 0).solve(solver="minizinc")
        with self.assertRaises(MinizincNameException):
            cp.Model(b == 0).solve(solver="minizinc")
        with self.assertRaises(MinizincNameException):
            cp.Model(c == 0).solve(solver="minizinc")

    @pytest.mark.skipif(not CPM_minizinc.supported(),
                        reason="MiniZinc not installed")
    def test_minizinc_inverse(self):
        from cpmpy.solvers.minizinc import CPM_minizinc

        fwd = cp.intvar(0, 9, shape=10)
        rev = cp.intvar(0, 9, shape=10)

        # Fixed value for `fwd`
        fixed_fwd = [9, 4, 7, 2, 1, 3, 8, 6, 0, 5]
        # Inverse of the above
        expected_inverse = [8, 4, 3, 5, 1, 9, 7, 2, 6, 0]

        model = cp.Model(cp.Inverse(fwd, rev), fwd == fixed_fwd)

        solver = CPM_minizinc(model)
        self.assertTrue(solver.solve())
        self.assertEqual(list(rev.value()), expected_inverse)

    @pytest.mark.skipif(not CPM_z3.supported(),
                        reason="Z3 not installed")
    def test_z3(self):
        bv = cp.boolvar(shape=3)
        iv = cp.intvar(0, 9, shape=3)
        # circular 'bigger then', UNSAT
        m = cp.Model([
            bv[0].implies(iv[0] > iv[1]),
            bv[1].implies(iv[1] > iv[2]),
            bv[2].implies(iv[2] > iv[0])
        ])
        s = cp.SolverLookup.get("z3", m)
        self.assertFalse(s.solve(assumptions=bv))
        self.assertTrue(len(s.get_core()) > 0)

        m = cp.Model(~(iv[0] != iv[1]))
        s = cp.SolverLookup.get("z3", m)
        self.assertTrue(s.solve())

        m = cp.Model((iv[0] == 0) & ((iv[0] != iv[1]) == 0))
        s = cp.SolverLookup.get("z3", m)
        self.assertTrue(s.solve())

        m = cp.Model([~bv, ~((iv[0] + abs(iv[1])) == sum(iv))])
        s = cp.SolverLookup.get("z3", m)
        self.assertTrue(s.solve())

        x = cp.intvar(0, 1)
        m = cp.Model((x >= 0.1) & (x != 1))
        s = cp.SolverLookup.get("z3", m)
        self.assertFalse(s.solve()) # upgrade z3 with pip install --upgrade z3-solver

    def test_pow(self):
        iv1 = cp.intvar(2,9)
        for i in [0,1,2]:
            self.assertTrue( cp.Model( iv1**i >= 0 ).solve() )

    def test_objective(self):
        iv = cp.intvar(0,10, shape=2)
        m = cp.Model(iv >= 1, iv <= 5, maximize=sum(iv))
        self.assertTrue( m.solve() )
        self.assertEqual( m.objective_value(), 10 )

        m = cp.Model(iv >= 1, iv <= 5, minimize=sum(iv))
        self.assertTrue( m.solve() )
        self.assertEqual( m.objective_value(), 2 )

    def test_only_objective(self):
        # from test_sum_unary and #95
        v = cp.intvar(1,9)
        model = cp.Model(minimize=sum([v]))
        self.assertTrue(model.solve())
        self.assertEqual(v.value(), 1)

    # minizinc: ignore inconsistency warning when deliberately testing unsatisfiable model
    @pytest.mark.filterwarnings("ignore:model inconsistency detected")
    def test_false(self):
        m = cp.Model([cp.boolvar(), False])
        for name, cls in cp.SolverLookup.base_solvers():
            if cls.supported():
                self.assertFalse(m.solve(solver=name))

