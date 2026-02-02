import inspect
import importlib
import inspect
import unittest
import tempfile
import pytest
import numpy as np
import cpmpy as cp
from cpmpy.expressions.core import Operator
from cpmpy.expressions.utils import argvals

from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.solvers.pindakaas import CPM_pindakaas
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.solvers.z3 import CPM_z3
from cpmpy.solvers.minizinc import CPM_minizinc
from cpmpy.solvers.gurobi import CPM_gurobi
from cpmpy.solvers.exact import CPM_exact
from cpmpy.solvers.choco import CPM_choco
from cpmpy.solvers.cplex import CPM_cplex
from cpmpy import SolverLookup
from cpmpy.exceptions import MinizincNameException, NotSupportedError

from test_constraints import numexprs
from utils import skip_on_missing_pblib

pysat_available = CPM_pysat.supported()
pblib_available = importlib.util.find_spec("pypblib") is not None

class TestSolvers(unittest.TestCase):

    
    @pytest.mark.skip(reason="upstream bug, waiting on release for https://github.com/google/or-tools/issues/4640")
    def test_implied_linear(self):
        x,y,z = cp.intvar(0, 2, shape=3,name="xyz")
        p = cp.boolvar(name="p")
        user_vars = (x, y, z, p)
        test_solve(cp.Model(cp.BoolVal(True)).solveAll(), None, user_vars)

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
        self.assertGreater(x[0].value(), x[1].value())
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
        self.assertGreater(x[0].value(), x[1].value())
        self.assertEqual(cb.solcount, 6)


        # intermediate solutions
        m_opt = cp.Model([x[0] > x[1]], maximize=sum(x))
        s = CPM_ortools(m_opt)
        cpm_status = s.solve(solution_callback=cb)
        self.assertEqual(s.objective_value(), 5.0)

        self.assertGreater(x[0].value(), x[1].value())


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

    # this test fails on OR-tools version <9.6
    def test_ortools_version(self):

        a,b,c,d = [cp.intvar(0,3, name=n) for n in "abcd"]
        p,q,r,s = [cp.intvar(0,3, name=n) for n in "pqrs"]

        bv1, bv2, bv3 = [cp.boolvar(name=f"bv{i}") for i in range(1,4)]

        model = cp.Model()

        model += b != 1
        model += b != 2

        model += c != 0
        model += c != 3

        model += d != 0

        model += p != 2
        model += p != 3

        model += q != 1

        model += r != 1

        model += s != 2

        model += cp.AllDifferent([a,b,c,d])
        model += cp.AllDifferent([p,q,r,s])

        model += bv1.implies(a == 0)
        model += bv2.implies(r == 0)
        model += bv3.implies(a == 2)
        model += (~bv1).implies(p == 0)

        model += bv2 | bv3

        self.assertTrue(model.solve(solver="ortools")) # this is a bug in ortools version 9.5, upgrade to version >=9.6 using pip install --upgrade ortools

    def test_ortools_real_coeff(self):

        m = cp.Model()
        # this works in OR-Tools
        x,y,z = cp.boolvar(shape=3, name=tuple("xyz"))
        m.maximize(0.3 * x + 0.5 * y + 0.6 * z)
        assert m.solve()
        assert m.objective_value() == 1.4
        # this does not
        m += 0.7 * x + 0.8 * y >= 1
        self.assertRaises(TypeError, m.solve)

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
        self.assertSetEqual(set(ps2.get_core()), set([mayo,~mayo]))

        # check get core, more realistic
        self.assertFalse(ps2.solve(assumptions=[mayo]+[v for v in inds]))
        self.assertSetEqual(set(ps2.get_core()), set([mayo,inds[6],inds[9]]))

    @pytest.mark.skipif(not CPM_pysat.supported(),
                        reason="PySAT not installed")
    def test_pysat_subsolver(self):
        pysat = CPM_pysat(subsolver="pysat:cadical195")
        x, y = cp.boolvar(shape=2)
        pysat += x | y
        pysat += ~x | ~y
        assert pysat.solve()

    @pytest.mark.skipif(not (pysat_available and pblib_available), reason="`pysat` is not installed" if not pysat_available else "`pypblib` not installed")
    @skip_on_missing_pblib()
    def test_pysat_card(self):
        b = cp.boolvar()
        x = cp.boolvar(shape=5)

        cons = [sum(x) > 3, sum(x) <= 2, sum(x) == 4, (sum(x) <= 1) & (sum(x) != 2),
                b.implies(sum(x) > 3), b == (sum(x) != 2), (sum(x) >= 3).implies(b)]
        for c in cons:
            self.assertTrue(cp.Model(c).solve("pysat"))
            self.assertTrue(c.value())

    @pytest.mark.skipif(not CPM_pindakaas.supported(),
                        reason="pindakaas not installed")
    def test_pindakaas(self):
        # Construct the model.

        b = cp.boolvar()
        x = cp.boolvar(shape=5)
        y = cp.intvar(0, 2, shape=3)
        model = cp.Model([
            cp.any((x[0], x[1])),
            cp.any((~x[0], ~x[1])),
            2*x[0] + 3*x[1] + 5*x[2] <= 6,
            b.implies(2*x[0] + 3*x[1] + 5*x[2] <= 6),
            2*y[0] + 3*y[1] + 5*y[2] <= 6,
            (cp.Xor([x[0], x[1], x[2]])) >= (cp.BoolVal(True))
        ])

        # any solver
        self.assertTrue(model.solve())
        
        # direct solver
        ps = CPM_pindakaas(model)
        self.assertTrue(ps.solve())


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

    @pytest.mark.skipif(not CPM_minizinc.supported(),
                        reason="MiniZinc not installed")
    def test_minizinc_gcc(self):
        from cpmpy.solvers.minizinc import CPM_minizinc

        iv = cp.intvar(-8, 8, shape=5)
        occ = cp.intvar(0, len(iv), shape=3)
        val = [1, 4, 5]
        model = cp.Model([cp.GlobalCardinalityCount(iv, val, occ)])
        solver = CPM_minizinc(model)
        self.assertTrue(solver.solve())
        self.assertTrue(cp.GlobalCardinalityCount(iv, val, occ).value())
        self.assertTrue(all(cp.Count(iv, val[i]).value() == occ[i].value() for i in range(len(val))))
        occ = [2, 3, 0]
        model = cp.Model([cp.GlobalCardinalityCount(iv, val, occ), cp.AllDifferent(val)])
        solver = CPM_minizinc(model)
        self.assertTrue(solver.solve())
        self.assertTrue(cp.GlobalCardinalityCount(iv, val, occ).value())
        self.assertTrue(all(cp.Count(iv, val[i]).value() == occ[i] for i in range(len(val))))
        self.assertTrue(cp.GlobalCardinalityCount([iv[0],iv[2],iv[1],iv[4],iv[3]], val, occ).value())

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


    def test_only_objective(self):
        # from test_sum_unary and #95
        v = cp.intvar(1,9)
        model = cp.Model(minimize=sum([v]))
        self.assertTrue(model.solve())
        self.assertEqual(v.value(), 1)


    @pytest.mark.skipif(not CPM_exact.supported(),
                        reason="Exact not installed")
    def test_exact(self):
        bv = cp.boolvar(shape=3)
        iv = cp.intvar(0, 9, shape=3)
        # circular 'bigger then', UNSAT
        m = cp.Model([
            bv[0].implies(iv[0] > iv[1]),
            bv[1].implies(iv[1] > iv[2]),
            bv[2].implies(iv[2] > iv[0])
        ])
        s = cp.SolverLookup.get("exact", m)
        self.assertFalse(s.solve(assumptions=bv))
        self.assertTrue({x for x in s.get_core()}=={x for x in bv})

        m = cp.Model(~(iv[0] != iv[1]))
        s = cp.SolverLookup.get("exact", m)
        self.assertTrue(s.solve())

        m = cp.Model((iv[0] == 0) & ((iv[0] != iv[1]) == 0))
        s = cp.SolverLookup.get("exact", m)
        self.assertTrue(s.solve())

        m = cp.Model([~bv, ~((iv[0] + abs(iv[1])) == sum(iv))])
        s = cp.SolverLookup.get("exact", m)
        self.assertTrue(s.solve())


        def _trixor_callback():
            assert (bv[0]+bv[1]+bv[2]).value() >= 1

        m = cp.Model([bv[0] | bv[1] | bv[2]])
        s = cp.SolverLookup.get("exact", m)
        self.assertEqual(s.solveAll(display=_trixor_callback),7)

    @pytest.mark.skipif(not CPM_exact.supported(), 
                        reason="Exact not installed")
    def test_parameters_to_exact(self):
    
        # php with 5 pigeons, 4 holes
        p,h = 40,39
        x = cp.boolvar(shape=(p,h))
        m = cp.Model(x.sum(axis=1) >= 1, x.sum(axis=0) <= 1)

        # this should raise a warning
        with self.assertWarns(UserWarning):
            self.assertFalse(m.solve(solver="exact", verbosity=10))
        
        # can we indeed set a parameter? Try with prooflogging
        proof_file = tempfile.NamedTemporaryFile(delete=False).name
        
        # taken from https://gitlab.com/nonfiction-software/exact/-/blob/main/python_examples/proof_logging.py
        options = {"proof-log": proof_file, "proof-assumptions":"0"}
        exact = cp.SolverLookup.get("exact",m, **options)
        self.assertFalse(exact.solve())

        with open(proof_file+".proof", "r") as f:
            self.assertEqual(f.readline()[:-1], "pseudo-Boolean proof version 1.1") # check header of proof-file

    @pytest.mark.skipif(not CPM_choco.supported(),
                        reason="pychoco not installed")
    def test_choco(self):
        bv = cp.boolvar(shape=3)
        iv = cp.intvar(0, 9, shape=3)
        # circular 'bigger then', UNSAT
        m = cp.Model([
            bv[0].implies(iv[0] > iv[1]),
            bv[1].implies(iv[1] > iv[2]),
            bv[2].implies(iv[2] > iv[0])
        ])
        m += sum(bv) == len(bv)
        s = cp.SolverLookup.get("choco", m)

        self.assertFalse(s.solve())

        m = cp.Model(~(iv[0] != iv[1]))
        s = cp.SolverLookup.get("choco", m)
        self.assertTrue(s.solve())

        m = cp.Model((iv[0] == 0) & ((iv[0] != iv[1]) == 0))
        s = cp.SolverLookup.get("choco", m)
        self.assertTrue(s.solve())

        m = cp.Model([~bv, ~((iv[0] + abs(iv[1])) == sum(iv))])
        s = cp.SolverLookup.get("choco", m)
        self.assertTrue(s.solve())

    @pytest.mark.skipif(not CPM_choco.supported(),
                        reason="pychoco not installed")
    def test_choco_element(self):

        # test 1-D
        iv = cp.intvar(-8, 8, 3)
        idx = cp.intvar(-8, 8)
        # test directly the constraint
        constraints = [cp.Element(iv, idx) == 8]
        model = cp.Model(constraints)
        s = cp.SolverLookup.get("choco", model)
        self.assertTrue(s.solve())
        self.assertTrue(iv.value()[idx.value()] == 8)
        self.assertTrue(cp.Element(iv, idx).value() == 8)
        # test through __get_item__
        constraints = [iv[idx] == 8]
        model = cp.Model(constraints)
        s = cp.SolverLookup.get("choco", model)
        self.assertTrue(s.solve())
        self.assertTrue(iv.value()[idx.value()] == 8)
        self.assertTrue(cp.Element(iv, idx).value() == 8)
        # test 2-D
        iv = cp.intvar(-8, 8, shape=(3, 3))
        idx = cp.intvar(0, 3)
        idx2 = cp.intvar(0, 3)
        constraints = [iv[idx, idx2] == 8]
        model = cp.Model(constraints)
        s = cp.SolverLookup.get("choco", model)
        self.assertTrue(s.solve())
        self.assertTrue(iv.value()[idx.value(), idx2.value()] == 8)

    @pytest.mark.skipif(not CPM_choco.supported(),
                        reason="pychoco not installed")
    def test_choco_gcc_alldiff(self):

        iv = cp.intvar(-8, 8, shape=5)
        occ = cp.intvar(0, len(iv), shape=3)
        val = [1, 4, 5]
        model = cp.Model([cp.GlobalCardinalityCount(iv, val, occ)])
        solver = cp.SolverLookup.get("choco", model)
        self.assertTrue(solver.solve())
        self.assertTrue(cp.GlobalCardinalityCount(iv, val, occ).value())
        self.assertTrue(all(cp.Count(iv, val[i]).value() == occ[i].value() for i in range(len(val))))
        occ = [2, 3, 0]
        model = cp.Model([cp.GlobalCardinalityCount(iv, val, occ), cp.AllDifferent(val)])
        solver = cp.SolverLookup.get("choco", model)
        self.assertTrue(solver.solve())
        self.assertTrue(cp.GlobalCardinalityCount(iv, val, occ).value())
        self.assertTrue(all(cp.Count(iv, val[i]).value() == occ[i] for i in range(len(val))))
        self.assertTrue(cp.GlobalCardinalityCount([iv[0],iv[2],iv[1],iv[4],iv[3]], val, occ).value())

    @pytest.mark.skipif(not CPM_choco.supported(),
                        reason="pychoco not installed")
    def test_choco_inverse(self):
        from cpmpy.solvers.ortools import CPM_ortools

        fwd = cp.intvar(0, 9, shape=10)
        rev = cp.intvar(0, 9, shape=10)

        # Fixed value for `fwd`
        fixed_fwd = [9, 4, 7, 2, 1, 3, 8, 6, 0, 5]
        # Inverse of the above
        expected_inverse = [8, 4, 3, 5, 1, 9, 7, 2, 6, 0]

        model = cp.Model(cp.Inverse(fwd, rev), fwd == fixed_fwd)

        solver = cp.SolverLookup.get("choco", model)
        self.assertTrue(solver.solve())
        self.assertEqual(list(rev.value()), expected_inverse)

    @pytest.mark.skipif(not CPM_choco.supported(),
                        reason="pychoco not installed")
    def test_choco_objective(self):
        iv = cp.intvar(0,10, shape=2)
        m = cp.Model(iv >= 1, iv <= 5, maximize=sum(iv))
        s = cp.SolverLookup.get("choco", m)
        self.assertTrue( s.solve() )
        self.assertEqual( s.objective_value(), 10)

        m = cp.Model(iv >= 1, iv <= 5, minimize=sum(iv))
        s = cp.SolverLookup.get("choco", m)
        self.assertTrue( s.solve() )
        self.assertEqual(s.objective_value(), 2)

    @pytest.mark.skipif(not CPM_gurobi.supported(),
                        reason="Gurobi not installed")
    def test_gurobi_element(self):
        # test 1-D
        iv = cp.intvar(-8, 8, 3)
        idx = cp.intvar(-8, 8)
        # test directly the constraint
        constraints = [cp.Element(iv,idx) == 8]
        model = cp.Model(constraints)
        s = cp.SolverLookup.get("gurobi", model)
        self.assertTrue(s.solve())
        self.assertTrue(iv.value()[idx.value()] == 8)
        self.assertTrue(cp.Element(iv,idx).value() == 8)
        # test through __get_item__
        constraints = [iv[idx] == 8]
        model = cp.Model(constraints)
        s = cp.SolverLookup.get("gurobi", model)
        self.assertTrue(s.solve())
        self.assertTrue(iv.value()[idx.value()] == 8)
        self.assertTrue(cp.Element(iv, idx).value() == 8)
        # test 2-D
        iv = cp.intvar(-8, 8, shape=(3, 3))
        idx = cp.intvar(0, 3)
        idx2 = cp.intvar(0, 3)
        constraints = [iv[idx,idx2] == 8]
        model = cp.Model(constraints)
        s = cp.SolverLookup.get("gurobi", model)
        self.assertTrue(s.solve())
        self.assertTrue(iv.value()[idx.value(), idx2.value()] == 8)

    @pytest.mark.skipif(not CPM_gurobi.supported(),
                        reason="Gurobi not installed")
    def test_gurobi_float_objective(self):
        """Test that Gurobi properly handles float objective values."""
        # Test case with float coefficients that should result in a float objective value
        x, y, z = cp.boolvar(shape=3, name=tuple("xyz"))

        # Create a model with float coefficients - this can happen with DirectVar
        # or when using floats as coefficients
        m = cp.Model()
        m.maximize(0.3 * x + 0.7 * y + 1.5 * z)

        s = cp.SolverLookup.get("gurobi", m)
        self.assertTrue(s.solve())

        # The optimal solution should be x=True, y=True, z=True with objective = 2.5
        expected_obj = 2.5
        actual_obj = s.objective_value()

        # Verify that the objective value is returned as a float (not int)
        self.assertIsInstance(actual_obj, float)
        self.assertAlmostEqual(actual_obj, expected_obj, places=5)


    @pytest.mark.skipif(not CPM_cplex.supported(),
                        reason="cplex not installed")
    def test_cplex(self):
        bv = cp.boolvar(shape=3)
        iv = cp.intvar(0, 10, shape=3)
        # circular 'bigger then', UNSAT
        m = cp.Model([
            bv[0].implies(iv[0] > iv[1]),
            bv[1].implies(iv[1] > iv[2]),
            bv[2].implies(iv[2] > iv[0])
        ])
        m += sum(bv) == len(bv)
        s = cp.SolverLookup.get("cplex", m)
        self.assertFalse(s.solve())

        m = cp.Model(~(iv[0] != iv[1]))
        s = cp.SolverLookup.get("cplex", m)
        self.assertTrue(s.solve())

        m = cp.Model((iv[0] == 0) & ((iv[0] != iv[1]) == 0))
        s = cp.SolverLookup.get("cplex", m)
        self.assertTrue(s.solve())

        m = cp.Model([~bv, ~((iv[0] + abs(iv[1])) == sum(iv))])
        s = cp.SolverLookup.get("cplex", m)
        self.assertTrue(s.solve())


    @pytest.mark.skipif(not CPM_cplex.supported(),
                        reason="cplex not installed")
    def test_cplex_float_objective(self):
        """Test that cplex properly handles float objective values."""
        # Test case with float coefficients that should result in a float objective value
        x, y, z = cp.boolvar(shape=3, name=tuple("xyz"))

        # Create a model with float coefficients - this can happen with DirectVar
        # or when using floats as coefficients
        m = cp.Model()
        m.maximize(0.3 * x + 0.7 * y + 1.5 * z)

        s = cp.SolverLookup.get("cplex", m)
        self.assertTrue(s.solve())

        # The optimal solution should be x=True, y=True, z=True with objective = 2.5
        expected_obj = 2.5
        actual_obj = s.objective_value()

        # Verify that the objective value is returned as a float (not int)
        self.assertIsInstance(actual_obj, float)
        self.assertAlmostEqual(actual_obj, expected_obj, places=5)

    @pytest.mark.skipif(not CPM_cplex.supported(),
                        reason="cplex not installed")
    def test_cplex_solveAll(self):
        iv = cp.intvar(0,5, shape=3)
        m = cp.Model(cp.AllDifferent(iv))
        s = cp.SolverLookup.get("cplex", m)
        sol_count = s.solveAll(solution_limit=10)
        self.assertTrue(sol_count == 10)
        self.assertEqual(s.status().exitstatus, ExitStatus.FEASIBLE)

    @pytest.mark.skipif(not CPM_cplex.supported(),
                        reason="cplex not installed")
    def test_cplex_objective(self):
        iv = cp.intvar(0,10, shape=2)
        m = cp.Model(iv >= 1, iv <= 5, maximize=sum(iv))
        s = cp.SolverLookup.get("cplex", m)
        self.assertTrue( s.solve() )
        self.assertEqual( s.objective_value(), 10)

        m = cp.Model(iv >= 1, iv <= 5, minimize=sum(iv))
        s = cp.SolverLookup.get("cplex", m)
        self.assertTrue( s.solve() )
        self.assertEqual(s.objective_value(), 2)

    @pytest.mark.skipif(not CPM_minizinc.supported(),
                        reason="Minizinc not installed")
    def test_count_mzn(self):
        # bug #461
        from cpmpy.expressions.core import Operator

        iv = cp.intvar(0,10, shape=3)
        x = cp.intvar(0,1)
        y = cp.intvar(0,1)
        wsum = Operator("wsum", [[1,2,3],[x,y,cp.Count(iv,3)]])

        m = cp.Model([x + y == 2, wsum == 9])
        self.assertTrue(m.solve(solver="minizinc"))


@pytest.mark.usefixtures("solver")
class TestSupportedSolvers:
    def test_installed_solvers(self, solver):
        # basic model
        v = cp.boolvar(3)
        x, y, z = v

        model = cp.Model(
                    x.implies(y & z),
                    y | z,
                    ~ z
                )

        model.solve(solver=solver)
        assert [int(a) for a in v.value()] == [0, 1, 0]

        s = cp.SolverLookup.get(solver)
        s.solve()
        assert [int(a) for a in v.value()] == [0, 1, 0]

    def test_time_limit(self, solver):
        if solver == "pysdd": # pysdd does not support time limit
            pytest.skip("time limit not supported")
        
        x = cp.boolvar(shape=3)
        m = cp.Model(x[0] | x[1] | x[2])
        assert m.solve(solver=solver, time_limit=1)

        try:
            m.solve(solver=solver, time_limit=-1)
            assert False
        except ValueError:
            pass

    def test_installed_solvers_solveAll(self, solver):
        # basic model
        v = cp.boolvar(3)
        x, y, z = v

        model = cp.Model(
                    x.implies(y & z),
                    y | z
                )

        time_limit, solution_limit = None, None
        if solver in ("gurobi", "cplex"):
            solution_limit = 5
        if solver == "hexaly":
            time_limit =2

        assert model.solveAll(solver=solver, time_limit=time_limit, solution_limit=solution_limit) == 4

    def test_objective(self, solver):
        iv = cp.intvar(0, 10, shape=2)
        m = cp.Model(iv >= 1, iv <= 5)

        try:
            m.maximize(sum(iv))
            assert m.solve(solver=solver)
            assert m.objective_value() == 10
        except NotSupportedError:
            return None
        
        if solver == "rc2":
            pytest.skip("does not support re-optimisation")

        # if the above works, so should everything below
        m.minimize(sum(iv))
        assert m.solve(solver=solver)
        assert m.objective_value() == 2

        # something slightly more exotic
        m.maximize(cp.min(iv))
        assert m.solve(solver=solver)
        assert m.objective_value() == 5

    def test_value_cleared(self, solver):
        x, y, z = cp.boolvar(shape=3)
        sat_model = cp.Model(cp.any([x,y,z]))
        unsat_model = cp.Model([x | y | z, ~x, ~y,~z])

        assert sat_model.solve(solver=solver)
        for v in (x,y,z):
            assert v.value() is not None
        assert not unsat_model.solve(solver=solver)
        for v in (x,y,z):
            assert v.value() is None

    def test_incremental_objective(self, solver):
        x = cp.intvar(0,10,shape=3)

        if solver == "choco":
            """
            Choco does not support first optimizing and then adding a constraint.
            During optimization, additional constraints get added to the solver,
            which removes feasible solutions.
            No straightforward way to resolve this for now.
            """
            return
        if solver == "gcs":
            return
        s = cp.SolverLookup.get(solver)
        try:
            s.minimize(cp.sum(x))
        except (NotSupportedError, NotImplementedError): # solver does not support optimization
            return

        assert s.solve()
        assert s.objective_value() == 0
        if solver == "rc2":
            return # RC2 only supports setting obj once
        s += x[0] == 5
        s.solve()
        assert s.objective_value() == 5
        s.maximize(cp.sum(x))
        assert s.solve()
        assert s.objective_value() == 25

    def test_incremental(self, solver):
        x, y, z = cp.boolvar(shape=3, name="x")
        s = cp.SolverLookup.get(solver)
        s += [x]
        s += [y | z]
        assert s.solve()
        assert x.value(), (y | z).value()
        s += ~y | ~z
        assert s.solve()
        assert x.value()
        assert y.value() + z.value() == 1


    def test_incremental_assumptions(self, solver):
        x, y, z = cp.boolvar(shape=3, name=["x","y","z"])
        s = cp.SolverLookup.get(solver)
        if "assumptions" not in inspect.signature((s.solve)).parameters:
            return # solver does not support solving under assumptions
        
        if solver == "pysdd":
            return # not implemented in pysdd
        
        s += x | y
        assert s.solve(assumptions=[x])
        assert x.value()
        assert (s.solve(assumptions=[~x]))
        assert not (x.value())
        assert (y.value())

        s += ~x | z
        assert s.solve(assumptions=[x,~y])
        assert z.value()

        s += y | ~z
        assert not s.solve(assumptions=[~x, ~y])

        core = s.get_core()
        assert len(core) > 0
        assert ~y in core
        assert cp.Model([x | y, ~x | z, y | ~z] + core).solve() is False # ensure it is indeed unsat

        assert s.solve(assumptions=[])

    def test_vars_not_removed(self, solver):

        bvs = cp.boolvar(shape=3)
        m = cp.Model([cp.any(bvs) <= 2])

        # reset value for vars
        bvs.clear()
        assert m.solve(solver=solver)
        for v in bvs:
            assert v.value() is not None
        #test solve_all
        sols = set()
        time_limit, solution_limit = None, None
        if solver in ("gurobi", "cplex"):
            solution_limit = 10
        if solver == "hexaly":
            time_limit = 2
        # test number of solutions is valid
        assert m.solveAll(solver=solver, time_limit=time_limit, solution_limit=solution_limit,
                          display=lambda: sols.add(tuple([x.value() for x in bvs]))) == 8
        #test number of solutions is valid, no display
        assert m.solveAll(solver=solver, solution_limit=solution_limit, time_limit=time_limit) == 8
        #test unique sols, should be same number
        assert len(sols) == 8


    # minizinc: ignore inconsistency warning when deliberately testing unsatisfiable model
    @pytest.mark.filterwarnings("ignore:model inconsistency detected")
    def test_false(self, solver):
        assert not cp.Model([cp.boolvar(), False]).solve(solver=solver)

    def test_partial_div_mod(self, solver):
        if solver in ("pysdd", "pysat", "pindakaas", "pumpkin", "rc2"):  # don't support div or mod with vars
            return
        if solver == 'cplex':
            pytest.skip("skip for cplex, cplex supports solveall only for MILPs, and this is not linear.")
        x,y,d,r = cp.intvar(-5, 5, shape=4,name=['x','y','d','r'])
        vars = [x,y,d,r]
        m = cp.Model()
        # modulo toplevel
        m += x / y == d
        m += x % y == r
        sols = set()
        solution_limit = None
        time_limit = None
        if solver == 'gurobi':
            solution_limit = 15 # Gurobi does not like this model, and gets stuck finding all solutions
        if solver == "hexaly":
            time_limit = 5
        m.solveAll(solver=solver, solution_limit=solution_limit, time_limit=time_limit, display=lambda: sols.add(tuple(argvals(vars))))
        for sol in sols:
            xv, yv, dv, rv = sol
            assert dv * yv + rv == xv
            assert (cp.Division(xv, yv)).value() == dv
            assert (cp.Modulo(xv, yv)).value() == rv


    def test_status(self, solver):

        bv = cp.boolvar(shape=3, name="bv")
        m = cp.Model(cp.any(bv))

        assert m.status().exitstatus == ExitStatus.NOT_RUN
        assert m.solve(solver=solver)
        assert m.status().exitstatus == ExitStatus.FEASIBLE

        try: # now try optimization, not supported for all solvers
            m.maximize(cp.sum(bv))
            assert m.solve(solver=solver)
            assert m.status().exitstatus == ExitStatus.OPTIMAL
        except NotSupportedError:
            return

        # now making a tricky problem to solve
        np.random.seed(0)
        start = cp.intvar(0,100, shape=50)
        dur = np.random.randint(1,5, size=50)
        end = cp.intvar(0,100, shape=50)
        demand  = np.random.randint(10,15, size=50)

        m += cp.Cumulative(start, dur, end,demand, 30)
        m.minimize(cp.max(end))
        m.solve(solver=solver, time_limit=1)
        # normally, should not be able to solve within 1s...
        assert m.status().exitstatus in (ExitStatus.FEASIBLE, ExitStatus.UNKNOWN)

        # now trivally unsat
        m += cp.sum(bv) <= 0
        m.solve(solver=solver)
        assert m.status().exitstatus == ExitStatus.UNSATISFIABLE



    def test_status_solveall(self, solver):
        if solver == "hexaly":
            pytest.skip("hexaly cannot proveably find all solutions, so status is never OPTIMAL")

        bv = cp.boolvar(shape=3, name="bv")
        m = cp.Model(cp.any(bv))

        limit = None
        if solver in ("gurobi", "cplex"): limit = 100000

        num_sols = m.solveAll(solver=solver, solution_limit=limit)
        assert num_sols == 7
        assert m.status().exitstatus == ExitStatus.OPTIMAL  # optimal



        # adding a bunch of variables to increase nb of sols
        try:
            x = cp.boolvar(shape=32, name="x")
            m = cp.Model(cp.any(x))
            num_sols = m.solveAll(solver=solver, time_limit=1, solution_limit=limit)
            assert m.status().exitstatus == ExitStatus.FEASIBLE

            num_sols = m.solveAll(solver=solver, solution_limit=10)
            assert num_sols == 10
            assert m.status().exitstatus == ExitStatus.FEASIBLE

            # edge-case: nb of solutions is exactly the sol limit
            m = cp.Model(cp.any(bv))
            num_sols = m.solveAll(solver=solver, solution_limit=7)
            assert num_sols ==  7
            assert m.status().exitstatus in (ExitStatus.OPTIMAL, ExitStatus.FEASIBLE) # which of the two?

        except NotImplementedError:
            pass # not all solvers support time/solution limits

        # making the problem unsat
        if solver != "pysdd": # constraint not supported by pysdd
            m  = cp.Model([cp.sum(bv) <= 0, cp.any(bv)])
            num_sols = m.solveAll(solver=solver, solution_limit=limit)
            assert num_sols == 0
            assert m.status().exitstatus == ExitStatus.UNSATISFIABLE


    def test_hidden_user_vars(self, solver):
        """
        Tests whether decision variables which are part of a constraint that never gets posted to the underlying solver
        still get correctly captured and posted.
        """
        if solver == 'pysdd':
            pytest.skip(reason=f"{solver} does not support integer decision variables")
        
        x = cp.intvar(1, 4, shape=1)
        # Dubious constraint which enforces nothing, gets decomposed to empty list
        # -> resulting CP model is empty
        m = cp.Model([cp.AllDifferentExceptN([x], 1)])
        s = cp.SolverLookup().get(solver, m)
        assert len(s.user_vars) == 1 # check if var captured as a user_var

        kwargs = dict()
        if solver in ("gurobi", "cplex"):
            kwargs['solution_limit'] = 5
        if solver == "hexaly":
            kwargs['time_limit'] = 2
        assert s.solveAll(**kwargs ) == 4   # check if still correct number of solutions, even though empty model

    def test_model_no_vars(self, solver):

        kwargs = dict()
        if solver in ("gurobi", "cplex"):
            kwargs['solution_limit'] = 10
        elif solver == "hexaly":
            kwargs['time_limit'] = 2

        # empty model
        num_sols = cp.Model().solveAll(solver=solver, **kwargs)
        assert num_sols == 1    

        # model with one True constant
        num_sols = cp.Model(cp.BoolVal(True)).solveAll(solver=solver,**kwargs)
        assert num_sols == 1        

        # model with two True constants
        num_sols = cp.Model(cp.BoolVal(True), cp.BoolVal(True)).solveAll(solver=solver, **kwargs)
        assert num_sols == 1

        # model with one False constant
        num_sols = cp.Model(cp.BoolVal(False)).solveAll(solver=solver, **kwargs)
        assert num_sols == 0

    def test_version(self, solver):
        solver_version = SolverLookup.lookup(solver).version()
        assert solver_version is not None
        assert isinstance(solver_version, str)

    def test_optimisation_direction(self, solver):
        x = cp.intvar(0, 10, shape=1)
        m = cp.Model(x >= 5)

        # TODO: in the future this might be simplified to a filter using pytest markers, first #780 needs to be merged
        # 1) Maximisation - model
        try: # one try-except to detect if the solver supports optimisation
            m.maximize(x)
            assert m.solve(solver=solver)
        except (NotImplementedError, NotSupportedError):
            pytest.skip(reason=f"{solver} does not support optimisation")
            return
        assert m.objective_value() == 10

        # 2) Maximisation - solver
        s = cp.SolverLookup.get(solver, m)
        assert s.solve()
        assert s.objective_value() == 10

        # 3) Minimisation - model
        m.minimize(x)
        assert m.solve(solver=solver)
        assert m.objective_value() == 5

        # 4) Minimisation - solver
        s = cp.SolverLookup.get(solver, m)
        assert s.solve()
        assert s.objective_value() == 5

    @skip_on_missing_pblib()
    def test_bug810(self, solver):
        if solver == "pysdd":  # non-supported constraint
            pytest.skip(reason=f"{solver} does not support int*boolvar")

        kwargs = {}
        if solver in ("gurobi", "cplex"):
            kwargs["solution_limit"] = 10
        p, q = cp.boolvar(2)
        model = cp.Model(p.implies(3 * q == 2))
        assert model.solve(solver)
        assert model.solveAll(solver, **kwargs) == 2


@pytest.mark.generate_constraints.with_args(numexprs)
def test_objective_numexprs(solver, constraint):

    model = cp.Model(cp.intvar(0, 10, shape=3) >= 1) # just to have some constraints
    try:
        model.minimize(constraint)
        assert model.solve(solver=solver, time_limit=3)
        assert constraint.value() < constraint.get_bounds()[1] # bounds are not always tight, but should be smaller than ub for sure
        model.maximize(constraint)
        assert model.solve(solver=solver)
        assert constraint.value() > constraint.get_bounds()[0] # bounds are not always tight, but should be larger than lb for sure
    except NotSupportedError:
        pytest.skip(reason=f"{solver} does not support optimisation")
