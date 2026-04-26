import numpy as np
import pytest

import cpmpy as cp
from cpmpy.solvers import CPM_gurobi, CPM_pysat, CPM_minizinc, CPM_pysdd, CPM_z3, CPM_exact, CPM_choco, CPM_hexaly


class TestDirectORTools:

    def test_direct_automaton(self):
        trans_vars = cp.boolvar(shape=4, name="trans")
        trans_tabl = [ # corresponds to regex 0* 1+ 0+
            (0, 0, 0),
            (0, 1, 1),
            (1, 1, 1),
            (1, 0, 2),
            (2, 0, 2)
        ]

        model = cp.SolverLookup.get("ortools")

        model += cp.DirectConstraint("AddAutomaton", (trans_vars, 0, [2], trans_tabl),
                                     novar=[1, 2, 3])  # optional, what not to scan for vars

        assert model.solveAll() == 6


@pytest.mark.skipif(not CPM_exact.supported(), reason="Exact not installed")
class TestDirectExact:

    def test_direct_left_reif(self):
        x,y = cp.boolvar(2)

        model = cp.SolverLookup.get("exact")
        print(model)
        # add x -> y>=1
        model += cp.DirectConstraint("addRightReification", (x, 1, [(1, y)], 1), novar=[1,3])
        print(model)
        assert model.solveAll() == 3


@pytest.mark.skipif(not CPM_pysat.supported(),
                    reason="PySAT not installed")
class TestDirectPySAT:

    def test_direct_clause(self):
        x,y = cp.boolvar(2)

        model = cp.SolverLookup.get("pysat")

        model += cp.DirectConstraint("add_clause", [x, y])

        assert model.solve()
        assert x.value() or y.value()

@pytest.mark.skipif(not CPM_pysdd.supported(),
                    reason="PySDD not installed")
class TestDirectPySDD:

    def test_direct_clause(self):
        x,y = cp.boolvar(2)

        model = cp.SolverLookup.get("pysdd")

        model += cp.DirectConstraint("conjoin", (x, y))

        assert model.solve()
        assert x.value() or y.value()

@pytest.mark.skipif(not CPM_z3.supported(),
                    reason="Z3py not installed")
class TestDirectZ3:

    def test_direct_clause(self):
        iv = cp.intvar(1,9, shape=3)

        model = cp.SolverLookup.get("z3")

        model += cp.DirectConstraint("Distinct", iv)

        assert model.solve()
        assert cp.AllDifferent(iv).value()

@pytest.mark.skipif(not CPM_minizinc.supported(),
                    reason="MinZinc not installed")
class TestDirectMiniZinc:

    def test_direct_clause(self):
        iv = cp.intvar(1,9, shape=3)

        model = cp.SolverLookup.get("minizinc")

        # MiniZinc is oddly different for DirectConstraint, because it is a text-based language
        # so, as DirectConstraint name, you need to give the name of a text-based constraint,
        # NOT a name of a function of the mzn_model class...

        # this just to demonstrate, there are no 0's in the domains...
        model += cp.DirectConstraint("alldifferent_except_0", iv)

        assert model.solve()
        assert cp.AllDifferent(iv).value()


@pytest.mark.skipif(not CPM_gurobi.supported(),
                    reason="Gurobi not installed")
class TestDirectGurobi:

    def test_direct_poly(self):

        x = cp.intvar(0,10,name="x")
        y = cp.intvar(0,100,name="y")

        model = cp.SolverLookup.get("gurobi")

        # y = 2 x^3 + 1.5 x^2 + 1
        p = [2, 1.5, 0, 1]
        model += cp.DirectConstraint("addGenConstrPoly", (x, y, p),
                                  novar=[2])  # optional, what not to scan for vars

        assert model.solve()

        x_val = x.value()
        x_terms = [x_val**3, x_val**2, x_val**1, x_val**0]
        poly_val = sum(np.array(p)*x_terms)

        assert y.value() == poly_val

@pytest.mark.skipif(not CPM_choco.supported(),
                    reason="pychoco not installed")
class TestDirectChoco:

    def test_direct_global(self):
        iv = cp.intvar(1,9, shape=3)

        model = cp.SolverLookup.get("choco")

        model += cp.DirectConstraint("increasing", iv)
        model += iv[1] < iv[0]

        assert not model.solve()


@pytest.mark.skipif(not CPM_hexaly.supported(),
                    reason="hexaly is not installed")
class TestDirectHexaly:

    def test_direct_distance(self):

        a,b = cp.intvar(0,10,shape=2, name=tuple("ab"))

        model = cp.SolverLookup.get("hexaly")

        model += cp.DirectConstraint("dist",(a,b)) >= 3 # model distance between two variables
        assert model.solve()

        assert abs(a.value() - b.value()) >=3

    def test_floatsum_objective(self):
        x, y, z = cp.boolvar(shape=3, name=tuple("xyz"))
        m = cp.Model(maximize=cp.FloatSum([0.3, 0.5, 0.6], [x, y, z]))
        assert m.solve(solver="hexaly")
        assert m.objective_value() == pytest.approx(1.4, abs=1e-05)
