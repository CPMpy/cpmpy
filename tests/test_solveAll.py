import pytest
import cpmpy as cp
from cpmpy.exceptions import NotSupportedError
from cpmpy.solvers.solver_interface import ExitStatus

@pytest.mark.usefixtures("solver")
class TestSolveAll:

    def test_status_solveall(self, solver):
        if solver == "hexaly":
            pytest.skip("hexaly cannot proveably find all solutions, so status is never OPTIMAL")

        bv = cp.boolvar(shape=3, name="bv")
        m = cp.Model(cp.any(bv))

        limit = None
        if solver in ("gurobi", "cplex"):
            limit = 100000

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
            assert num_sols == 7
            assert m.status().exitstatus in (ExitStatus.OPTIMAL, ExitStatus.FEASIBLE)  # which of the two?

        except NotImplementedError:
            pass  # not all solvers support time/solution limits

        # making the problem unsat
        if solver != "pysdd":  # constraint not supported by pysdd
            m = cp.Model([cp.sum(bv) <= 0, cp.any(bv)])
            num_sols = m.solveAll(solver=solver, solution_limit=limit)
            assert num_sols == 0
            assert m.status().exitstatus == ExitStatus.UNSATISFIABLE

    def test_solveall_no_obj(self, solver):
        if solver == "rc2":
            pytest.skip(reason="rc2 does not support decision problems (solveAll)")
        a, b = cp.boolvar(shape=2)
        m = cp.Model(a | b)

        sols = set()
        add_sol = lambda: sols.add(str([a.value(), b.value()]))

        s = cp.SolverLookup.get(solver, model=m)

        kwargs = dict(display=add_sol)
        if solver in ("gurobi", "cplex"):
            kwargs["solution_limit"] = 1000
        elif solver == "hexaly":
            kwargs["time_limit"] = 5

        count = s.solveAll(**kwargs)
        assert 3 == count
        assert sols == {"[True, True]", "[True, False]", "[False, True]"}

    def test_solveall_with_obj(self, solver):
        x = cp.intvar(0, 3, shape=3)
        m = cp.Model(cp.sum(x) >= 1, minimize=cp.sum(x))

        try:
            sols = set()
            add_sol = lambda: sols.add(str(x.value().tolist()))

            kwargs = dict(display=add_sol)
            if solver in ("gurobi", "cplex"):
                kwargs["solution_limit"] = 1000
            elif solver == "hexaly":
                kwargs["time_limit"] = 5

            count = m.solveAll(solver=solver, **kwargs)
            assert 3 == count
            assert sols == {"[1, 0, 0]", "[0, 1, 0]", "[0, 0, 1]"}
        except NotSupportedError:
            pytest.skip(reason=f"{solver} does not support finding all optimal solutions")

    def test_solve_all_keywords(self):
        a, b = cp.boolvar(shape=2)
        m = cp.Model(a | b)

        assert 3 == m.solveAll('ortools', log_search_progress=True)


@pytest.mark.skip()
class TestPrint:
    def test_solveAll_display_expr(self, solver, capsys):
        x = cp.boolvar(shape=3, name="x")
        m = cp.Model(cp.sum(x) == 1)

        n_sols = m.solveAll(solver=solver, display=x[0], solution_limit=3)  # should print 3 sols
        assert n_sols == 3
        out = capsys.readouterr().out
        assert {"True", "False"} == set([s for s in out.split("\n") if len(s)])

    def test_solveAll_display_ndvararray(self, solver, capsys):
        x = cp.boolvar(shape=3,name="x")
        m = cp.Model(cp.sum(x) == 1)

        m.solveAll(solver=solver, display=x, solution_limit=3) # should print 3 sols
        out = capsys.readouterr().out
        assert {"[True, False, False]", "[False, True, False]", "[False, False, True]"} == set([s for s in out.split("\n") if len(s)])

    def test_solveAll_display_list(self, solver, capsys):
        x = cp.boolvar(shape=3, name="x")
        m = cp.Model(cp.sum(x) == 1)

        m.solveAll(solver=solver, display=list(x), solution_limit=3)  # should print 3 sols
        out = capsys.readouterr().out
        assert {"[True, False, False]", "[False, True, False]", "[False, False, True]"} ==  set([s for s in out.split("\n") if len(s)])

