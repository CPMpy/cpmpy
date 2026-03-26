import cpmpy as cp
from cpmpy.exceptions import NotSupportedError


class TestSolveAll:


    def test_solveall_no_obj(self):

        a,b = cp.boolvar(shape=2)
        m = cp.Model(a | b)

        for name, solver in cp.SolverLookup.base_solvers():
            if not solver.supported():
                continue
            if name == "rc2":
                continue

            sols = set()
            add_sol = lambda: sols.add(str([a.value(), b.value()]))

            solver = cp.SolverLookup.get(name,model=m)

            # special case for some solvers
            kwargs = dict(display=add_sol)
            if name in ("gurobi", "cplex"):
                kwargs['solution_limit'] =  1000
            elif name == "hexaly":
                kwargs['time_limit'] = 5

            count = solver.solveAll(**kwargs)
            assert 3 == count
            assert sols == \
                                {"[True, True]", "[True, False]", "[False, True]"}

    def test_solveall_with_obj(self):

        x = cp.intvar(0, 3, shape=3)
        m = cp.Model(cp.sum(x) >= 1, minimize=cp.sum(x))

        for name in cp.SolverLookup.solvernames():
            try:
                sols = set()
                add_sol = lambda: sols.add(str(x.value().tolist()))

                kwargs = dict(display=add_sol)
                if name in ("gurobi", "cplex"):
                    kwargs['solution_limit'] = 1000
                elif name == "hexaly":
                    kwargs['time_limit'] = 5

                count = m.solveAll(solver=name,**kwargs)
                assert 3 == count
                assert sols == \
                                    {"[1, 0, 0]","[0, 1, 0]","[0, 0, 1]"}


            except NotSupportedError as e:
                pass # solver does not support finding all optimal solutions

    def test_solve_all_keywords(self):
        a, b = cp.boolvar(shape=2)
        m = cp.Model(a | b)

        assert 3 == m.solveAll('ortools', log_search_progress=True)

    def test_solver_specific_display_ndvararray(self, capsys):
        x = cp.boolvar(shape=3, name="x")
        m = cp.Model(cp.sum(x) == 1)
        expected = {"[True, False, False]", "[False, True, False]", "[False, False, True]"}

        for name in ("cpo", "hexaly"):
            if name not in cp.SolverLookup.solvernames():
                continue

            kwargs = {"display": x}
            if name == "hexaly":
                kwargs["time_limit"] = 5
            else:
                kwargs["solution_limit"] = 3

            n_sols = m.solveAll(solver=name, **kwargs)
            assert n_sols == 3

            out = capsys.readouterr().out
            assert expected == {s for s in out.split("\n") if s}
