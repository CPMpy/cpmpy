import unittest

import cpmpy as cp
from cpmpy.exceptions import NotSupportedError


class TestSolveAll(unittest.TestCase):


    def test_solveall_no_obj(self):

        a,b = cp.boolvar(shape=2)
        m = cp.Model(a | b)

        for name, solver in cp.SolverLookup.base_solvers():
            if not solver.supported():
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
            self.assertEqual(3, count)
            self.assertSetEqual(sols,
                                {"[True, True]", "[True, False]", "[False, True]"})

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
                self.assertEqual(3, count)
                self.assertSetEqual(sols,
                                    {"[1, 0, 0]","[0, 1, 0]","[0, 0, 1]"})


            except NotSupportedError as e:
                pass # solver does not support finding all optimal solutions

    def test_solveAll_keywords(self):
        a, b = cp.boolvar(shape=2)
        m = cp.Model(a | b)

        self.assertEqual(3, m.solveAll('ortools', log_search_progress=True))
