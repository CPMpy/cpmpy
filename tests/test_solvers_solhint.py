import unittest

import cpmpy as cp
from cpmpy.exceptions import NotSupportedError


class TestSolutionHinting(unittest.TestCase):


    def test_hints(self):

        a,b = cp.boolvar(shape=2)
        model = cp.Model(a | b)

        for n, solver_class in cp.SolverLookup.base_solvers():
            if not solver_class.supported():
                continue
            slv = solver_class(model)
            try:
                args = {"cp_model_presolve": False} if n == "ortools" else {}  # hints are not taken into account in presolve

                slv.solution_hint([a,b], [True, False])
                self.assertTrue(slv.solve(**args))
                self.assertEqual(a.value(), True)
                self.assertEqual(b.value(), False)

                slv.solution_hint([a,b], [False, True]) # check hints are correctly overwritten
                self.assertTrue(slv.solve(**args))
                self.assertEqual(a.value(), False)
                self.assertEqual(b.value(), True)

                slv.solution_hint([a,b], [False,False])
                self.assertTrue(slv.solve(**args)) # should also work with an UNSAT hint

                slv.solution_hint([a,[b]], [[[False]], True]) # check nested lists
                self.assertTrue(slv.solve(**args))

            except NotSupportedError:
                continue


