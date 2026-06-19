import cpmpy as cp
from cpmpy.exceptions import NotSupportedError
from cpmpy import SolverLookup

import pytest


@pytest.mark.usefixtures("solver")
class TestSolutionHinting:

    def test_hints(self, solver):
        if solver == "rc2":
            pytest.skip("does not support solution hints")

        a,b = cp.boolvar(shape=2)
        model = cp.Model(a | b)

        slv = cp.SolverLookup.get(solver, model)
        try:
            slv.solution_hint([], [])
        except NotSupportedError:
            pytest.skip(f"{solver} does not support solution hinting")
            return
        
        if solver == "gurobi":
            pytest.skip("Gurobi supports solution hinting, but simple models are solved too fast to see the effect")
            return
        
        if solver == "ortools":
            args = {"cp_model_presolve": False} # hints are not taken into account in presolve
        elif solver == "cplex":
            args = {"clean_before_solve": True} # will continue from previous solution otherwise
        else:
            args = {}

        slv.solution_hint([a,b], [True, False]) # check hints are used
        assert slv.solve(**args)
        assert a.value() == True
        assert b.value() == False

        slv.solution_hint([a,b], [False, True]) # check hints are correctly overwritten
        assert slv.solve(**args)
        assert a.value() == False
        assert b.value() == True

        slv.solution_hint([a,b], [False,False])
        assert slv.solve(**args) # should also work with an UNSAT hint

        # disabled, does not match type hints of `def solution_hint`
        # slv.solution_hint([a,[b]], [[[False]], True]) # check nested lists
        # assert slv.solve(**args)

