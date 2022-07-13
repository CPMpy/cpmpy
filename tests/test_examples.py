"""Tests all examples in the ../examples folder
with all the solvers available"""
from glob import glob
from os.path import join
import types
import importlib.machinery
import pytest
from cpmpy import *

EXAMPLES = glob(join("..", "examples", "*.py")) + glob(join(".", "examples", "*.py")) + glob(join(".", "examples/advanced", "*.py"))

@pytest.mark.parametrize("example", EXAMPLES)
def test_examples(example):
    """Loads example files and executes with default solver

class TestExamples(unittest.TestCase):

    Args:
        example ([string]): Loaded with parametrized example filename
    """
    # do not run, dependency local to that folder
    if example.endswith('explain_satisfaction.py'):
        return
    loader = importlib.machinery.SourceFileLoader("example", example)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    # run again with gurobi
    if any(x in example for x in ["npuzzle","tst_likevrp","ortools_presolve_propagate"]):
        return

    gbi_slv = SolverLookup.lookup("gurobi")
    if gbi_slv.supported():
        # temporarily brute-force overwrite SolverLookup.base_solvers
        f = SolverLookup.base_solvers
        try:
            SolverLookup.base_solvers = lambda: [('gurobi', gbi_slv)]
            loader.exec_module(mod)
        finally:
            SolverLookup.base_solvers = f

    # run again with minizinc, if installed on system
    if example in ['./examples/npuzzle.py', './examples/tsp_likevrp.py']:
        # except for these too slow ones
        return
    mzn_slv = SolverLookup.lookup('minizinc')
    if mzn_slv.supported():
        # temporarily brute-force overwrite SolverLookup.base_solvers
        f = SolverLookup.base_solvers
        try:
            SolverLookup.base_solvers = lambda: [('minizinc', mzn_slv)]
            loader.exec_module(mod)
        finally:
            SolverLookup.base_solvers = f
