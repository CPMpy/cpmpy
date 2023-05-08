"""
Tests all examples in the `examples` folder

Run from the CPMpy root directory with `python3 -m pytest tests/` to make
sure that you are testing your local version.

Will only run solver tests on solvers that are installed
"""
from glob import glob
from os.path import join
from os import getcwd
import types
import importlib.machinery
import pytest
from cpmpy import *

cwd = getcwd()
if 'y' in cwd[-2:]:
    EXAMPLES =  glob(join(".", "examples", "*.py")) + \
                glob(join(".", "examples", "advanced", "*.py")) + \
                glob(join(".", "examples", "csplib", "*.py"))
else:
    EXAMPLES = glob(join("..", "examples", "*.py")) + \
               glob(join("..", "examples", "advanced", "*.py")) + \
               glob(join("..", "examples", "csplib", "*.py"))

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

    # run again with gurobi, if installed on system
    gbi_slv = SolverLookup.lookup("gurobi")
    if any(x in example for x in ["npuzzle","tst_likevrp","ortools_presolve_propagate"]):
        # exclude those, too slow or solver specific
        return
    if gbi_slv.supported():
        # temporarily brute-force overwrite SolverLookup.base_solvers so our solver is default
        f = SolverLookup.base_solvers
        try:
            SolverLookup.base_solvers = lambda: [('gurobi', gbi_slv)]+f()
            loader.exec_module(mod)
        finally:
            SolverLookup.base_solvers = f

    # run again with minizinc, if installed on system
    if example in ['./examples/npuzzle.py', './examples/tsp_likevrp.py']:
        # except for these too slow ones
        return
    mzn_slv = SolverLookup.lookup('minizinc')
    if mzn_slv.supported():
        # temporarily brute-force overwrite SolverLookup.base_solvers so our solver is default
        f = SolverLookup.base_solvers
        try:
            SolverLookup.base_solvers = lambda: [('minizinc', mzn_slv)]+f()
            loader.exec_module(mod)
        finally:
            SolverLookup.base_solvers = f
