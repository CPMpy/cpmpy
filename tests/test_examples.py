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
from cpmpy import SolverLookup
from cpmpy.exceptions import NotSupportedError, TransformationNotImplementedError
import itertools

cwd = getcwd()
if 'y' in cwd[-2:]:
    EXAMPLES =  glob(join(".", "examples", "*.py")) + \
                glob(join(".", "examples", "advanced", "*.py")) + \
                glob(join(".", "examples", "csplib", "*.py"))
else:
    EXAMPLES = glob(join("..", "examples", "*.py")) + \
               glob(join("..", "examples", "advanced", "*.py")) + \
               glob(join("..", "examples", "csplib", "*.py"))

# SOLVERS = SolverLookup.supported()
SOLVERS = ["ortools", "gurobi", "minizinc", "pindakaas"]

@pytest.mark.parametrize(("solver", "example"), itertools.product(SOLVERS, EXAMPLES))
def test_examples(solver, example):
    """Loads example files and executes with default solver

    Args:
        solver ([string]): Loaded with parametrized solver name
        example ([string]): Loaded with parametrized example filename
    """
    if solver in ('gurobi', 'minizinc') and any(x in example for x in ["npuzzle", "tst_likevrp", "ortools_presolve_propagate", 'sudoku_ratrun1.py']):
        return pytest.skip(reason=f"exclude {example} for gurobi, too slow or solver specific")

    original_base_solver = SolverLookup.base_solvers
    try:
        solver_class = SolverLookup.lookup(solver)
        if not solver_class.supported():
            return pytest.skip(reason=f"solver {solver} not supported")

        # Overwrite SolverLookup.base_solvers so our solver is the only
        SolverLookup.base_solvers = lambda: [(solver, solver_class)] + original_base_solver()
        loader = importlib.machinery.SourceFileLoader("example", example)
        mod = types.ModuleType(loader.name)
        loader.exec_module(mod)  # this runs the scripts
    except (NotSupportedError, TransformationNotImplementedError) as e:
        if solver == 'ortools':  # `from` augments exception trace
            raise Exception("Example not supported by ortools, which is currently able to run all models, but raised") from e
        pytest.skip(reason=f"Skipped, solver or its transformation does not support model, raised {type(e).__name__}: {e}")
    except ValueError as e:
        if hasattr(e, 'message') and e.message.contains("Unknown solver"):
            pytest.skip(reason=f"Skipped, example uses specific solver, raised: {e}")
        else:  # still fail for other reasons
            raise e
    except ModuleNotFoundError as e:
        pytest.skip('Skipped, module {} is required'.format(str(e).split()[-1]))
    finally:
        SolverLookup.base_solvers = original_base_solver


