"""
Tests all examples in the `examples` folder

Run from the CPMpy root directory with `python3 -m pytest tests/` to make
sure that you are testing your local version.

Will only run solver tests on solvers that are installed
"""
from glob import glob
from os.path import join
from os import getcwd
import sys

import runpy
import pytest
from cpmpy import SolverLookup
from cpmpy.exceptions import NotSupportedError, TransformationNotImplementedError
import itertools

prefix = '.' if 'y' in getcwd()[-2:] else '..'

TO_SKIP = [
    "prob001_convert_data.py"
]

EXAMPLES = glob(join(prefix, "examples", "*.py")) + \
           glob(join(prefix, "examples", "csplib", "*.py"))
EXAMPLES = [e for e in EXAMPLES if not any(x in e for x in TO_SKIP)]

ADVANCED_EXAMPLES = glob(join(prefix, "examples", "advanced", "*.py"))

# SOLVERS = SolverLookup.supported()
SOLVERS = [
    "ortools",
    "gurobi",
    "minizinc",
]

@pytest.mark.parametrize(("solver", "example"), itertools.product(SOLVERS, EXAMPLES))  # run the test for each combination of solver and example
@pytest.mark.timeout(60)  # 60-second timeout for each test
def test_example(solver, example):
    """Loads the example file and executes its __main__ block with the given solver being set as default.

    Args:
        solver ([string]): Loaded with parametrized solver name
        example ([string]): Loaded with parametrized example filename
    """
    if solver in ('gurobi', 'minizinc') and any(x in example for x in ["npuzzle.py", "tst_likevrp.py", 'sudoku_', 'pareto_optimal.py', 'prob009_perfect_squares.py', 'blocks_world.py', 'flexible_jobshop.py']):
        return pytest.skip(reason=f"exclude {example} for {solver}, too slow or solver-specific")

    base_solvers = SolverLookup.base_solvers
    try:
        solver_class = SolverLookup.lookup(solver)
        if not solver_class.supported():
            # check this here, as unsupported solvers can fail the example for various reasons
            return pytest.skip(reason=f"solver {solver} not supported")

        # Overwrite SolverLookup.base_solvers to set the target solver first, making it the default
        SolverLookup.base_solvers = lambda: sorted(base_solvers(), key=lambda s: s[0] == solver, reverse=True)
        sys.argv = [example]  # avoid pytest arguments being passed the executed module
        runpy.run_path(example, run_name="__main__")  # many examples won't do anything `__name__ != "__main__"`
    except (NotSupportedError, TransformationNotImplementedError) as e:
        if solver == 'ortools':  # `from` augments exception trace
            raise Exception(
                "Example not supported by ortools, which is currently able to run all models, but raised") from e
        pytest.skip(
            reason=f"Skipped, solver or its transformation does not support model, raised {type(e).__name__}: {e}")
    except ValueError as e:
        if "Unknown solver" in str(e):
            pytest.skip(reason=f"Skipped, example uses specific solver, raised: {e}")
        else:  # still fail for other reasons
            raise e
    except ModuleNotFoundError as e:
        pytest.skip('Skipped, module {} is required'.format(str(e).split()[-1]))
    finally:
        SolverLookup.base_solvers = base_solvers


@pytest.mark.parametrize("example", ADVANCED_EXAMPLES)
@pytest.mark.timeout(30)
def test_advanced_example(example):
    """Loads the advanced example file and executes its __main__ block with no default solver set."""
    try:
        sys.argv = [example]
        runpy.run_path(example, run_name="__main__")
    except Exception as e:
        if "CPM_exact".lower() in str(e).lower():
            pytest.skip(reason=f"Skipped, example uses Exact but is not installed, raised: {e}")
        else:
            raise e