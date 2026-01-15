"""
Tests all examples in the `examples` folder

Run from the CPMpy root directory with `python3 -m pytest tests/` to make
sure that you are testing your local version.

Will only run solver tests on solvers that are installed
"""
from glob import glob
from os.path import join
import sys

import runpy
import pytest
from cpmpy import SolverLookup
from cpmpy.exceptions import NotSupportedError, TransformationNotImplementedError
import itertools

EXAMPLES = glob(join("examples", "*.py")) + glob(join("examples", "csplib", "*.py"))
ADVANCED_EXAMPLES = glob(join("examples", "advanced", "*.py"))

EXAMPLES = sorted(EXAMPLES)
ADVANCED_EXAMPLES = sorted(ADVANCED_EXAMPLES)

SKIPPED_EXAMPLES = [
                    "ocus_explanations.py", # waiting for issues to be resolved 
                    "psplib.py", # randomly fails on github due to file creation
                    "nurserostering.py",
                    "test_incremental_solving.py",  # 30s timeout for some solver
                    ]

SKIP_MIP = ['npuzzle.py', 'tst_likevrp.py', 'sudoku_', 'pareto_optimal.py',
            'prob009_perfect_squares.py', 'blocks_world.py', 'flexible_jobshop.py',
            'mario', 'pareto_optimal','prob006_golomb.py', 'tsp.py', 'prob028_bibd.py', 'prob001_car_sequence.py'
            ]

SKIP_MZN = ['blocks_world.py', 'flexible_jobshop.py', 'pareto_optimal.py', 'npuzzle.py', 'sudoku_']


# SOLVERS = SolverLookup.supported()
SOLVERS = [
    "ortools",
    "gurobi",
    "minizinc",
]


# run the test for each combination of solver and example
@pytest.mark.usefixtures("solver")
@pytest.mark.requires_solver(*SOLVERS)
@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.timeout(60)  # 60-second timeout for each test
def test_example(solver, example):
    """Loads the example file and executes its __main__ block with the given solver being set as default.

    Args:
        solver ([string]): Loaded with parametrized solver name
        example ([string]): Loaded with parametrized example filename
    """
    if any(skip_name in example for skip_name in SKIPPED_EXAMPLES):
        pytest.skip(f"Skipped {example}, waiting for issues to be resolved")
    if solver in ('gurobi',) and any(x in example for x in SKIP_MIP):
        return pytest.skip(reason=f"exclude {example} for {solver}, too slow or solver-specific")
    if solver == 'minizinc' and any(x in example for x in SKIP_MZN):
        return pytest.skip(reason=f"exclude {example} for {solver}, too slow or solver-specific")

    base_solvers = SolverLookup.base_solvers
    try:
        if solver:
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
        pytest.skip(f'Skipped {example}, module {str(e).split()[-1]} is required')
    finally:
        SolverLookup.base_solvers = base_solvers

@pytest.mark.parametrize("example", ADVANCED_EXAMPLES)
@pytest.mark.timeout(30)
@pytest.mark.depends_on_solver # let pytest know this test indirectly depends on the solver fixture
def test_advanced_example(example):
    """Loads the advanced example file and executes its __main__ block with no default solver set."""
    if any(skip_name in example for skip_name in SKIPPED_EXAMPLES):
        pytest.skip(f"Skipped {example}, waiting for issues to be resolved")
    test_example(None, example)
