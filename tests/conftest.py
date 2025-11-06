import pytest
import cpmpy as cp

def pytest_addoption(parser):
    """
    Adds cli arguments to the pytest command
    """
    parser.addoption(
        "--solver", type=str, action="store", default=None, help="Only run the tests on this particular solver."
    )

@pytest.fixture
def solver(request):
    """
    Limit tests to a specific solver.

    By providing the cli argument `--solver=<SOLVER_NAME>`, two things will happen:
    - non-solver-specific tests which make a `.solve()` call will now use `SOLVER_NAME` as backend (instead of the default OR-Tools)
    - solver-specific tests, like the ones produced through `_generate_inputs`, will be filtered if they don't match `SOLVER_NAME`

    By not providing a value for ``--solver`, the default behaviour will be to run non-solver-specific on the default solver (OR-Tools),
    and to run all solver-specific tests for which the solver has been installed on the system.
    """
    request.cls.solver = request.config.getoption("--solver")
    return request.config.getoption("--solver")

def pytest_configure(config):
    # Register custom marker for documentation and linting
    config.addinivalue_line(
        "markers",
        "requires_solver(name): mark test as requiring a specific solver", # to filter tests when required solver is not installed
    )
    config.addinivalue_line(
        "markers",
        "requires_dependency(name): mark test as requiring a specific dependency", # to filter tests when required solver is not installed
    )


def pytest_collection_modifyitems(config, items):
    """
    Centrally apply filters and skips to test targets.

    For now, only solver-based filtering gets applied.
    """
    cmd_solver = config.getoption("--solver") # get cli `--solver`` arg

    filtered = []
    for item in items:
        required_solver_marker = item.get_closest_marker("requires_solver")
        required_dependency_marker = item.get_closest_marker("requires_dependency")

        # --------------------------------- Dependency filtering --------------------------------- #
        if required_dependency_marker:
            if not all(importlib.util.find_spec(dependency) is not None for dependency in required_dependency_marker.args):
                skip = pytest.mark.skip(reason=f"Dependency {required_dependency_marker.args} not installed")
                item.add_marker(skip)
                continue

        # --------------------------------- Solver filtering --------------------------------- #
        if required_solver_marker:
            required_solvers = required_solver_marker.args

            # --------------------------------- Filtering -------------------------------- #
            
            # when a solver is specified on the command line, 
            # only run solver-specific tests that require that solver
            if cmd_solver:
                if cmd_solver in required_solvers:
                    filtered.append(item)
                else:
                    continue
            # instance has survived filtering
            else:
                filtered.append(item)

            # --------------------------------- Skipping --------------------------------- #

            # skip test if the required solver is not installed
            if not {k:v for k,v in cp.SolverLookup.base_solvers()}[required_solvers[0]].supported():
                skip = pytest.mark.skip(reason=f"Solver {cmd_solver} not installed")
                item.add_marker(skip)
            
            continue # skip rest of the logic for this test
            
        # ------------------------------ More filtering ------------------------------ #

        # Only filter tests that are parameterized with a 'solver' (through `_generate_inputs`)
        if hasattr(item, "callspec"):
            if cmd_solver:
                if "solver" in item.callspec.params:
                    solver = item.callspec.params["solver"]
                    if solver == cmd_solver:
                        filtered.append(item)
                if "solver_name" in item.callspec.params:
                    solver = item.callspec.params["solver_name"]
                    if solver == cmd_solver:
                        filtered.append(item)
            else:
                filtered.append(item)
        else:
            # keep non-parametrized tests
            filtered.append(item)

    items[:] = filtered