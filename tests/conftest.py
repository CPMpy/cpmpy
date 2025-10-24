import pytest

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
    - non-solver-specific tests which make a `.solve()` call will now use `SOLVER_NAME` as backend
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
        "requires_solver(name): mark test as requiring a specific solver",
    )


def pytest_collection_modifyitems(config, items):
    """
    Centrally apply filters and skips to test targets.

    For now, only solver-based filtering gets applied.
    """
    cmd_solver = config.getoption("--solver") # get cli `--solver`` arg
    if not cmd_solver:
        return  # no filtering needed

    filtered = []
    for item in items:
        marker = item.get_closest_marker("requires_solver")

        if marker:
            required_solvers = marker.args
            if cmd_solver in required_solvers:
                filtered.append(item)
            continue
            
        # Only filter tests that are parameterized with a 'solver' (through `_generate_inputs`)
        if hasattr(item, "callspec"):
            if "solver" in item.callspec.params:
                solver = item.callspec.params["solver"]
                if solver == cmd_solver:
                    filtered.append(item)
            if "solver_name" in item.callspec.params:
                solver = item.callspec.params["solver_name"]
                if solver == cmd_solver:
                    filtered.append(item)
        else:
            # keep non-parametrized tests
            filtered.append(item)

    items[:] = filtered