import pytest
import cpmpy as cp
import importlib
import warnings
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def _parse_solver_option(solver_option: Optional[str] , filter_not_installed: bool = True) -> Optional[list[str]]:
    """
    Parse the --solver option into a list of solvers.
    Returns 'None' if no solver was specified, otherwise returns a list of solver names.
    Supports the special "all" keyword to expand to all installed solvers.
    Supports the special "None" keyword to skip all solver-parametrized tests.
    
    Arguments:
        solver_option (str): The solver option string from command line
        filter_not_installed (bool): If True, filter out non-installed solvers from the result

    Returns:
        list[str] | None:
            A list of solver names, or 'None' if no solver was specified
            Returns empty list [] if "None" was explicitly specified or solver was specified but all were filtered out
            If 'filter_not_installed' is True, the list will only contain installed solvers
    """
    if solver_option is None:
        return None

    # Split by comma and strip whitespace
    original_solvers = [s.strip() for s in solver_option.split(",") if s.strip()]
    if not original_solvers: # no solver specified
        warnings.warn('--solver option set, but no solver specified. Using default solver (OR-Tools).')
        return None

    # Handle special "None" keyword - skip all solver-parametrized tests (no solver at all)
    if "None" in original_solvers or "none" in original_solvers:
        if len(original_solvers) == 1:
            # Only "None" specified, return empty list to skip all solver-parametrized tests
            # Non-solver tests will still run
            return []
        else:
            # "None" mixed with other solvers - remove it and warn
            original_solvers = [s for s in original_solvers if s.lower() != "none"]
            warnings.warn('Special "None" solver was specified along with other solvers. Ignoring "None" and using specified solvers.')

    # Expand "all" to all installed solvers
    if "all" in original_solvers:
        solvers = cp.SolverLookup.supported() 
        if filter_not_installed:
            warnings.warn('Option "all" already expands to all installed solvers. Ignoring filter for "filter_not_installed".')
    else:            
        solvers = original_solvers.copy()
        # Filter out non-installed solvers if requested
        if filter_not_installed:
            solvers = [s for s in solvers if s in cp.SolverLookup.supported()]
    
    # If solver was provided but all were filtered out, return empty list (not None)
    # This distinguishes "no solver specified" from "solver specified but not available"
    if not solvers:
        return []
    
    return solvers if solvers else None

def pytest_addoption(parser):
    """
    Adds cli arguments to the pytest command
    """
    parser.addoption(
        "--solver", type=str, action="store", default=None, help="Only run the tests on these solvers. Can be a single solver, a comma-separated list (e.g., 'ortools,cplex'), 'all' to use all installed solvers, or 'None' to skip all solver-parametrized tests."
    )

@pytest.fixture
def solver(request):
    """
    Limit tests to specific solvers.

    By providing the cli argument `--solver=<SOLVER_NAME>`, `--solver=<SOLVER1,SOLVER2,...>`, `--solver=all`, or `--solver=None`, two things will happen:
    - non-solver-specific tests which make a `.solve()` call will now run against all specified solvers (instead of just the default OR-Tools)
    - solver-specific tests, like the ones produced through `_generate_inputs`, will be filtered if they don't match any of the specified solvers

    Special values:
    - "all" expands to all installed solvers from SolverLookup
    - "None" skips all solver-parametrized tests (no solver at all), only runs tests that don't depend on solver parametrization

    By not providing a value for ``--solver`, the default behaviour will be to run non-solver-specific on the default solver (OR-Tools),
    and to run all solver-specific tests for which the solver has been installed on the system.
    """
    # Check if solver was parametrized for test (via pytest_generate_tests or explicit parametrization)
    if hasattr(request, "param"):
        solver_value = request.param
    else:
        # Not parametrized, use command line option
        # This branch is reached when:
        # - Single solver provided (will use that solver)
        # - No solver provided (will use None/default)
        # - Empty list returned (all solvers filtered out) - use default solver (OR-Tools)
        # - Multiple solvers provided but test wasn't parametrized (shouldn't happen, but uses first solver as fallback)
        solver_option = request.config.getoption("--solver")
        parsed_solvers = _parse_solver_option(solver_option)
        # Handle empty list (all solvers filtered out) same as None
        solver_value = parsed_solvers[0] if parsed_solvers else None
    
    # Set solver value on class if available (for tests using self.solver)
    if hasattr(request, "cls") and request.cls:
        request.cls.solver = solver_value
    
    return solver_value

@pytest.fixture
def constraint(request):
    if not hasattr(request, "param"):
        raise RuntimeError(
            "The 'constraint' fixture must be parametrized via pytest_generate_tests"
        )
    return request.param

def pytest_configure(config):
    # Configure logging for test filtering information
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Register custom marker for pytest test collecting
    config.addinivalue_line(
        "markers",
        "requires_solver(name): mark test as requiring a specific solver", # to filter tests when required solver is not installed
    )
    config.addinivalue_line(
        "markers",
        "requires_dependency(name): mark test as requiring a specific dependency", # to filter tests when required solver is not installed
    )
    config.addinivalue_line(
        "markers",
        "generate_constraints(generator): something", 
    )
    
    # Check for non-installed solvers and issue warnings
    solver_option = config.getoption("--solver")
    if solver_option:
        # Parse without filtering to check original list
        parsed_solvers_unfiltered = _parse_solver_option(solver_option, filter_not_installed=False)
        if parsed_solvers_unfiltered:
            installed_solvers = cp.SolverLookup.supported()        # installed base solvers
            not_installed_solvers = list(set(parsed_solvers_unfiltered) - set(installed_solvers))
            if not_installed_solvers:
                warnings.warn(
                    f"The following solvers are not installed and will not be tested: {', '.join(not_installed_solvers)}. "
                    f"Only installed solvers will be used for testing.",
                    UserWarning,
                    stacklevel=2
                )
            logger.info(f"Using solvers: {', '.join(parsed_solvers_unfiltered)}")
        else:
            if parsed_solvers_unfiltered is None:
                logger.info("No solvers specified, using default behavior (parametrised using OR-Tools and solver-specific tests).")
            else:
                logger.info("No solvers available to run tests with. Install some solvers or choose different solvers.")


def generate_inputs(generator, solvers):
    result = []
    if solvers is None:
        installed_solvers = cp.SolverLookup.supported() 
        solvers = [installed_solvers[0]]
    for solver in solvers:
        result += [(solver, expr) for expr in generator(solver)]
    return result

def pytest_generate_tests(metafunc):
    """
    Dynamically parametrize non-solver-specific tests with all provided solvers.
    
    When multiple solvers are provided via --solver, tests that use the 'solver' fixture
    but are not already parametrized will be parametrized to run against all provided solvers.
    """

    # Early exists
    # 1) Check if this test uses the 'solver' fixture
    if "solver" not in metafunc.fixturenames:
        return
    # 2) Check if test is already parametrized with solver
    #    Check callspec (for tests parametrized programmatically)
    if hasattr(metafunc, "callspec") and metafunc.callspec and "solver" in metafunc.callspec.params:
        return

    # Get solvers from command line option
    solver_option = metafunc.config.getoption("--solver")
    parsed_solvers = _parse_solver_option(solver_option)

    # Handle 'generate_constraints' marker
    constraint_generator_marker = metafunc.definition.get_closest_marker("generate_constraints")
    if constraint_generator_marker:
        generator = constraint_generator_marker.args[0]
        metafunc.parametrize(("solver","constraint"), list(generate_inputs(generator, parsed_solvers)),  ids=str)
        return   
    
    # Check parametrize markers (for tests parametrized via @pytest.mark.parametrize)
    #  i.e. test that have already been explicitly parametrized with solver
    for marker in metafunc.definition.iter_markers("parametrize"):
        # marker.args[0] is the argnames (can be string or tuple/list)
        argnames, argvalues = marker.args if marker.args else None
        if argnames:
            # Handle both string "solver" and tuple ("solver", ...) cases
            if isinstance(argnames, str):
                # if argnames == "solver" or "solver" in argnames.split(","):
                #     return
                if argnames == "generator":
                    generator = argvalues
                    print(list(generate_inputs(generator)))
                    metafunc.parametrize(("solver","constraint"), list(generate_inputs(generator)),  ids=str)
            elif isinstance(argnames, (tuple, list)):
                if "solver" in argnames:
                    return
    
    # Check if test has requires_solver marker (solver-specific tests)
    if metafunc.definition.get_closest_marker("requires_solver"):
        return
    
    # Only parametrize if multiple solvers are explicitly provided
    # When parsed_solvers is None (no --solver specified), don't parametrize - use default solver
    # When parsed_solvers is empty list (all filtered out), don't parametrize
    # Only parametrize when we have 2+ solvers explicitly specified
    if parsed_solvers is not None and len(parsed_solvers) > 1:
        metafunc.parametrize("solver", parsed_solvers)


def pytest_collection_modifyitems(config, items):
    """
    Centrally apply filters and skips to test targets.

    For now, only solver-based filtering gets applied.
    """
    initial_count = len(items)
    logger.info(f"Test suite size before filtering: {initial_count} tests")
    
    cmd_solver_option = config.getoption("--solver") # get cli `--solver`` arg
    cmd_solvers = _parse_solver_option(cmd_solver_option)  # parse into list
    
    # Uncomment for debugging
    # if cmd_solver_option:
    #     if cmd_solvers is None:
    #         logger.info(f"Solver option '{cmd_solver_option}' parsed to None (using default solver)")
    #     elif len(cmd_solvers) == 0:
    #         logger.info(f"Solver option '{cmd_solver_option}' resulted in empty list (all solvers filtered out)")
    #     else:
    #         logger.info(f"Solver option '{cmd_solver_option}' parsed to: {cmd_solvers}")
    # else:
    #     logger.info("No --solver option provided (using default behavior)")

    filtered = []
    skipped_dependency = 0
    skipped_solver_specific = 0
    skipped_parametrized = 0
    skipped_solver_fixture = 0
    
    for item in items:
        required_solver_marker = item.get_closest_marker("requires_solver")
        required_dependency_marker = item.get_closest_marker("requires_dependency")

        # --------------------------------- Dependency filtering --------------------------------- #

        # Skip test if the required dependency is not installed
        if required_dependency_marker:
            if not all(importlib.util.find_spec(dependency) is not None for dependency in required_dependency_marker.args):
                skip = pytest.mark.skip(reason=f"Dependency {required_dependency_marker.args} not installed")
                item.add_marker(skip)
                skipped_dependency += 1
                continue

        # --------------------------------- Solver filtering --------------------------------- #

        # Skip test if the required solver is not installed (for solver-specific tests)
        if required_solver_marker:
            required_solvers = required_solver_marker.args

            # --------------------------------- Filtering -------------------------------- #
            
            # when solvers are specified on the command line, 
            # only run solver-specific tests that require any of those solvers
            if cmd_solvers is not None:
                # If cmd_solvers is empty (all filtered out), skip solver-specific tests
                if len(cmd_solvers) == 0:
                    skipped_solver_specific += 1
                    continue
                # If required solver is in the list of specified solvers on the command line, run the test
                if any(cmd_solver in required_solvers for cmd_solver in cmd_solvers):
                    filtered.append(item)
                # If required solver is not in the list of specified solvers on the command line, filter the test to be skipped
                else:
                    skipped_solver_specific += 1
                    continue
            # instance has survived filtering
            else:
                filtered.append(item)

            # --------------------------------- Skipping --------------------------------- #

            # skip test if the required solver is not installed
            if not {k:v for k,v in cp.SolverLookup.base_solvers()}[required_solvers[0]].supported():
                skip = pytest.mark.skip(reason=f"Solver {required_solvers[0]} not installed")
                item.add_marker(skip)
            
            continue # skip rest of the logic for this test
            
        # ------------------------------ More filtering ------------------------------ #

        # Check if test uses solver fixture (for non-parametrized tests)
        uses_solver_fixture = hasattr(item, "_fixtureinfo") and "solver" in getattr(item._fixtureinfo, "names_closure", [])
        
        # Only filter tests that are parameterized with a 'solver' (through `_generate_inputs`)
        #   i.e. filter externally parametrized tests
        if hasattr(item, "callspec"):
            if cmd_solvers is not None:
                # If cmd_solvers is empty (all filtered out), skip solver-dependent tests
                if len(cmd_solvers) == 0:
                    # Skip tests parametrized with solver, but keep others that don't depend on solver
                    if "solver" not in item.callspec.params and "solver_name" not in item.callspec.params:
                        filtered.append(item)
                    else:
                        skipped_parametrized += 1
                    continue
                # If test is parametrized with solver, filter it if it is not in the list of specified solvers on the command line
                if "solver" in item.callspec.params:
                    solver = item.callspec.params["solver"]
                    if solver in cmd_solvers:
                        filtered.append(item)
                    else:
                        skipped_parametrized += 1
                # If test is parametrized with solver_name, filter it if it is not in the list of specified solvers on the command line
                elif "solver_name" in item.callspec.params:
                    solver = item.callspec.params["solver_name"]
                    if solver in cmd_solvers:
                        filtered.append(item)
                    else:
                        skipped_parametrized += 1
                # If test is not parametrized with solver or solver_name, keep it
                else:
                    # Parametrized but not with solver - keep it
                    filtered.append(item)
            else:
                filtered.append(item)
        # Non-parametrized tests:
        else:
            # Filter out test if it uses solver fixture and cmd_solvers is empty
            if cmd_solvers is not None and len(cmd_solvers) == 0:
                if uses_solver_fixture:
                    skipped_solver_fixture += 1
                    continue  # Skip tests that use solver fixture when no solvers available
            # keep non-parametrized tests that don't depend on solver
            filtered.append(item)

    # Uncomment for debugging
    # final_count = len(filtered)
    # logger.info(f"Test suite filtering summary:")
    # logger.info(f"  Initial tests: {initial_count}")
    # if skipped_dependency > 0:
    #     logger.info(f"  Skipped (missing dependency): {skipped_dependency}")
    # if skipped_solver_specific > 0:
    #     logger.info(f"  Skipped (solver-specific, not matching): {skipped_solver_specific}")
    # if skipped_parametrized > 0:
    #     logger.info(f"  Skipped (parametrized solver, not matching): {skipped_parametrized}")
    # if skipped_solver_fixture > 0:
    #     logger.info(f"  Skipped (solver fixture, no solvers): {skipped_solver_fixture}")
    # logger.info(f"  Final tests: {final_count} ({final_count - initial_count:+d})")
    
    items[:] = filtered # replace the original list with the filtered list