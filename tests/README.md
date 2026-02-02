# Test Suite

CPMpy has an extensive test suite, covering all major components including variables, constraints, models, solvers, transformations, and tools.

## Running Tests

### Basic Usage

Run all tests:
```bash
pytest tests/
```

Run a specific test file:
```bash
pytest tests/test_model.py
```

Run a specific test:
```raw
pytest tests/test_model.py::TestModel::test_ndarray
                                |            |
                       (name of the class)   |
                                 (name of the test method)
```

### Parallelisation

Through the `pytest-xdist` pytest plugin, running tests can be parallelised. 
E.g. running with 40 workers:
```console
pytest -n 40 tests/test_model.py
```

Or letting pytest auto-decide how many workers to use based on the number of available cores on your machine:
```console
pytest -n auto tests/test_model.py
```

Install using:
```console
pip install pytest-xdist
```

### Solver Selection

The test suite supports changing the solver backend used to run the tests via the `--solver` command-line option.

For now, this only affects tests/test_constraints.py, but it will gradually be added to the entire test-suite.

#### Single Solver
Run tests with a specific solver:
```bash
pytest tests/ --solver=gurobi
```

#### Multiple Solvers
Run tests with multiple solvers
- certain non-solver-specific tests (test_constraints, test_solverinterface, test_solvers_solhint) will run against all specified solvers
- other non-solver-specific tests will only run against the default solver (OR-Tools)
- solver-specific tests will be filtered on specified solvers
```bash
pytest tests/ --solver=ortools,cplex,gurobi
```

#### All Installed Solvers
Run tests with all installed solvers:
```bash
pytest tests/ --solver=all
```

This automatically detects all installed solvers from `SolverLookup` and parametrises the subset of non-solver-specific tests to run against each one.

#### Skip Solver Tests
Skip all solver-parametrised tests (only run tests that don't depend on solver parametrisation).
I.e., tests that do not rely on solving a model. Examples are tests that evaluate constructors of expressions.
```bash
pytest --solver=None
```

#### Default Behavior

If no `--solver` option is provided:
- Non-solver-specific tests run with the default solver (OR-Tools)
- All solver-specific tests run for their respective declared solver (if installed)

## Test Organization

### Test Files

- **`test_model.py`** - Model creation, manipulation, and I/O
- **`test_expressions.py`** - Expression types and operations (comparisons, operators, sums, etc.)
- **`test_constraints.py`** - Constraint types and validation (boolean, comparison, reification, implication)
- **`test_globalconstraints.py`** - Global constraint implementations (AllDifferent, Circuit, Cumulative, etc.)
- **`test_solvers.py`** - Solver interface and functionality (high-level solver tests)
- **`test_solverinterface.py`** - Low-level solver interface tests (constructor, native model, solve methods)
- **`test_variables.py`** - Variable types (intvar, boolvar, shapes, naming)
- **`test_builtins.py`** - Python builtin functions (max, min, all, any)
- **`test_cse.py`** - Common subexpression elimination
- **`test_direct.py`** - Direct solver constraints (automaton, etc.)
- **`test_flatten.py`** - Model flattening transformations
- **`test_int2bool.py`** - Integer to boolean transformation
- **`test_pysat_*.py`** - PySAT-specific tests (cardinality, interrupt, weighted sum)
- **`test_solveAll.py`** - solveAll functionality across solvers
- **`test_solvers_solhint.py`** - Solver hints functionality
- **`test_tocnf.py`** - Conversion to CNF (Conjunctive Normal Form)
- **`test_tool_dimacs.py`** - DIMACS format tools
- **`test_trans_*.py`** - Transformation tests (linearize, safen, simplify)
- **`test_transf_*.py`** - Additional transformation tests (comp, decompose, reif)
- **`test_tools_*.py`** - Tool functionality (MUS, tuning, etc.)
- **`test_examples.py`** - Run examples as a testsuite

### Test Markers

Tests can be marked with special markers:

- **`@pytest.mark.requires_solver("solver_name_1", "solver_name_2", ...)`** - Test requires a specific solver, one of the listed names
- **`@pytest.mark.requires_dependency("package_name")`** - Test requires a specific Python package
- **`@pytest.mark.generate_constraints.with_args(generator_function)`** - Parametrise test's "constraint" argument using the provided generator
- **`@pytest.mark.depends_on_solver`** - Test indirectly depends on solvers


Examples:
```python
@pytest.mark.requires_solver("cplex")
def test_cplex_specific_feature():
    # This test only runs if cplex is available
    pass
```

```python
def randomly_sample_expressions(solver)
    return [...]

@pytest.mark.generate_constraints.with_args(randomly_sample_expressions)
def test_bool_constraints(solver, constraint):
    ...
```
(for a complete example, have a look at `/tests/test_constraints.py`)

## Writing Tests

### Basic Test Structure

```python
import pytest
import cpmpy as cp

def test_basic_model():
    x = cp.intvar(0, 10, name="x")
    m = cp.Model(x >= 5)
    assert m.solve()
    assert x.value() >= 5
```

### Using the Solver Fixture

For tests that should run with different solvers:

```python
from utils import TestCase

@pytest.mark.usefixtures("solver")
class TestMyFeature(TestCase):
    def test_with_solver(self):
        x = cp.intvar(0, 10)
        m = cp.Model(x >= 5)
        self.assertTrue(m.solve(solver=self.solver))
        self.assertGreaterEqual(x.value(), 5)
```

When multiple solvers are provided via `--solver`, these tests will automatically be parametrised to run against each solver. The `self.solver` attribute is automatically set by the test framework.

### Solver-Parametrised Tests

For tests that are explicitly parametrised with a selection of solvers:

```python
@pytest.mark.parametrise("solver", ["ortools", "cplex", "gurobi"])
def test_with_explicit_solvers(solver):
    x = cp.intvar(0, 10)
    m = cp.Model(x >= 5)
    assert m.solve(solver=solver)
```

### Solver-Specific Tests

For tests that only work with specific solvers:

```python
@pytest.mark.requires_solver("cplex")
def test_cplex_feature():
    # Test cplex-specific functionality
    pass
```

You can pass multiple solvers, for all of which the test will be run.

## Contributing

When adding new tests:

1. Follow existing test patterns
2. Use appropriate markers for solver-specific tests
3. Ensure tests work with multiple solvers when possible
4. Add docstrings explaining what the test validates
5. Use descriptive test names
