# CPMpy Test Suite

This directory contains the comprehensive test suite for CPMpy, covering all major components including variables, constraints, models, solvers, transformations, and tools.

## Running Tests

### Basic Usage

Run all tests:
```bash
pytest
```

Run a specific test file:
```bash
pytest tests/test_model.py
```

Run a specific test:
```bash
pytest tests/test_model.py::TestModel::test_ndarray
```

### Parallelisation

Through the `pytest-xdist` pytest plugin, running tests can be parallelised. 
E.g. running with 40 workers:
```console
pytest -n 40 tests/test_model.py
```

Install using:
```console
pip install pytest-xdist
```

### Solver Selection

The test suite supports changing the solver backend used to run the tests via the `--solver` command-line option:

#### Single Solver
Run tests with a specific solver:
```bash
pytest --solver=gurobi
```

#### Multiple Solvers
Run tests with multiple solvers
- non-solver-specific tests will run against all specified solvers
- solver-specific tests will be filtered on specified solvers
```bash
pytest --solver=ortools,cplex,gurobi
```

#### All Installed Solvers
Run tests with all installed solvers:
```bash
pytest --solver=all
```

This automatically detects all installed solvers from `SolverLookup` and parametrises non-solver-specific tests to run against each one.

#### Skip Solver Tests
Skip all solver-parametrised tests (only run tests that don't depend on solver parametrisation):
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
- **`test_tocnf.py`** - Conversion to CNF (conjunctive normal form)
- **`test_tool_dimacs.py`** - DIMACS format tools
- **`test_trans_*.py`** - Transformation tests (linearize, safen, simplify)
- **`test_transf_*.py`** - Additional transformation tests (comp, decompose, reif)
- **`test_tools_*.py`** - Tool functionality (MUS, tuning, etc.)
- **`test_examples.py`** - Run examples as a testsuite

### Test Markers

Tests can be marked with special markers:

- **`@pytest.mark.requires_solver("solver_name")`** - Test requires a specific solver
- **`@pytest.mark.requires_dependency("package_name")`** - Test requires a specific Python package

Example:
```python
@pytest.mark.requires_solver("cplex")
def test_cplex_specific_feature():
    # This test only runs if cplex is available
    pass
```

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
@pytest.mark.usefixtures("solver")
class TestMyFeature:
    def test_with_solver(self):
        x = cp.intvar(0, 10)
        m = cp.Model(x >= 5)
        assert m.solve(solver=self.solver)
```

When multiple solvers are provided via `--solver`, these tests will automatically be parametrised to run against each solver.

### Solver-Parametrised Tests

For tests that explicitly parametrise with solvers:

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

## Contributing

When adding new tests:

1. Follow existing test patterns
2. Use appropriate markers for solver-specific tests
3. Ensure tests work with multiple solvers when possible
4. Add docstrings explaining what the test validates
5. Use descriptive test names