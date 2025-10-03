import unittest

import pytest

from cpmpy.expressions.core import Operator, Comparison
from cpmpy.solvers import CPM_pysat, CPM_ortools, CPM_minizinc, CPM_gurobi
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.solvers.utils import SolverLookup
from cpmpy import *
from cpmpy.expressions.variables import NegBoolView
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.expressions.utils import is_any_list
from cpmpy.exceptions import NotSupportedError
from utils import skip_on_missing_pblib

# Get all supported solvers
SOLVERNAMES = [name for name, solver in SolverLookup.base_solvers() if solver.supported()]


@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_empty_constructor(solver_name):
    solver_class = SolverLookup.lookup(solver_name)
    solver = solver_class()

    assert hasattr(solver, "status")
    assert solver.status() is not None
    assert solver.status().exitstatus == ExitStatus.NOT_RUN
    assert solver.status().solver_name != "dummy"


@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_constructor(solver_name):
    solver_class = SolverLookup.lookup(solver_name)
    
    bvar = boolvar(shape=3)
    x, y, z = bvar

    m = Model([x & y])
    solver = solver_class(m)

    assert solver.status() is not None
    assert solver.status().exitstatus == ExitStatus.NOT_RUN
    assert solver.status().solver_name != "dummy"


@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_native_model(solver_name):
    solver_class = SolverLookup.lookup(solver_name)
    
    bvar = boolvar(shape=3)
    x, y, z = bvar

    m = Model([x & y])
    solver = solver_class(m)
    assert solver.native_model is not None


@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_add_var(solver_name):
    solver_class = SolverLookup.lookup(solver_name)
    solver = solver_class()

    bvar = boolvar(shape=3)
    x, y, z = bvar

    solver += x

    assert len(solver.user_vars) == 1


@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_add_constraint(solver_name):

    solver_class = SolverLookup.lookup(solver_name)
    solver = solver_class()

    bvar = boolvar(shape=3)
    x, y, z = bvar

    solver += [x & y]
    assert len(solver.user_vars) == 2

    # Skip pysdd as it doesn't support sum
    if solver_name == "pysdd":
        return

    solver += [sum(bvar) == 2]
    assert len(solver.user_vars) == 3


@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_solve(solver_name):
    solver_class = SolverLookup.lookup(solver_name)
    solver = solver_class()

    bvar = boolvar(shape=3)
    x, y, z = bvar

    solver += x.implies(y & z)
    solver += y | z
    solver += ~ z

    assert solver.solve()
    assert solver.status().exitstatus == ExitStatus.FEASIBLE

    assert [x.value(), y.value(), z.value()] == [0, 1, 0]


@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_solve_infeasible(solver_name):
    solver_class = SolverLookup.lookup(solver_name)
    solver = solver_class()

    bvar = boolvar(shape=3)
    x, y, z = bvar

    solver += x.implies(y & z)

    assert solver.solve()
    assert solver.status().exitstatus == ExitStatus.FEASIBLE

    solver += ~ z
    solver += x

    assert not solver.solve()
    assert solver.status().exitstatus == ExitStatus.UNSATISFIABLE


@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_minimize(solver_name):
    """Test minimize functionality"""
    solver_class = SolverLookup.lookup(solver_name)
    solver = solver_class() if solver_name != "z3" else solver_class(subsolver="opt")

    ivar = intvar(1, 10)

    try:
        solver.minimize(ivar)
    except NotImplementedError:
        return

    assert hasattr(solver, "objective_value_")
    assert solver.solve()
    assert solver.objective_value() == 1
    assert solver.status().exitstatus == ExitStatus.OPTIMAL

@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_maximize(solver_name):
    """Test maximize functionality"""
    solver_class = SolverLookup.lookup(solver_name)
    if solver_name == "z3":
        return
    solver = solver_class() if solver_name != "z3" else solver_class(subsolver="opt")

    ivar = intvar(1, 10)    

    try:
        solver.maximize(ivar)
    except NotImplementedError:
        return

    assert solver.solve()
    assert solver.objective_value() == 10
    assert solver.status().exitstatus == ExitStatus.OPTIMAL

# solver_var() tests
@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_solver_var(solver_name):
    """Test basic solver_var functionality with different variable types"""
    solver_class = SolverLookup.lookup(solver_name)
    solver = solver_class()
    
    # Test with boolean variable
    bool_var = boolvar(name="test_bool")
    solver_bool = solver.solver_var(bool_var)
    
    # Should return something (not None)
    assert solver_bool is not None
    
    # Test if it is cashed correctly    
    # Should return the same object/reference
    assert solver_bool is solver.solver_var(bool_var) if not is_any_list(solver_bool) else solver_bool == solver.solver_var(bool_var), f"Solver {solver_name} did not cache bool variable properly"

    # Test with negative boolean view
    neg_bool_var = ~bool_var    
    
    
    try:
        solver_bool = solver.solver_var(bool_var)
        solver_neg_bool = solver.solver_var(neg_bool_var)
        
        # Both should return something
        assert solver_bool is not None
        assert solver_neg_bool is not None
    
    except (NotSupportedError, ValueError) as e: # TODO: fix consistency among solvers
        # Some solvers might not support NegBoolView in solver_var
        # That's potentially OK if they handle it elsewhere
        print(f"Solver {solver_name} raised exception for NegBoolView: {e}")

    # Test with integer variable

    # Skip pysdd as it doesn't support sum
    if solver_name == "pysdd":
        return

    # Test with integer variable
    int_var = intvar(1, 10, name="test_int")
    solver_int = solver.solver_var(int_var)
    
    # Should return something (not None)
    assert solver_int is not None

    # Test if it is cashed correctly    
    assert solver_int is solver.solver_var(int_var) if not is_any_list(solver_int) else solver_int == solver.solver_var(int_var), f"Solver {solver_name} did not cache int variable properly"    


@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_solver_vars(solver_name):
    """Test solver_vars (plural) function with arrays"""
    solver_class = SolverLookup.lookup(solver_name)
    solver = solver_class()
    
    # Test with array of boolean variables
    bool_array = boolvar(shape=3, name="bool_array")
    solver_bool_array = solver.solver_vars(bool_array)
    
    # Should return list of same length
    assert len(solver_bool_array) == 3
    assert all(var is not None for var in solver_bool_array)
    
    # Test with nested arrays
    nested_array = boolvar(shape=(2, 2), name="nested_array")
    solver_nested = solver.solver_vars(nested_array)
    
    # Should preserve structure
    assert len(solver_nested) == 2
    assert len(solver_nested[0]) == 2
    assert len(solver_nested[1]) == 2
    
    # Test with single variable (should work too)
    single_var = boolvar(name="single")
    solver_single = solver.solver_vars(single_var)
    assert solver_single is not None


@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_time_limit(solver_name):
    """Test time limit functionality"""
    solver_class = SolverLookup.lookup(solver_name)
    solver = solver_class()
    
    # Skip pysdd as it doesn't support time limits
    if solver_name == "pysdd":
        return
    
    bvar = boolvar(shape=3)
    x, y, z = bvar
    solver += [x | y | z]
    
    # Test with positive time limit
    assert solver.solve(time_limit=1.0)
    assert solver.status().exitstatus == ExitStatus.FEASIBLE
    
    # Test with negative time limit should raise ValueError
    try:
        solver.solve(time_limit=-1)
        assert False, f"Solver {solver_name} should raise ValueError for negative time limit"
    except ValueError:
        pass  # Expected behavior


@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_has_objective(solver_name):
    """Test has_objective() method"""
    solver_class = SolverLookup.lookup(solver_name)
    solver = solver_class() if solver_name != "z3" else solver_class(subsolver="opt")
    
    # Initially should have no objective
    assert not solver.has_objective()
    
    # Add an objective if supported
    try:
        ivar = intvar(1, 10)
        solver.minimize(ivar)
        assert solver.has_objective()

        solver.maximize(ivar)
        assert solver.has_objective()
    except NotImplementedError:
        # Solver doesn't support objectives
        assert not solver.has_objective()

@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_runtime_tracking(solver_name):
    """Test that solver tracks runtime correctly"""
    solver_class = SolverLookup.lookup(solver_name)
    solver = solver_class()
    
    bvar = boolvar(shape=2)
    x, y = bvar
    
    solver += [x | y]
    
    # Before solving, runtime should be None
    assert solver.status().runtime is None
    
    # After solving, runtime should be recorded
    solver.solve()
    status = solver.status()
    assert status.runtime is not None
    assert status.runtime >= 0  # Should be non-negative


@pytest.mark.parametrize("solver_name", SOLVERNAMES)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_solveall_basic(solver_name):
    """Test solveAll functionality if supported"""
    solver_class = SolverLookup.lookup(solver_name)
    solver = solver_class()
    
    bvar = boolvar(shape=2)
    x, y = bvar
    
    # Create a problem with multiple solutions
    solver += [x | y]  # 3 solutions: (T,T), (T,F), (F,T)
    
    try:
        # Test solveAll with solution limit
        solution_count = 0
        def count_solution():
            nonlocal solution_count
            solution_count += 1
            
        if solver_name == "pysdd":
            # pysdd doesn't support solution_limit
            total = solver.solveAll(display=count_solution)
        elif solver_name == "hexaly":
            # set time limit, hexaly cannot prove UNSAT at last call
            total = solver.solveAll(display=count_solution, solution_limit=10, time_limit=5)
        else:
            total = solver.solveAll(display=count_solution, solution_limit=10)
        
        assert total == 3  # Should find all 3 solutions
        assert solution_count == 3

    except NotSupportedError:
        # Solver doesn't support solveAll with objectives or other limitations
        pass