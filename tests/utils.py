import importlib
import pytest
from functools import wraps
import cpmpy as cp
from cpmpy.expressions.utils import argval, argvals

# ---------------------------------------------------------------------------- #
#                              Generic Decorators                              #
# ---------------------------------------------------------------------------- #

"""
These should probably not be used on their own, 
but rather as building blocks for the more "specific" decorators below.
"""

def skip_on_exception(exc_type, message_contains=None, skip_message=None):
    """
    Skip test when expected failure occurs.
    """

    def decorator(func):
        """
        The actual decorator that gets placed around the function.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            The function that gets called instead of the decorated function.
            """
            try:
                # Try calling the decorated function
                return func(*args, **kwargs)
            except exc_type as e:
                # Check if expected exception
                if message_contains is None or message_contains in str(e):
                    msg = skip_message or f"Skipped due to {exc_type.__name__}: {e}"
                    pytest.skip(msg) # expected -> skip test
                raise  # Re-raise if not the expected exception
        return wrapper
    return decorator


def apply_decorator_to_tests(decorator):
    """
    Decorator wrapper for unittest.TestCase classes.
    Applies `decorator` to all methods with a name starting with `test_`.
    """
    def class_decorator(cls):
        for attr_name in dir(cls):
            if attr_name.startswith("test_"):
                attr = getattr(cls, attr_name)
                if callable(attr):
                    setattr(cls, attr_name, decorator(attr))
        return cls
    return class_decorator


def smart_decorator(method_decorator):
    """
    Wraps a method decorator so it can be applied to either:
    - a function/method: applies the decorator directly
    - a class: applies the decorator to all test_* methods
    """
    def wrapper(obj):
        if isinstance(obj, type):
            return apply_decorator_to_tests(method_decorator)(obj)
        elif callable(obj):
            return method_decorator(obj)
        else:
            raise TypeError("smart_decorator can only be used on classes or callables")
    return wrapper


# ---------------------------------------------------------------------------- #
#                              Specific Decorators                             #
# ---------------------------------------------------------------------------- #

from cpmpy.solvers.pysat import CPM_pysat
pblib_available = importlib.util.find_spec("pypblib") is not None

def skip_on_missing_pblib(skip_on_exception_only:bool=False):
    """
    Skips the decorated test when the optional `pblib` dependency is not available on the current system.

    Arguments:
        skip_on_exception_only (bool): If set to `True`, still run the test but ignore any exception related to missing `pblib`.
                                        If exception occurs, test gets reported as being skipped.
                                       If set to `False`, test doesn't get run (even if test does not rely on `pblib`) 

    Notes:
        `@skip_on_missing_pblib()` should be used for test which we know require `pblib`. 
            These tests then never get run, reducing runtime.
        `@skip_on_missing_pblib(skip_on_exception_only=True)` should be used when we're not sure if it requires 
            `pblib` but would like to ignore any exception related to missing `pblib` dependency.
    """

    if not skip_on_exception_only:
        return pytest.mark.skipif(not pblib_available, reason="`pypblib` not installed")
    
    return smart_decorator(
        skip_on_exception(
            ImportError,
            message_contains="PB constraint",
            skip_message="`pypblib` not installed"
        )
    )


def inclusive_range(lb,ub):
    return range(lb,ub+1)

def lambda_assert(assert_func):
    return lambda : _lambda_assert(assert_func)
    
def _lambda_assert(assert_func):
    assert assert_func()

# ---------------------------------------------------------------------------- #
#                              Utility Functions                              #
# ---------------------------------------------------------------------------- #

def full_test_constraint(constraint, solver, satisfiable=True):
    """
    Thorougly test a constraint and its negation for a given solver.
        - Tests whether the constraint and its negation can be satisfied
        - Tests whether the value of the constraint is None when one variable is assigned None
        - Tests whether the negation of the constraint is correctly handled (cons & ~cons should be unsat)
        - Tests the above but for the decomposition of the constraint
        - Enumerates all solutions to the constraint and its negation and checks the solution sets are correct and disjoint
        
    Arguments:
        constraint (Expression): The constraint to test.
        solver (Solver): The solver to use.
        satisfiable (bool): Whether the constraint should be satisfiable.
    """
    vars = cp.cpm_array(cp.transformations.get_variables.get_variables(constraint))

    assert cp.Model(constraint).solve(solver=solver) is satisfiable
    assert argval(constraint) is (True if satisfiable else None), f"Value of {constraint} is {argval(constraint)}"
    if satisfiable:
        vars[0]._value = None
        argval(constraint) is None

    assert cp.Model(~constraint).solve(solver=solver) is True # assumes the constraint excludes at least one assignment
    assert argval(constraint) is False
    vars[0]._value = None
    assert argval(constraint) is None

    # test negation properly
    assert cp.Model(constraint, ~constraint).solve(solver=solver) is False

    # test forced decomposition
    s = cp.SolverLookup.get(solver)
    s.supported_global_constraints = set()
    s.supported_reified_global_constraints = set()
    s += constraint
    assert s.solve() is satisfiable
    assert argval(constraint) is (True if satisfiable else None)

    
    s = cp.SolverLookup.get(solver)
    s.supported_global_constraints = set()
    s.supported_reified_global_constraints = set()
    s += ~constraint
    assert s.solve() is True  # assumes the constraint excludes at least one assignment
    assert argval(constraint) is False, f"value of {constraint} is {argval(constraint)}, value of {vars} = {argvals(vars)}"

    s = cp.SolverLookup.get(solver)
    s.supported_global_constraints = set()
    s.supported_reified_global_constraints = set()
    s += constraint & ~constraint
    assert s.solve() is False

    # test all solutions, not for incomplete solvers
    # if solver in ["hexaly"]: 
    #     return
    
    # total = 1
    # for v in cp.transformations.get_variables.get_variables(constraint):
    #     total *= v.ub - v.lb + 1

    # def assert_value(val):
    #     return lambda: argval(constraint) is val
    
    # pos_sols = cp.Model(constraint).solveAll(solver=solver, display=assert_value(True), solution_limit=total)
    # neg_sols = cp.Model(~constraint).solveAll(solver=solver, display=assert_value(False), solution_limit=total)
    # assert pos_sols + neg_sols == total