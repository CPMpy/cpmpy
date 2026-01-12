import importlib
import pytest
import unittest
from functools import wraps

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


# ---------------------------------------------------------------------------- #
#                                TestCase class                                #
# ---------------------------------------------------------------------------- #

class TestCase:
    """
    Custom TestCase class that provides unittest-style assertions.

    Does NOT inherit from unittest.TestCase to avoid conflicts with pytest's
    pytest_generate_tests parametrization. Instead, copies all attributes
    (methods and class variables) from unittest.TestCase.
    """

    @pytest.fixture(autouse=True)
    def _init_unittest_attrs(self, request):
        """
        Initialize instance attributes needed by unittest assertions.
        """

        # Initialize the type equality funcs dictionary used by assertions
        self._type_equality_funcs = {}
        self.addTypeEqualityFunc(dict, 'assertDictEqual')
        self.addTypeEqualityFunc(list, 'assertListEqual')
        self.addTypeEqualityFunc(tuple, 'assertTupleEqual')
        self.addTypeEqualityFunc(set, 'assertSetEqual')
        self.addTypeEqualityFunc(frozenset, 'assertSetEqual')
        # Initialize other attributes that unittest.TestCase sets
        self._outcome = None

        # Set solver on instance if the test uses solver fixture
        if 'solver' in request.fixturenames:
            self.solver = request.getfixturevalue('solver')

    def setup_method(self, method=None):
        """
        Empty setup_method to allow subclasses to call super().setup_method()
        """
        pass

    def teardown_method(self, method=None):
        """
        Empty teardown_method to allow subclasses to call super().teardown_method()
        """
        pass


# Copy all attributes (methods and class variables) from unittest.TestCase to TestCase
for _attr_name in dir(unittest.TestCase):
    if not _attr_name.startswith('__'):  # Skip dunder methods
        _attr = getattr(unittest.TestCase, _attr_name)
        if not hasattr(TestCase, _attr_name):
            setattr(TestCase, _attr_name, _attr)    

