import pytest
from functools import wraps

def skip_on_exception(exc_type, message_contains=None, skip_message=None):
    """
    Skip test when expected failure occurs.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exc_type as e:
                if message_contains is None or message_contains in str(e):
                    msg = skip_message or f"Skipped due to {exc_type.__name__}: {e}"
                    pytest.skip(msg)
                raise  # Re-raise if the message doesn't match
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

def skip_on_missing_pblib(func):
    return smart_decorator(
        skip_on_exception(
            ImportError,
            message_contains="PB constraint",
            skip_message="`pypblib` not installed"
        )
    )
