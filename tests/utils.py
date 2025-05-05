import pytest
from functools import wraps

def skip_on_exception(exc_type, message_contains=None, skip_message=None):
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
    def class_decorator(cls):
        for attr_name in dir(cls):
            if attr_name.startswith("test_"):
                attr = getattr(cls, attr_name)
                if callable(attr):
                    setattr(cls, attr_name, decorator(attr))
        return cls
    return class_decorator
