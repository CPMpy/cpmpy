import contextvars

__perf_context = contextvars.ContextVar("perf_context")
__timer_context = contextvars.ContextVar("timer_context")