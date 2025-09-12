
import resource
import sys
import time
import warnings
import psutil


TIME_BUFFER = 5 # seconds
# TODO : see if good value
MEMORY_BUFFER_SOFT = 2 # MiB
MEMORY_BUFFER_HARD = 0 # MiB
MEMORY_BUFFER_SOLVER = 20 # MB



def set_memory_limit(mem_limit):
    """
    Set memory limit (Virtual Memory Size). 
    """
    if mem_limit is not None:
        soft = max(_mib_as_bytes(mem_limit) - _mib_as_bytes(MEMORY_BUFFER_SOFT), _mib_as_bytes(MEMORY_BUFFER_SOFT))
        hard = max(_mib_as_bytes(mem_limit) - _mib_as_bytes(MEMORY_BUFFER_HARD), _mib_as_bytes(MEMORY_BUFFER_HARD))
        if sys.platform != "win32":
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard)) # limit memory in number of bytes
        else:
            warnings.warn("Memory limits using `resource` are not supported on Windows. Skipping hard limit.")

def disable_memory_limit():
    if sys.platform != "win32":
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # set a very high soft limit
        resource.setrlimit(resource.RLIMIT_AS, (hard, hard))

def set_time_limit(time_limit, verbose:bool=False):
    """
    Set time limit (CPU time in seconds).
    """
    if time_limit is not None:
        if sys.platform != "win32":
            soft = time_limit
            hard = resource.RLIM_INFINITY
            resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
        else:
            warnings.warn("CPU time limits using `resource` are not supported on Windows. Skipping hard limit.")

def _wall_time(p: psutil.Process):
    return time.time() - p.create_time()

def _mib_as_bytes(mib: int) -> int:
    return mib * 1024 * 1024

def _mb_as_bytes(mb: int) -> int:
    return mb * 1000 * 1000

def _bytes_as_mb(bytes: int) -> int:
    return bytes // (1000 * 1000)

def _bytes_as_gb(bytes: int) -> int:
    return bytes // (1000 * 1000 * 1000)

def _bytes_as_mb_float(bytes: int) -> float:
    return bytes / (1000 * 1000)

def _bytes_as_gb_float(bytes: int) -> float:
    return bytes / (1000 * 1000 * 1000)