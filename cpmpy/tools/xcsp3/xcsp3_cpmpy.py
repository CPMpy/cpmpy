"""
Deprecated CLI for solving XCSP3 instances via CPMpy.

New code should use XCSP3Adapter or run_benchmark.py::

    # Single instance
    python -m cpmpy.tools.benchmark.cpbenchy.adapter.xcsp3 \\
        instance.xml.lzma --solver ortools --time_limit 60

    # Batch / dataset
    python -m cpmpy.tools.benchmark.cpbenchy.runner.run_benchmark \\
        --dataset cpmpy.tools.datasets.xcsp3.XCSP3Dataset \\
        --dataset-year 2024 --dataset-track CSP --dataset-download \\
        --runner xcsp3 --solver ortools --time_limit 60 --workers 4

This module is kept for backward compatibility.  The __main__ block preserves
the original competition CLI flags (-s/-l/-m/-c/--solver/--intermediate/…).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional

if sys.platform != "win32":
    import resource


# ---------------------------------------------------------------------------- #
#                            XCSP3 Output formatting                           #
# ---------------------------------------------------------------------------- #

class ExitStatus(Enum):
    unsupported: str = "UNSUPPORTED"
    sat: str = "SATISFIABLE"
    optimal: str = "OPTIMUM" + chr(32) + "FOUND"
    unsat: str = "UNSATISFIABLE"
    unknown: str = "UNKNOWN"


def print_status(status: ExitStatus) -> None:
    print("s" + chr(32) + status.value, end="\n", flush=True)


def print_value(value: str) -> None:
    value = value[:-2].replace("\n", "\nv" + chr(32)) + value[-2:]
    print("v" + chr(32) + value, end="\n", flush=True)


def print_objective(objective: int) -> None:
    print("o" + chr(32) + str(objective), end="\n", flush=True)


def print_comment(comment: str) -> None:
    print("c" + chr(32) + comment.rstrip("\n"), end="\r\n", flush=True)


# ---------------------------------------------------------------------------- #
#                            Signal / resource helpers                         #
# ---------------------------------------------------------------------------- #

def sigterm_handler(_signo, _stack_frame):
    print_status(ExitStatus.unknown)
    print_comment("SIGTERM raised.")
    sys.exit(0)


def rlimit_cpu_handler(_signo, _stack_frame):
    print_status(ExitStatus.unknown)
    print_comment("SIGXCPU raised.")
    print(flush=True)
    sys.exit(0)


def init_signal_handlers():
    """Configure OS signal handlers (kept for backward compatibility)."""
    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGABRT, sigterm_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGXCPU, rlimit_cpu_handler)
    else:
        warnings.warn("Windows does not support setting SIGXCPU signal")


# ---------------------------------------------------------------------------- #
#                          CLI argument type checker                           #
# ---------------------------------------------------------------------------- #

def dir_path(path):
    if os.path.isfile(path):
        return Path(path)
    raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


# ---------------------------------------------------------------------------- #
#                   Deprecated shim — delegates to XCSP3Adapter               #
# ---------------------------------------------------------------------------- #

def xcsp3_cpmpy(
    benchname: str,
    seed: Optional[int] = None,
    time_limit: Optional[int] = None,
    mem_limit: Optional[int] = None,
    cores: int = 1,
    solver: str = None,
    time_buffer: int = 0,
    intermediate: bool = False,
    verbose: bool = False,
    tmpdir=None,
    **kwargs,
):
    """Deprecated. Use XCSP3Adapter().run() or run_benchmark.py instead."""
    warnings.warn(
        "xcsp3_cpmpy() is deprecated; use XCSP3Adapter().run() or run_benchmark.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if tmpdir is not None:
        warnings.warn(
            "--tmpdir is not used by the new runner and will be ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
    if time_buffer:
        warnings.warn(
            f"--time-buffer={time_buffer} is not propagated to the new runner (uses its own default). "
            "Pass time_buffer via XCSP3Adapter directly if needed.",
            DeprecationWarning,
            stacklevel=2,
        )

    from cpmpy.tools.benchmark.cpbenchy.adapter.xcsp3 import XCSP3Adapter
    XCSP3Adapter().run(
        instance=str(benchname),
        solver=solver,
        time_limit=time_limit,
        mem_limit=mem_limit,
        seed=seed,
        cores=cores,
        intermediate=bool(intermediate),
        verbose=bool(verbose),
    )


# ---------------------------------------------------------------------------- #
#                   Competition CLI — flags kept unchanged                     #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":

    parser = argparse.ArgumentParser("CPMpy XCSP3 executable")

    parser.add_argument("benchname", type=dir_path,
                        help="XCSP3 XML file to parse, with full path and extension.")
    parser.add_argument("-s", "--seed", required=False, type=int, default=None,
                        help="Random seed (integer between 0 and 4294967295).")
    parser.add_argument("-l", "--time-limit", required=False, type=int, default=None,
                        help="Time limit in seconds.")
    parser.add_argument("-m", "--mem-limit", required=False, type=int, default=None,
                        help="Memory limit in MiB (1 MiB = 1024 x 1024 bytes).")
    parser.add_argument("-t", "--tmpdir", required=False, type=dir_path,
                        help="Directory for temporary read/write operations (ignored by new runner).")
    parser.add_argument("-c", "--cores", required=False, type=int, default=None,
                        help="Number of processing units to use.")
    parser.add_argument("--solver", required=False, type=str, default="ortools",
                        help="Underlying CPMpy solver to use (can be 'solver:subsolver').")
    parser.add_argument("--time-buffer", required=False, type=int, default=0,
                        help="Time buffer in seconds (not propagated to new runner).")
    parser.add_argument("--intermediate", action=argparse.BooleanOptionalAction,
                        help="Whether to print intermediate solutions.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction,
                        help="Enable verbose output.")

    args = parser.parse_args()
    if args.verbose:
        print_comment(f"Arguments: {args}")

    try:
        # Signal handling and LZMA decompression are managed by cpbenchy
        # (HandlerObserver and the XCSP3Adapter reader respectively).
        xcsp3_cpmpy(**vars(args))
    except Exception as e:
        print_comment(f"{type(e).__name__} -- {e}")
