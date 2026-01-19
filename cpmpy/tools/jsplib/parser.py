"""
Parser for the JSPLib format.


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_jsplib
"""


import os
import sys
import lzma
import argparse
import cpmpy as cp
import numpy as np
from io import StringIO
from typing import Union


_std_open = open
def read_jsplib(jsp: Union[str, os.PathLike], open=open) -> cp.Model:
    """
    Parser for JSPLib format. Reads in an instance and returns its matching CPMpy model.

    Arguments: 
        jsp (str or os.PathLike):
            - A file path to a JSPlib file
            - OR a string containing the JSPLib content directly
        open: (callable):
            If jsp is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        cp.Model: The CPMpy model of the JSPLib instance.
    """
    # If rcpsp is a path to a file -> open file
    if isinstance(jsp, (str, os.PathLike)) and os.path.exists(jsp):
        if open is not None:
            f = open(jsp)
        else:
            f = _std_open(jsp, "rt")
    # If rcpsp is a string containing a model -> create a memory-mapped file
    else:
        f = StringIO(jsp)


    task_to_machines, task_durations = _parse_jsplib(f)
    model, (start, makespan) = _model_jsplib(task_to_machines=task_to_machines, task_durations=task_durations)
    return model


def _parse_jsplib(f):
    """
    Parse a JSPLib instance file
    Returns two matrices:
        - task to machines indicating on which machine to run which task
        - task durations: indicating the duration of each task
    """

    line = f.readline()
    while line.startswith("#"):
        line = f.readline()
    n_jobs, n_tasks = map(int, line.strip().split(" "))
    matrix = np.fromstring(f.read(), sep=" ", dtype=int).reshape((n_jobs, n_tasks*2))

    task_to_machines = np.empty(dtype=int, shape=(n_jobs, n_tasks))
    task_durations = np.empty(dtype=int, shape=(n_jobs, n_tasks))

    for t in range(n_tasks):
        task_to_machines[:, t] = matrix[:, t*2]
        task_durations[:, t] = matrix[:, t*2+1]

    return task_to_machines, task_durations



def _model_jsplib(task_to_machines, task_durations):

    task_to_machines = np.array(task_to_machines)
    dur = np.array(task_durations)

    assert task_to_machines.shape == task_durations.shape

    n_jobs, n_tasks = task_to_machines.shape

    start = cp.intvar(0, task_durations.sum(), name="start", shape=(n_jobs,n_tasks)) # extremely bad upperbound... TODO
    end = cp.intvar(0, task_durations.sum(), name="end", shape=(n_jobs,n_tasks)) # extremely bad upperbound... TODO
    makespan = cp.intvar(0, task_durations.sum(), name="makespan") # extremely bad upperbound... TODO

    model = cp.Model()
    model += start + dur == end
    model += end[:,:-1] <= start[:,1:] # precedences

    for machine in set(task_to_machines.flat):
        model += cp.NoOverlap(start[task_to_machines == machine],
                              dur[task_to_machines == machine],
                              end[task_to_machines == machine])

    model += end <= makespan
    model.minimize(makespan)

    return model, (start, makespan)



def main():
    parser = argparse.ArgumentParser(description="Parse and solve a JSPLib model using CPMpy")
    parser.add_argument("model", help="Path to a JSPLib file (or raw RCPSP string if --string is given)")
    parser.add_argument("-s", "--solver", default=None, help="Solver name to use (default: CPMpy's default)")
    parser.add_argument("--string", action="store_true", help="Interpret the first argument (model) as a raw JSPLib string instead of a file path")
    parser.add_argument("-t", "--time-limit", type=int, default=None, help="Time limit for the solver in seconds (default: no limit)")
    args = parser.parse_args()

    # Build the CPMpy model
    try:
        if args.string:
            model = read_jsplib(args.model)
        else:
            model = read_jsplib(os.path.expanduser(args.model))
    except Exception as e:
        sys.stderr.write(f"Error reading model: {e}\n")
        sys.exit(1)

    # Solve the model
    try:
        if args.solver:
            result = model.solve(solver=args.solver, time_limit=args.time_limit)
        else:
            result = model.solve(time_limit=args.time_limit)
    except Exception as e:
        sys.stderr.write(f"Error solving model: {e}\n")
        sys.exit(1)

    # Print results
    print("Status:", model.status())
    if result is not None:
        if model.has_objective():
            print("Objective:", model.objective_value())
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()