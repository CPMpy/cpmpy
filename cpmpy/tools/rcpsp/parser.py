"""
Parser for the PSPLIB RCPSP format.


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_rcpsp
"""


import os
import sys
import lzma
import argparse
import cpmpy as cp
from io import StringIO
from typing import Union


_std_open = open
def read_rcpsp(rcpsp: Union[str, os.PathLike], open=open) -> cp.Model:
    """
    Parser for PSPLIB RCPSP format. Reads in an instance and returns its matching CPMpy model.

    Arguments: 
        rcpsp (str or os.PathLike):
            - A file path to a PSPLIB RCPSP file
            - OR a string containing the RCPSP content directly
        open: (callable):
            If rcpsp is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        cp.Model: The CPMpy model of the PSPLIB RCPSP instance.
    """
    # If rcpsp is a path to a file -> open file
    if isinstance(rcpsp, (str, os.PathLike)) and os.path.exists(rcpsp):
        if open is not None:
            f = open(rcpsp)
        else:
            f = _std_open(rcpsp, "rt")
    # If rcpsp is a string containing a model -> create a memory-mapped file
    else:
        f = StringIO(rcpsp)


    table, capacities = _parse_rcpsp(f)
    model, (start, end, makespan) = _model_rcpsp(job_data=table, capacities=capacities)
    return model

def _parse_rcpsp(f):

    data = dict()

    line = f.readline()
    while not line.startswith("PRECEDENCE RELATIONS:"):
        line = f.readline()
    
    f.readline() # skip keyword line
    line = f.readline() # first line of table, skip
    while not line.startswith("*****"):
        jobnr, n_modes, n_succ, *succ = [int(x) for x in line.split(" ") if len(x.strip())]
        assert len(succ) == n_succ, "Expected %d successors for job %d, got %d" % (n_succ, jobnr, len(succ))
        data[jobnr] = dict(num_modes=n_modes, successors=succ)
        line = f.readline()

    # skip to job info
    while not line.startswith("REQUESTS/DURATIONS:"):
        line = f.readline()

    line = f.readline()
    _j, _m, _d, *_r = [x.strip() for x in line.split(" ") if len(x.strip())] # first line of table
    resource_names = [f"{_r[i]}{_r[i+1]}" for i in range(0,len(_r),2)]
    line = f.readline() # first line of table
    if line.startswith("----") or line.startswith("*****"): # intermediate line in table...
        line = f.readline() # skip

    while not line.startswith("*****"):
        jobnr, mode, duration, *resources = [int(x) for x in line.split(" ") if len(x.strip())]
        assert len(resources) == len(resource_names), "Expected %d resources for job %d, got %d" % (len(resource_names), jobnr, len(resources))
        data[jobnr].update(dict(mode=mode, duration=duration))
        data[jobnr].update({name : req for name, req in zip(resource_names, resources)})
        line = f.readline()
    
    # read resource availabilities
    while not line.startswith("RESOURCEAVAILABILITIES:"):
        line = f.readline()
    
    f.readline() # skip header
    capacities = [int(x) for x in f.readline().split(" ") if len(x)]

    import pandas as pd
    df =pd.DataFrame([dict(jobnr=k ,**info) for k, info in data.items()], 
                        columns=["jobnr", "mode", "duration", "successors", *resource_names])
    df.set_index("jobnr", inplace=True)

    return df, dict(zip(resource_names, capacities))

def _model_rcpsp(job_data, capacities):

    model = cp.Model()

    horizon = job_data.duration.sum() # worst case, all jobs sequential on a machine
    makespan = cp.intvar(0, horizon, name="makespan")

    start = cp.intvar(0, horizon, name="start", shape=len(job_data))
    end = cp.intvar(0, horizon, name="end", shape=len(job_data))

    # ensure capacity is not exceeded
    for rescource, capa in capacities.items():
        model += cp.Cumulative(
            start = start,
            duration = job_data['duration'].tolist(),
            end = end,
            demand = job_data[rescource].tolist(),
            capacity = capa
        )

    # enforce precedences
    for idx, (jobnr, info) in enumerate(job_data.iterrows()):
        for succ in info['successors']:
            model += end[idx] <= start[succ-1] # job ids start at idx 1

    model += end <= makespan
    model.minimize(makespan)

    return model, (start, end, makespan)


def main():
    parser = argparse.ArgumentParser(description="Parse and solve a PSPLIB RCPSP model using CPMpy")
    parser.add_argument("model", help="Path to a PSPLIB RCPSP file (or raw RCPSP string if --string is given)")
    parser.add_argument("-s", "--solver", default=None, help="Solver name to use (default: CPMpy's default)")
    parser.add_argument("--string", action="store_true", help="Interpret the first argument (model) as a raw RCPSP string instead of a file path")
    parser.add_argument("-t", "--time-limit", type=int, default=None, help="Time limit for the solver in seconds (default: no limit)")
    args = parser.parse_args()

    # Build the CPMpy model
    try:
        if args.string:
            model = read_rcpsp(args.model)
        else:
            model = read_rcpsp(os.path.expanduser(args.model))
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