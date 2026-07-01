#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## rcpsp.py
##
"""
Helper functions for the PSPLIB RCPSP format.

The PSPLIB RCPSP format is a textual format to represent Resource-Constrained Project Scheduling problems.
More can be read about it here: 

- https://www.om-db.wi.tum.de/psplib/geninfo.php


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_rcpsp
    parse_rcpsp
    to_dataframe
"""


import os
import builtins
import cpmpy as cp
from typing import Union, Callable, TextIO, Any
from cpmpy.tools.io.utils import _handle_loader_input

# Optional dependencies
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def load_rcpsp(rcpsp: Union[str, os.PathLike, TextIO], open:Callable=builtins.open) -> cp.Model:
    """
    Loader for PSPLIB RCPSP format. Loads an instance and returns its matching CPMpy model.

    Arguments: 
        rcpsp (str or os.PathLike or TextIO):
            - A file path to a PSPLIB RCPSP file
            - OR a string containing the RCPSP content directly
            - OR a TextIO object already open for reading
        open (Callable):
            If rcpsp is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        cp.Model: The CPMpy model of the PSPLIB RCPSP instance.
    """
    with _handle_loader_input(rcpsp, open=open) as f:
        data = parse_rcpsp(f)
    model, _ = _model_rcpsp(**data)
    return model

def parse_rcpsp(f: TextIO) -> dict[str, Any]:
    """
    Parse a PSPLIB RCPSP instance file.

    Arguments:
        f (TextIO): The file to parse.

    Returns:
        dict[str, Any]: Native Python structures with keys ``jobs``, ``resource_names``,
            and ``capacities``.

    Note:
        Use :func:`to_dataframe` to convert job data to a pandas DataFrame if needed.
    """
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

    return {
        "jobs": data,
        "resource_names": resource_names,
        "capacities": dict(zip(resource_names, capacities)),
    }


def to_dataframe(data: dict[str, Any]):
    """
    Convert parsed RCPSP job data to a pandas DataFrame.

    Arguments:
        data (dict[str, Any]): Dictionary returned by :func:`parse_rcpsp`.

    Returns:
        pd.DataFrame: Job data indexed by job number.

    Raises:
        ImportError: If pandas is not installed.
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas is required for to_dataframe(). Install it with: pip install pandas")

    resource_names = data["resource_names"]
    df = pd.DataFrame(
        [dict(jobnr=k, **info) for k, info in data["jobs"].items()],
        columns=["jobnr", "mode", "duration", "successors", *resource_names],
    )
    return df.set_index("jobnr")


def _model_rcpsp(jobs: dict[int, dict[str, Any]], resource_names: list[str], capacities: dict[str, int]):

    model = cp.Model()

    job_order = list(jobs.keys())
    durations = [jobs[j]["duration"] for j in job_order]
    horizon = sum(durations)  # worst case, all jobs sequential on a machine
    makespan = cp.intvar(0, horizon, name="makespan")

    start = cp.intvar(0, horizon, name="start", shape=len(job_order))
    end = cp.intvar(0, horizon, name="end", shape=len(job_order))

    # ensure capacity is not exceeded
    for resource in resource_names:
        model.add(cp.Cumulative(
            start=start,
            duration=durations,
            end=end,
            demand=[jobs[j][resource] for j in job_order],
            capacity=capacities[resource],
        ))

    # enforce precedences
    for idx, jobnr in enumerate(job_order):
        for succ in jobs[jobnr]["successors"]:
            model.add(end[idx] <= start[succ - 1])  # job ids start at idx 1

    model.add(end <= makespan)
    model.minimize(makespan)

    return model, (start, end, makespan)
