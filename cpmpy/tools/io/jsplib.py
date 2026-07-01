#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## jsplib.py
##
"""
Loader for the JSPLib format.


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_jsplib
"""


import os
import builtins
import cpmpy as cp
import numpy as np
from typing import Union, Callable, TextIO

from cpmpy.expressions.variables import NDVarArray, _IntVarImpl
from cpmpy.tools.io.utils import _handle_loader_input


def load_jsplib(jsp: Union[str, os.PathLike, TextIO], open:Callable=builtins.open) -> cp.Model:
    """
    Loader for JSPLib format. Loads an instance and returns its matching CPMpy model.

    Arguments: 
        jsp (str or os.PathLike or TextIO):
            - A file path to a JSPlib file, or
            - A string containing the JSPLib content directly, or
            - A TextIO object already open for reading
        open (Callable):
            If jsp is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        cp.Model: The CPMpy model of the JSPLib instance.
    """
    with _handle_loader_input(jsp, open=open) as f:
        task_to_machines, task_durations = _parse_jsplib(f)
    model, (start, makespan) = _model_jsplib(task_to_machines=task_to_machines, task_durations=task_durations)
    return model


def _parse_jsplib(f: TextIO) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a JSPLib instance file

    Arguments:
        f (TextIO): The file to parse.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two matrices:
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


def _model_jsplib(task_to_machines: np.ndarray, task_durations: np.ndarray) -> tuple[cp.Model, tuple[NDVarArray, _IntVarImpl]]:
    """
    Model a JSPLib instance

    Arguments:
        task_to_machines (np.ndarray): The task to machines matrix
        task_durations (np.ndarray): The task durations matrix

    Returns:
        tuple[cp.Model, tuple[NDVarArray, _IntVarImpl]]: The model and the start and makespan variables

    Raises:
        AssertionError: If the shapes of the matrices are not compatible
    """

    # Check if the shapes of the matrices are compatible
    assert task_to_machines.shape == task_durations.shape

    n_jobs, n_tasks = task_to_machines.shape

    start = cp.intvar(0, task_durations.sum(), name="start", shape=(n_jobs,n_tasks)) # extremely bad upperbound... TODO
    end = cp.intvar(0, task_durations.sum(), name="end", shape=(n_jobs,n_tasks)) # extremely bad upperbound... TODO
    makespan = cp.intvar(0, task_durations.sum(), name="makespan") # extremely bad upperbound... TODO

    model = cp.Model()
    model.add(end[:,:-1] <= start[:,1:]) # precedences

    for machine in set(task_to_machines.flat):
        model.add(cp.NoOverlap(start[task_to_machines == machine],
                              task_durations[task_to_machines == machine],
                              end[task_to_machines == machine]))

    model.add(end <= makespan)
    model.minimize(makespan)

    return model, (start, makespan)
