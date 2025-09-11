#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## __init__.py
##
"""
Set of utilities for working with WCNF-formatted CP models.


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_wcnf
"""


import os
import lzma
import cpmpy as cp
from io import StringIO
from typing import Union


def _get_var(i, vars_dict):
    """
    Returns CPMpy boolean decision variable matching to index `i` if exists, else creates a new decision variable.

    Arguments:
        i: index
        vars_dict (dict): dictionary to keep track of previously generated decision variables
    """
    if i not in vars_dict:
        vars_dict[i] = cp.boolvar(name=f"x{i}") # <- be carefull that name doesn't clash with generated variables during transformations / user variables
    return vars_dict[i]


def read_wcnf(wcnf: Union[str, os.PathLike]) -> cp.Model:
    """
    Parser for WCNF format. Reads in an instance and returns its matching CPMpy model.

    Arguments: 
        wcnf (str or os.PathLike): A string containing a WCNF-formatted model, or a path to a file containing containing the same.

    Returns:
        cp.Model: The CPMpy model of the WCNF instance.
    """
    # If wcnf is a path to a file -> open file
    if isinstance(wcnf, (str, os.PathLike)) and os.path.exists(wcnf):
        f_open = lzma.open if str(wcnf).endswith(".xz") else open
        f = f_open(wcnf, "rt")
    # If wcnf is a string containing a model -> create a memory-mapped file
    else:
        f = StringIO(wcnf)

    model = cp.Model()
    vars = {}
    soft_terms = []

    for raw in f:
        line = raw.strip()

        # Empty line or a comment -> skip
        if not line or line.startswith("c"):
            continue

        # Hard clause
        if line[0] == "h":
            literals = map(int, line[1:].split())
            clause = [_get_var(i, vars) if i > 0 else ~_get_var(-i, vars)
                      for i in literals if i != 0]
            model.add(cp.any(clause))

        # Soft clause (weight first)
        else:
            parts = line.split()
            weight = int(parts[0])
            literals = map(int, parts[1:])
            clause = [_get_var(i, vars) if i > 0 else ~_get_var(-i, vars)
                    for i in literals if i != 0]
            soft_terms.append(weight * cp.any(clause))

    # Objective = sum of soft clause terms
    if soft_terms:
        model.maximize(sum(soft_terms))

    return model