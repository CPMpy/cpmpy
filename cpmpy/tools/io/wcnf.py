#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## wcnf.py
##
"""
Helper functions for the WCNF format.

WCNF is a textual format to represent MaxSAT problems.
More can be read about it here:

- https://maxsat-evaluations.github.io/

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_wcnf

For writing, use :func:`write_dimacs`.
"""


import os
import builtins
import cpmpy as cp
from typing import Union, Callable, TextIO

from cpmpy.expressions.variables import _BoolVarImpl
from cpmpy.tools.io.utils import _handle_loader_input


def _get_var(i: int, vars_dict: dict[int, _BoolVarImpl]) -> _BoolVarImpl:
    """
    Returns CPMpy boolean decision variable matching to index `i` if exists, else creates a new decision variable.

    Arguments:
        i (int): index
        vars_dict (dict): dictionary to keep track of previously generated decision variables
    Returns:
        cp.BoolVar: The CPMpy boolean decision variable matching to index `i`.
    """
    if i not in vars_dict:
        vars_dict[i] = cp.boolvar(name=f"x{i}") # <- be carefull that name doesn't clash with generated variables during transformations / user variables
    return vars_dict[i]

def load_wcnf(wcnf: Union[str, os.PathLike, TextIO], open:Callable=builtins.open) -> cp.Model:
    """
    Loader for WCNF format. Loads an instance and returns its matching CPMpy model.

    Arguments: 
        wcnf (str or os.PathLike or TextIO):
            - A file path to an WCNF file (optionally LZMA-compressed with `.xz`), or
            - A string containing the WCNF content directly, or
            - A TextIO object already open for reading
        open (Callable): callable to open the file for reading (default: builtin ``open``).

    Returns:
        cp.Model: The CPMpy model of the WCNF instance.
    """

    with _handle_loader_input(wcnf, open=open) as f:

        model = cp.Model()
        vars: dict[int, _BoolVarImpl] = {}
        nr_vars_declared = None
        unsatisfied_soft_terms = []
        for raw in f:
            line = raw.strip()

            # Empty line or a comment -> skip
            if not line or line.startswith("c"):
                continue

            # Problem line
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 4:
                    nr_vars_declared = int(parts[2])
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
                unsatisfied_soft_terms.append(weight * ~cp.any(clause))

        # WCNF/MaxSAT objective: minimize the sum of unsatisfied soft weights.
        if unsatisfied_soft_terms:
            model.minimize(sum(unsatisfied_soft_terms))

        setattr(model, "wcnf_max_var", nr_vars_declared if nr_vars_declared is not None else max(vars, default=0))

        return model
