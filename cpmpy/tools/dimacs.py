#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## dimacs.py
##
"""
    Dimacs tooling has moved to the io module. This file is kept for backward compatibility.

    .. deprecated:: 1.0.0
          Please use :mod:`cpmpy.tools.io.dimacs` instead.

    This file implements helper functions for exporting CPMpy models from and to DIMACS format.
    DIMACS is a textual format to represent CNF problems.
    The header of the file should be formatted as ``p cnf <n_vars> <n_constraints>``.
    If the number of variables and constraints are not given, it is inferred by the parser.

    Each remaining line of the file is formatted as a list of integers.
    An integer represents a Boolean variable and a negative Boolean variable is represented using a `'-'` sign.
"""

import warnings

from cpmpy.tools.io.dimacs import write_dimacs
from cpmpy.tools.io.dimacs import load_dimacs

def read_dimacs(fname):
    """
    .. deprecated:: 1.0.0
          Please use :func:`load_dimacs` instead.

    Read a CPMpy model from a DIMACS formatted file strictly following the specification:
    https://web.archive.org/web/20190325181937/https://www.satcompetition.org/2009/format-benchmarks2009.html
    
    .. note::
        The p-line has to denote the correct number of variables and clauses
    
    :param fname: the name of the DIMACS file
    :param sep: optional, separator used in the DIMACS file, will try to infer if None
    """
    warnings.warn("Deprecated, use load_dimacs instead", DeprecationWarning)
    return load_dimacs(fname)



