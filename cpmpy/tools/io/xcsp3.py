#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## xcsp3.py
##
"""
Helper functions for loading CPMpy models from XCSP3-core formatted files.

XCSP3 is an XML-based format for constraint satisfaction and optimisation problems.
More can be read about it here: 

- https://xcsp.org/specifications/

For additional XCSP3 utilities (benchmarking, analysis, CLI, …),
see :mod:`cpmpy.tools.xcsp3`.

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_xcsp3
"""

import os
from typing import Union, TextIO, Callable
import builtins


import cpmpy as cp


def load_xcsp3(xcsp3: Union[str, os.PathLike, TextIO], open: Callable = builtins.open) -> cp.Model:
    """
    Loads an XCSP3 instance (.xml or .xml.lzma) and returns its matching CPMpy model.

    Arguments:
        xcsp3 (str or os.PathLike or TextIO):
            - A file path to an XML file (optionally LZMA-compressed with `.lzma`), or
            - A string containing the XML content directly, or
            - A TextIO object already open for reading
        open: (callable):
            If xcsp3 is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        The XCSP3 instance loaded as a CPMpy model.
    """
    from cpmpy.tools.xcsp3.parser import load_xcsp3 as load_xcsp3_parser

    return load_xcsp3_parser(xcsp3, open=open)
