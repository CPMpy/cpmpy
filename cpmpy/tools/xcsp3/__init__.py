#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## __init__.py
##
"""
Set of utilities for working with XCSP3-formatted CP models.


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_xcsp3

========================
List of helper functions
========================

.. autosummary::
    :nosignatures:

    _parse_xcsp3
    _load_xcsp3

==================
List of submodules
==================

.. autosummary::
    :nosignatures:

    parser_callbacks
    parser
    analyze
    benchmark
    xcsp3_cpmpy
    dataset
    globals
"""

import os
import lzma
from io import StringIO
from typing import Union, TextIO, Callable
import builtins
from warnings import deprecated

import cpmpy as cp
from cpmpy.tools.datasets.xcsp3 import XCSP3Dataset  # for easier importing
from .parser import load_xcsp3 


# Backward compatibility alias
@deprecated("Use load_xcsp3 instead")
def read_xcsp3(xcsp3: Union[str, os.PathLike, TextIO], open: Callable = builtins.open) -> cp.Model:
    """
    Reads in an XCSP3 instance (.xml or .xml.lzma) and returns its matching CPMpy model.

    Arguments:
        xcsp3 (str or os.PathLike or TextIO):
            - A file path to an XCSP3 instance file (.xml or .xml.lzma), or
            - A string containing the XCSP3 content directly, or
            - A TextIO object already open for reading
        open (Callable): callable to open the file for reading (default: builtin ``open``).
            Use for decompression, e.g. ``lambda p: lzma.open(p, 'rt')`` for ``.xml.lzma``.

    Returns:
        The XCSP3 instance loaded as a CPMpy model.
    """
    return load_xcsp3(xcsp3, open=open)


def decompress_lzma(path: os.PathLike) -> StringIO:
    """
    Decompresses a .lzma file.

    Arguments:
        path: Location of .lzma file

    Returns:
        Memory-mapped decompressed file
    """
    # Decompress the XZ file
    with lzma.open(path, "rt", encoding="utf-8") as f:
        return StringIO(f.read())  # read to memory-mapped file


__all__ = [
    "XCSP3Dataset",
    "decompress_lzma",
    "read_xcsp3",
]
