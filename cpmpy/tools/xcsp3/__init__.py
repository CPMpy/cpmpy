#!/usr/bin/env python
#-*- coding:utf-8 -*-
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
        analyze
        benchmark
        xcsp3_cpmpy
        dataset
        globals
"""
from io import StringIO
import lzma
import os
import cpmpy as cp

# Special case for optional cpmpy dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3

from .dataset import XCSP3Dataset # for easier importing

def _parse_xcsp3(path: os.PathLike) -> "ParserXCSP3":
    """
    Parses an XCSP3 instance file (.xml) and returns a `ParserXCSP3` instance.
    
    Arguments:
        path: location of the XCSP3 instance to read (expects a .xml file).
    
    Returns:
        A parser object.
    """
    try:
        from pycsp3.parser.xparser import ParserXCSP3
    except ImportError as e:
        raise ImportError("The 'pycsp3' package is required to parse XCSP3 files. "
                          "Please install it with `pip install pycsp3`.") from e
    
    parser = ParserXCSP3(path)
    return parser

def _load_xcsp3(parser: "ParserXCSP3") -> cp.Model:
    """
    Takes in a `ParserXCSP3` instance and loads its captured model as a CPMpy model.

    Arguments:
        parser (ParserXCSP3): A parser object to load from.

    Returns:
        The XCSP3 instance loaded as a CPMpy model.
    """
    from .parser_callbacks import CallbacksCPMPy
    from pycsp3.parser.xparser import CallbackerXCSP3
    callbacks = CallbacksCPMPy()
    callbacks.force_exit = True
    callbacker = CallbackerXCSP3(parser, callbacks)
    callbacker.load_instance()
    model = callbacks.cpm_model
   
    return model


def read_xcsp3(path: os.PathLike) -> cp.Model:
    """
    Reads in an XCSP3 instance (.xml or .xml.lzma) and returns its matching CPMpy model.

    Arguments:
        path: location of the XCSP3 instance to read (expects a .xml or .xml.lzma file).

    Returns:
        The XCSP3 instance loaded as a CPMpy model.
    """
    # Decompress on the fly if still in .lzma format
    if str(path).endswith(".lzma"):
        path = decompress_lzma(path)

    # Parse and create CPMpy model
    parser = _parse_xcsp3(path)
    model = _load_xcsp3(parser)
    return model

def decompress_lzma(path: os.PathLike) -> StringIO:
    """
    Decompresses a .lzma file.

    Arguments:
        path: Location of .lzma file

    Returns:
        Memory-mapped decompressed file
    """
    # Decompress the XZ file
    with lzma.open(path, 'rt', encoding='utf-8') as f:
        return StringIO(f.read()) # read to memory-mapped file
        

    