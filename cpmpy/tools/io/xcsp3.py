#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## xcsp3.py
##
"""
XCSP3 parser.

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_xcsp3
"""

from io import StringIO
import os
from typing import Union

import cpmpy as cp

# Special case for optional cpmpy dependencies
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pycsp3.parser.xparser import ParserXCSP3

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


_std_open = open
def load_xcsp3(xcsp3: Union[str, os.PathLike], open=open) -> cp.Model:
    """
    Loads an XCSP3 instance (.xml or .xml.lzma) and returns its matching CPMpy model.

    Arguments:
        xcsp3 (str or os.PathLike):
            - A file path to an XCSP3 file (optionally LZMA-compressed with `.lzma`)
            - OR a string containing the XCSP3 content directly
        open: (callable):
            If xcsp3 is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        The XCSP3 instance loaded as a CPMpy model.
    """
    # If xcsp3 is a path to a file -> open file
    if isinstance(xcsp3, (str, os.PathLike)) and os.path.exists(xcsp3):
        if open is not None:
            f = open(xcsp3)
        else:
            f = _std_open(xcsp3, "rt")
    # If xcsp3 is a string containing a model -> create a memory-mapped file
    else:
        f = StringIO(xcsp3)

    # Parse and create CPMpy model
    parser = _parse_xcsp3(f)
    model = _load_xcsp3(parser)
    return model