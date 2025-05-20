import os, io, sys
from pathlib import Path
import lzma

import cpmpy as cp

from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3
from .parser_callbacks import CallbacksCPMPy

# from . import xcsp3_dataset

def _parse_xcsp3(path: os.PathLike) -> ParserXCSP3:
    """
    Parses an XCSP3 instance file (.xml) and returns a `ParserXCSP3` instance.
    
    Arguments:
        path: location of the XCSP3 instance to read (expects a .xml file).
    
    Returns:
        A parser object.
    """
    parser = ParserXCSP3(path)
    return parser

def _load_xcsp3(parser: ParserXCSP3) -> cp.Model:
    """
    Takes in a `ParserXCSP3` instance and loads its captured model as a CPMpy model.

    Arguments:
        parser (ParserXCSP3): A parser object to load from.

    Returns:
        The XCSP3 instance loaded as a CPMpy model.
    """
    callbacks = CallbacksCPMPy()
    callbacks.force_exit = True
    callbacker = CallbackerXCSP3(parser, callbacks)
    callbacker.load_instance()
    model = callbacks.cpm_model
   
    return model


def read_xcsp3(path: os.PathLike) -> cp.Model:
    """
    Reads in an XCSP3 instance (.xml) and returns its matching CPMpy model.

    Arguments:
        path: location of the XCSP3 instance to read (expects a .xml file).

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
        

    