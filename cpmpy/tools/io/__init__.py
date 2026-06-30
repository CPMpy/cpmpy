"""
IO tools for CPMpy.

This module provides tools to load and write models in various formats.
Use the generic `load(..., format="...")` and `write(..., format="...")` functions to load and write 
models in one of the supported formats.

Some formats can be auto-detected from the file extension, so only a file path is required as argument.
"""

# Cross-format loaders and writers + utility functions
from .writer import write, write_formats
from .loader import load, load_formats
from .utils import get_extension, get_format

# Problem-specific loaders
from .jsplib import load_jsplib        
from .nurserostering import load_nurserostering
from .rcpsp import load_rcpsp

# Standard format loaders and writers
from .opb import load_opb, write_opb
from .scip_formats import load_scip, write_scip
from .wcnf import load_wcnf
from .dimacs import load_dimacs, write_dimacs
from .xcsp3 import load_xcsp3

_all__ = [
    "load",
    "load_formats",
    "write",
    "write_formats",
    "load_opb",
    "write_opb",
    "load_scip",
    "write_scip",
    "load_dimacs",
    "write_dimacs",
    "load_wcnf",
    "load_xcsp3",
    "load_jsplib",
    "load_rcpsp",
    "load_nurserostering",
    "get_extension",
    "get_format",
]