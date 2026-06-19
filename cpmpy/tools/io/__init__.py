"""
IO tools for CPMpy.

This module provides tools to read and write models in various formats.
Use the generic `read(..., format="...")` and `write(..., format="...")` functions to read and write 
models in one of the supported formats.

Some formats can be auto-detected from the file extension, so only a file path is required as argument.
"""

# Cross-format readers and writers + utility functions
from .writer import write, write_formats
from .loader import load, load_formats
from .utils import get_extension, get_format

# Problem-specific loaders
from .jsplib import load_jsplib        
from .nurserostering import load_nurserostering
from .rcpsp import load_rcpsp

# Standard format loaders and writers
from .opb import load_opb, write_opb
from .scip import load_scip, write_scip
from .wcnf import load_wcnf
from ..xcsp3 import read_xcsp3