"""
IO tools for CPMpy.

This module provides tools to read and write models in various formats.
Use the generic `read(..., format="...")` and `write(..., format="...")` functions to read and write 
models in one of the supported formats.

Some formats can be auto-detected from the file extension, so only a file path is required as argument.
"""

from .writer import write, write_formats
from .reader import read, read_formats
from .utils import get_extension, get_format

# Problem datasets
from .jsplib import read_jsplib        
from .nurserostering import read_nurserostering
from .rcpsp import read_rcpsp

# Model datasets
from .opb import read_opb, write_opb
from .scip import read_scip, write_scip
from .wcnf import read_wcnf
from ..xcsp3 import read_xcsp3