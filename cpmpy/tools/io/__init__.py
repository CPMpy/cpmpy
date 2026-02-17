"""
IO tools for CPMpy.

This module provides tools to load and write models in various formats.
Use the generic `load(..., format="...")` and `write(..., format="...")` functions to load and write 
models in one of the supported formats.

Some formats can be auto-detected from the file extension, so only a file path is required as argument.
"""

from .writer import write, write_formats
from .reader import load, read, read_formats  # read is alias for backward compatibility
from .utils import get_extension, get_format

# Problem datasets
from .jsplib import load_jsplib, read_jsplib  # read_jsplib is alias for backward compatibility
from .nurserostering import load_nurserostering, read_nurserostering  # read_nurserostering is alias
from .rcpsp import load_rcpsp, read_rcpsp  # read_rcpsp is alias

# Model datasets
from .opb import load_opb, read_opb, write_opb  # read_opb is alias
from .scip import load_scip, read_scip, write_scip  # read_scip is alias
from .wcnf import load_wcnf, read_wcnf  # read_wcnf is alias