"""
IO tools for CPMpy.

This module provides tools to read and write models in various formats.
Use the generic `read(..., format="...")` and `write(..., format="...")` functions to read and write 
models in one of the supported formats.
"""

from .writer import write, write_formats
from .reader import read, read_formats
from .utils import get_extension, get_format

from .jsplib import read_jsplib
# TODO: this tool is just a wrapper around read_scip and write_scip, 
# do we want such a wrapper for each format scip provides? 
# You can already use the generic `read()` and `write()` to read and write any format scip provides.
from .mps import read_mps, write_mps            
from .nurserostering import read_nurserostering
from .opb import read_opb
from .rcpsp import read_rcpsp
from .scip import read_scip, write_scip
from .wcnf import read_wcnf