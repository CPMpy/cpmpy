"""
IO tools for CPMpy.

This module provides tools to load and write models in various formats.
Use the generic `load(..., format="...")` and `write(..., format="...")` functions to load and write 
models in one of the supported formats.

Some formats can be auto-detected from the file extension, so only a file path is required as argument.
"""

from .writer import write, write_formats
from .loader import load, load_formats
from .utils import get_extension, get_format

# Problem datasets
from .xcsp3 import load_xcsp3