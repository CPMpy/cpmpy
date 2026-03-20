#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## dimacs.py (re-export from cpmpy.tools.io.dimacs)
##
"""
    DIMACS read/write support.

    This module re-exports from :mod:`cpmpy.tools.io.dimacs` for backward
    compatibility. New code should import from ``cpmpy.tools.io`` or
    ``cpmpy.tools.io.dimacs``.
"""

from cpmpy.tools.io.dimacs import load_dimacs, read_dimacs, write_dimacs

__all__ = ["load_dimacs", "read_dimacs", "write_dimacs"]
