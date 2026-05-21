#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## __init__.py
##
"""
Set of utilities for working with XCSP3-formatted CP models.

==================
List of submodules
==================

.. autosummary::
    :nosignatures:

    parser
    parser_callbacks
    analyze
    benchmark
    xcsp3_cpmpy
    dataset
    globals
"""


from cpmpy.tools.datasets.xcsp3 import XCSP3Dataset  # canonical location; dataset.py is deprecated
from .parser import load_xcsp3, read_xcsp3  # read_xcsp3 is alias for backward compatibility