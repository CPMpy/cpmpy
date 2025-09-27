#!/usr/bin/env python
#-*- coding:utf-8 -*-
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


from .dataset import XCSP3Dataset # for easier importing
from .parser import read_xcsp3