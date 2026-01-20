#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## __init__.py
##
"""
Set of utilities for working with MPS-formatted LP/MIP models.


==================
List of submodules
==================

.. autosummary::
    :nosignatures:

    parser
"""

from .parser import read_mps
from .parser import write_mps