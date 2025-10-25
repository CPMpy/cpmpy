#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## __init__.py
##
"""
Set of utilities for working with OPB-formatted CP models.

Currently only the restricted OPB PB24 format is supported (without WBO).

==================
List of submodules
==================

.. autosummary::
    :nosignatures:

    parser
"""

from .parser import read_opb
