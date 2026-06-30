#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## xcsp3.py
##
"""
Helper functions for loading CPMpy models from XCSP3-core formatted files.

XCSP3 is an XML-based format for constraint satisfaction and optimisation problems.
More can be read about it here: 
- https://xcsp.org/specifications/

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_xcsp3
"""

import os
from typing import Union, TextIO, Callable
import builtins


import cpmpy as cp


def load_xcsp3(xcsp3: Union[str, os.PathLike, TextIO], open: Callable = builtins.open) -> cp.Model:
    from cpmpy.tools.xcsp3.parser import load_xcsp3 as load_xcsp3_parser

    return load_xcsp3_parser(xcsp3, open=open)
