#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## xcsp3.py
##
"""
XCSP3 parser.

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_xcsp3
"""

import os
from typing import Union


import cpmpy as cp

from cpmpy.tools.xcsp3.parser import load_xcsp3 as load_xcsp3_parser
_std_open = open
def load_xcsp3(xcsp3: Union[str, os.PathLike], open=open) -> cp.Model:
    return load_xcsp3_parser(xcsp3, open=open)

