#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## __init__.py
##
"""
Collection of tools for explanation techniques.

=============
List of tools
=============

.. autosummary::
    :nosignatures:

    mus
    mss
    mcs
    marco
    utils
"""

from .marco import marco
from .mcs import mcs, mcs_grow, mcs_grow_naive, mcs_opt
from .mss import mss, mss_grow, mss_grow_naive, mss_opt
from .mus import (
    mus,
    mus_native,
    mus_naive,
    ocus,
    ocus_naive,
    optimal_mus,
    optimal_mus_naive,
    quickxplain,
    quickxplain_naive,
    smus,
)
from .utils import OCUSException

__all__ = [
    "OCUSException",
    "marco",
    "mcs",
    "mcs_grow",
    "mcs_grow_naive",
    "mcs_opt",
    "mss",
    "mss_grow",
    "mss_grow_naive",
    "mss_opt",
    "mus",
    "mus_native",
    "mus_naive",
    "ocus",
    "ocus_naive",
    "optimal_mus",
    "optimal_mus_naive",
    "quickxplain",
    "quickxplain_naive",
    "smus",
]
