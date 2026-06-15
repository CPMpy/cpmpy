#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## __init__.py
##
"""
PyTorch-style dataset interface for Constraint Optimisation (CO) benchmarks.

CPMpy provides a PyTorch-style dataset interface for loading and iterating over
benchmark instance collections. Each dataset handles downloading, file discovery,
metadata collection, and decompression automatically.

===============
List of classes
===============

.. autosummary::
    :nosignatures:

    Dataset
    FileDataset

================
List of datasets
================

.. autosummary::
    :nosignatures:

    XCSP3Dataset
"""

from .core import (
    Dataset,
    FileDataset,
)
from .xcsp3 import XCSP3Dataset


__all__ = [
    # Base
    "Dataset",
    "FileDataset",
    # Datasets
    "XCSP3Dataset",
]

