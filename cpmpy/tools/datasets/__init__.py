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

    JSPLibDataset
    MaxSATEvalDataset
    MIPLibDataset
    NurseRosteringDataset
    OPBDataset
    PSPLibDataset
    SATDataset
    ScaledSudokuDataset
    XCSP3Dataset
"""

from .core import (
    Dataset,
    FileDataset,
)
from .xcsp3 import XCSP3Dataset
from .jsplib import JSPLibDataset
from .psplib import PSPLibDataset
from .miplib import MIPLibDataset
from .mse import MaxSATEvalDataset
from .opb import OPBDataset
from .sat import SATDataset
from .nurserostering import NurseRosteringDataset
from .scaledsudoku import ScaledSudokuDataset


__all__ = [
    # Base
    "Dataset",
    "FileDataset",
    # Datasets
    "XCSP3Dataset",
    "JSPLibDataset",
    "PSPLibDataset",
    "MIPLibDataset",
    "MaxSATEvalDataset",
    "OPBDataset",
    "SATDataset",
    "NurseRosteringDataset",
    "ScaledSudokuDataset",
]

