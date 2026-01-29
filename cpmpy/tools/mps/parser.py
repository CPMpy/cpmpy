"""
MPS parser.

This file implements helper functions for reading and writing MPS-formatted LP/MIP models.

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_mps
    write_mps
"""


from typing import Optional, Union
import os

import cpmpy as cp
from cpmpy.tools.scip.parser import read_scip


def read_mps(mps: Union[str, os.PathLike], open=open, assume_integer:bool=False) -> cp.Model:
    return read_scip(mps, open, assume_integer)

def write_mps(model: cp.Model, file_path: Optional[str] = None) -> str:
    pass

