#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## utils.py
##
"""
    Utilities for explanation techniques

    =================
    List of functions
    =================

    .. autosummary::
        :nosignatures:

        make_assump_model
"""

import copy
import cpmpy as cp
from cpmpy.expressions.utils import is_any_list
from cpmpy.expressions.variables import NegBoolView
from cpmpy.transformations.normalize import toplevel_list

def make_assump_model(soft, hard=[], name=None):
    """
        Construct implied version of all soft constraints.
        Can be used to extract cores (see :func:`tools.mus() <cpmpy.tools.explain.mus.mus>`).
        Provide name for assumption variables with `name` param.
    """
    # ensure toplevel list
    soft2 = toplevel_list(soft, merge_and=False)

    # make assumption variables
    assump = cp.boolvar(shape=(len(soft2),), name=name)

    # hard + implied soft constraints
    hard = toplevel_list(hard)
    model = cp.Model(hard + [assump.implies(soft2)])  # each assumption variable implies a candidate

    return model, soft2, assump

def replace_cons_with_assump(cpm_cons, assump_map):
    """
        Replace soft constraints with assumption variables in a Boolean CPMpy expression.
    """

    if is_any_list(cpm_cons):
        return [replace_cons_with_assump(c, assump_map) for c in cpm_cons]
    
    if cpm_cons in assump_map:
        return assump_map[cpm_cons]
    
    elif hasattr(cpm_cons, "args"):
        cpm_cons = copy.copy(cpm_cons)
        cpm_cons.update_args(replace_cons_with_assump(cpm_cons.args, assump_map))
        return cpm_cons

    elif isinstance(cpm_cons, NegBoolView):
        return ~replace_cons_with_assump(cpm_cons._bv, assump_map)
    return cpm_cons

class OCUSException(Exception):
    pass