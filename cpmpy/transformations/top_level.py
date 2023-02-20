from cpmpy.expressions.variables import _BoolVarImpl, _NumVarImpl, bool_val
from cpmpy.expressions.utils import is_num, is_any_list
from ..expressions.core import *


def to_flat_list(expr):
    if isinstance(expr, bool):
        if expr:
            return []
        else:
            return [bool_val(expr)]
    elif isinstance(expr, _BoolVarImpl):
        return [expr]
    elif is_num(expr) or isinstance(expr, _NumVarImpl):
        raise Exception("Numeric constants or numeric variables not allowed as base constraint")

    # recursively flatten list of constraints
    if is_any_list(expr):
        basecons = []
        for e in expr:
            basecons += to_flat_list(e)  # add all at end
        return basecons
    # recursively flatten top-level 'and'
    if isinstance(expr, Operator) and expr.name == 'and':
        basecons = []
        for e in expr.args:
            basecons += to_flat_list(e)  # add all at end
        return basecons
    return [expr]
