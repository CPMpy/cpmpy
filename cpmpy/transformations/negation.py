import copy

from .normalize import toplevel_list
from ..expressions.core import Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl
def push_down_negation(lst_of_expr):

    newlist = []
    for expr in lst_of_expr:
        if expr.name == "not":
            newlist.append(recurse_negation(expr.args[0], negative_context=True))
        else:
            newlist.append(recurse_negation(expr, negative_context=False))

    return toplevel_list(newlist)

def recurse_negation(expr, negative_context=True):
    """
        Negate 'expr' by pushing the negation down into it and its args

        Comparison: swap comparison sign
        Operator.is_bool(): apply DeMorgan
        Global: leave "NOT" operator before global constraint. Use `decompose_globals` for this (AFTER ISSUE #293)

        This function only ensures 'negated normal' for the top-level
        constraint (negating arguments recursively as needed),
        it does not ensure flatness (except if the input is flat)
    """

    if isinstance(expr, (_BoolVarImpl,BoolVal)):
        return ~expr if negative_context else expr

    if isinstance(expr, Comparison):

        newexpr = copy.copy(expr)
        newexpr.args = [recurse_negation(arg, False) for arg in newexpr.args] # check if new 'not' is present in arguments
        if negative_context is True:
            if   expr.name == '==': newexpr.name = '!='
            elif expr.name == '!=': newexpr.name = '=='
            elif expr.name == '<=': newexpr.name = '>'
            elif expr.name == '<':  newexpr.name = '>='
            elif expr.name == '>=': newexpr.name = '<'
            elif expr.name == '>':  newexpr.name = '<='
        return newexpr

    elif isinstance(expr, Operator):
        assert(not negative_context or expr.is_bool()), f"Can only negate boolean expressions but got {expr}"

        if expr.name == "not":
            return recurse_negation(expr.args[0], not negative_context)

        if negative_context and expr.name == "->":
            # XXX this might create a top-level and
            return expr.args[0] & recurse_negation(expr.args[1], True)

        newexpr = copy.copy(expr)
        newexpr.args = [recurse_negation(arg, negative_context) for arg in expr.args]
        if negative_context:
            if   expr.name == "and": newexpr.name = "or"
            elif expr.name == "or": newexpr.name = "and"
            else:
                raise ValueError(f"Unknown operator to negate {expr}")
        return newexpr

    # global constraints
    if hasattr(expr, "decompose"):
        newexpr = copy.copy(expr)
        newexpr.args = [recurse_negation(arg, negative_context=False) for arg in expr.args]
        return ~newexpr if negative_context else newexpr

    # numvars or direct constraint
    if negative_context:
        raise ValueError(f"Unsupported expression to negate: {expr}")

    return expr
