import copy

import numpy as np

from ..expressions.core import BoolVal, Expression, Comparison, Operator
from ..expressions.utils import eval_comparison, is_false_cst, is_true_cst, is_boolexpr, is_num
from ..expressions.variables import NDVarArray
from ..exceptions import NotSupportedError
from ..expressions.globalconstraints import GlobalConstraint

def toplevel_list(cpm_expr, merge_and=True):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression with `.is_bool()` true.

    - cpm_expr: Expression or list of Expressions
    - merge_and: if True then a toplevel 'and' will have its arguments merged at top level
    """
    # very efficient version with limited function lookups and list operations
    def unravel(lst, append):
        for e in lst:
            if isinstance(e, Expression):
                if isinstance(e, NDVarArray):  # sometimes does not have .name
                    unravel(e.flat, append)
                elif merge_and and e.name == "and":
                    unravel(e.args, append)
                else:
                    assert (e.is_bool()), f"Only boolean expressions allowed at toplevel, got {e}"
                    append(e) # presumably the most frequent case
            elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
                unravel(e, append)
            elif e is False or e is np.False_:
                append(BoolVal(e))
            elif e is not True and e is not np.True_:  # if True: pass
                raise NotSupportedError(f"Expression {e} is not a valid CPMpy constraint")

    newlist = []
    append = newlist.append
    unravel((cpm_expr,), append)

    return newlist


def simplify_boolean(lst_of_expr, num_context=False):
    """
    removes boolean constants from all CPMpy expressions
    only resulting boolean constant is literal 'false'
    - list_of_expr: list of CPMpy expressions
    """
    from .negation import recurse_negation # avoid circular import
    newlist = []
    for expr in lst_of_expr:

        if isinstance(expr, bool):
            # not sure if this should happen here or at construction time
            expr = BoolVal(expr)

        if isinstance(expr, BoolVal):
            newlist.append(int(expr.value()) if num_context else expr)

        elif isinstance(expr, Operator):
            args = simplify_boolean(expr.args, num_context=not expr.is_bool())

            if expr.name == "or":
                if any(is_true_cst(arg) for arg in args):
                    newlist.append(1 if num_context else BoolVal(True))
                else:
                    filtered_args = [arg for arg in args if not isinstance(arg, BoolVal)]
                    if len(filtered_args):
                        newlist.append(Operator("or", filtered_args))
                    else:
                        newlist.append(BoolVal(False))

            elif expr.name == "and":
                if any(is_false_cst(arg) for arg in args):
                    newlist.append(0 if num_context else BoolVal(False))
                else:
                    filtered_args = [arg for arg in args if not isinstance(arg, BoolVal)]
                    if len(filtered_args):
                        newlist.append(Operator("and", filtered_args))
                    else:
                        newlist.append(BoolVal(True))

            elif expr.name == "->":
                cond, bool_expr = args
                if is_false_cst(cond) or is_true_cst(bool_expr):
                    newlist.append(BoolVal(True))
                elif is_true_cst(cond):
                    newlist.append(bool_expr)
                elif is_false_cst(bool_expr):
                    newlist += simplify_boolean([recurse_negation(cond)])
                else:
                    newlist.append(cond.implies(bool_expr))

            elif expr.name == "not":
                if is_true_cst(args[0]):
                    newlist.append(BoolVal(False))
                elif is_false_cst(args[0]):
                    newlist.append(BoolVal(True))
                else:
                    newlist.append(~args[0])

            else: # numerical expressions
                newlist.append(Operator(expr.name, args))

        elif isinstance(expr, Comparison):
            lhs, rhs = simplify_boolean(expr.args, num_context=True)
            name = expr.name
            if is_num(lhs) and is_boolexpr(rhs): # flip arguments of comparison to reduct nb of cases
                if name == "<": name = ">"
                elif name == ">": name=  "<"
                elif name == "<=" : name = ">="
                elif name == ">=" : name = "<="
                lhs, rhs = rhs, lhs
            """
            Simplify expressions according to this table:
            x  | <0	 0  0.n	 1	>1
            ----------------------
            == |  F	~x	 F   x	 F
            != |  T	 x	 T  ~x	 T
            >  |  T	 x	 x   F	 F
            <  |  F	 F	~x  ~x	 T
            >= |  T	 T	 x   x	 F   
            <= |  F	~x	~x   T	 T
            """
            if is_boolexpr(lhs) and is_num(rhs):
                # direct simplification of boolean comparisons
                if rhs < 0:
                    newlist.append(BoolVal(name in  {"!=", ">", ">="})) # all other operators evaluate to False
                elif rhs == 0:
                    if name == "!=" or name == ">":
                        newlist.append(lhs)
                    if name == "==" or name == "<=":
                        newlist.append(recurse_negation(lhs))
                    if name == "<":
                        newlist.append(BoolVal(False))
                    if name == ">=":
                        newlist.append(BoolVal(True))
                elif 0 < rhs < 1:
                    # a floating point value
                    if name == "==":
                        newlist.append(BoolVal(False))
                    if name == "!=":
                        newlist.append(BoolVal(True))
                    if name == "<" or name == "<=":
                        newlist.append(recurse_negation(lhs))
                    if name == ">" or name == ">=":
                        newlist.append(lhs)
                elif rhs == 1:
                    if name == "==" or name == ">=":
                        newlist.append(lhs)
                    if name == "!=" or name == "<":
                        newlist.append(recurse_negation(lhs))
                    if name == ">":
                        newlist.append(BoolVal(False))
                    if name == "<=":
                        newlist.append(BoolVal(True))
                elif rhs > 1:
                    newlist.append(BoolVal(name in  {"!=", "<", "<="})) # all other operators evaluate to False
            else:
                newlist.append(eval_comparison(name, lhs, rhs))
        elif isinstance(expr, GlobalConstraint):
            expr = copy.copy(expr)
            expr.args = simplify_boolean(expr.args) # TODO: how to determine boolean or numerical context?
            newlist.append(expr)
        else: # variables/constants/direct constraints
            newlist.append(expr)
    return newlist
