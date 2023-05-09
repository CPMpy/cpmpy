import numpy as np

from ..expressions.core import BoolVal, Expression, Comparison, Operator
from ..expressions.utils import eval_comparison
from ..expressions.variables import NDVarArray
from ..exceptions import NotSupportedError

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
            elif e is False:
                append(BoolVal(e))
            elif e is not True:  # if True: pass
                raise NotSupportedError(f"Expression {e} is not a valid CPMpy constraint")

    newlist = []
    append = newlist.append
    unravel((cpm_expr,), append)

    return newlist


def simplify_bool_const(lst_of_expr, num_context=False):
    """
    removes boolean constants from all CPMpy expressions
    only resulting boolean constant is literal 'false'
    - list_of_expr: list of CPMpy expressions
    """

    def _is_false(expr):
        return isinstance(expr, BoolVal) and expr.args[0] is False

    def _is_true(expr):
        return isinstance(expr, BoolVal) and expr.args[0] is True

    newlist = []
    for expr in lst_of_expr:

        if isinstance(expr, bool):
            # not sure if this should happen here or at construction time
            expr = BoolVal(expr)

        if isinstance(expr, BoolVal):
            newlist.append(int(expr.value()) if num_context else expr)

        elif isinstance(expr, Operator):
            args = simplify_bool_const(expr.args, num_context=not expr.is_bool())

            if expr.name == "or":
                if any(_is_true(arg) for arg in args):
                    newlist.append(1 if num_context else BoolVal(True))
                else:
                    newlist.append(Operator("or", [arg for arg in args if not isinstance(arg, BoolVal)]))

            elif expr.name == "and":
                if any(_is_false(arg) for arg in args):
                    newlist.append(0 if num_context else BoolVal(False))
                else:
                    newlist.append(Operator("and", [arg for arg in args if not isinstance(arg, BoolVal)]))

            elif expr.name == "->":
                cond, bool_expr = args
                if _is_false(cond) or _is_true(bool_expr):
                    newlist.append(BoolVal(True))
                elif _is_true(cond):
                    newlist.append(bool_expr)
                elif _is_false(bool_expr):
                    newlist += simplify_bool_const([~cond])
                else:
                    newlist.append(cond.implies(bool_expr))

            elif expr.name == "not":
                if isinstance(args[0], BoolVal):
                    newlist.append(BoolVal(not args[0]))
                else:
                    newlist.append(~args[0])

            else: # numerical expressions
                newlist.append(Operator(expr.name, args))

        elif isinstance(expr, Comparison):
            lhs, rhs = simplify_bool_const(expr.args, num_context=True)
            newlist.append(eval_comparison(expr.name, lhs, rhs))
        else: # global constraints
            newlist.append(expr)

    return newlist
