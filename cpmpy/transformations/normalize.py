import copy

import numpy as np

from ..expressions.core import BoolVal, Expression, Comparison, Operator
from ..expressions.utils import eval_comparison, is_false_cst, is_true_cst, is_boolexpr, is_num, is_bool
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


# def needs_simplify(expr):
#     if hasattr(expr, 'args'):
#         args = set()
#         for arg in expr.args:
#             if is_bool(arg):
#                 return True  # boolean constants can be simplified away
#             args.add(is_boolexpr(arg))
#         return len(args) > 1  # mixed types should be simplified -> constant ints?
#     else:
#         return False


def simplify_boolean(lst_of_expr, num_context=False, filter=None):
    """
    removes boolean constants from all CPMpy expressions
    only resulting boolean constant is literal 'false'
    - list_of_expr: list of CPMpy expressions
    - filter: a list of booleans, indicating which expressions can be skipped
    """
    from .negation import recurse_negation # avoid circular import
    newlist = []

    for expr in lst_of_expr:
   
        if isinstance(expr, bool):
            # not sure if BoolVal creation should happen here or at construction time
            newlist.append(expr if num_context else BoolVal(expr))

        elif isinstance(expr, BoolVal):
            newlist.append(int(expr.value()) if num_context else expr)

        # Detect branch in expression tree where no boolean constants will follow
        #  Thomas > it causes a  [... <= -(BoolVal(True))] not to be detected TODO
        # elif not needs_simplify(expr): # not expr.has_nested_boolean_constants():
        #     newlist.append(expr)

        elif isinstance(expr, Operator):
            args = simplify_boolean(expr.args, num_context=not expr.is_bool(), filter=expr.nested_boolean_constants())

            if expr.name == "or":
                if any(is_true_cst(arg) for arg in args): # expr | True -> True
                    newlist.append(1 if num_context else BoolVal(True)) # Why a boolval and not just True?
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
            lhs, rhs = simplify_boolean(expr.args, num_context=True, filter=expr.nested_boolean_constants())
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
                if isinstance(rhs, BoolVal):
                    rhs = int(rhs.value()) # ensure proper comparisons below
                if rhs < 0:
                    newlist.append(BoolVal(name in  {"!=", ">", ">="})) # all other operators evaluate to False
                elif rhs == 0:
                    if name == "!=" or name == ">":
                        newlist.append(lhs)
                    if name == "==" or name == "<=":
                        newlist.append(recurse_negation(lhs))
                    if name == "<":
                        newlist.append(0 if num_context else BoolVal(False))
                    if name == ">=":
                        newlist.append(1 if num_context else BoolVal(True))
                elif 0 < rhs < 1:
                    # a floating point value
                    if name == "==":
                        newlist.append(0 if num_context else BoolVal(False))
                    if name == "!=":
                        newlist.append(1 if num_context else BoolVal(True))
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
                        newlist.append(0 if num_context else BoolVal(False))
                    if name == "<=":
                        newlist.append(1 if num_context else BoolVal(True))
                elif rhs > 1:
                    newlist.append(BoolVal(name in  {"!=", "<", "<="})) # all other operators evaluate to False
            else:
                newlist.append(eval_comparison(name, lhs, rhs))

        
        #elif isinstance(expr, GlobalConstraint):
            # TODO: how to determine boolean or numerical context?
            # not sure this is needed at all... maybe in a XOR? but could damage it?
            #if any(needs_simplify(a) for a in expr.args):
            #    expr = copy.copy(expr)
            #    expr.update_args(simplify_boolean(expr.args))
        else: # variables/constants/direct constraints
            newlist.append(expr)
    return newlist
