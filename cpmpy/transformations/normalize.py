"""
    Normalizing the constraints given to a CPMpy model.
"""

import copy

import numpy as np
import cpmpy as cp

from ..expressions.core import BoolVal, Expression, Comparison, Operator
from ..expressions.utils import eval_comparison, is_false_cst, is_true_cst, is_boolexpr, is_num, is_bool
from ..expressions.variables import NDVarArray, _BoolVarImpl
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

    newlist = []
    for expr in lst_of_expr:

        if isinstance(expr, Operator):
            if expr.has_subexpr():
                newexpr = copy.copy(expr)
                newexpr.update_args(simplify_boolean(expr.args, num_context=not expr.is_bool()))
                newlist.append(newexpr)

            else:  # no need to recurse, will check constants from here
                args = expr.args
                if expr.name == "or":
                    i = 0
                    while i < len(args):
                        a = args[i]
                        if isinstance(a, _BoolVarImpl):
                            i += 1 # default path
                        elif is_true_cst(a):
                            break
                        elif is_false_cst(a):
                            if args is expr.args: # will remove this one, need to copy args...
                                args = expr.args.copy()
                            args.pop(i)
                        else:  # subexpression, should not happen here...
                            raise ValueError(f"Unexpected argument {a} of expression {expr}, did not expect a subexpr")

                    if i != len(args): # found "True" along the way, early exit
                        newlist.append(1 if num_context else BoolVal(True))
                    elif args is not expr.args: # removed something
                        newexpr = copy.copy(expr)
                        newexpr.update_args(args)
                        newlist.append(newexpr)
                    else: # no changes
                        newlist.append(expr)

                elif expr.name == "and":
                    # filter out True/False constants
                    i = 0
                    while i < len(args):
                        a = args[i]
                        if isinstance(a, _BoolVarImpl):
                            i += 1 # default path
                        elif is_false_cst(a):
                            break
                        elif is_true_cst(a):
                            if args is expr.args:  # will remove this one, need to copy args...
                                args = expr.args.copy()
                            args.pop(i)
                        else:  # subexpression, should not happen here...
                            raise ValueError(f"Unexpected argument {a} of expression {expr}, did not expect a subexpr")

                    if i != len(args): # found "False" along the way, early exit
                        newlist.append(0 if num_context else BoolVal(False))
                    elif args is not expr.args: # removed something
                        newexpr = copy.copy(expr)
                        newexpr.update_args(args)
                        newlist.append(newexpr)
                    else: # no changes
                        newlist.append(expr)

                elif expr.name == "->":
                    cond, bool_expr = args
                    if is_false_cst(cond) or is_true_cst(bool_expr):
                        newlist.append(1 if num_context else BoolVal(True))
                    elif is_true_cst(cond):
                        newlist.append(bool_expr)
                    elif is_false_cst(bool_expr):
                        newlist += simplify_boolean([cp.transformations.negation.recurse_negation(cond)])
                    else:
                        newlist.append(cond.implies(bool_expr))

                elif expr.name == "not":
                    if is_true_cst(args[0]):
                        newlist.append(0 if num_context else BoolVal(False))
                    elif is_false_cst(args[0]):
                        newlist.append(1 if num_context else BoolVal(True))
                    else:
                        newlist.append(expr)

                # numerical expressions
                elif expr.name == "wsum":
                    weights, vars = args
                    newvars = [int(v) if is_bool(v) else v for v in vars]
                    if any(v1 is not v2 for v1,v2 in zip(vars, newvars)):
                        newexpr = copy.copy(expr)
                        newexpr.update_args([weights, newvars])
                        newlist.append(newexpr)
                    else:
                        newlist.append(expr)
                else:
                    newargs = [int(a) if is_bool(a) else a for a in args]
                    if any(v1 is not v2 for v1, v2 in zip(args, newargs)):
                        newexpr = copy.copy(expr)
                        newexpr.update_args(newargs)
                        newlist.append(newexpr)
                    else:
                        newlist.append(expr)


        elif isinstance(expr, Comparison):
            lhs, rhs = simplify_boolean(expr.args, num_context=True)
            name = expr.name
            if is_num(lhs) and is_boolexpr(rhs):  # flip arguments of comparison to reduct nb of cases
                if name == "<":
                    name = ">"
                elif name == ">":
                    name = "<"
                elif name == "<=":
                    name = ">="
                elif name == ">=":
                    name = "<="
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
                    rhs = int(rhs.value())
                if rhs < 0:
                    newlist.append(BoolVal(name in  {"!=", ">", ">="})) # all other operators evaluate to False
                elif rhs == 0:
                    if name == "!=" or name == ">":
                        newlist.append(lhs)
                    if name == "==" or name == "<=":
                        newlist.append(cp.transformations.negation.recurse_negation(lhs))
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
                        newlist.append(cp.transformations.negation.recurse_negation(lhs))
                    if name == ">" or name == ">=":
                        newlist.append(lhs)
                elif rhs == 1:
                    if name == "==" or name == ">=":
                        newlist.append(lhs)
                    if name == "!=" or name == "<":
                        newlist.append(cp.transformations.negation.recurse_negation(lhs))
                    if name == ">":
                        newlist.append(0 if num_context else BoolVal(False))
                    if name == "<=":
                        newlist.append(1 if num_context else BoolVal(True))
                elif rhs > 1:
                    newlist.append(BoolVal(name in  {"!=", "<", "<="})) # all other operators evaluate to False
            else:
                newlist.append(eval_comparison(name, lhs, rhs))

        elif isinstance(expr, GlobalConstraint):
            expr = copy.copy(expr)
            expr.update_args(simplify_boolean(expr.args)) # TODO: how to determine boolean or numerical context? also i this even needed?
            newlist.append(expr)
        elif is_bool(expr): # very unlikely base-case
            newlist.append(int(expr) if num_context else BoolVal(expr))
        elif isinstance(expr, DirectConstraint):
            newlist.append(expr)

        else:
            raise ValueError(f"Unexpected expression to normalize: {expr}, please report on github.")
    return newlist
