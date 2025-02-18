"""
    Normalizing the constraints given to a CPMpy model.
"""

import copy

import numpy as np
import cpmpy as cp

from ..expressions.core import BoolVal, Expression, Comparison, Operator
from ..expressions.utils import eval_comparison, is_false_cst, is_true_cst, is_boolexpr, is_num, is_bool, is_int
from ..expressions.variables import NDVarArray
from ..exceptions import NotSupportedError
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint


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
                args = simplify_boolean(expr.args, num_context=not expr.is_bool())
            else:
                args = list(expr.args)

            if expr.name == "or":
                for i,a in enumerate(list(args)):
                    if is_true_cst(a):
                        newlist.append(1 if num_context else BoolVal(True))
                        break
                    elif is_false_cst(a):
                        args.pop(i)
                else: # did not find True constant, might have removed False
                    if len(args) == 0:
                        newlist.append(BoolVal(False))
                    if len(args) != expr.args:
                        expr.update_args(args)
                    newlist.append(expr)
                    continue

            elif expr.name == "and":
                for i,a in enumerate(list(args)):
                    if is_false_cst(a):
                        newlist.append(0 if num_context else BoolVal(False))
                        break
                    elif is_true_cst(a):
                        args.pop(i)
                else: # did not find False constant, might have removed True
                    if len(args) == 0:
                        newlist.append(BoolVal(True))
                        continue
                    elif len(args) != expr.args:
                        expr.update_args(args)
                    newlist.append(expr)

            elif expr.name == "->":
                cond, bool_expr = args
                if is_false_cst(cond) or is_true_cst(bool_expr):
                    newlist.append(BoolVal(True))
                elif is_true_cst(cond):
                    newlist.append(bool_expr)
                elif is_false_cst(bool_expr):
                    newlist += simplify_boolean([cp.transformations.negation.recurse_negation(cond)])
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
                assert not expr.is_bool()
                if expr.name == "wsum":
                    weights, vars = expr.args
                    vars = [int(v) if is_int(v) else v for v in vars]
                    args = [weights, vars]
                else:
                    args = [int(a) if is_int(a) else a for a in args]
                newlist.append(Operator(expr.name, args))

        elif isinstance(expr, Comparison):
            lhs, rhs = expr.args
            if isinstance(lhs, Expression) and lhs.has_subexpr():
                lhs_args = simplify_boolean(expr.args)
                expr.update_args(lhs_args) # TODO check if args changed?
            if isinstance(rhs, Expression) and rhs.has_subexpr():
                rhs_args = simplify_boolean(expr.args)
                expr.update_args(rhs_args)

            if is_bool(lhs): lhs = int(lhs)
            if is_bool(rhs): rhs = int(rhs)
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
