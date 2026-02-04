"""
    Normalize a toplevel list, or simplify Boolean expressions.
"""

import copy

import numpy as np
import cpmpy as cp

from ..expressions.core import BoolVal, Expression, Comparison, Operator
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.utils import eval_comparison, is_false_cst, is_true_cst, is_boolexpr, is_num, is_bool
from ..expressions.variables import NDVarArray, _BoolVarImpl
from ..exceptions import NotSupportedError
from ..expressions.globalconstraints import GlobalConstraint


def toplevel_list(cpm_expr, merge_and=True):
    """
    Unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression with :func:`~cpmpy.expressions.core.Expression.is_bool()` true.

    Arguments:
        cpm_expr:   Expression or list of Expressions
        merge_and:  if True then a toplevel 'and' will have its arguments merged at top level
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
    Removes boolean constants from all CPMpy expressions, except for constants in global constraints/functions.
    Solver interfaces are expected to implement special cases of typing for global constraints themselves.

    Only resulting Boolean constant is literal 'false'.
    Boolean constants are promoted to `int` if in a numerical context,
    `ints` are never converted to `bool`.
    
    Arguments:
        list_of_expr: list of CPMpy expressions
    """

    newlist = []
    for expr in lst_of_expr:

        if isinstance(expr, Operator):
            if expr.has_subexpr():
                expr_args = simplify_boolean(expr.args, num_context=not expr.is_bool())
            else:
                expr_args = expr.args

            args = expr_args
            if expr.name == "or":
                i = 0
                while i < len(args):
                    a = args[i]
                    if isinstance(a, _BoolVarImpl):
                        i += 1 # default path
                    elif is_true_cst(a):
                        break
                    elif is_false_cst(a):
                        if args is expr_args: # will remove this one, need to copy args...
                            args = args.copy()
                        args.pop(i)
                    else:
                        i += 1

                if i != len(args): # found "True" along the way, early exit
                    newlist.append(1 if num_context else BoolVal(True))
                elif args is not expr.args: # removed something, or changed due to subexpr
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
                        if args is expr_args:  # will remove this one, need to copy args...
                            args = args.copy()
                        args.pop(i)
                    else:  # subexpression, should not happen here...
                        i += 1

                if i != len(args): # found "False" along the way, early exit
                    newlist.append(0 if num_context else BoolVal(False))
                elif args is not expr.args: # removed something, or changed due to subexpr
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
                if isinstance(args[0], _BoolVarImpl): # fast path
                    newlist.append(~args[0])
                elif is_true_cst(args[0]):
                    newlist.append(0 if num_context else BoolVal(False))
                elif is_false_cst(args[0]):
                    newlist.append(1 if num_context else BoolVal(True))
                elif args is not expr.args:
                    expr = copy.copy(expr)
                    expr.update_args(expr_args)
                    newlist.append(expr)
                else: # nothing changed
                    newlist.append(expr)

            # numerical expressions
            elif expr.name == "wsum":
                weights, vars = args
                newvars = [int(v) if is_bool(v) else v for v in vars]
                if args is not expr.args or any(v1 is not v2 for v1,v2 in zip(vars, newvars)):
                    newexpr = copy.copy(expr)
                    newexpr.update_args([weights, newvars])
                    newlist.append(newexpr)
                else:
                    newlist.append(expr)

            else: # default case, check if anything is Bool that should be int
                newargs = [int(a) if is_bool(a) else a for a in args]
                if expr_args is not expr.args or any(v1 is not v2 for v1, v2 in zip(args, newargs)):
                    newexpr = copy.copy(expr)
                    newexpr.update_args(newargs)
                    newlist.append(newexpr)
                else:
                    newlist.append(expr)

        elif isinstance(expr, Comparison):
            lhs, rhs = simplify_boolean(expr.args, num_context=True)
            name = expr.name
            if is_num(lhs) and is_boolexpr(rhs):  # flip arguments of comparison to reduct nb of cases
                if name == "<":    name = ">"
                elif name == ">":  name = "<"
                elif name == "<=": name = ">="
                elif name == ">=": name = "<="
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
                    newlist.append(BoolVal(name in {"!=", "<", "<="})) # all other operators evaluate to False
            else:
                res = eval_comparison(name, lhs, rhs)
                if is_bool(res): # Result is a Boolean constant
                    newlist.append(int(res) if num_context else BoolVal(res))
                else: # Result is an expression
                    newlist.append(res)    

        elif isinstance(expr, (GlobalConstraint, GlobalFunction)):
            newargs = simplify_boolean(expr.args) # TODO: how to determine which are Bool/int?
            if any(a1 is not a2 for a1,a2 in zip(expr.args, newargs)):
                expr = copy.copy(expr)
                expr.update_args(newargs)
            newlist.append(expr)
        elif is_bool(expr):  # unlikely base-case (Boolean constant)
            newlist.append(int(expr) if num_context else BoolVal(expr))
        elif isinstance(expr, list):  # nested list in args (like for wsum)
            newlist.append(simplify_boolean(expr))
        else:  # variables/constants/direct constraints
            newlist.append(expr)
    return newlist
