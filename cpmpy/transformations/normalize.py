"""
    Normalize a toplevel list, or simplify Boolean expressions.
"""

import copy
from typing import Any

import numpy as np

from ..expressions.core import BoolVal, Expression, Comparison, Operator
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.utils import eval_comparison, is_false_cst, is_true_cst, is_boolexpr, is_num, is_bool
from ..expressions.variables import _BoolVarImpl
from ..exceptions import NotSupportedError
from ..expressions.globalconstraints import GlobalConstraint
from .negation import recurse_negation


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
                if merge_and and e.name == "and":
                    unravel(e.args, append)
                else:
                    assert (e.is_bool()), f"Only boolean expressions allowed at toplevel, got {e}"
                    append(e) # presumably the most frequent case
            elif isinstance(e, np.ndarray):
                unravel(e.flat, append)  # return iterator over flat elements
            elif isinstance(e, (list, tuple, np.flatiter)):
                unravel(e, append)
            elif e is False or e is np.False_:
                append(BoolVal(e))
            elif e is not True and e is not np.True_:  # if True: pass
                raise NotSupportedError(f"Expression {e} is not a valid CPMpy constraint")

    newlist = []
    append = newlist.append
    unravel((cpm_expr,), append)

    return newlist


def simplify_boolean(lst_of_expr: list[Expression], num_context=False) -> list[Expression]:
    """
    Removes boolean constants from all CPMpy expressions, except for constants in global constraints/functions.
    Solver interfaces are expected to implement special cases of typing for global constraints themselves.

    Only resulting Boolean constant is literal 'false'.
    Boolean constants are promoted to `int` if in a numerical context,
    `ints` are never converted to `bool`.
    
    Arguments:
        list_of_expr: list of CPMpy expressions
    """

    newlist: list[Expression] = []
    for expr in lst_of_expr:
        changed, newexpr = _simplify_boolean_expr(expr, num_context=num_context)
        if changed:
            assert not isinstance(newexpr, int)
            newlist.append(newexpr)
        else:
            newlist.append(expr)
    return newlist


def _simplify_boolean_expr(expr: Expression, num_context=False) -> tuple[bool, Expression|int]:
    changed = False
    if isinstance(expr, Operator):
        args = expr.args
        if expr.has_subexpr():
            changed, rec_args = _simplify_boolean_args(args, num_context=not expr.is_bool())
            if changed:
                args = rec_args

        if expr.name == "or":
            newargs = None
            for i, a in enumerate(args):
                if is_true_cst(a):
                    return True, BoolVal(True)
                elif is_false_cst(a):
                    changed = True
                    if newargs is None:
                        newargs = list(args[:i])
                elif newargs is not None:
                    newargs.append(a)

            if changed:
                if newargs is None:
                    newargs = list(args)
                newexpr = copy.copy(expr)
                newexpr.update_args(newargs)
                return True, newexpr
            return False, expr

        elif expr.name == "and":
            newargs = None
            for i, a in enumerate(args):
                if is_false_cst(a):
                    return True, BoolVal(False)
                elif is_true_cst(a):
                    changed = True
                    if newargs is None:
                        newargs = list(args[:i])
                elif newargs is not None:
                    newargs.append(a)

            if changed:
                if newargs is None:
                    newargs = list(args)
                newexpr = copy.copy(expr)
                newexpr.update_args(newargs)
                return True, newexpr
            return False, expr

        elif expr.name == "->":
            cond, bool_expr = args
            if is_false_cst(cond) or is_true_cst(bool_expr):
                return True, BoolVal(True)
            elif is_true_cst(cond):
                if is_bool(bool_expr):
                    return True, BoolVal(bool_expr)
                return True, bool_expr
            elif is_false_cst(bool_expr):
                _, newcond = _simplify_boolean_expr(recurse_negation(cond), num_context=False)
                # always changed, because negated cond
                return True, newcond

            if changed:
                return True, cond.implies(bool_expr)
            return False, expr

        elif expr.name == "not":
            if isinstance(args[0], _BoolVarImpl):
                return True, ~args[0]
            elif is_true_cst(args[0]):
                return True, BoolVal(False)
            elif is_false_cst(args[0]):
                return True, BoolVal(True)

            if changed:
                newexpr = copy.copy(expr)
                newexpr.update_args(args)
                return True, newexpr
            return False, expr

        elif expr.name == "wsum":
            weights, vars = args
            newvars = None
            for i, v in enumerate(vars):
                if is_bool(v):
                    if newvars is None:
                        newvars = list(vars[:i])
                    changed = True
                    newvars.append(int(v))
                elif newvars is not None:
                    newvars.append(v)
            if changed:
                if newvars is not None:
                    vars = newvars
                newexpr = copy.copy(expr)
                newexpr.update_args((weights, vars))
                return True, newexpr
            return False, expr

        # numerical operators and default operator case:
        # cast booleans to integers in numeric context
        newargs = None
        for i, a in enumerate(args):
            if is_bool(a):
                if newargs is None:
                    newargs = list(args[:i])
                changed = True
                newargs.append(int(a))
            elif newargs is not None:
                newargs.append(a)
        if changed:
            if newargs is None:
                newargs = list(args)
            newexpr = copy.copy(expr)
            newexpr.update_args(newargs)
            return True, newexpr

    elif isinstance(expr, Comparison):
        changed, (lhs, rhs) = _simplify_boolean_args(expr.args, num_context=True)
        name = expr.name
        if is_num(lhs) and is_boolexpr(rhs):  # flip arguments of comparison to reduct nb of cases
            if name == "<":    name = ">"
            elif name == ">":  name = "<"
            elif name == "<=": name = ">="
            elif name == ">=": name = "<="
            lhs, rhs = rhs, lhs
            changed = True
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
        if isinstance(lhs, Expression) and lhs.is_bool() and is_num(rhs):
            # direct simplification of boolean comparisons
            if isinstance(rhs, BoolVal):
                rhs = int(rhs.value())
            if rhs < 0:
                return True, BoolVal(name in  {"!=", ">", ">="}) # all other operators evaluate to False
            elif rhs == 0:
                if name == "!=" or name == ">":
                    return True, lhs
                if name == "==" or name == "<=":
                    return True, recurse_negation(lhs)
                if name == "<":
                    return True, (0 if num_context else BoolVal(False))
                if name == ">=":
                    return True, (1 if num_context else BoolVal(True))
            elif 0 < rhs < 1:
                # a floating point value
                if name == "==":
                    return True, (0 if num_context else BoolVal(False))
                if name == "!=":
                    return True, (1 if num_context else BoolVal(True))
                if name == "<" or name == "<=":
                    return True, recurse_negation(lhs)
                if name == ">" or name == ">=":
                    return True, lhs
            elif rhs == 1:
                if name == "==" or name == ">=":
                    return True, lhs
                if name == "!=" or name == "<":
                    return True, recurse_negation(lhs)
                if name == ">":
                    return True, (0 if num_context else BoolVal(False))
                if name == "<=":
                    return True, (1 if num_context else BoolVal(True))
            elif rhs > 1:
                return True, BoolVal(name in {"!=", "<", "<="}) # all other operators evaluate to False

        # normalize comparison orientation to keep expression on lhs
        if is_num(lhs) and isinstance(rhs, Expression):
            if name == "<":
                name = ">"
            elif name == ">":
                name = "<"
            elif name == "<=":
                name = ">="
            elif name == ">=":
                name = "<="
            lhs, rhs = rhs, lhs
            changed = True

        if changed or (is_num(lhs) and is_num(rhs)):
            newexpr = eval_comparison(name, lhs, rhs)
            if is_bool(newexpr): # Result is a Boolean constant
                return True, (int(newexpr) if num_context else BoolVal(newexpr))
            else: # Result is an expression
                return True, newexpr

    elif isinstance(expr, (GlobalConstraint, GlobalFunction)):
        rec_changed, rec_args = _simplify_boolean_args(expr.args, num_context=False)
        if rec_changed:
            newexpr = copy.copy(expr)
            newexpr.update_args(rec_args)
            return True, newexpr

    return False, expr


def _simplify_boolean_args(args: list[Any]|tuple[Any, ...], num_context=False) -> tuple[bool, list[Any]|tuple[Any, ...]]:
    changed = False
    newargs: list[Any] = []

    for arg in args:
        if isinstance(arg, Expression):
            if isinstance(arg, BoolVal):
                changed = True
                newargs.append(int(arg.value()) if num_context else arg)
                continue
            rec_changed, rec_arg = _simplify_boolean_expr(arg, num_context=num_context)
            if rec_changed:
                changed = True
                newargs.append(rec_arg)
                continue
        elif isinstance(arg, (list, tuple)):
            rec_changed, rec_arg = _simplify_boolean_args(arg, num_context=num_context)
            if rec_changed:
                changed = True
                newargs.append(rec_arg)
                continue
        elif is_bool(arg):
            changed = True
            newargs.append(int(arg) if num_context else BoolVal(arg))
            continue

        newargs.append(arg)

    if changed:
        return True, newargs
    return False, args
