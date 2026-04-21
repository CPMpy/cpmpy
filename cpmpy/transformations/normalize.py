"""
    Normalize a toplevel list, or simplify Boolean expressions.
"""

import copy
from typing import Any, Literal, cast, overload
from typing_extensions import Optional

import numpy as np
import cpmpy as cp

from ..expressions.core import BoolVal, Expression, Comparison, Operator
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.utils import is_false_cst, is_true_cst, is_boolexpr, is_num, is_bool
from ..expressions.python_builtins import cpm_array
from ..expressions.variables import _BoolVarImpl, NDVarArray
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


def simplify_boolean(lst_of_expr: list[Expression], num_context: Optional[bool] = None) -> list[Expression]:
    """
    Removes boolean constants from all CPMpy expressions, except for constants in global constraints/functions.
    Solver interfaces are expected to implement special cases of typing for global constraints themselves.

    Only resulting Boolean constant is literal 'false'.
    Boolean constants are promoted to `int` if in a numerical context, `ints` are never converted to `bool`.
    
    Arguments:
        list_of_expr: list of CPMpy expressions
        num_context: Deprecated, do not use this parameter anymore.
    
    Returns:
        list[Expression]: list of CPMpy expressions with Boolean constants removed
    """
    if num_context is not None:
        raise DeprecationWarning("The num_context parameter is deprecated and will be removed in the future. Do not use it anymore.")

    newlist: list[Expression] = []
    for expr in lst_of_expr:
        changed, newexpr = _simplify_boolean_expr(expr, num_context=False) # toplevel call, so always Boolean context
        if changed:
            if newexpr.name == "and":
                for b in newexpr.args:
                    if isinstance(b, Expression):
                        newlist.append(b)
                    elif not b: # either BoolVal(False) or False
                        newlist.append(BoolVal(b))
                        break  # stop early if a False
            elif isinstance(newexpr, BoolVal):
                if not newexpr:
                    newlist.append(BoolVal(False))
                    break # stop early if a False
                else: # True, skip
                    pass
            else:
                newlist.append(newexpr)
        else:
            newlist.append(expr)
    
    return newlist

@overload
def _simplify_boolean_expr(expr: Expression, num_context: Literal[False]) -> tuple[bool, Expression]: ...
@overload
def _simplify_boolean_expr(expr: Expression, num_context: Literal[True]) -> tuple[bool, Expression | int]: ...
@overload
def _simplify_boolean_expr(expr: Expression, num_context: bool = False) -> tuple[bool, Expression | int]: ...
def _simplify_boolean_expr(expr: Expression, num_context: bool = False) -> tuple[bool, Expression | int]:
    """
    Well-typed helper function to eliminate boolean constants in an Expression.
    Removes Boolean constants from logical operators, and replaces Boolean constants with ints in a numerical context (sum, comparisons, etc.)
    Uses :func:`_simplify_boolean_args()` to recurse into the arguments of all other expressions.
    
    Arguments:
        expr: Expression to eliminate boolean constants from
        num_context: Whether the expression is used as a numerical argument in another expression.

    Returns:
        tuple[bool, Expression]: (changed, newexpr)

    """

    changed = False

    if isinstance(expr, Operator):

        rec_changed, expr_args = _simplify_boolean_args(expr.args, num_context=not expr.is_bool())
        if rec_changed:
            changed = True
        
        if expr.name == "or": # filter out True/False constants
            i = 0
            while i < len(expr_args):
                a = expr_args[i]
                if isinstance(a, _BoolVarImpl):
                    i += 1 # fast path for clauses -- avoid isinstance checks below
                elif is_true_cst(a):
                    return True, 1 if num_context else cp.BoolVal(True)
                elif is_false_cst(a):
                    if changed is False: # we are operating on the original args list, so need to copy to a list first
                        expr_args = list(expr_args)
                        changed = True
                    expr_args = cast(list[Any], expr_args) # changed is True, so expr_args is a list
                    expr_args.pop(i)
                else: # some subexpression, already simplified above
                    i += 1

        elif expr.name == "and": # filter out True/False constants
            i = 0
            while i < len(expr_args):
                a = expr_args[i]
                if isinstance(a, _BoolVarImpl):
                    i += 1 # default path
                elif is_false_cst(a):
                    return True, 0 if num_context else cp.BoolVal(False)
                elif is_true_cst(a):
                    if changed is False: # we are operating on the the original args tuple, so need to copy to a list first
                        expr_args = list(expr_args)
                        changed = True
                    expr_args = cast(list[Any], expr_args) # changed is True, so expr_args is a list
                    expr_args.pop(i)
                else:  # subexpression, already simplified above
                    i += 1

        elif expr.name == "->":
            cond, bool_expr = expr_args
            if is_false_cst(cond) or is_true_cst(bool_expr):
                return True, 1 if num_context else cp.BoolVal(True)
            elif is_true_cst(cond):
                return True, bool_expr
            elif is_false_cst(bool_expr): 
                neg_expr = recurse_negation(cond)
                # recursing negation can introduce new boolean constants, so we need to simplify again
                neg_changed, new_neg_expr = _simplify_boolean_expr(neg_expr, num_context=num_context)
                if neg_changed:
                    return True, new_neg_expr
                return True, neg_expr
            else:
                pass # nothing changed

        elif expr.name == "not":
            if is_true_cst(expr_args[0]):
                return True, 0 if num_context else cp.BoolVal(False)
            elif is_false_cst(expr_args[0]):
                return True, 1 if num_context else cp.BoolVal(True)
            else:
                pass # nothing changed

        # numerical expressions
        elif expr.name == "wsum":
            i = 0
            while i < len(expr_args[1]):
                var = expr_args[1][i]
                if is_bool(var):
                    if changed is False: # we are operating on the the original args list, so need to copy first
                        expr_args = (expr_args[0], list(expr_args[1]))
                        changed = True
                    expr_args = cast(tuple[Any, list[Any]], expr_args) # changed is True, so this is safe to do
                    expr_args[1][i] = int(var)
                i += 1
                    
        else: 
            # other operators: "sum", "sub", "-" have flat list as arguments 
            # check if anything is Bool that should be int
            i = 0
            while i < len(expr_args):
                a = expr_args[i]
                if is_bool(a):
                    if changed is False: # we are operating on the the original args list, so need to copy first
                        expr_args = list(expr_args)
                        changed = True
                    expr_args = cast(list[Any], expr_args) # changed is True, so expr_args is a list
                    expr_args[i] = int(a)
                i += 1
            
        if changed:
            expr = copy.copy(expr)
            expr.update_args(expr_args)
            return True, expr
        else:
            return False, expr

    elif isinstance(expr, Comparison): 

        rec_changed, expr_args = _simplify_boolean_args(expr.args, num_context=True) # Comparisons are supposed to be numerical
        if rec_changed:
            changed = True

        # simplify comparisons with Boolean arg: <BoolExpr> <Comparison> <Constant>
        lhs, rhs = expr_args
        name = expr.name
        already_checked = False
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
            already_checked = True # avoid-checking with isinstance below
        """
        Simplify expressions according to this table:
        x  | <0	 0  1  >1
        ----------------------
        == |  F	~x  x   F
        != |  T	 x ~x   T
        >  |  T	 x  F   F
        <  |  F	 F ~x   T
        >= |  T	 T  x   F   
        <= |  F	~x  T   T
        """
        if already_checked or (is_boolexpr(lhs) and is_num(rhs)):
            # direct simplification of boolean comparisons
            if isinstance(rhs, BoolVal):
                rhs = int(rhs.value())

            result: Optional[bool|Expression] = None
            if rhs < 0:
                result = (name in  {"!=", ">", ">="}) # all other operators evaluate to False
            elif rhs == 0:
                if name == "!=" or name == ">":
                    result = lhs
                elif name == "==" or name == "<=":
                    result = recurse_negation(lhs)
                elif name == "<":
                    result = False
                elif name == ">=":
                    result = True
            elif rhs == 1:
                if name == "==" or name == ">=":
                    result = lhs
                elif name == "!=" or name == "<":
                    result = recurse_negation(lhs)
                elif name == ">":
                    result = False
                elif name == "<=":
                    result = True
            elif rhs > 1:
                result = name in {"!=", "<", "<="} # all other operators evaluate to False

            assert result is not None
            if result is True or result is False:
                return True, result if num_context else BoolVal(result)
            else: # lhs or its negation
                return True, result

        if changed:
            expr = copy.copy(expr)
            expr.update_args((lhs, rhs))
            return True, expr
        else:
            return False, expr
        
    elif isinstance(expr, (GlobalConstraint, GlobalFunction)):
        if expr.has_subexpr() is False: # did not handle args above yet
            # TODO: how to determine wich argument are int/bool?
            # For now, assume all arguments are in Numerical context, this can cause issues with XOR and ITE global constraints
            # Solver interfaces are expected to handle these cases themselves before posting to the solver.
            rec_changed, expr_args = _simplify_boolean_args(expr.args, num_context=True)
            if rec_changed:
                changed = True

        if changed:
            expr = copy.copy(expr)
            expr.update_args(expr_args)
            return True, expr
        else:
            return False, expr
    
    elif isinstance(expr, BoolVal):
        if num_context:
            return True, int(expr)
        else:
            return False, expr

    # BoolVarImpl, DirectConstraint
    return False, expr

def _simplify_boolean_args(args: list[Any]|tuple[Any, ...], num_context: bool) -> tuple[bool, list[Any]|tuple[Any, ...]]:
    """
    Well-typed helper function to eliminate boolean constants in a list of arguments.
    Removes Boolean constants from logical operators, and replaces Boolean constants with ints in a numerical context (sum, comparisons, etc.)
    Uses :func:`_simplify_boolean_expr()` to recurse into the arguments of all other expressions.
    
    Arguments:
        args: list of arguments to eliminate boolean constants from
    """
    changed = False
    newargs: list[Any] = []
    for arg in args:
        if isinstance(arg, Expression):
            rec_changed, rec_newexpr = _simplify_boolean_expr(arg, num_context=arg.is_bool())
            if rec_changed:
                changed = True
                newargs.append(rec_newexpr)
                continue

        elif isinstance(arg, np.ndarray):
            if isinstance(arg, NDVarArray):
                # optimization for NDVarArray: only if it contains subexpressions
                if arg.has_subexpr():
                    rec_changed, rec_newargs = _simplify_boolean_args(tuple(arg.flat), num_context=num_context)
                    if rec_changed:
                        changed = True
                        newargs.append(cpm_array(rec_newargs).reshape(arg.shape)) # reshape to original
                        continue
            elif arg.dtype == object:
                # user can create an np.array with Expressions, without converting to NDVarArray first
                rec_changed, rec_newargs = _simplify_boolean_args(tuple(arg.flat), num_context=num_context)
                if rec_changed:
                    changed = True
                    newargs.append(np.array(rec_newargs).reshape(arg.shape))
                    continue

            # ndarray with constants, keep as is
            newargs.append(arg)
            continue

        elif isinstance(arg, (list, tuple)):
            rec_changed, rec_newarg = _simplify_boolean_args(arg, num_context=num_context)
            if rec_changed:
                changed = True
                newargs.append(rec_newarg)
                continue
            
        # handle non-Expression Boolean constants
        if is_true_cst(arg):
            if num_context:
                newargs.append(1)
                changed = True
            else:
                newargs.append(BoolVal(True))
            continue
        if is_false_cst(arg):
            if num_context:
                newargs.append(0)
                changed = True
            else:
                newargs.append(BoolVal(False))
            continue

        # all the rest: not allowed to contain expressions
        newargs.append(arg)

    if changed:
        return True, newargs
    else:
        return False, args