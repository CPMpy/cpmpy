"""
    Transformations dealing with negations (used by other transformations).
"""
import copy
import warnings  # for deprecation warning
import numpy as np
from typing import Any
from cpmpy.expressions.globalconstraints import GlobalConstraint

from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, NDVarArray, cpm_array
from ..expressions.utils import is_boolexpr

def push_down_negation(lst_of_expr: list[Expression], toplevel=True) -> list[Expression]:
    """
        Recursively simplifies expressions by pushing down negation into the arguments.
        E.g., not(x >= 3 | y == 2) is simplified to (x < 3) & (y != 2).

        Input is expected to be a flat list of Expressions.
        Argument `toplevel` is deprecated and will be removed in a future version.

        Return:
            list of Expressions
    """
    newlist: list[Expression] = []
    for expr in lst_of_expr:
        changed, newexpr = _push_down_negation_expr(expr)
        if changed:
            if toplevel and newexpr.name == "and":
                # TODO: newexpr.args are ExprLike, check for constants
                for b in newexpr.args:
                    if isinstance(b, Expression):
                        newlist.append(b)
                    elif not b:
                        newlist.append(BoolVal(b))
                        break  # stop early if a False
            else:
                newlist.append(newexpr)
        else:
            newlist.append(expr)
    return newlist


def _push_down_negation_expr(expr: Expression) -> tuple[bool, Expression]:
    # special cases, _recurse_negation() will handle recursive calls into the args
    if expr.name == "not":
        # the negative case, negate
        return True, recurse_negation(expr.args[0])

    # rewrite 'BoolExpr != BoolExpr' to normalized 'BoolExpr == ~BoolExpr'
    elif expr.name == '!=':
        lexpr, rexpr = expr.args
        lhs_bool = is_boolexpr(lexpr)
        rhs_bool = is_boolexpr(rexpr)
        if lhs_bool and rhs_bool:
            if isinstance(lexpr, (_BoolVarImpl, BoolVal)):
                rhs_changed, rhs_newexpr = _push_down_negation_expr(rexpr)
                if rhs_changed:
                    rexpr = rhs_newexpr
                return True, (~lexpr) == rexpr

            elif isinstance(rexpr, (_BoolVarImpl, BoolVal)):
                lhs_changed, lhs_newexpr = _push_down_negation_expr(lexpr)
                if lhs_changed:
                    lexpr = lhs_newexpr
                return True, lexpr == (~rexpr)

            else:
                # change lhs, keep/recurse rhs
                lexpr = recurse_negation(lexpr)
                rhs_changed, rhs_newexpr = _push_down_negation_expr(rexpr)
                if rhs_changed:
                    rexpr = rhs_newexpr
                return True, lexpr == rexpr
    
    if expr.has_subexpr():
        rec_changed, rec_newargs = _push_down_negation_args(expr.args)
        if rec_changed:
            newexpr = copy.copy(expr)
            newexpr.update_args(rec_newargs)
            return True, newexpr
    
    return False, expr


def _push_down_negation_args(args: tuple[Any, ...]) -> tuple[bool, tuple[Any, ...]]:
    changed = False
    newargs: list[Any] = []
    for arg in args:
        if isinstance(arg, Expression):
            rec_changed, rec_newexpr = _push_down_negation_expr(arg)
            if rec_changed:
                changed = True
                newargs.append(rec_newexpr)
                continue

        elif isinstance(arg, np.ndarray):
            if isinstance(arg, NDVarArray):
                # NDVarArray: only if it contains subexpressions
                if arg.has_subexpr():
                    rec_changed, rec_newargs = _push_down_negation_args(tuple(arg.flat))
                    if rec_changed:
                        changed = True
                        newargs.append(cpm_array(rec_newargs).reshape(arg.shape))
                        continue
            elif arg.dtype == object:
                # not sure we need this... an np.array that might contain expressions (or we only allow NDVarArray in that case?)
                rec_changed, rec_newargs = _push_down_negation_args(tuple(arg.flat))
                if rec_changed:
                    changed = True
                    newargs.append(np.array(rec_newargs).reshape(arg.shape))
                    continue

            # if still here, only data
            newargs.append(arg)
            continue

        elif isinstance(arg, list):
            rec_changed, rec_newarg = _push_down_negation_args(tuple(arg))
            if rec_changed:
                changed = True
                newargs.append(list(rec_newarg))
                continue
        
        elif isinstance(arg, tuple):  # unlikely but allowed
            rec_changed, rec_newarg = _push_down_negation_args(arg)
            if rec_changed:
                changed = True
                newargs.append(rec_newarg)
                continue
        
        # all the rest: not allowed to contain expressions
        newargs.append(arg)

    if changed:
        return True, tuple(newargs)
    else:
        return False, args

def recurse_negation(expr: Expression|bool|np.bool_) -> Expression:
    """
        Negate `expr` by pushing the negation down into it and its arguments.

        The following cases are handled:
        - Boolean variables and constants: negate the variable or constant
        - Comparisons: swap comparison sign
        - Boolean operators (and/or/implies): apply DeMorgan
        - Global constraints: calls :func:`~cpmpy.expressions.globalconstraints.GlobalConstraint.negate()` to negate the global constraint.
          Depending on the implementation, this may leave a "NOT" operator before the global constraint. 
          Use :func:`~cpmpy.transformations.decompose_global.decompose_in_tree()` to decompose the negated global constraint into simpler constraints if needed.

        Ensures no "NOT" operator is left in the expression tree of `expr`, apart from negated global constraints.

        Returns the negated expression.
    """
    if isinstance(expr, (_BoolVarImpl,BoolVal)):
        return ~expr
    
    # recurse_negation is called before simplify_boolean, 
    # so handle Boolean constants here too
    elif isinstance(expr, bool|np.bool_):
        return BoolVal(not bool(expr))

    elif isinstance(expr, Comparison):
        new_comp: Comparison = copy.copy(expr)
        if   expr.name == '==': 
            new_comp.name = '!='
        elif expr.name == '!=': 
            new_comp.name = '=='
        elif expr.name == '<=': 
            new_comp.name = '>'
        elif expr.name == '<':  
            new_comp.name = '>='
        elif expr.name == '>=': 
            new_comp.name = '<'
        elif expr.name == '>':  
            new_comp.name = '<='
        else: 
            raise ValueError(f"Unknown comparison to negate {expr}")

        # args are positive now, still check if no 'not' in its arguments
        rec_changed, rec_newargs = _push_down_negation_args(expr.args)
        if rec_changed:
            new_comp.update_args(rec_newargs)
        return new_comp
        
    elif isinstance(expr, Operator):
        assert(expr.is_bool()), f"Can only negate boolean expressions but got {expr}"

        if expr.name == "not":
            # negation while in negative context = switch back to positive case
            _, rec_args = _push_down_negation_args(expr.args)
            assert len(rec_args) == 1, f"not has only 1 argument but got {rec_args}"
            return rec_args[0]

        elif expr.name == "->":
            # ~(x -> y) :: x & ~y
            # arg0 remains positive, but check its arguments
            # (must wrap awkwardly in a list, but can make no assumption about expr.args[0] has .args)
            lhs = expr.args[0]
            lhs_changed, lhs_new = _push_down_negation_expr(lhs)
            if lhs_changed:
                lhs = lhs_new
            return lhs & recurse_negation(expr.args[1])

        elif expr.name == "and":
            # ~(x & y) :: ~x | ~y -- negate all arguments
            # copy experession to avoid init checks and keep _has_subexpr
            new_op = copy.copy(expr)
            new_op.name = "or"
            new_op.update_args([recurse_negation(a) for a in expr.args])
            return new_op
        
        elif expr.name == "or":
            # ~(x | y) :: ~x & ~y -- negate all arguments
            # copy experession to avoid init checks and keep _has_subexpr
            new_op = copy.copy(expr)
            new_op.name = "and"
            new_op.update_args([recurse_negation(a) for a in expr.args])
            return new_op

        else:
            raise ValueError(f"Unsupported operator to negate {expr}")
        
    # global constraints
    elif isinstance(expr, GlobalConstraint):
        new_glob = copy.copy(expr)
        rec_changed, rec_args = _push_down_negation_args(expr.args)
        if rec_changed:
            new_glob.update_args(rec_args)
        return new_glob.negate()
           
    else:
        raise ValueError(f"Unsupported expression to negate: {expr}")


def negated_normal(expr):
    """
    .. deprecated:: 0.9.16
          Please use :func:`recurse_negation()` instead.
    """
    warnings.warn("Deprecated, use `recurse_negation()` instead which will negate and push down all negations in "
                  "the expression (or use `push_down_negation` on the full expression tree); will be removed in "
                  "stable version", DeprecationWarning)
    return recurse_negation(expr)
