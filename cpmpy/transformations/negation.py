"""
    Transformations dealing with negations (used by other transformations).
"""
import copy
import warnings  # for deprecation warning
import numpy as np

from cpmpy.expressions.globalconstraints import GlobalConstraint

from ..expressions.core import Expression, Comparison, Operator, BoolVal, ListLike
from ..expressions.variables import _BoolVarImpl, _NumVarImpl
from ..expressions.utils import is_any_list, is_bool, is_boolexpr

def push_down_negation(lst_of_expr: ListLike[Expression], toplevel=True):
    """
        Recursively simplifies expressions by pushing down negation into the arguments.
        E.g., not(x >= 3 | y == 2) is simplified to (x < 3) & (y != 2).

        Input is expected to be a flat list of Expressions.
        Argument `toplevel` is deprecated and will be removed in a future version.

        Return:
            list of Expressions
    """
    changed, newlist = _push_down_negation(lst_of_expr, toplevel=True)
    if not changed:
        return lst_of_expr
    return newlist

def _push_down_negation(lst_of_expr: ListLike[Expression], toplevel=False) -> tuple[bool, ListLike]:
    
    newlist: list = []
    changed = False
    
    if isinstance(lst_of_expr, np.ndarray) and not (lst_of_expr.dtype == object):
        # shortcut for data array, return as is
        return changed, lst_of_expr
    
    for expr in lst_of_expr:
        if is_any_list(expr):
            # can be a nested list with expressions?
            rec_changed, rec_newlist = _push_down_negation(expr)
            if rec_changed:
                changed = True
                newlist.extend(rec_newlist)
            else:
                newlist.append(expr)
        
        if isinstance(expr, Expression):

            # special cases, _recurse_negation() will handle recursive calls into the args
            if expr.name == "not":
                # the negative case, negate
                changed = True
                expr = recurse_negation(expr.args[0])
                if toplevel and expr.name == "and":
                    newlist.extend(expr.args)
                    continue

            # rewrite 'BoolExpr != BoolExpr' to normalized 'BoolExpr == ~BoolExpr'
            elif expr.name == '!=' and is_boolexpr(expr.args[0]) and is_boolexpr(expr.args[1]):

                lexpr, rexpr = expr.args

                if isinstance(lexpr, (_BoolVarImpl, BoolVal)):
                    rhs_changed, rhs_newlist = _push_down_negation((rexpr,))
                    if rhs_changed:
                        rexpr = rhs_newlist[0]
                    expr = (~lexpr) == rexpr
                    changed = True

                elif isinstance(rexpr, (_BoolVarImpl, BoolVal)):
                    lhs_changed, lhs_newlist = _push_down_negation((lexpr,))
                    if lhs_changed:
                        lexpr = lhs_newlist[0]
                    expr = lexpr == (~rexpr)
                    changed = True

                elif is_boolexpr(lexpr) and is_boolexpr(rexpr):
                    lhs_changed, lhs_newlist = _push_down_negation((lexpr,))
                    if lhs_changed:
                        rexpr = lhs_newlist[0]
                    # recurse_negation will handle rexpr
                    expr = lexpr == recurse_negation(rexpr)
                    changed = True
            
            elif expr.has_subexpr():
                rec_changed, rec_newlist = _push_down_negation(expr.args)
                if rec_changed:
                    changed = True
                    copy.copy(expr)
                    expr.update_args(rec_newlist)

        # default case: vars, constants, direct constraints
        newlist.append(expr)        

    return changed, newlist

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
        rec_changed, rec_newargs = _push_down_negation(expr.args)
        if rec_changed:
            new_comp.update_args(rec_newargs)
        return new_comp
        
    elif isinstance(expr, Operator):
        assert(expr.is_bool()), f"Can only negate boolean expressions but got {expr}"

        if expr.name == "not":
            # negation while in negative context = switch back to positive case
            rec_changed, rec_args = _push_down_negation(expr.args)
            assert len(rec_args) == 1, f"not has only 1 argument but got {rec_args}"
            return rec_args[0]

        elif expr.name == "->":
            # ~(x -> y) :: x & ~y
            # arg0 remains positive, but check its arguments
            # (must wrap awkwardly in a list, but can make no assumption about expr.args[0] has .args)
            x_changed, x_arg_lst = _push_down_negation([expr.args[0]])
            if x_changed:
                new_x = x_arg_lst[0]
            else:
                new_x = expr.args[0]
            return new_x & recurse_negation(expr.args[1])

        elif expr.name == "and":
            # ~(x & y) :: ~x | ~y -- negate all arguments
            # copy experession to avoid init checks and keep _has_subexpr
            new_op = copy.copy(expr)
            new_op.name = "or"
            new_op.update_args([recurse_negation(a) for a in expr.args])
            new_op._has_subexpr = expr._has_subexpr
            return new_op
        
        elif expr.name == "or":
            # ~(x | y) :: ~x & ~y -- negate all arguments
            # copy experession to avoid init checks and keep _has_subexpr
            new_op = copy.copy(expr)
            new_op.name = "and"
            new_op.update_args([recurse_negation(a) for a in expr.args])
            new_op._has_subexpr = expr._has_subexpr
            return new_op

        else:
            raise ValueError(f"Unsupported operator to negate {expr}")
        
    # global constraints
    elif isinstance(expr, GlobalConstraint):
        new_glob = copy.copy(expr)
        rec_changed, rec_args = _push_down_negation(expr.args)
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
