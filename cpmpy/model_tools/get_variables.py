from ..model import *
from ..expressions import *
from ..variables import *

"""
 Model transformation, read-only
 Returns an (ordered by appearance) list of all variables in the model
"""
def get_variables(model):
    # want an ordered set. Emulate with full list that is uniquified
    vars_cons = vars_expr(model.constraints)
    vars_obj = vars_expr(model.objective)

    # mimics an ordered set, manually...
    return uniquify(vars_cons+vars_obj)

# https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
def uniquify(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# TODO: rename this function more publicly, more like in flatten or so
def vars_expr(expr):
    if isinstance(expr, NegBoolView):
        # this is just a view, return the actual variable
        return [expr._bv]
        
    if isinstance(expr, NumVarImpl):
        # a real var, do our thing
        return [expr]

    vars_ = []
    # if list or Expr: recurse
    if is_any_list(expr):
        for subexpr in expr:
            vars_ += vars_expr(subexpr)
    elif isinstance(expr, Expression):
        for subexpr in expr.args:
            vars_ += vars_expr(subexpr)
    # else: every non-list, non-expression
    return vars_
