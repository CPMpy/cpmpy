from ..model import *
from ..expressions import *
from ..variables import *

"""
 Model transformation, read-only
 Returns an (ordered by appearance) list of all variables in the model
"""
def get_variables(model):
    # want an ordered set. Emulate with full list that is uniquified
    vars_ = vars_expr(model.constraints)

    # mimics an ordered set, manually...
    return uniquify(vars_)

# https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
def uniquify(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def vars_expr(expr):
    # a var, do our thing
    if isinstance(expr, NumVarImpl):
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
