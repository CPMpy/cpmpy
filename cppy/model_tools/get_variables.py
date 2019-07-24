from ..model import *
from ..expressions import *
from ..variables import *

"""
 Model transformation, read-only
 Returns an (ordered by appearance) list of all variables in the model
"""
def get_variables(model):
    # want an ordered set. Emulate with full list that is uniquified
    vars_ = []
    if model.constraints:
        vars_ += vars_expr(model.constraints)
    if model.objective:
        vars_ += vars_expr(model.objective)
    # mimics an ordered set, manually...
    return uniquify(vars_)

# https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
def uniquify(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def vars_expr(expr):
    # check if iterable, if so, do
    try:
        vars_ = []
        for subexpr in iter(expr):
            vars_ += vars_expr(subexpr)
        return vars_
    except TypeError:
        # not iterable, base element
        if not isinstance(expr, Expression):
            # no expr, no vars for sure
            return []

        # a var
        if isinstance(expr, NumVarImpl):
            return [expr]

        # classes storing left/right
        if isinstance(expr, (MathOperator,Comparison)):
            return vars_expr(expr.left) + vars_expr(expr.right)

        # classes storing elems
        if isinstance(expr, (Sum,WeightedSum,BoolOperator,GlobalConstraint)):
            return vars_expr(expr.elems)

        # classes storing args (possibly nested)
        if isinstance(expr, Objective):
            return vars_expr(expr.expr)

        raise Exception("Expression {} unknown to variable extractor".format(expr))
