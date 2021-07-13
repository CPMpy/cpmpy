import warnings # for deprecation warning
from ..expressions.core import Expression
from ..expressions.variables import _NumVarImpl,NegBoolView
from ..expressions.utils import is_any_list

"""
Returns an (ordered by appearance) list of all variables in the model or expressions

Does not modify any expression
"""
def get_variables_model(model):
    """
        Get variables of a model (constraints and objective)

        This is a separate function because we can not import
        `Model` without a circular dependency...
    """
    # want an ordered set. Emulate with full list that is uniquified
    vars_cons = get_variables(model.constraints)
    vars_obj = get_variables(model.objective)

    # mimics an ordered set, manually...
    return _uniquify(vars_cons+vars_obj)

def vars_expr(expr):
    warnings.warn("Deprecated, use get_variables() instead, will be removed in stable version", DeprecationWarning)
    return get_variables(expr)
def get_variables(expr):
    """
        Get variables of an expression
    """
    if isinstance(expr, NegBoolView):
        # this is just a view, return the actual variable
        return [expr._bv]
        
    if isinstance(expr, _NumVarImpl):
        # a real var, do our thing
        return [expr]

    vars_ = []
    # if list or Expr: recurse
    if is_any_list(expr):
        for subexpr in expr:
            vars_ += get_variables(subexpr)
    elif isinstance(expr, Expression):
        for subexpr in expr.args:
            vars_ += get_variables(subexpr)
    # else: every non-list, non-expression
    return vars_

# https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
def _uniquify(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

