"""
Returns an list of all variables in the model or expressions

Variables are ordered by appearance, e.g. first encountered first
"""
import warnings # for deprecation warning
from ..expressions.core import Expression
from ..expressions.variables import _NumVarImpl,NegBoolView
from ..expressions.utils import is_any_list

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

def print_variables(expr_or_model):
    """
        Print variables _and their domains_

        argument 'expr_or_model' can be an expression or a model
    """
    vars_ = None
    if isinstance(expr_or_model, Expression) or is_any_list(expr_or_model):
        vars_ = get_variables(expr_or_model)
    else:
        vars_ = get_variables_model(expr_or_model)

    # TODO: variables with the same prefix name will have the same domain
    # group them for clarity?
    # Currently: in order of appearance in the constraints, helps debugging too...
    print("Variables:")
    for var in vars_:
        print(f"    {var}: {var.lb}..{var.ub}")

# https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
def _uniquify(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

