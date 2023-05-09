"""
Returns an list of all variables in the model or expressions

Variables are ordered by appearance, e.g. first encountered first
"""
import warnings # for deprecation warning
import numpy as np

from ..expressions.core import Expression
from ..expressions.variables import _NumVarImpl, NegBoolView, NDVarArray, _DirectVarImpl
from ..expressions.utils import is_any_list

def get_variables_model(model):
    """
        Get variables of a model (constraints and objective)

        This is a separate function because we can not import
        `Model` without a circular dependency...
    """
    # want an ordered set. Emulate with full list that is uniquified
    vars_ = get_variables(model.constraints)

    # then append to it from objective
    seen = frozenset(vars_)
    return vars_ + [x for x in get_variables(model.objective_) if not x in seen]


def vars_expr(expr):
    warnings.warn("Deprecated, use get_variables() instead, will be removed in stable version", DeprecationWarning)
    return get_variables(expr)
def get_variables(expr, collect=None):
    """
        Get variables of an expression

        - expr: Expression or list of expressions
        - collect: optional set, variables will be added to this set of given
   """
    def extract(lst, append):
        for e in lst:
            if isinstance(e, Expression):
                if isinstance(e, _NumVarImpl):
                    if isinstance(e, NegBoolView):
                        # this is just a view, return the actual variable
                        e = e._bv
                    append(e)
                elif isinstance(e, NDVarArray):  # sometimes does not have a .name
                    if e.dtype == object:
                        extract(e.flat, append)
                    # else: all const, skip
                elif e.name == "wsum":
                    extract(e.args[1], append)  # skip data in arg0
                elif e.name == "table":
                    extract(e.args[0], append)  # skip data in arg1
                elif isinstance(e, _DirectVarImpl) and e.novar is not None:
                    # custom variables, skip novar arguments
                    extract([a for i,a in enumerate(e.args) if i not in e.novar], append)
                else:
                    extract(e.args, append)
            elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
                extract(e, append)

    if collect is not None:
        # add to given set
        append = collect.add
        extract((expr,), append)
        return collect

    # no 'collect' given, return ordered list
    vars_ = []
    append = vars_.append
    extract((expr,), append)

    # mimics an ordered set, manually...
    # (looks expensive but surprisingly little overhead)
    # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    seen = set()
    seen_add = seen.add
    return [x for x in vars_ if not (x in seen or seen_add(x))]


def print_variables(expr_or_model):
    """
        Print variables _and their domains_

        argument 'expr_or_model' can be an expression or a model
    """
    from ..model import Model
    if isinstance(expr_or_model, Model):
        vars_ = get_variables_model(expr_or_model)
    else:
        vars_ = get_variables(expr_or_model)

    # TODO: variables with the same prefix name will have the same domain
    # group them for clarity?
    # Currently: in order of appearance in the constraints, helps debugging too...
    print("Variables:")
    for var in vars_:
        print(f"    {var}: {var.lb}..{var.ub}")


# https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
def _uniquify(seq):
    warnings.warn("Deprecated, copy inline if used, will be removed in stable version", DeprecationWarning)
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

