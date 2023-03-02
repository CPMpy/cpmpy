
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.utils import is_any_list, eval_comparison
from ..expressions.python_builtins import all
from .flatten_model import flatten_constraint

def decompose_global(lst_of_expr, supported={}):

    """
        Decomposes any global constraint not supported by the solver
        Accepts a list of flat constraints as input
        Returns a list of flat constraints
    """

    if not is_any_list(lst_of_expr):
        lst_of_expr= [lst_of_expr]

    newlist = []
    for cpm_expr in lst_of_expr:
        assert isinstance(cpm_expr, Expression), f"Expected CPMpy expression but got {cpm_expr}, run 'cpmpy.transformations.normalize.toplevel_list' first"

        # 1: Base case boolean global constraint
        if hasattr(cpm_expr, "decompose") and cpm_expr.name not in supported:
            cpm_expr = cpm_expr.decompose()
        # 2: global constraint on lhs of a comparison
        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            if isinstance(lhs, GlobalConstraint) and lhs.name not in supported:
                cpm_expr = _decompose_global_comp(cpm_expr)
        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            cond, subexpr = cpm_expr.args
            if hasattr(subexpr, "decompose") and subexpr.name not in supported: # probably most frequent case
                # 3: Boolean global constraint on rhs of implies
                cpm_expr = [cond.implies(all(subexpr.decompose()))]
            elif hasattr(cond, "decompose") and cond.name not in supported:
                # 4: Boolean global constraint on lhs of implies
                cpm_expr = [all(cond.decompose()).implies(subexpr)]
            elif isinstance(cond, Comparison) and \
                    isinstance(cond.args[0], GlobalConstraint) and cond.args[0].name not in supported:
                # 5: global constraint on lhs of comparison on lhs of implies
                cpm_expr = [all(_decompose_global_comp(cond)).implies(subexpr)]
            elif isinstance(subexpr, Comparison) and \
                    isinstance(subexpr.args[0], GlobalConstraint) and subexpr.args[0].name not in supported:
                # 6: Numerical global constraint on lhs of comparison on rhs of implies
                cpm_expr = [cond.implies(all(_decompose_global_comp(subexpr)))]

        if isinstance(cpm_expr, list): # some decomposition happened, have to run again as decomp can contain new global
            newlist.extend(decompose_global(flatten_constraint(cpm_expr), supported=supported))
        else:
            # default
            newlist.append(cpm_expr)

    return newlist


def _decompose_global_comp(cpm_expr):
    """
        Helper function for decomposing global constraints in comparisons
    """
    assert isinstance(cpm_expr, Comparison), f"expected CPMpy Comparison, got {cpm_expr}"

    lhs, rhs = cpm_expr.args
    if hasattr(lhs, "decompose_comparison"):
        # numerical global
        return lhs.decompose_comparison(cpm_expr.name, rhs)
    if hasattr(lhs, "decompose"):
        # boolean global
        return [Comparison(cpm_expr.name, all(lhs.decompose()), rhs)]

    raise ValueError(f"Expected global constraint on lhs of comparison {cpm_expr}, got {lhs}")