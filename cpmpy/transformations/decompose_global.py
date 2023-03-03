import copy

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
    def _is_supported(cpm_expr):
        if isinstance(cpm_expr, GlobalConstraint) and cpm_expr.name not in supported:
            return False
        if isinstance(cpm_expr, Comparison) and isinstance(cpm_expr.args[0], GlobalConstraint) and cpm_expr.args[0].name not in supported:
            return False
        return True

    if not is_any_list(lst_of_expr):
        lst_of_expr= [lst_of_expr]

    newlist = []
    for cpm_expr in lst_of_expr:
        assert isinstance(cpm_expr, Expression), f"Expected CPMpy expression but got {cpm_expr}, run 'cpmpy.transformations.normalize.toplevel_list' first"
        decomp_idx = None

        if hasattr(cpm_expr, "decompose") and cpm_expr.is_bool() and cpm_expr.name not in supported:
            cpm_expr = cpm_expr.decompose() # base boolean global constraints
        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            if cpm_expr.name == "==" and not _is_supported(lhs): # can be both boolean or numerical globals
                decomp_idx = 0

            if not _is_supported(cpm_expr):
                cpm_expr = do_decompose(cpm_expr) # base global constraint in comparison
                decomp_idx = None

        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            lhs, rhs = cpm_expr.args
            if not _is_supported(rhs):
                decomp_idx = 1 # reified (numerical) global constraint, probably most frequent case
            elif not _is_supported(lhs):
                decomp_idx = 0

        if decomp_idx is not None:
            cpm_expr = copy.copy(cpm_expr)
            cpm_expr.args[decomp_idx] = all(do_decompose(cpm_expr.args[decomp_idx]))
            cpm_expr = [cpm_expr]

        if isinstance(cpm_expr, list): # some decomposition happened, have to run again as decomp can contain new global
            flat = flatten_constraint(cpm_expr)
            newlist.extend(decompose_global(flat, supported=supported))
        else:
            # default
            newlist.append(cpm_expr)

    return newlist




def do_decompose(cpm_expr):
    """
        Helper function for decomposing global constraints
        - cpm_expr: Global constraint or comparison containing a global constraint on lhs
    """
    if isinstance(cpm_expr, Comparison):
        lhs, rhs = cpm_expr.args
        if lhs.is_bool():
            return [Comparison(cpm_expr.name, all(lhs.decompose()), rhs)]
        else:
            return lhs.decompose_comparison(cpm_expr.name, rhs)

    return cpm_expr.decompose()