import copy

from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.utils import is_any_list, eval_comparison
from ..expressions.python_builtins import all
from .flatten_model import flatten_constraint

def decompose_global(lst_of_expr, supported=set(), supported_reif=set()):

    """
        Decomposes any global constraint not supported by the solver
        Accepts a list of flat constraints as input
        Returns a list of flat constraints

        - supported: a set of supported global constraints or global functions
        - supported_reified: a set of supported global constraints within a reification

        Unsupported global constraints are decomposed into equivalent simpler constraints.
        Numerical global constraints are left reified even if not part of the `supported_reified` set
            as they can be rewritten using `cpmpy.transformations.reification.reify_rewrite`...
                ... reified partial functions are decomposed when unsupported by `supported_reified`
                The following `bv -> NumExpr <comp> Var/Const` can always be rewritten as ...
                                [bv -> IV0 <comp> Var/Const, NumExpr == IV0].
                So even if numerical constraints are not supported in reified context, we can rewrite them to non-reified versions.
                TODO: decide if we want to do the rewrite of unsupported globals here or leave it to `reify_rewrite`.
                    Currently, we left it for `reify_rewrite`



    """
    def _is_supported(cpm_expr, reified):
        if isinstance(cpm_expr, GlobalConstraint) and cpm_expr.is_bool():
            if reified and cpm_expr.name not in supported_reif:
                return False
            if not reified and cpm_expr.name not in supported:
                return False
        if isinstance(cpm_expr, Comparison) and isinstance(cpm_expr.args[0], GlobalConstraint):
            if not cpm_expr.args[0].name in supported:
                # reified numerical global constraints can be rewritten to non-reified versions
                #  so only have to check for 'supported' set
                return False
            if reified and not cpm_expr.args[0].is_total() and cpm_expr.args[0].name not in supported_reif:
                # edge case for partial functions as they cannot be rewritten to non-reified versions
                #  have to decompose to total functions
                #  ASSUMPTION: the decomposition for partial global functions is total! (for now only Element)
                return False

        return True

    assert supported_reif <= supported, "`supported_reif` set is assumed to be a subset of the `supported` set"
    if not is_any_list(lst_of_expr):
        lst_of_expr= [lst_of_expr]

    newlist = []
    for cpm_expr in lst_of_expr:
        assert isinstance(cpm_expr, Expression), f"Expected CPMpy expression but got {cpm_expr}, run 'cpmpy.transformations.normalize.toplevel_list' first"
        decomp_idx = None

        if hasattr(cpm_expr, "decompose") and not _is_supported(cpm_expr, reified=False):
            cpm_expr = cpm_expr.decompose() # base boolean global constraints
        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            if cpm_expr.name == "==" and not _is_supported(lhs, reified=True): # rhs will always be const/var
                decomp_idx = 0 # this means it is an unsupported reified boolean global

            if not _is_supported(cpm_expr, reified=False):
                cpm_expr = do_decompose(cpm_expr) # unsupported numerical constraint in comparison
                decomp_idx = None

        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            lhs, rhs = cpm_expr.args # BV -> Expr or Expr -> BV as flat normal form is required
            if not _is_supported(rhs, reified=True):
                decomp_idx = 1 # reified (numerical) global constraint, probably most frequent case
            elif not _is_supported(lhs, reified=True):
                decomp_idx = 0

        if decomp_idx is not None:
            cpm_expr = copy.deepcopy(cpm_expr) # need deepcopy as we are changing args of list inplace
            cpm_expr.args[decomp_idx] = all(do_decompose(cpm_expr.args[decomp_idx]))
            cpm_expr = [cpm_expr]

        if isinstance(cpm_expr, list): # some decomposition happened, have to run again as decomp can contain new global
            flat = flatten_constraint(cpm_expr) # most of the time will already be flat... do check here?
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
            return [eval_comparison(cpm_expr.name, all(lhs.decompose()), rhs)]
        else:
            return lhs.decompose_comparison(cpm_expr.name, rhs)

    return cpm_expr.decompose()