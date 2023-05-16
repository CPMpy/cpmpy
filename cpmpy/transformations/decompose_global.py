import copy

from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl
from ..expressions.utils import is_any_list, eval_comparison
from ..expressions.python_builtins import all
from .flatten_model import flatten_constraint


def decompose_tree(lst_of_expr, supported=set(), supported_reif=set(), nested=False):
    """
        Decomposes any global constraint not supported by the solver
        Accepts a list of CPMpy expressions as input
        Returns a list of CPMpy expressions

        - supported: a set of supported global constraints or global functions
        - supported_reified: as set of supported global constraints/functions within a reification

        Unsupported constraints are decomposed in equivalent simpler constraints.
        Special care taken for partial global constraints in reified contexts and constraints in negative contexts

        Supported numerical global functions are left in reified contexts as they can be rewritten using
            `cpmpy.transformations.reification.reify_rewrite`
            The following `bv -> NumExpr <comp> Var/Const` can always be rewritten as  [bv -> IV0 <comp> Var/Const, NumExpr == IV0].
            So even if numerical constraints are not supported in reified context, we can rewrite them to non-reified versions.
    """

    newlist = []
    for expr in lst_of_expr:

        if isinstance(expr, (_BoolVarImpl, BoolVal, DirectConstraint)):
            newlist.append(expr)
            continue

        if isinstance(expr, Operator):
            # tricky here, what if some argument is a numerical global constraint? Need to create a new auxiliary variable
            # for example: min(x,y,z) + iv <= 10 and min has to be decomposed
            #   then we should call min(x,y,z).decompose_comparison('==', AUX) and put AUX into the SUM as argument
            pass #TODO



        if isinstance(expr, Comparison):
            lhs, rhs = expr.args

            if hasattr(lhs, "decompose_comparison"):
                # numerical global constraint
                should_decompose = lhs.name not in supported_reif if nested else lhs.name not in supported
                if should_decompose:
                    expr = lhs.decompose_comparison(expr.name, rhs)

            elif hasattr(rhs, "decompose_comparison"):
                # numerical global constraint
                should_decompose = rhs.name not in supported_reif if nested else lhs.name not in supported
                if should_decompose:
                    flipmap = {"==":"==", "!=":"!=","<":">", "<=":">="}
                    expr = rhs.decompose_comparison(flipmap[expr.name], lhs)

            if isinstance(expr, list):# decomposition has triggered above
                newlist.extend(decompose_tree(expr, supported, supported_reif, nested)) # no new level of nesting
            else: # recurse into arguments of comparison
                decomp_lhs = decompose_tree([lhs], supported, supported_reif, nested=True)
                decomp_rhs = decompose_tree([rhs], supported, supported_reif, nested=True)
                if len(decomp_lhs) >= 1:
                    assert hasattr(lhs, "decompose") and lhs.is_bool(), "unexpected argument, lhs should be boolean global constraint if decomposition was triggered"
                    decomp_lhs = [all(decomp_lhs)]
                if len(decomp_rhs) >= 1:
                    assert hasattr(rhs, "decompose") and rhs.is_bool(), "unexpected argument, rhs should be boolean global constraint if decomposition was triggered"
                    decomp_rhs = [all(decomp_rhs)]

                newlist.append(Comparison(expr.name, decomp_lhs[0], decomp_rhs[0]))
            continue


        if hasattr(expr, "decompose"):
            # boolean global constraint
            should_decompose = expr.name not in supported_reif if nested else expr.name not in supported
            if should_decompose:
                expr = do_decompose(expr)

            if isinstance(expr, list): #decomposition has triggered above
                newlist.extend(decompose_tree(expr, supported, supported_reif, nested))
            else:# recurse into arguments
                # tricky case, similar to Operator
                # E.g. AllDifferent(min(x,y,z), a,b) where min should be decomposed!
                pass # TODO




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




def do_decompose(cpm_expr, negative_context=False):
    """
        Helper function for decomposing global constraints
        - cpm_expr: Global constraint or comparison containing a global constraint on lhs
        - negative_context: Boolean indicating if the global constraints should be decomposed using `decompose_negation`
    """
    if isinstance(cpm_expr, Comparison):
        lhs, rhs = cpm_expr.args
        if lhs.is_bool():
            return [eval_comparison(cpm_expr.name, all(lhs.decompose()), rhs)]
        else:
            return lhs.decompose_comparison(cpm_expr.name, rhs)

    if negative_context:
        return cpm_expr.decompose_negation()
    return cpm_expr.decompose()