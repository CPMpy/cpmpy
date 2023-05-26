import copy

from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, intvar, boolvar, _NumVarImpl, cpm_array, NDVarArray
from ..expressions.utils import is_any_list, eval_comparison
from ..expressions.python_builtins import all
from .flatten_model import flatten_constraint


def decompose_in_tree(lst_of_expr, supported=set(), supported_nested=set(), nested=False, root_call=False):
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

    newlist = [] # decomposed constraints will go here
    newcons = [] # new constraints to be added toplevel

    flipmap = {"==":"==","!=":"!=","<":">","<=":">=",">":"<",">=":"<="}

    for expr in lst_of_expr:

        if is_any_list(expr):
            assert nested is True, "Cannot have nested lists without passing trough an expression, make sure to run cpmpy.transformations.normalize.toplevel_list first."
            newexpr, newcons[len(newcons):] = decompose_in_tree(expr, supported, supported_nested, nested=True)
            if isinstance(expr, NDVarArray):
                newlist.append(cpm_array(newexpr))
            else:
                newlist.append(newexpr)
            continue

        elif isinstance(expr, Operator):
            # just recurse into arguments
            expr = copy.copy(expr)
            expr.args, newcons[len(newcons):] = decompose_in_tree(expr.args, supported, supported_nested, nested=True)
            newlist.append(expr)

        elif isinstance(expr, GlobalConstraint):
            expr = copy.copy(expr)
            expr.args, newcons[len(newcons):] = decompose_in_tree(expr.args, supported, supported_nested, nested=True)

            if expr.is_bool():
                # boolean global constraints
                is_supported = (nested is False and expr.name in supported) or (nested is True and expr.name in supported_nested)
                if not is_supported:
                    if nested is False or expr.equivalent_decomposition():
                        expr, newcons[len(newcons):] = decompose_in_tree(expr.decompose(), supported, supported_nested, nested=True)
                        newlist.append(all(expr))
                    else:
                        bv = boolvar()
                        newlist.append(bv)
                        impl, newcons[len(newcons):] = decompose_in_tree([bv.implies(d) for d in expr.decompose()], supported, supported_nested, nested=True)
                        neg_impl, newcons[len(newcons):] = decompose_in_tree([(~bv).implies(d) for d in expr.decompose_negation()], supported, supported_nested, nested=True)
                        newcons.extend([all(impl), all(neg_impl)])
                else:
                    newlist.append(expr)
            else:
                # numerical global function (e.g. min, max, count, element...) in non-comparison context
                #   if the constraint is supported, it is also supported nested as it
                #   can always be rewritten as non-nested (i.e., top-level comparison)
                #    EXCEPT for non-total functions!! They should be decomposed if not explicitly supported in nested contexts
                is_supported = (expr.is_total() and expr.name in supported) or (not expr.is_total() and nested is True and expr.name in supported_nested)
                if not is_supported:
                    aux = intvar(*expr.get_bounds())
                    newlist.append(aux)
                    newcons.extend(expr.decompose_comparison("==",rhs=aux))
                else:
                    newlist.append(expr)

        elif isinstance(expr, Comparison):
            # Tricky case, if we just recurse into arguments, we risk creating unnesecary auxiliary variables
            #  e.g., min(x,y,z) <= a would become `min(x,y,z).decompose_comparison('==',aux) + [aux <= a]`
            #   while more optimally, its just `min(x,y,z).decompose_comparison('<=', a)`
            #    so have to be careful here and operate from 'one level above'

            lhs, rhs = expr.args
            if hasattr(lhs, "args"):
                # recurse into arguments of lhs
                lhs = copy.copy(lhs)
                lhs.args, newcons[len(newcons):] = decompose_in_tree(lhs.args, supported, supported_nested, nested=True)
            if hasattr(rhs, "args"):
                # recurse into arguments of rhs
                rhs = copy.copy(rhs)
                rhs.args, newcons[len(newcons):] = decompose_in_tree(rhs.args, supported, supported_nested, nested=True)

            if isinstance(lhs, GlobalConstraint):
                if lhs.is_bool() and lhs.name not in supported_nested: # boolean global constraint so comparison means nested!!
                    lhs = all(lhs.decompose())

                if not lhs.is_bool() and lhs.name not in supported: # TODO: handle partial?
                    newexpr = lhs.decompose_comparison(expr.name, rhs)
                    expr, newcons[len(newcons):] = decompose_in_tree(newexpr, supported, supported_nested, nested=True) # handle rhs
                    newlist.append(all(expr))
                    continue

            if isinstance(rhs, GlobalConstraint):
                # by now we know lhs is supported, so just flip comparison if unsupported
                rhs_supported = (rhs.is_bool() and rhs.name in supported_nested) or (not rhs.is_bool() and rhs.name in supported)
                if not rhs_supported:
                    newexpr, newcons[len(newcons):] = decompose_in_tree(eval_comparison(flipmap[expr.name], rhs, lhs), supported, supported_nested, nested)
                    newlist.append(all(newexpr))
                    continue

            # recreate original comparison
            newlist.append(eval_comparison(expr.name, lhs, rhs))
            continue

        else:  # constants, variables, direct constraints
            newlist.append(expr)

    if root_call and len(newcons):
        return newlist + decompose_in_tree(newcons, supported, supported_nested, nested=False)
    elif root_call:
        return newlist
    else:
        return newlist, newcons




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