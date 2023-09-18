import copy
import warnings  # for deprecation warning

from .normalize import toplevel_list
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, intvar, boolvar, _NumVarImpl, cpm_array, NDVarArray
from ..expressions.utils import is_any_list, eval_comparison
from ..expressions.python_builtins import all
from .flatten_model import flatten_constraint, normalized_numexpr


def decompose_in_tree(lst_of_expr, supported=set(), supported_reified=set(), _toplevel=None, nested=False):
    """
        Decomposes any global constraint not supported by the solver
        Accepts a list of CPMpy expressions as input and returns a list of CPMpy expressions,
            if nested is True, new constraints will have been added to the `_toplevel` list too

        - supported: a set of supported global constraints or global functions
        - supported_reified: a set of supported reified global constraints (globals with Boolean return type only)
        - toplevel: a list of constraints that should be added toplevel, carried as pass by reference to recursive calls

        Special care taken for unsupported global constraints in reified contexts and for numeric global constraints
            in a comparison.

        Supported numerical global functions remain in the expression tree as is. They can be rewritten using
            `cpmpy.transformations.reification.reify_rewrite`
            The following `bv -> NumExpr <comp> Var/Const` can be rewritten as  [bv -> IV0 <comp> Var/Const, NumExpr == IV0].
            So even if numerical constraints are not supported in reified context, we can rewrite them to non-reified versions.
    """
    if _toplevel is None:
        _toplevel = []

    flipmap = {"==": "==", "!=": "!=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}

    newlist = []  # decomposed constraints will go here
    for expr in lst_of_expr:

        if is_any_list(expr):
            assert nested is True, "Cannot have nested lists without passing trough an expression, make sure to run cpmpy.transformations.normalize.toplevel_list first."
            newexpr = decompose_in_tree(expr, supported, supported_reified, _toplevel, nested=True)
            if isinstance(expr, NDVarArray):
                newlist.append(cpm_array(newexpr))
            else:
                newlist.append(newexpr)
            continue

        elif isinstance(expr, Operator):
            if any(isinstance(a,GlobalFunction) for a in expr.args):
                expr, base_con = normalized_numexpr(expr)
                _toplevel.extend(base_con)  # should be added toplevel
            # recurse into arguments, recreate through constructor (we know it stores no other state)
            args = decompose_in_tree(expr.args, supported, supported_reified, _toplevel, nested=True)
            newlist.append(Operator(expr.name, args))

        elif isinstance(expr, GlobalConstraint) or isinstance(expr, GlobalFunction):
            # first create a fresh version and recurse into arguments
            expr = copy.copy(expr)
            expr.args = decompose_in_tree(expr.args, supported, supported_reified, _toplevel, nested=True)

            is_supported = (expr.name in supported)
            if nested and expr.is_bool():
                # special case: reified (Boolean) global
                is_supported = (expr.name in supported_reified)

            if is_supported:
                newlist.append(expr)
            else:
                if expr.is_bool():
                    assert isinstance(expr, GlobalConstraint)
                    # boolean global constraints
                    dec = expr.decompose()
                    if not isinstance(dec, tuple):
                        warnings.warn("Decomposing an old-style global that does not return a tuple, which is deprecated. Support for old-style globals will be removed in stable version", DeprecationWarning)
                        dec = (dec, [])
                    decomposed, define = dec

                    _toplevel.extend(define)  # definitions should be added toplevel
                    # the `decomposed` expression might contain other global constraints, check it
                    decomposed = decompose_in_tree(decomposed, supported, supported_reified, [], nested=nested)
                    newlist.append(all(decomposed))

                else:
                    # global function, replace by a fresh variable and decompose the equality to this
                    assert isinstance(expr, GlobalFunction)
                    lb,ub = expr.get_bounds()
                    aux = intvar(lb, ub)

                    dec = expr.decompose_comparison("==", aux)
                    if not isinstance(dec, tuple):
                        warnings.warn("Decomposing an old-style global that does not return a tuple, which is deprecated. Support for old-style globals will be removed in stable version", DeprecationWarning)
                        dec = (dec, [])
                    auxdef, otherdef = dec

                    _toplevel.extend(auxdef + otherdef)  # all definitions should be added toplevel
                    newlist.append(aux)  # replace original expression by aux

        elif isinstance(expr, Comparison):
            # if one of the two children is a (numeric) global constraint, we can decompose the comparison directly
            # otherwise e.g., min(x,y,z) == a would become `min(x,y,z).decompose_comparison('==',aux) + [aux == a]`
            lhs, rhs = expr.args
            exprname = expr.name  # can change when rhs needs decomp

            decomp_lhs = isinstance(lhs, GlobalFunction) and not lhs.name in supported
            decomp_rhs = isinstance(rhs, GlobalFunction) and not rhs.name in supported

            if not decomp_lhs:
                if not decomp_rhs:
                    # nothing special, create a fresh version and recurse into arguments
                    expr = copy.copy(expr)
                    expr.args = decompose_in_tree(expr.args, supported, supported_reified, _toplevel, nested=True)
                    newlist.append(expr)

                else:
                    # only rhs needs decomposition, so flip comparison to make lhs needing the decomposition
                    exprname = flipmap[expr.name]
                    lhs, rhs = rhs, lhs
                    decomp_lhs, decomp_rhs = True, False  # continue into next 'if'

            if decomp_lhs:
                # recurse into lhs args
                lhs = copy.copy(lhs)
                lhs.args = decompose_in_tree(lhs.args, supported, supported_reified, _toplevel, nested=True)

                # decompose comparison of lhs and rhs
                dec = lhs.decompose_comparison(exprname, rhs)
                if not isinstance(dec, tuple):
                    warnings.warn("Decomposing an old-style global that does not return a tuple, which is deprecated. Support for old-style globals will be removed in stable version", DeprecationWarning)
                    dec = (dec, [])
                decomposed, define = dec

                _toplevel.extend(define)  # definitions should be added toplevel
                # the `decomposed` expression (and rhs) might contain other global constraints, check it
                decomposed = decompose_in_tree(decomposed, supported, supported_reified, _toplevel, nested=True)
                newlist.append(all(decomposed))

        else:  # constants, variables, direct constraints
            newlist.append(expr)

    if nested:
        return newlist

    if len(_toplevel) == 0:
        return toplevel_list(newlist)
    else:
        # we are toplevel and some new constraints are introduced, decompose new constraints!
        return toplevel_list(newlist) + decompose_in_tree(_toplevel, supported, supported_reified, nested=False)


# DEPRECATED!
# old way of doing decompositions post-flatten
# will be removed in any future version!
def decompose_global(lst_of_expr, supported=set(), supported_reif=set()):
    warnings.warn("Deprecated, use `decompose_in_tree()` instead, will be removed in stable version", DeprecationWarning)
    """
        DEPRECATED!!! USE `decompose_in_tree()` instead!
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
        if isinstance(cpm_expr, GlobalConstraint):
            if reified and cpm_expr.name not in supported_reif:
                return False
            if not reified and cpm_expr.name not in supported:
                return False
        if isinstance(cpm_expr, Comparison) and isinstance(cpm_expr.args[0], GlobalFunction):
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
    warnings.warn("Deprecated, never meant to be used outside this transformation; will be removed in stable version", DeprecationWarning)
    """
        DEPRECATED
        Helper function for decomposing global constraints
        - cpm_expr: Global constraint or comparison containing a global constraint on lhs
    """
    if isinstance(cpm_expr, Comparison):
        lhs, rhs = cpm_expr.args
        if lhs.is_bool():
            return [eval_comparison(cpm_expr.name, all(lhs.decompose()), rhs)]
        else:
            dec = lhs.decompose_comparison(cpm_expr.name, rhs)
            # new style is a tuple of size 2, old style is a single list (which captures less cases)
            if isinstance(dec, tuple):
                dec = dec[0]+dec[1]
            return dec

    dec = cpm_expr.decompose()
    # new style is a tuple of size 2, old style is a single list (which captures less cases)
    if isinstance(dec, tuple):
        dec = dec[0]+dec[1]
    return dec
