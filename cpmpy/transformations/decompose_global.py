"""
    Decompose global constraints not supported by the solver.
"""

import copy
import warnings  # for deprecation warning

from .normalize import toplevel_list
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.variables import intvar, cpm_array, NDVarArray
from ..expressions.utils import is_any_list, eval_comparison
from ..expressions.python_builtins import all as cpm_all
from .flatten_model import flatten_constraint, normalized_numexpr


def decompose_in_tree(lst_of_expr, supported=set(), supported_reified=set(), _toplevel=None, nested=False, csemap=None):
    """
        Decomposes any global constraint not supported by the solver.
        Accepts a list of CPMpy expressions as input and returns a list of CPMpy expressions.
            
        :param supported: a set of supported global constraints or global functions
        :param supported_reified: a set of supported reified global constraints (globals with Boolean return type only)
        :param _toplevel: a list of constraints that should be added toplevel, carried as pass by reference to recursive calls
        :param nested: if True, new constraints will have been added to the `_toplevel` list too
        
        Special care taken for unsupported global constraints in reified contexts and for numeric global constraints
        in a comparison.

        Supported numerical global functions remain in the expression tree as is. They can be rewritten using
        :func:`cpmpy.transformations.reification.reify_rewrite`
        The following ``bv -> NumExpr <comp> Var/Const`` can be rewritten as  ``[bv -> IV0 <comp> Var/Const, NumExpr == IV0]``.
        So even if numerical constraints are not supported in reified context, we can rewrite them to non-reified versions if they are total.
    """
    if _toplevel is None:
        _toplevel = []

    newlist = []  # decomposed constraints will go here
    for expr in lst_of_expr:

        if is_any_list(expr):
            assert nested is True, "Cannot have nested lists without passing trough an expression, make sure to run " \
                                   "cpmpy.transformations.normalize.toplevel_list first. "
            if isinstance(expr, NDVarArray):  # NDVarArray is also an expression,
                                              # so we can call has_subexpr on it for a possible early-exit
                if expr.has_subexpr():
                    newexpr = decompose_in_tree(expr, supported, supported_reified, _toplevel, nested=True, csemap=csemap)
                    newlist.append(cpm_array(newexpr))
                else:
                    newlist.append(expr)
            else: # a normal list-like (list, tuple, np.ndarray), must be called recursively and check all elements
                newexpr = decompose_in_tree(expr, supported, supported_reified, _toplevel, nested=True, csemap=csemap)
                newlist.append(newexpr)
            continue

        elif isinstance(expr, Operator):

            if not expr.has_subexpr(): # Only recurse if there are nested expressions
                newlist.append(expr)
                continue

            # if any(isinstance(a,GlobalFunction) for a in expr.args):
            #     expr, base_con = normalized_numexpr(expr, csemap=csemap)
            #     _toplevel.extend(base_con)  # should be added toplevel
            # recurse into arguments, recreate through constructor (we know it stores no other state)
            args = decompose_in_tree(expr.args, supported, supported_reified, _toplevel, nested=True, csemap=csemap)
            newlist.append(Operator(expr.name, args))

        elif isinstance(expr, GlobalConstraint) or isinstance(expr, GlobalFunction):
            # Can't early-exit here, need to check if constraint in itself is even supported
            if nested and expr.is_bool():
                # special case: reified (Boolean) global
                is_supported = (expr.name in supported_reified)
            else:
                is_supported = (expr.name in supported)

            if is_supported:
                # If no nested expressions, don't recurse the arguments
                if not expr.has_subexpr():
                    newlist.append(expr)
                    continue
                # Recursively decompose the subexpression arguments
                else:
                    expr = copy.copy(expr)
                    expr.update_args(decompose_in_tree(expr.args, supported, supported_reified, _toplevel, nested=True, csemap=csemap))
                    newlist.append(expr)

            else: # unsupported, need to decompose
                # check if it is in the csemap
                if csemap is not None and expr in csemap:
                    newlist.append(csemap[expr])
                    continue # no need to decompose, just use the variable we already have

                dec = expr.decompose()
                if not isinstance(dec, tuple):
                    warnings.warn(f"Decomposing an old-style global ({expr}) that does not return a tuple, which is "
                                  "deprecated. Support for old-style globals will be removed in stable version",
                                  DeprecationWarning)
                    dec = (dec, [])
                value, define = dec
                if not is_any_list(value):
                    value = [value]

                _toplevel.extend(define)  # definitions should be added toplevel
                # the `decomposed` expression might contain other global constraints, check it
                decomposed = decompose_in_tree(value, supported, supported_reified, _toplevel, nested=nested, csemap=csemap)
                if expr.is_bool():
                    value = cpm_all(decomposed)

                else:
                    assert len(decomposed) == 1, "Global functions should return a single numerical value, not a list"
                    value = decomposed[0]

                newlist.append(value)
                if csemap is not None:
                    csemap[expr] = value


        elif isinstance(expr, Comparison):
            if not expr.has_subexpr(): # Only recurse if there are nested expressions
                newlist.append(expr)
                continue

            expr = copy.copy(expr)
            expr.update_args(decompose_in_tree(expr.args, supported, supported_reified, _toplevel, nested=True, csemap=csemap))
            newlist.append(expr)

        else:  # constants, variables, direct constraints
            newlist.append(expr)

    if nested:
        return newlist

    if len(_toplevel) == 0:
        return toplevel_list(newlist)
    else:
        # we are toplevel and some new constraints are introduced, decompose new constraints!
        return toplevel_list(newlist) + decompose_in_tree(toplevel_list(_toplevel), supported, supported_reified, nested=False, csemap=csemap)


def decompose_objective(expr, supported=set(), supported_reified=set(), csemap=None):
    if is_any_list(expr):
        raise ValueError(f"Expected a numerical expression as objective but got a list {expr}")

    toplevel = []
    decomp_expr = decompose_in_tree([expr], supported=supported, supported_reified=supported_reified,
                                    _toplevel=toplevel, nested=True, csemap=csemap)
    assert len(decomp_expr) == 1, f"Expected {expr} to be decomposed into a single expression, but got {decomp_expr}.\nPlease report on github."
    return decomp_expr[0], toplevel


# DEPRECATED!
# old way of doing decompositions post-flatten
# will be removed in any future version!
def decompose_global(lst_of_expr, supported=set(), supported_reif=set()):
    """
    .. deprecated:: 0.9.16
          Please use :func:`decompose_in_tree()` instead.
    """
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
    """
    .. deprecated:: 0.9.13
          Please use :func:`decompose_in_tree()` instead.
    """
    warnings.warn("Deprecated, never meant to be used outside this transformation; will be removed in stable version",
                  DeprecationWarning)
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
