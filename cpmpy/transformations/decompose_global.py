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
        When root call is True, returns a list of CPMpy expressions,
            otherwise, returns a list of CPMpy expressions and new constraints to be added toplevel.

        - supported: a set of supported global constraints or global functions
        - supported_nested: as set of supported global constraints/functions within a reification

        Unsupported constraints are decomposed in equivalent simpler constraints.
        Special care taken for global constraints in nested contexts.

        Supported numerical global functions are left in nested contexts as they can be rewritten using
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
                if expr.name not in supported:
                    aux = intvar(*expr.get_bounds())
                    newlist.append(aux)
                    newexpr, newcons[len(newcons):] = decompose_in_tree(expr.decompose_comparison("==",rhs=aux),
                                                                        supported, supported_nested, nested)
                    newcons.extend(newexpr)
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
                    lhs, newcons[len(newcons):] = decompose_in_tree(lhs.decompose(), supported, supported_nested, True)
                    lhs = all(lhs)

                if not lhs.is_bool() and lhs.name not in supported:
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