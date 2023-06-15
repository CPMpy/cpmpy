import copy

from .normalize import toplevel_list
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, intvar, boolvar, _NumVarImpl, cpm_array, NDVarArray
from ..expressions.utils import is_any_list, eval_comparison
from ..expressions.python_builtins import all


def decompose_in_tree(lst_of_expr, supported=set(), supported_nested=set(), _toplevel=None, nested=False):
    """
        Decomposes any global constraint not supported by the solver
        Accepts a list of CPMpy expressions as input and returns a list of CPMpy expressions,
            otherwise, returns a list of CPMpy expressions and new constraints to be added toplevel.

        - supported: a set of supported global constraints or global functions
        - supported_nested: as set of supported global constraints/functions within other expressions
                                these will be part of reifications after flattening
        - toplevel: a list of constraints to be added toplevel, carried as pass by reference to recursive calls

        The basic idea of the algorithm is to traverse the expression tree and
            replace unsupported expressions inplace. Some new expressions should be added
            toplevel to link new auxiliary variables created by the decomposition of constraints.

        Special care taken for unsupported global constraints in nested contexts.

        Supported numerical global functions are left in nested contexts as they can be rewritten using
            `cpmpy.transformations.reification.reify_rewrite`
            The following `bv -> NumExpr <comp> Var/Const` can always be rewritten as  [bv -> IV0 <comp> Var/Const, NumExpr == IV0].
            So even if numerical constraints are not supported in reified context, we can rewrite them to non-reified versions.
    """
    if _toplevel is None:
        _toplevel = []

    flipmap = {"==": "==", "!=": "!=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}

    newlist = []  # decomposed constraints will go here
    for expr in lst_of_expr:

        if is_any_list(expr):
            assert nested is True, "Cannot have nested lists without passing trough an expression, make sure to run cpmpy.transformations.normalize.toplevel_list first."
            newexpr = decompose_in_tree(expr, supported, supported_nested, _toplevel, nested=True)
            if isinstance(expr, NDVarArray):
                newlist.append(cpm_array(newexpr))
            else:
                newlist.append(newexpr)
            continue

        elif isinstance(expr, Operator):
            # recurse into arguments, recreate through constructor (we know it stores no other state)
            args = decompose_in_tree(expr.args, supported, supported_nested, _toplevel, nested=True)
            newlist.append(Operator(expr.name, args))

        elif isinstance(expr, GlobalConstraint):
            # first create a fresh version and recurse into arguments
            expr = copy.copy(expr)
            expr.args = decompose_in_tree(expr.args, supported, supported_nested, _toplevel, nested=True)

            is_supported = (nested and expr.name in supported_nested) or (not nested and expr.name in supported)
            if is_supported:
                newlist.append(expr)
            else:
                if expr.is_bool():
                    # boolean global constraints
                    decomposed, define = expr.decompose()

                    _toplevel.extend(define)  # definitions should be added toplevel
                    # the `decomposed` expression might contain other global constraints, check it
                    decomposed = decompose_in_tree(decomposed, supported, supported_nested, _toplevel, nested=nested)
                    newlist.append(all(decomposed))

                else:
                    # numeric global constraint, replace by a fresh variable and decompose the equality to this
                    lb,ub = expr.get_bounds()
                    aux = intvar(lb, ub)

                    auxdef, otherdef = expr.decompose_comparison("==", aux)
                    _toplevel.extend(auxdef + otherdef)  # all definitions should be added toplevel
                    newlist.append(aux)  # replace original expression by aux

        elif isinstance(expr, Comparison):
            # if one of the two children is a (numeric) global constraint, we can decompose the comparison directly
            # otherwise e.g., min(x,y,z) == a would become `min(x,y,z).decompose_comparison('==',aux) + [aux == a]`
            lhs, rhs = expr.args
            exprname = expr.name  # can change when rhs needs decomp

            # TODO: change hasattr to global functions class check
            decomp_lhs = hasattr(lhs, "decompose_comparison") and not lhs.name in supported_nested
            decomp_rhs = hasattr(rhs, "decompose_comparison") and not rhs.name in supported_nested

            if not decomp_lhs:
                if not decomp_rhs:
                    # nothing special, create a fresh version and recurse into arguments
                    expr = copy.copy(expr)
                    expr.args = decompose_in_tree(expr.args, supported, supported_nested, _toplevel, nested=True)
                    newlist.append(expr)

                else:
                    # only rhs needs decomposition, so flip comparison to make lhs needing the decomposition
                    exprname = flipmap[expr.name]
                    lhs, rhs = rhs, lhs
                    decomp_lhs, decomp_rhs = True, False  # continue into next 'if'

            if decomp_lhs:
                # recurse into lhs args
                lhs = copy.copy(lhs)
                lhs.args = decompose_in_tree(lhs.args, supported, supported_nested, _toplevel, nested=True)

                # decompose comparison of lhs and rhs
                decomposed, define = lhs.decompose_comparison(exprname, rhs)
                _toplevel.extend(define)  # definitions should be added toplevel
                # the `decomposed` expression (and rhs) might contain other global constraints, check it
                decomposed = decompose_in_tree(decomposed, supported, supported_nested, _toplevel, nested=nested)
                newlist.append(all(decomposed))

        else:  # constants, variables, direct constraints
            newlist.append(expr)

    if nested is True:
        return newlist
    elif len(_toplevel) > 0:
        # we are toplevel and some new constraints are introduced, decompose new constraints!
        return toplevel_list(newlist) + decompose_in_tree(_toplevel, supported, supported_nested, nested=False)
    else:
        return toplevel_list(newlist)  # TODO, check for top-level ANDs in transformation?
