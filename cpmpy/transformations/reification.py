import copy
from ..expressions.core import Operator, Comparison, Expression
from ..expressions.globalconstraints import GlobalConstraint, Element
from ..expressions.variables import _BoolVarImpl, _NumVarImpl
from ..expressions.python_builtins import all
from ..expressions.utils import is_any_list
from .flatten_model import flatten_constraint, negated_normal, get_or_make_var

"""
  Transformations regarding reification constraints.

  There are three types of reification (BV=BoolVar, BE=BoolExpr):
    - BV -> BE      single implication, from var to expression
    - BV <- BE      single implication, from expression to var
    - BE == BV      full reification / double implication (e.g. BV <-> BE)

  Using logical operations, they can be decomposed and rewritten to each other.

  This file implements:
    - only_bv_implies():    transforms all reifications to BV -> BE form
    - reify_rewrite():      rewrites reifications not supported by a solver to ones that are
"""

def only_bv_implies(constraints):
    """
        Transforms all reifications to BV -> BE form

        More specifically:
            BE -> BV :: ~BV -> ~BE
            BE == BV :: ~BV -> ~BE, BV -> BE

        Assumes all constraints are in 'flat normal form'. Hence only apply
        AFTER `flatten()`
    """
    if not is_any_list(constraints):
        # assume list, so make list
        constraints = [constraints]

    newcons = []
    for cpm_expr in constraints:
        # Operators: check BE -> BV
        if isinstance(cpm_expr, Operator) and \
                cpm_expr.name == '->' and \
                cpm_expr.args[0].is_bool() and \
                not isinstance(cpm_expr.args[0], _BoolVarImpl) and \
                isinstance(cpm_expr.args[1], _BoolVarImpl):
            # BE -> BV :: ~BV -> ~BE
            negbvar = ~(cpm_expr.args[1])
            negexpr = negated_normal(cpm_expr.args[0])
            newcons.append(negbvar.implies(negexpr))

        # Comparisons: check BE == BV
        elif isinstance(cpm_expr, Comparison) and \
                cpm_expr.name == '==' and \
                cpm_expr.args[0].is_bool() and \
                isinstance(cpm_expr.args[1], _BoolVarImpl):
            # BV == BV special case
            if isinstance(cpm_expr.args[0], _BoolVarImpl):
                l,r = cpm_expr.args
                newcons.append(l.implies(r))
                newcons.append(r.implies(l))
            else:
                # BE == BV :: ~BV -> ~BE, BV -> BE
                expr,bvar = cpm_expr.args
                newcons.append((~bvar).implies(negated_normal(expr)))
                newcons.append(bvar.implies(expr))

        else:
            # all other flat normal form expressions are fine
            newcons.append(cpm_expr)
    
    return newcons


def reify_rewrite(constraints, supported=frozenset(['sum', 'wsum'])):
    """
        Rewrites reified constraints not natively supported by a solver,
        to a version that uses standard constraints and reification over equalities between variables.

        Input is expected to be in Flat Normal Form (so after `flatten_constraint()`)
        Output will also be in Flat Normal Form

        argument 'supported' is a list (or set) of expression names that support reification in the solver
    """
    if not is_any_list(constraints):
        # assume list, so make list
        constraints = [constraints]

    newcons = []
    for cpm_expr in constraints:
        if not isinstance(cpm_expr, Expression):
            continue
        # check if reif, get (the index of) the Boolean subexpression BE
        boolexpr_index = None
        if cpm_expr.name == '->':
            if not isinstance(cpm_expr.args[0], _BoolVarImpl):  # BE -> BV
                boolexpr_index = 0
            elif not isinstance(cpm_expr.args[1], _BoolVarImpl):  # BV -> BE
                boolexpr_index = 1
            else:
                pass  # both BV
        elif cpm_expr.name == '==' and \
                cpm_expr.args[0].is_bool() and \
                not isinstance(cpm_expr.args[0], _BoolVarImpl) and \
                isinstance(cpm_expr.args[1], _BoolVarImpl):  # BE == BV
            boolexpr_index = 0

        if boolexpr_index is None:  # non-reified or variable-only reification
            newcons.append(cpm_expr)
        else:  # reification, check for rewrite
            boolexpr = cpm_expr.args[boolexpr_index]
            if isinstance(boolexpr, Operator):
                # Case 1, BE is Operator (and, or, ->)
                #   assume supported, return as is
                newcons.append(cpm_expr)
                #   could actually rewrite into list of clauses like to_cnf() does... not for here
            elif isinstance(boolexpr, GlobalConstraint):
                # Case 2, BE is a GlobalConstraint
                # replace BE by its decomposition, then flatten
                reifexpr = copy.copy(cpm_expr)
                reifexpr.args[boolexpr_index] = all(boolexpr.decompose())  # decomp() returns list
                newcons += flatten_constraint(reifexpr)
            elif isinstance(boolexpr, Comparison):
                # Case 3, BE is Comparison(OP, LHS, RHS)
                op,(lhs,rhs) = boolexpr.name, boolexpr.args
                #   have list of supported lhs's such as sum and wsum...
                #   at the very least, (iv1 == iv2) == bv has to be supported (or equivalently, sum: (iv1 - iv2 == 0) == bv)
                if isinstance(lhs, _NumVarImpl) or lhs.name in supported:
                    newcons.append(cpm_expr)
                elif isinstance(lhs, Element) and (lhs.args[1].lb < 0 or lhs.args[1].ub >= len(lhs.args[0])):
                    # special case: (Element(arr,idx) <OP> RHS) == BV (or -> in some way)
                    # if the domain of 'idx' is larger than the range of 'arr', then 
                    # this is allowed and BV should be false if it takes a value there
                    # so we can not use Element (which would restruct the domain of idx)
                    # and have to work with an element-wise decomposition instead
                    reifexpr = copy.copy(cpm_expr)
                    reifexpr.args[boolexpr_index] = all(lhs.decompose_comparison(op,rhs))  # decomp() returns list
                    newcons += flatten_constraint(reifexpr)
                else:  #   other cases (assuming LHS is a total function):
                    #     (AUX,c) = get_or_make_var(LHS)
                    #     return c+[Comp(OP,AUX,RHS) == BV] or +[Comp(OP,AUX,RHS) -> BV] or +[Comp(OP,AUX,RHS) <- BV]
                    (auxvar, cons) = get_or_make_var(lhs)
                    newcons += cons
                    reifexpr = copy.copy(cpm_expr)
                    reifexpr.args[boolexpr_index] = Comparison(op, auxvar, rhs)  # Comp(OP,AUX,RHS)
                    newcons.append(reifexpr)
            else:
                # don't think this will be reached
                newcons.append(cpm_expr)

    return newcons
