import copy
from ..expressions.core import Operator, Comparison, Expression
from ..expressions.globalconstraints import GlobalConstraint, Element
from ..expressions.variables import _BoolVarImpl, _NumVarImpl
from ..expressions.python_builtins import all
from ..expressions.utils import is_any_list
from .flatten_model import flatten_constraint, get_or_make_var, negated_normal

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
    newcons = []
    for cpm_expr in constraints:
        # Operators: check BE -> BV
        if cpm_expr.name == '->':
            a0,a1 = cpm_expr.args
            if not isinstance(a0, _BoolVarImpl) and \
                    isinstance(a1, _BoolVarImpl):
                # BE -> BV :: ~BV -> ~BE
                newexpr = (~a1).implies(negated_normal(a0))
                #newexpr = (~a1).implies(~a0)  # XXX when push_down_neg is separate, negated_normal no longer needed separately
                newcons.extend(only_bv_implies(flatten_constraint(newexpr)))
            elif isinstance(a1, Comparison) and \
                    a1.name == '==' and a1.args[0].is_bool():
                # BV0 -> BV2 == BV3 :: BV0 -> (BV2->BV3 & BV3->BV2)
                #                   :: BV0 -> (BV2->BV3) & BV0 -> (BV3->BV2)
                #                   :: BV0 -> (~BV2|BV3) & BV0 -> (~BV3|BV2)
                bv2,bv3 = a1.args
                newexpr = [a0.implies(~bv2|bv3), a0.implies(~bv3|bv2)]
                newcons.extend(only_bv_implies(flatten_constraint(newexpr)))
            else:
                newcons.append(cpm_expr)

        # Comparisons: check BE == BV
        elif cpm_expr.name == '==' and cpm_expr.args[0].is_bool():
            a0,a1 = cpm_expr.args
            if isinstance(a0, _BoolVarImpl) and isinstance(a1, _BoolVarImpl):
                # BVar0 == BVar1 special case, no need to re-transform
                newcons.append(a0.implies(a1))
                newcons.append(a1.implies(a0))
            else:
                # BE0 == BVar1 :: ~BVar1 -> ~BE0, BVar1 -> BE0
                newexprs = ((~a1).implies(negated_normal(a0)), a1.implies(a0))
                #newexprs = ((~a1).implies(~a0), a1.implies(a0))  # XXX when push_down_neg is separate, negated_normal no longer needed separately
                newcons.extend(only_bv_implies(flatten_constraint(newexprs)))
            # XXX there used to be a weird
            # BE0 == IVar1 :: IVar1 = BVarX, ~BVarX -> ~BE, BVarX -> BE
        else:
            # all other flat normal form expressions are fine
            newcons.append(cpm_expr)
    
    return newcons


def reify_rewrite(constraints, supported=frozenset()):
    """
        Rewrites reified constraints not natively supported by a solver,
        to a version that uses standard constraints and reification over equalities between variables.

        Input is expected to be in Flat Normal Form (so after `flatten_constraint()`)
        Output will also be in Flat Normal Form

        Boolean expressions 'and', 'or', and '->' and comparison expression 'IV1==IV2' are assumed to support reification
        (actually currently all comparisons <op> in {'==', '!=', '<=', '<', '>=', '>'},
         IV1 <op> IV2 are assumed to support reification BV -> (IV1 <op> IV2))

        :param supported  a (frozen)set of expression names that support reification in the solver, including
                          supported 'Left Hand Side (LHS)' expressions in reified comparisons, e.g. BV -> (LHS == V)
    """
    if not is_any_list(constraints):
        # assume list, so make list
        constraints = [constraints]

    newcons = []
    for cpm_expr in constraints:
        assert isinstance(cpm_expr, Expression), f"Expected CPMpy Expression but got {cpm_expr}, run transformations.normalize.make_cpm_expr first!"
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
                if boolexpr.name in supported:
                    newcons.append(cpm_expr)
                else:
                    reifexpr = copy.copy(cpm_expr)
                    reifexpr.args[boolexpr_index] = all(boolexpr.decompose())  # decomp() returns list
                    newcons += flatten_constraint(reifexpr)
            elif isinstance(boolexpr, Comparison):
                # Case 3, BE is Comparison(OP, LHS, RHS)
                op,(lhs,rhs) = boolexpr.name, boolexpr.args
                #   have list of supported lhs's such as sum and wsum...
                #   at the very least, (iv1 == iv2) == bv has to be supported
                if isinstance(lhs, _NumVarImpl) or lhs.name in supported:
                    newcons.append(cpm_expr)
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
