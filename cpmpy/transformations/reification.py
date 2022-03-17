from ..expressions.core import Operator, Comparison
from ..expressions.variables import _BoolVarImpl
from ..expressions.utils import is_any_list
from .flatten_model import negated_normal
"""
  Transformations regarding reification constraints.

  There are three types of reification (BV=BoolVar, BE=BoolExpr):
    - BV -> BE      single implication, from var to expression
    - BV <- BE      single implication, from expression to var
    - BE == BV      full reification / double implication (e.g. BV <-> BE)

  Using logical operations, they can be decomposed and rewritten to each other.

  This file implements:
    - only_bv_implies():    transforms all reifications to BV -> BE form
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
