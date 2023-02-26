import copy

from .flatten_model import get_or_make_var
from ..expressions.core import Comparison
from ..expressions.variables import _NumVarImpl

"""
  Transformations regarding Comparison constraints.
  
  Comparisons in Flat Normal Form are of the kind
    - NumExpr == IV
    - BoolExpr == BV
    
  The latter is a reified expression, not considered here.
  (for handling of all reified expressions, see `reification.py` transformations)
  
  This file implements:
    - only_numexpr_equality():    transforms `NumExpr <op> IV` to `(NumExpr == A) & (A <op> IV)` if not supported
"""

def only_numexpr_equality(constraints, supported=frozenset()):
    """
        transforms `NumExpr <op> IV` to `(NumExpr == A) & (A <op> IV)` if not supported

        :param supported  a (frozen)set of expression names that supports all comparisons in the solver
    """

    # shallow copy (could support inplace too this way...)
    newcons = copy.copy(constraints)

    for i,con in enumerate(newcons):
        if isinstance(con, Comparison) and con.name != '==':
            # LHS <op> IV    with <op> one of !=,<,<=,>,>=
            lhs = con.args[0]
            if not isinstance(lhs, _NumVarImpl) and not lhs.name in supported:
                # LHS is unsupported for LHS <op> IV, rewrite to `(LHS == A) & (A <op> IV)`
                (lhsvar, lhscons) = get_or_make_var(lhs)
                # replace comparison by A <op> IV
                newcons[i] = Comparison(con.name, lhsvar, con.args[1])
                # add lhscon(s), which will be [(LHS == A)]
                assert(len(lhscons) == 1), "only_numexpr_eq: lhs surprisingly non-flat"
                newcons.insert(i, lhscons[0])

    return newcons
