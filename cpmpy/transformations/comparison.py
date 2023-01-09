import copy

from cpmpy.expressions.utils import *

from .flatten_model import get_or_make_var
from ..expressions.core import Comparison, Operator
from ..expressions.variables import _NumVarImpl, _IntVarImpl

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

def only_numexpr_equality(constraints, supported=frozenset(["sum","wsum"])):
    """
        transforms `NumExpr <op> IV` to `(NumExpr == A) & (A <op> IV)` if not supported

        argument 'supported' is a list (or set) of expression names that supports all comparisons in the solver
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


def nbc_main(constraints):
    '''
    Input must be a list of 2 elements, where the first one keeps track of assignment of helper IV's that should not be
    affected by nested constraints. The second element keeps the regular constraints.

    Do not call this function directly, instead call 'no_boolean_comparisons'
    '''
    aux, con = constraints
    if isinstance(con, Comparison):
        lhs, rhs = con.args
        if con.name == '==' or con.name == '!=':
            if not is_boolexpr(rhs) and lhs.is_bool():
                #introduce aux var
                iv = _IntVarImpl(0, 1)  # 0,1 is the valid domain, as this represents true or false
                #deal with nested constraints
                nbc_lhs = nbc_main([[], lhs])
                nbc_rhs = nbc_main([[], rhs])
                return [aux + nbc_lhs[0] + nbc_rhs[0] + [nbc_lhs[1] == iv], Comparison(con.name, iv, nbc_rhs[1])]
        elif lhs.is_bool():
            iv = _IntVarImpl(0, 1)
            nbc_lhs = nbc_main([[], lhs])
            nbc_rhs = nbc_main([[], rhs])
            #other comparisons:
            assert con.name == '<=' or con.name == '<' or con.name == '>=' or con.name == '>'
            return [aux + nbc_lhs[0] + nbc_rhs[0] + [nbc_lhs[1] == iv], Comparison(con.name, iv, nbc_rhs[1])]

        else:
            #allowed comparison, just do the nested call
            nbc_lhs = nbc_main([[], lhs])
            nbc_rhs = nbc_main([[], rhs])
            return[aux + nbc_lhs[0] + nbc_rhs[0], Comparison(con.name, nbc_lhs[1], nbc_rhs[1])]
    elif isinstance(con, Operator):
        nbc_args = []
        for arg in con.args:
            nbc_arg = nbc_main([[], arg])
            aux += nbc_arg[0]
            nbc_args += [nbc_arg[1]]
        return [aux, Operator(con.name, nbc_args)]
    else:
        return [aux, con] #base case


def no_boolean_comparisons(constraints):
    '''
    Transforms a set of constraints such that a boolexpr is never compared with a numerical expression
    furthermore 2 boolean expressions are never compared with <,>,<= or >=
    (!= and == are allowed between 2 boolean expressions)
    Boolexpr == Numexpr will be transformed to Boolexpr == IV and IV == Numexpr
    where IV is an additional intvar with domain 0,1
    '''
    if is_any_list(constraints):
        return [no_boolean_comparisons(con) for con in constraints]
    return flatlist(nbc_main([[], constraints]))
    #main function expects a list of 2 elements, where the first element keeps track of helper variables.