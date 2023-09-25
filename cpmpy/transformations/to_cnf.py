from ..expressions.core import Operator, Comparison
from ..expressions.variables import _BoolVarImpl, NegBoolView
from .reification import only_bv_implies
from .flatten_model import flatten_constraint
"""
  Converts the logical constraints into disjuctions using the tseitin transform,
        including flattening global constraints that are is_bool() and not in `supported`.
  
  Other constraints are copied verbatim so this transformation
  can also be used in non-pure CNF settings

  The implementation first converts the list of constraints
  to 'flat normal form', this already flattens subexpressions using
  auxiliary variables.

  What is then left to do is to tseitin encode the following into CNF:
  - BV with BV a BoolVar (or NegBoolView)
  - or([BV]) constraint
  - and([BV]) constraint
  - BE != BV  with BE :: BV|or()|and()|BV!=BV|BV==BV|BV->BV
  - BE == BV
  - BE -> BV
  - BV -> BE
"""

def to_cnf(constraints):
    """
        Converts all logical constraints into Conjunctive Normal Form

        Arguments:
        - constraints: list[Expression] or Operator
        - supported: (frozen)set of global constraint names that do not need to be decomposed
    """
    fnf = flatten_constraint(constraints)
    fnf = only_bv_implies(fnf)
    return flat2cnf(fnf)

def flat2cnf(constraints):
    """
        Converts from 'flat normal form' all logical constraints into Conjunctive Normal Form,
        including flattening global constraints that are is_bool() and not in `supported`.

        What is now left to do is to tseitin encode:

  - BV with BV a BoolVar (or NegBoolView)
  - or([BV]) constraint
  - and([BV]) constraint
  - BE != BV  with BE :: BV|or()|and()|BV!=BV|BV==BV|BV->BV
  - BE == BV
  - BE -> BV
  - BV -> BE

        We do it in a principled way for each of the cases. (in)equalities
        get transformed into implications, everything is modular.

        Arguments:
        - constraints: list[Expression] or Operator
    """
    cnf = []
    for expr in constraints:
        # BE -> BE
        if expr.name == '->':
            a0,a1 = expr.args

            # BoolVar() -> BoolVar()
            if isinstance(a1, _BoolVarImpl) or \
                    (isinstance(a1, Operator) and a1.name == 'or'):
                cnf.append(~a0 | a1)
                continue

        # all other cases added as is...
        # TODO: we should raise here? is not really CNF...
        cnf.append(expr)

    return cnf
