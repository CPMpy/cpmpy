from ..expressions.core import Operator, Comparison
from ..expressions.variables import _BoolVarImpl, NegBoolView
from .flatten_model import flatten_constraint, negated_normal
"""
  Converts the logical constraints into disjuctions using the tseitin transform.
  
  Other constraints are copied verbatim so this transformation
  can also be used in non-pure CNF settings

  The implementation first converts the list of constraints
  to 'flat normal form', this already flattens subexpressions using
  auxiliary variables.

  What is then left to do is to tseitin encode the following into CNF:
  - BV with BV a BoolVar (or NegBoolView)
  - or([BV]) constraint
  - and([BV]) constraint
  - xor(BV,BV) constraint (length-2 only for now)
  - BE != BV  with BE :: BV|or()|and()|xor()|BV!=BV|BV==BV|BV->BV
  - BE == BV
  - BE -> BV
  - BV -> BE
"""

def to_cnf(constraints):
    """
        Converts all logical constraints into Conjunctive Normal Form

        Arguments:

        - constraints: list[Expression] or Operator
    """
    if isinstance(constraints, Operator): 
        if constraints.name == "and":
            # and() is same as a list of its elements
            constraints = constraints.args
        elif constraints.name in ['or', '->']:
            # make or() into [or()] as result will be cnf anyway
            constraints = [constraints]

    if not isinstance(constraints, (list,tuple)):
        # catch rest, single object to singleton object
        constraints = [constraints]

    fnf = flatten_constraint(constraints)
    return flat2cnf(fnf)

def flat2cnf(constraints):
    """
        Converts from 'flat normal form' all logical constraints into Conjunctive Normal Form

        What is now left to do is to tseitin encode:

  - BV with BV a BoolVar (or NegBoolView)
  - or([BV]) constraint
  - and([BV]) constraint
  - xor(BV,BV) constraint (length-2 only for now)
  - BE != BV  with BE :: BV|or()|and()|xor()|BV!=BV|BV==BV|BV->BV
  - BE == BV
  - BE -> BV
  - BV -> BE

        We do it in a principled way for each of the cases. (in)equalities
        get transformed into implications, everything is modular.
    """
    cnf = []
    for expr in constraints:
        is_operator = isinstance(expr, Operator)
        # base cases
        if isinstance(expr, (bool,int)):
            # python convention: 1 is true, rest is false
            if expr:
                continue # skip this trivially true expr
            else:
                return [False]

        # BoolVar()
        elif isinstance(expr, _BoolVarImpl): # includes NegBoolView
            cnf.append(expr)
            continue

        # or() constraint
        elif is_operator and expr.name == "or":
            # top-level OR constraint, easy
            cnf.append(expr)
            continue

        # and() constraints
        elif is_operator and expr.name == "and":
            # special case: top-level AND constraint,
            # flatten into toplevel conjunction
            cnf += expr.args
            continue

        # xor() constraints
        elif is_operator and expr.name == "xor":
            if len(expr.args) == 2:
                a0,a1 = expr.args
                cnf += flat2cnf([(a0|a1), (~a0|~a1)]) # one true and one false
                continue
            else:
                # xor(x,y,z) = (~x&y&z) | (x&~y~z) | (x&y&~z)
                # need to flatten and tseitin that accordingly
                raise NotImplementedError("TODO: nary xor")

        # BE != BE (same as xor)
        elif isinstance(expr, Comparison) and expr.name == "!=":
            a0,a1 = expr.args
            # using 'implies' means it will recursively work for BE's too
            cnf += flat2cnf([a0.implies(~a1), (~a0).implies(a1)]) # one true and one false
            continue

        # BE == BE
        elif isinstance(expr, Comparison) and expr.name == "==":
            a0,a1 = expr.args
            # using 'implies' means it will recursively work for BE's too
            cnf += flat2cnf([a0.implies(a1), a1.implies(a0)]) # a0->a1 and a1->a0
            continue

        # BE -> BE
        elif is_operator and expr.name == '->':
            a0,a1 = expr.args

            # BoolVar() -> BoolVar()
            if isinstance(a0, _BoolVarImpl) and isinstance(a1, _BoolVarImpl):
                cnf.append(~a0 | a1)
                continue
            # BoolVar() -> BE
            elif isinstance(a0, _BoolVarImpl):
                # flat, so a1 must itself be a base constraint
                subcnf = flat2cnf([a1]) # will return a list that is flat
                cnf += [~a0 | a1sub for a1sub in subcnf]
                continue
            # BE -> BoolVar()
            elif isinstance(a1, _BoolVarImpl):
                # a0 is the base constraint, negate to ~a1 -> ~a0
                subcnf = flat2cnf([negated_normal(a0)])
                cnf += [a0sub | a1 for a0sub in subcnf]
                continue

        # all other cases not covered (e.g. not continue'd)
        # pass verbatim
        cnf.append(expr)

    return cnf
