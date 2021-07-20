from ..model import Model
from ..expressions.core import Operator
from ..expressions.variables import BoolVarImpl, NegBoolView
from .flatten_model import flatten_constraint, negated_normal
"""
  Converts the logical constraints in a list of constraints,
  into disjuctions using the tseitin transform.
  
  Other constraints are copied verbatim so this transformation
  can also be used in non-pure CNF settings (e.g. linear constraints)

  The implementation first converts the list of constraints
  to 'flat normal form', this already flattens subexpressions using
  auxiliary variables.

  What is then left to do is to tseitin encode
  - and() constraints
  - xor() constraints
  - BE == BoolVar() with BE :: BoolVar()|and()|or()
  - BE -> BoolVar()
  - BoolVar() -> BE
"""

def to_cnf(constraints):
    """
        Converts all logical constraints into Conjunctive Normal Form

        - constraints: list[Expression] or Model or Operator
    """
    if isinstance(constraints, Model):
        # transform model's constraints
        return to_cnf(constraints.constraints)
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
    cnf = flat2cnf(fnf)

def flat2cnf(constraints):
    """
        Converts from 'flat normal form' all logical constraints into Conjunctive Normal Form

        What is now left to do is to tseitin encode
  - BoolVar()
  - and() constraints
  - xor() constraints
  - BE == BoolVar() with BE :: BoolVar()|and()|or()|(BV == BV)
  - BE -> BoolVar()
  - BoolVar() -> BE
    """
    cnf = []
    for expr in constraints:
        # base cases
        if isinstance(expr, (bool,int)):
            # python convention: 1 is true, rest is false
            if expr:
                continue # skip this trivially true expr
            else:
                return [False]

        # BoolVar()
        elif isinstance(expr, BoolVarImpl): # includes NegBoolView
            cnf.append(expr)
            continue

        # and() constraints
        elif isinstance(expr, Operator) and expr.name == "and":
            # special case: top-level AND constraint,
            # flatten into toplevel conjunction
            cnf += expr.args
            continue

        # xor() constraints
        elif isinstance(expr, Operator) and expr.name == "xor":
            # xor(x,y,z) = (~x&y&z) | (x&~y~z) | (x&y&~z)
            # need to flatten and tseitin that accordingly
            raise NotImplementedError("TODO")

        # BoolVar() -> BoolVar()
        # BE -> BoolVar() with BE :: and()|or()
        # BoolVar() -> BE
        elif isinstance(expr, Operator) and expr.name == '->':
            a0,a1 = expr.args

            if isinstance(a0, BoolVarImpl) and isinstance(a1, BoolVarImpl):
                cnf.append(~a0 | a1)
                continue
            elif isinstance(a0, BoolVarImpl):
                # BV -> BE
                if isinstance(a1, Operator) and a1.name == 'or':
                    # trivial clause
                    cnf.append( Operator("or", [~a0] + [~var for var in a1.args]) )
                    continue
                elif isinstance(a1, Operator) and a1.name == 'and':
                    # BV -> and()
                    # do tseitin on fresh var: [~BV | aux] + tseitin(aux == and())
                    aux = BoolVar()
                    subcnf = flat2cnf(aux == a1)
                    cnf += [~a0 | aux] + subcnf
                    continue
            elif isinstance(a1, BoolVarImpl):
                # BE -> BV
                if isinstance(a0, Operator) and a1.name == 'and':
                    # trivial clause
                    cnf.append( Operator("or", [~var for var in a0.args] + [a1]) )
                    continue
                elif isinstance(a0, Operator) and a1.name == 'or':
                    # or -> BV
                    # do tseitin on fresh var: [~aux | BV] + tseitin(aux == or())
                    aux = BoolVar()
                    subcnf = flat2cnf(aux == a0)
                    cnf += [~aux | a1] + subcnf
                    continue

        # BE == BoolVar() with BE :: BoolVar()|and()|or()
        elif isinstance(expr, Comparison) and expr.name == '==':
            a0,a1 = expr.args
            if isinstance(a0, BoolVarImpl) and \ # includes NegBoolView
               isinstance(a1, BoolVarImpl):
                    # BV == BV :: BV <-> BV
                    cnf += [~a0|a1, ~a1|a0] # [a0 -> a1, a1 -> a0]
                    continue
            elif isinstance(expr, Operator) and expr.name in ('and','or'):
                # BE == BoolVar()
                subvars = a0.expr
                if a0.name == "and":
                    # Tseitin of and(subvars) <-> a1:
                    #   a1 or ~subvar1 or ~subvar2 or ...
                    #   ~subvar1 or a1
                    #   ~subvar2 or a1
                    #   ...
                    cnf.append( Operator("or", [a1] + [~var for var in subvars]) )
                    for var in subvars:
                        cnf.append( ~a1 | var )
                    continue
                elif a0.name == "or":
                     # Tseitin of or(subvars) <-> a1:
                     #   ~a1 or subvar1 or subvar2 or ...
                     #   a1 or ~subvar1
                     #   a1 or ~subvar2                        #   ...
                     cnf.append( Operator("or", [~a1] + [var for var in subvars]) )
                     for var in subvars:
                        cnf.append( a1 | ~var )
                     continue

        # all other cases not covered (e.g. not continue'd)
        # pass verbatim
        cnf.append(expr)

    # TODO:
    if expr.name == '==':
        # Aux :: A <-> B
        # Aux A B
        # 1   1 1
        # 1   0 0
        # 0   0 1
        # 0   1 0
        A = subvars[0]
        B = subvars[1]
        
        cnf = [(Aux | A | B), (Aux | ~A | ~B), (~Aux | ~A | B), (~Aux | A | ~B)]

        
