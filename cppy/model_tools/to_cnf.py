from ..model import *
from ..expressions import *
from ..variables import *
"""
 Do tseitin transform on list of constraints
 Only supports [], and, or, -, ->, == for now
"""
def to_cnf(constraints):
    # 'constraints' should be list, but lets add some special cases
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


    cnf = []
    
    for expr in constraints:
        # special cases first
        if isinstance(expr, Operator):
            if expr.name == '->':
                # turn into OR constraint, a -> b =:= ~a | b
                expr.args[0] = ~expr.args[0]
                expr.name = 'or'

                # Optimisation: if RHS is an 'or', then merge it in:
                # a -> or([b,c,...,d]) =:= or([~a,b,c,...,d])
                if expr.args[1].name == 'or':
                    expr.args = [expr.args[0]] + expr.args[1].args

            if expr.name == "or":
                # special case: OR constraint, shortcut to disjunction of subexprs
                subvarcnfs = [tseitin_transform(subexpr) for subexpr in expr.args]
                cnf.append( Operator("or", [subv for (subv,_) in subvarcnfs]) )
                cnf += [clause for (_,subcnf) in subvarcnfs for clause in subcnf]
            elif expr.name == "and":
                # special case: AND constraint, flatten into toplevel conjunction
                subcnf = to_cnf(expr.args)
                cnf += subcnf
        elif isinstance(expr, Comparison) and expr.name == '==' and not isinstance(expr.args[1], (bool,int)):
            # [..., a <-> b, ...] :: [..., a -> b, b -> a, ...]
            # reuse variables, no global cache yet
            a,cnf_a = tseitin_transform(expr.args[0])
            b,cnf_b = tseitin_transform(expr.args[1])
            cnf += [~a|b, ~b|a] # [a -> b, b -> a]
            cnf += cnf_a + cnf_b
        elif isinstance(expr, (bool,int)):
            continue
        elif isinstance(expr, list):
            # same special case as 'AND': flatten into top-level
            subcnf = to_cnf(expr)
            cnf += subcnf
        else:
            # generic case
            newvar, newcnf = tseitin_transform(expr)
            cnf.append(newvar)
            cnf += newcnf
    return cnf


def tseitin_transform(expr):
    # base cases
    if isinstance(expr, bool):
        return (expr, [])
    if isinstance(expr, int):
        # python convention: 1 is true, rest is false
        if expr == 1:
            return (True, [])
        return (False, [])
    if isinstance(expr, BoolVarImpl):
        return (expr, [])

    # e == 0 and e == 1
    if isinstance(expr, Comparison) and expr.name == '==' and isinstance(expr.args[1], (int,bool)):
        if expr.args[1] == 1 or expr.args[1] is True:
            return tseitin_transform(expr.args[0])
        elif expr.args[1] == 0 or expr.args[1] is False:
            (var,cnf) = tseitin_transform(expr.args[0])
            return (~var, cnf)
        else:
            raise Exception("Tseitin: e == '"+str(expr.args[1])+"' not supported yet")

    if isinstance(expr, Comparison) and expr.name != '==':
        raise Exception("Tseitin: Expression '"+str(expr)+"' not supported yet:", type(expr))

    # rest, with exception of '=='
    if not isinstance(expr, Operator) and not (isinstance(expr, Comparison) and expr.name == '=='):
        raise Exception("Tseitin: Expression '"+str(expr)+"' not supported yet:", type(expr))

    # Operators:
    implemented = ['-', 'and', 'or', '->', '==']
    if not expr.name in implemented:
        raise Exception("Tseitin: Operator '"+expr.name+"' not implemented")

    # recursively transform the arguments first and merge their cnfs
    subvarcnfs = [tseitin_transform(subexpr) for subexpr in expr.args]
    cnf = [clause for (_,subcnf) in subvarcnfs for clause in subcnf]
    subvars = [subvar for (subvar,_) in subvarcnfs]

    if isinstance(expr, Operator):
        # special case: unary -, negate single argument var
        if expr.name == '-':
            return (~subvars[0], cnf)

    Aux = BoolVarImpl()
    if expr.name == "and":
        cnf.append( Operator("or", [Aux] + [~var for var in subvars]) )
        for var in subvars:
            cnf.append( ~Aux | var )

    if expr.name == "or":
        cnf.append( Operator("or", [~Aux] + [var for var in subvars]) )
        for var in subvars:
            cnf.append( Aux | ~var )

    if expr.name  == "->":
        # Implication is treated as if it were "or": A -> B <=> ~A or B
        A = subvars[0]
        B = subvars[1]
        cnf = [(~Aux | ~A | B), (Aux | A), (Aux | ~B)]

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

    return Aux, cnf

