from .expressions import GlobalConstraint, Operator, Expression
import numpy as np
# in one file for easy overview, does not include interpretation

def alldifferent(variables):
    return GlobalConstraint("alldifferent", [variables])

def circuit(variables):
    return GlobalConstraint("circuit", [variables])

# implication constraint: a -> b
# Python does not offer relevant syntax...
# I am considering overloading bitshift >>
# for double implication, use equivalence a == b
def implies(a, b):
    # both constant
    if type(a) == bool and type(b) == bool:
        return (~a | b)
    # one constant
    if a is True:
        return b
    if a is False:
        return True
    if b is True:
        return True
    if b is False:
        return ~a

    return Operator('->', [a.boolexpr(), b.boolexpr()])

# all: listwise 'and'
def all(iterable):
    collect = [] # logical expressions
    for elem in iterable:
        if elem is False:
            return False # no need to create constraint
        elif elem is True:
            pass
        elif isinstance(elem, Expression):
            collect.append( elem.boolexpr() )
        else:
            raise "unknown argument to 'all'"
    if len(collect) == 1:
        return collect[0]
    if len(collect) >= 2:
        return Operator("and", collect)
    return True

# any: listwise 'or'
def any(iterable):
    collect = [] # logical expressions
    for elem in iterable:
        if elem is True:
            return True # no need to create constraint
        elif elem is False:
            pass
        elif isinstance(elem, Expression):
            collect.append( elem.boolexpr() )
        else:
            raise "unknown argument to 'any'"
    if len(collect) == 1:
        return collect[0]
    if len(collect) >= 2:
        return Operator("or", collect)
    return False

# min: listwise 'min'
def min(iterable):
    # constants only?
    if not any(isinstance(elem, Expression) for elem in iterable):
        return np.min(iterable)
    return GlobalConstraint("min", list(iterable))

def max(iterable):
    # constants only?
    if not any(isinstance(elem, Expression) for elem in iterable):
        return np.max(iterable)
    return GlobalConstraint("max", list(iterable))
