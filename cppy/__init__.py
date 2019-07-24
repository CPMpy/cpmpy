# Tias Guns, 2019

from .variables import *
from .expressions import *
from .globalconstraints import *
from .model import *


### variable creation, objects in .variables

# N-dimensional array of Boolean Decision Variables
def BoolVar(shape=None):
    if shape is None or shape == 1:
        return BoolVarImpl()
    length = np.prod(shape)
    
    # create base data
    data = np.array([BoolVarImpl() for _ in range(length)]) # repeat new instances
    # insert into custom ndarray
    return NDVarArray(shape, dtype=object, buffer=data)

# N-dimensional array of Integer Decision Variables with lower-bound and upper-bound
def IntVar(lb, ub, shape=None):
    if shape is None or shape == 1:
        return IntVarImpl(lb,ub)
    length = np.prod(shape)
    
    # create base data
    data = np.array([IntVarImpl(lb,ub) for _ in range(length)]) # repeat new instances
    # insert into custom ndarray
    return NDVarArray(shape, dtype=object, buffer=data)


# implication constraint: a -> b
# Python does not offer relevant syntax...
# for double implication, use equivalence a == b
def implies(a, b):
    assert isinstance(a, LogicalExpression), "First argument must be a logical expression"
    assert isinstance(b, LogicalExpression), "Second argument must be a logical expression"
    return BoolOperator('->', [a, b])


# all: listwise 'and'
def all(iterable):
    collect = [] # logical expressions
    for elem in iterable:
        if isinstance(elem, LogicalExpression):
            collect.append( elem )
        elif not elem:
            return False
    if len(collect) > 0:
        return BoolOperator("and", collect)
    return True
        
# any: listwise 'or'
def any(iterable):
    collect = [] # logical expressions
    for elem in iterable:
        if isinstance(elem, LogicalExpression):
            collect.append( elem )
        elif elem:
            return True
    if len(collect) > 0:
        return BoolOperator("or", collect)
    return False
        
