# Tias Guns, 2019

from .variables import *
from .expressions import *
from .globalconstraints import *
from .model import *


### variable creation, objects in .variables

# N-dimensional array of Boolean Decision Variables
def BoolVar(shape=None):
    if shape == None or shape == 1:
        return BoolVarImpl()
    length = np.prod(shape)
    
    # create base data
    data = np.array([BoolVarImpl() for _ in range(length)]) # repeat new instances
    # insert into custom ndarray
    return NDVarArray(shape, dtype=object, buffer=data)

# N-dimensional array of Integer Decision Variables with lower-bound and upper-bound
def IntVar(lb, ub, shape=None):
    if shape == None or shape == 1:
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
    return BoolOperator('->', a, b)
