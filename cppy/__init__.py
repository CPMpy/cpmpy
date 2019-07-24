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

# N-dimensional wrapper, wrap a standard array (e.g. [1,2,3,4] whatever)
# so that we can do [1,2,3,4][var1] == var2, e.g. element([1,2,3,4],var1,var2)
# needed because standard arrays can not be indexed by non-constants
def cparray(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return NDVarArray(shape=arr.shape, dtype=type(arr.flat[0]), buffer=arr)


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
        

# min: listwise 'min'
def min(iterable):
    if not any(isinstance(elem, MathExpression) for elem in iterable):
        return np.min(iterable)
    return GlobalConstraint("min", list(iterable))
def max(iterable):
    if not any(isinstance(elem, MathExpression) for elem in iterable):
        return np.max(iterable)
    return GlobalConstraint("max", list(iterable))
