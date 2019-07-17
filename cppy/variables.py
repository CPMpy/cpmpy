import numpy as np
from .expressions import *

class NumVarImpl(NumericExpression):
    def __init__(self, lb, ub):
        assert(lb <= ub)
        self.lb = lb
        self.ub = ub
        self.value = None

# Maybe should have an abstract VarImpl for both Bool and Int?
class BoolVarImpl(NumVarImpl):
    counter = 0

    def __init__(self, lb=0, ub=1):
        assert(lb == 0 or lb == 1)
        assert(ub == 0 or ub == 1)
        super().__init__(lb, ub)
        
        self.id = BoolVarImpl.counter
        BoolVarImpl.counter = BoolVarImpl.counter + 1 # static counter
        
    def __repr__(self):
        return "BV{}[{},{}]".format(self.id, self.lb, self.ub)


class IntVarImpl(NumVarImpl):
    counter = 0

    def __init__(self, lb, ub):
        assert(lb >= 0 and ub >= 0)
        super().__init__(lb, ub)
        
        self.id = IntVarImpl.counter
        IntVarImpl.counter = IntVarImpl.counter + 1 # static counter
        self.value = None
    
    def __repr__(self):
        return "IV{}[{},{}]".format(self.id, self.lb, self.ub)


# subclass numericexpression for operators (first), ndarray for all the rest
class NDVarArray(NumericExpression, np.ndarray):
    # see ndarray docs, best place to add something to constructor
    def __array_finalize__(self, obj):
        self.value = None
        # no need to return anything
    
    #[index] 	  __getitem__(self, index) 	 Index operator
    # TODO: element constraint...
    #def __getitem__(self, index):
    #    elem = super(NDVarArray, self).__getitem__(index)
    #    return elem

    # TODO?
    #in	  __contains__(self, value) 	Check membership
    #object.__matmul__(self, other)
