import numpy as np
from .expressions import *

class NumVarImpl(Expression):
    def __init__(self, lb, ub):
        assert (lb <= ub)
        self.lb = lb
        self.ub = ub
        self._value = None

    def value(self):
        return self._value

    # for sets/dicts. Because IDs are unique, so is the str repr
    def __hash__(self):
        return hash(str(self))

class IntVarImpl(NumVarImpl):
    counter = 0

    def __init__(self, lb, ub, setid=True):
        assert (lb >= 0 and ub >= 0)
        assert (isinstance(lb, int) and isinstance(ub, int))
        super().__init__(lb, ub)
        
        self.id = IntVarImpl.counter
        IntVarImpl.counter = IntVarImpl.counter + 1 # static counter
    
    def __repr__(self):
        return "IV{}".format(self.id)

class BoolVarImpl(IntVarImpl):
    counter = 0

    def __init__(self, lb=0, ub=1):
        assert(lb == 0 or lb == 1)
        assert(ub == 0 or ub == 1)
        super().__init__(lb, ub, setid=False)
        
        self.id = BoolVarImpl.counter
        BoolVarImpl.counter = BoolVarImpl.counter + 1 # static counter
        
    def __repr__(self):
        return "BV{}".format(self.id)



# subclass numericexpression for operators (first), ndarray for all the rest
class NDVarArray(Expression, np.ndarray):
    def value(self):
        return np.reshape([x.value() for x in self], self.shape)
    
    def __getitem__(self, index):
        def is_var(x):
            return isinstance(x, IntVarImpl)

        if is_var(index):
            return GlobalConstraint("element", [self, index], add_equality_as_arg=True)

        if isinstance(index, tuple) and any(is_var(el) for el in index):
            index_rest = list(index)
            var = []
            for i in range(len(index)):
                if is_var(index[i]):
                    index_rest[i] = None
                    var.append(index[i])
            array_rest = self[tuple(index_rest)]
            assert (len(var)==1), "variable indexing (element) only supported with 1 variable at this moment"
            return GlobalConstraint("element", [array_rest, var[0]], add_equality_as_arg=True)
            
        return super().__getitem__(index)

    # TODO?
    #in	  __contains__(self, value) 	Check membership
    #object.__matmul__(self, other)
