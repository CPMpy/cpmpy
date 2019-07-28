import numpy as np
from .expressions import *

# Helpers for type checking
def is_int(arg):
    return isinstance(arg, (int, np.integer))

class NumVarImpl(Expression):
    def __init__(self, lb, ub):
        assert (is_num(lb) and is_num(ub))
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
        assert (is_int(lb) and is_int(ub))
        assert (lb >= 0 and ub >= 0)
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

    def __eq__(self, other):
        # (BV == 1) <-> BV
        if other == 1:
            return self
        return super().__eq__(other)

    # when redefining __eq__, must redefine custom__hash__
    # https://stackoverflow.com/questions/53518981/inheritance-hash-sets-to-none-in-a-subclass
    def __hash__(self): return super().__hash__()


# subclass numericexpression for operators (first), ndarray for all the rest
class NDVarArray(Expression, np.ndarray):
    def __init__(self, shape, **kwargs):
        # TODO: global name?
        # this is nice and sneaky, 'self' is the list_of_arguments!
        Expression.__init__(self, "NDVarArray", self)
        # somehow, no need to call ndarray constructor

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
