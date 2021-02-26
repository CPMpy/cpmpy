import numpy as np
from .expressions import *

# Helpers for type checking
def is_int(arg):
    return isinstance(arg, (int, np.integer))

def is_var(x):
    return isinstance(x, IntVarImpl)

class NumVarImpl(Expression):
    def __init__(self, lb, ub):
        assert (is_num(lb) and is_num(ub))
        assert (lb <= ub)
        self.lb = lb
        self.ub = ub
        self._value = None

    def value(self):
        return self._value

    # for sets/dicts. Because names are unique, so is the str repr
    def __hash__(self):
        return hash(str(self))

class IntVarImpl(NumVarImpl):
    counter = 0

    def __init__(self, lb, ub, setname=True):
        assert (is_int(lb) and is_int(ub))
        assert (lb >= 0 and ub >= 0)
        super().__init__(lb, ub)
        
        self.name = IntVarImpl.counter
        IntVarImpl.counter = IntVarImpl.counter + 1 # static counter
    
    def __repr__(self):
        return "IV{}".format(self.name)

class BoolVarImpl(IntVarImpl):
    counter = 0

    def __init__(self, lb=0, ub=1):
        assert(lb == 0 or lb == 1)
        assert(ub == 0 or ub == 1)
        IntVarImpl.__init__(self, lb, ub, setname=False)
        
        self.name = BoolVarImpl.counter
        BoolVarImpl.counter = BoolVarImpl.counter + 1 # static counter
        
    def __repr__(self):
        return "BV{}".format(self.name)

    def __invert__(self):
        return NegBoolView(self)

    def __eq__(self, other):
        # (BV == 1) <-> BV
        # if other == 1:
        # XXX: dangerous!
        # "=="" is overloaded 
        if other is 1 or other is True:
            return self
        return super().__eq__(other)

    # when redefining __eq__, must redefine custom__hash__
    # https://stackoverflow.com/questions/53518981/inheritance-hash-sets-to-none-in-a-subclass
    def __hash__(self): return super().__hash__()

class NegBoolView(BoolVarImpl):
    """
        Represents not(`var`), not an actual variable implementation!

        It stores a link to `var`'s BoolVarImpl
    """
    def __init__(self, bv):
        assert(isinstance(bv, BoolVarImpl))
        self._bv = bv

    def value(self):
        return not self._bv.value()

    def __repr__(self):
        return "~BV{}".format(self._bv.name)

    def __invert__(self):
        return self._bv


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
        # array access, check if variables are used in the indexing

        # index is single variable: direct element
        if is_var(index):
            return Element([self, index])

        # index is array/tuple with at least one var in it:
        # index non-var part, and create element on var part
        if isinstance(index, tuple) and any(is_var(el) for el in index):
            index_rest = list(index) # mutable view
            var = [] # collector of variables
            for i in range(len(index)):
                if is_var(index[i]):
                    index_rest[i] = Ellipsis # selects all remaining dimensions
                    var.append(index[i])
            array_rest = self[tuple(index_rest)] # non-var array selection
            assert (len(var)==1), "variable indexing (element) only supported with 1 variable at this moment"
            # single var, so flatten rest array
            return Element([array_rest, var[0]])
            
        return super().__getitem__(index)

    def sum(self, axis=None, out=None):
        """
            overwrite np.sum(NDVarArray) as people might use it

            does not actually support axis/out... todo?
        """
        if not axis is None or not out is None:
            raise NotImplementedError() # please report on github with usecase

        # return sum object
        return Operator("sum", self)

    # TODO?
    #in	  __contains__(self, value) 	Check membership
    #object.__matmul__(self, other)
