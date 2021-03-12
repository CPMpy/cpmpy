#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## variables.py
##
"""
    ===============
    List of classes
    ===============
    .. autosummary::
        :nosignatures:

        NumVarImpl
        IntVarImpl
        BoolVarImpl
        NegBoolView
        NDVarArray

    =================
    List of functions
    =================
    .. autosummary::
        :nosignatures:

        BoolVar
        IntVar
        cparray

    ==================
    Module description
    ==================

    This module is used for defining single variables as well as numpy-arrays of variables. There are 2 different types of variables: boolean variables, integer variables.
    
    Boolean Variables
    -----------------

    Boolean variables a.k.a `BoolVar` are variables that have a very specific domain. They take either the value `True` or `False` (1 or 0 respectively).
    The syntax is as follows:

    .. code-block:: python

        BoolVar([shape])

    - *optional* **shape**: integer value larger than 0 or tuple of integer values.

    The following examples show how to create a boolean variable with 3 use cases:

    - the creation of a single (unit-sized or non-vector) boolean variable.
        .. code-block:: python

            # creation of a unit Boolean variable
            x = BoolVar()

    - the creation of a vector boolean variables. 

        .. code-block:: python

            # creation of a vector Boolean variables
            x = BoolVar(3)

            # note that using the python unpacking you can assign them
            # to intermediate variables. THis allows for fine-grained use of variables when
            # defining the constraints of the model
            e,x,a,m,p,l = BoolVar(5)

    - the creation of array/tensor of boolean variables. 
        .. code-block:: python

            # creation of an __array__ of Boolean variables where (3, 8, 7) reflects
            # the dimensions of the tensor, a matrix of multiple-dimensions.
            # In this case, we create an 3D-array of dimensions 3 x 8 x 7.
            array_vars = BoolVar((3, 8, 7))

    Integer Variables
    -----------------

    Integer variables are variables that are given a lower bound and an upper bound, correpsonding to the values that they can take.
    The syntax is as follows:

    .. code-block:: python

        IntVar(lb, ub [, shape])
    
    - **lb**: lower bound
    - **ub**: upper bound
    - *optional* **shape**: integer value larger than 0 or tuple of integer values

    The following examples showcase how to instantiate integer variable with 3 use cases similar to `BoolVar`:

    - Creation of a single (unit-sized or non-vector) integer variable with a given lower bound (**lb**) of 3 and upper bound (**ub**) 8. Variable `x` can thus take values 3, 4, 5, 6, 7, 8 (upper bound included!).

        .. code-block:: python

            # creation of a unit integer variable with lowerbound of 3 and upperbound of 8 
            x = IntVar(3, 8)

    - Creation of a vector integer variables with all having the same given lower bound and upper bound:

        .. code-block:: python

            # creation of a vector Boolean of 5 variables with lowerbound of 3 and upperbound of 8 
            vecx = IntVar(3, 8, 5)

            # Similar `BoolVar`'s python unpacking can assign multiple intermediate variables at once
            e,x,a,m,p,l = IntVar(3, 8, 5)

    - Creation of a 4D-array/tensor (of dimensions 100 x 100 x 100 x 100) of boolean variables.
        .. code-block:: python

            arrx = IntVar(3, 8, (100, 100, 100, 100))

    Array of Variables
    ------------------

    N-dimensional array of cp-variables. Indexing an array with a variable is not allowed by standard numpy arrays, but it is allowed by cpmpy-numpy arrays. 
    First convert your numpy array to a cpmpy-numpy array with the `cparray()` wrapper:

    .. code-block:: python

        # Transforming a given numpy-array **m** into a cparray

        marr = np.array([
            [1, 2, 3, 4],
            [4, 8, 13, 15]
        ])

        m = cparray(marr)

    ==============
    Module details
    ==============
"""

from .utils.exceptions import NullShapeError
import numpy as np
from .expressions import Expression, Operator, is_num

# Helpers for type checking
def is_int(arg):
    return isinstance(arg, (int, np.integer))

def is_var(x):
    return isinstance(x, IntVarImpl)

class NumVarImpl(Expression):
    """
    **Continuous numerical** variable with given lowerbound and upperbound.
    """
    def __init__(self, lb, ub):
        assert (is_num(lb) and is_num(ub))
        assert (lb <= ub)
        self.lb = lb
        self.ub = ub
        self._value = None

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return False

    def value(self):
        return self._value

    # for sets/dicts. Because names are unique, so is the str repr
    def __hash__(self):
        return hash(str(self))

class IntVarImpl(NumVarImpl):
    """
    **Integer** constraint variable with given lowerbound and upperbound.
    """
    counter = 0

    def __init__(self, lb, ub, setname=True):
        assert is_int(lb), "IntVar lowerbound must be integer {} {}".format(type(lb),lb)
        assert is_int(ub), "IntVar upperbound must be integer {} {}".format(type(ub),ub)
        #assert (lb >= 0 and ub >= 0) # can be negative?
        super().__init__(int(lb), int(ub)) # explicit cast: can be numpy
        
        if setname:
            self.name = IntVarImpl.counter
            IntVarImpl.counter = IntVarImpl.counter + 1 # static counter
    
    def __repr__(self):
        return "IV{}".format(self.name)

class BoolVarImpl(IntVarImpl):
    """
    **Boolean** constraint variable with given lowerbound and upperbound.
    """
    counter = 0

    def __init__(self, lb=0, ub=1):
        assert(lb == 0 or lb == 1)
        assert(ub == 0 or ub == 1)
        IntVarImpl.__init__(self, lb, ub, setname=False)
        
        self.name = BoolVarImpl.counter
        BoolVarImpl.counter = BoolVarImpl.counter + 1 # static counter

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return True
        
    def __repr__(self):
        return "BV{}".format(self.name)

    def __invert__(self):
        return NegBoolView(self)

    def __eq__(self, other):
        # (BV == 1) <-> BV
        # if other == 1: XXX: dangerous because "=="" is overloaded 
        if other is 1 or other is True:
            return self
        if other is 0 or other is False:
            return ~self
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
        #assert(isinstance(bv, BoolVarImpl))
        self._bv = bv

    def value(self):
        return not self._bv.value()

    def __repr__(self):
        return "~BV{}".format(self._bv.name)

    def __invert__(self):
        return self._bv


# subclass numericexpression for operators (first), ndarray for all the rest
class NDVarArray(Expression, np.ndarray):
    """
    N-dimensional numpy array of variables.
    """
    def __init__(self, shape, **kwargs):
        # TODO: global name?
        # this is nice and sneaky, 'self' is the list_of_arguments!
        Expression.__init__(self, "NDVarArray", self)
        # somehow, no need to call ndarray constructor

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return False

    def value(self):
        return np.reshape([x.value() for x in self], self.shape)
    
    def __getitem__(self, index):
        from .globalconstraints import Element # here to avoid circular
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


# N-dimensional array of Boolean Decision Variables
def BoolVar(shape=None):
    """
    # N-dimensional array of Boolean Decision Variables
    """
    if shape is None or shape == 1:
        return BoolVarImpl()
    elif shape == 0:
        raise NullShapeError(shape)
    length = np.prod(shape)
    
    # create base data
    data = np.array([BoolVarImpl() for _ in range(length)]) # repeat new instances
    # insert into custom ndarray
    return NDVarArray(shape, dtype=object, buffer=data)


def IntVar(lb, ub, shape=None):
    """
    N-dimensional array of Integer Decision Variables with lower-bound `lb` and upper-bound `ub`
    """
    if shape is None or shape == 1:
        return IntVarImpl(lb,ub)
    elif shape == 0:
        raise NullShapeError(shape)
    length = np.prod(shape)
    
    # create base data
    data = np.array([IntVarImpl(lb,ub) for _ in range(length)]) # repeat new instances
    # insert into custom ndarray
    return NDVarArray(shape, dtype=object, buffer=data)


def cparray(arr):
    """
    N-dimensional wrapper, wraps a standard array.

    So that we can do [1,2,3,4][var1] == var2, e.g. element([1,2,3,4],var1,var2)
    needed because standard arrays can not be indexed by non-constants
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return NDVarArray(shape=arr.shape, dtype=type(arr.flat[0]), buffer=arr)
