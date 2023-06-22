#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## variables.py
##
"""
    Integer and Boolean decision variables (as n-dimensional numpy objects)

    =================
    List of functions
    =================
    .. autosummary::
        :nosignatures:

        boolvar
        intvar
        cpm_array

    ==================
    Module description
    ==================

    A decision variable is a variable whose value will be determined by the solver.

    Boolean and Integer decision variables are the key elements of a CP model.

    All variables in CPMpy are n-dimensional array objects and have defined dimensions. Following the numpy library, the dimension sizes of an n-dimenionsal array is called its __shape__. In CPMpy all variables are considered an array with a given shape. For 'single' variables the shape is '1'. For an array of length `n` the shape is 'n'. An `n*m` matrix has shape (n,m), and tensors with more than 2 dimensions are all supported too. For the implementation of this, CPMpy builts on numpy's n-dimensional ndarray and inherits many of its benefits (vectorized operators and advanced indexing).

    This module contains the cornerstone `boolvar()` and `intvar()` functions, which create (numpy arrays of) variables. There is also a helper function `cpm_array` for wrapping standard numpy arrays so they can be indexed by a variable. Apart from these 3 functions, none of the classes in this module should be directly created; they are created by these 3 helper functions.


    ===============
    List of classes
    ===============
    .. autosummary::
        :nosignatures:

        NullShapeError
        _NumVarImpl
        _IntVarImpl
        _BoolVarImpl
        NegBoolView
        NDVarArray

    ==============
    Module details
    ==============
"""
import math
from collections.abc import Iterable
import warnings # for deprecation warning
from functools import reduce

import numpy as np
from .core import Expression, Operator
from .utils import is_num, is_int, flatlist, is_boolexpr, is_true_cst, is_false_cst


def BoolVar(shape=1, name=None):
    warnings.warn("Deprecated, use boolvar() instead, will be removed in stable version", DeprecationWarning)
    return boolvar(shape=shape, name=name)


def boolvar(shape=1, name=None):
    """
    Boolean decision variables will take either the value `True` or `False`.
    
    Arguments:
    shape -- the shape of the n-dimensional array of variables (int or tuple of ints, default: 1)
    name -- name to give to the variables (string, default: None)

    If name is None then a name 'BV<unique number>' will be assigned to it.

    If shape is different from 1, then each element of the array will have the location
    of this specific variable in the array append to its name.

    For example, `print(boolvar(shape=3, name="x"))` will print `[x[0], x[1], x[2]]`


    The following examples show how to create Boolean variables of different shapes:

    - Creating a single (unit-sized or scalar) Boolean variable:
        .. code-block:: python

            # creation of a unit Boolean variable
            x = boolvar(name="x")

    - the creation of a vector boolean variables. 

        .. code-block:: python

            # creation of a vector of size 3 of Boolean variables
            x = boolvar(shape=3, name="x")

            # note that with Python's unpacking, you can assign them
            # to intermediate variables. This allows for fine-grained use of variables when
            # defining the constraints of the model
            e, x, a, m, p, l = boolvar(shape=6)

    - the creation of a matrix or higher-order tensor of Boolean variables. 
        .. code-block:: python

            # creation of a 9x9 matrix of Boolean variables:
            matrix = boolvar(shape=(9, 9), name="matrix")

            # creation of a __tensor of Boolean variables where (3, 8, 7) reflects
            # the dimensions of the tensor, a matrix of multiple-dimensions.
            # In this case, we create an 3D-array of dimensions 3 x 8 x 7.
            tensor = BoolVar(shape=(3, 8, 7), name="tensor")
    """
    if shape == 0 or shape is None:
        raise NullShapeError(shape)
    if shape == 1:
        return _BoolVarImpl(name=name)

    # create base data
    data = np.array([_BoolVarImpl(name=_genname(name, idxs)) for idxs in np.ndindex(shape)]) # repeat new instances
    # insert into custom ndarray
    return NDVarArray(shape, dtype=object, buffer=data)


def IntVar(lb, ub, shape=1, name=None):
    warnings.warn("Deprecated, use intvar() instead, will be removed in stable version", DeprecationWarning)
    return intvar(lb, ub, shape=shape, name=name)


def intvar(lb, ub, shape=1, name=None):
    """
    Integer decision variables are constructed by specifying the lowest (lb)
    the decision variable can take, as well as the highest value (ub).

    Arguments:
    lb -- lower bound on the values the variable can take (int)
    ub -- upper bound on the values the variable can take (int)
    shape -- the shape of the n-dimensional array of variables (int or tuple of ints, default: 1)
    name -- name to give to the variables (string, default: None)

    The range of values between lb..ub is called the __domain__ of the integer variable.
    All variables in an array start from the same domain.
    Specific values in the domain of individual variables can be forbidden with constraints.

    If name is None then a name 'IV<unique number>' will be assigned to it.

    If shape is different from 1, then each element of the array will have the location
    of this specific variable in the array append to its name.

    The following examples show how to create integer variables of different shapes:

    - Creation of a single (unit-sized or scalar) integer variable with a given lower bound (**lb**) of 3 and upper bound (**ub**) 8. Variable `x` can thus take values 3, 4, 5, 6, 7, 8 (upper bound included!).

        .. code-block:: python

            # creation of a unit integer variable with lowerbound of 3 and upperbound of 8 
            x = intvar(3, 8, name="x")

    - Creation of a vector of integer variables with all having the same given lower bound and upper bound:

        .. code-block:: python

            # creation of a vector Boolean of 5 variables with lowerbound of 3 and upperbound of 8 
            x = intvar(3, 8, shape=5, name="x")

            # Python's unpacking can assign multiple intermediate variables at once
            e, x, a, m, p, l = intvar(3, 8, shape=5)

    - Creation of a 4D-array/tensor (of dimensions 100 x 100 x 100 x 100) of integer variables.
        .. code-block:: python

            arrx = intvar(3, 8, shape=(100, 100, 100, 100), name="arrx")

    """
    if shape == 0 or shape is None:
        raise NullShapeError(shape)
    if shape == 1:
        return _IntVarImpl(lb, ub, name=name)

    # create base data
    data = np.array([_IntVarImpl(lb, ub, name=_genname(name, idxs)) for idxs in np.ndindex(shape)]) # repeat new instances
    # insert into custom ndarray
    return NDVarArray(shape, dtype=object, buffer=data)


def cparray(arr):
    warnings.warn("Deprecated, use cpm_array() instead, will be removed in stable version", DeprecationWarning)
    return cpm_array(arr)


def cpm_array(arr):
    """
    N-dimensional wrapper, to wrap standard numpy arrays or lists.

    In CP modeling languages, indexing an array by an integer variable is common, e.g. `[1,2,3,4][var1] == var2`.
    This is called an __element__ constraint. Python does not allow expressing it on standard arrays,
    but CPMpy-numpy arrays do allow it, so you first have to wrap the array.

    Note that 'arr' will be transformed to vector and indexed as such, 2-dimensional indexing is not supported (yet?).

    .. code-block:: python

        # Transforming a given numpy-array **m** into a cparray

        iv1, iv2 = intvar(0, 9, shape=2)

        data = [1, 2, 3, 4]
        data = cpm_array(data)

        Model([ data[iv1] == iv2 ])

    As an alternative, you can also write the `Element` constraint directly on `data`: `Element(data, iv1) == iv2`
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return NDVarArray(shape=arr.shape, dtype=arr.dtype, buffer=arr)


class NullShapeError(Exception):
    """
    Error returned when providing an empty or size 0 shape for numpy arrays of variables
    """
    def __init__(self, shape, message="Shape should be non-zero"):
        self.shape = shape
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f'{self.shape}: {self.message}'


class _NumVarImpl(Expression):
    """
    Abstract **continuous numerical** variable with given lowerbound and upperbound.

    Abstract class, only mean to be subclassed
    """
    def __init__(self, lb, ub, name):
        assert (is_num(lb) and is_num(ub))
        assert (lb <= ub)
        self.lb = lb
        self.ub = ub
        self.name = name
        self._value = None

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return False

    def value(self):
        """ the value obtained in the last solve call
            (or 'None')
        """
        return self._value

    def get_bounds(self):
        """ the lower and upper bounds"""
        return self.lb, self.ub

    def clear(self):
        """ clear the value obtained from the last solve call
        """
        self._value = None

    def __repr__(self):
        return self.name

    # for sets/dicts. Because names are unique, so is the str repr
    def __hash__(self):
        return hash(self.name)


class _IntVarImpl(_NumVarImpl):
    """
    **Integer** variable with given lowerbound and upperbound.

    Do not create this object directly, use `intvar()` instead
    """
    counter = 0

    def __init__(self, lb, ub, name=None):
        assert is_int(lb), "IntVar lowerbound must be integer {} {}".format(type(lb), lb)
        assert is_int(ub), "IntVar upperbound must be integer {} {}".format(type(ub), ub)

        if name is None:
            name = "IV{}".format(_IntVarImpl.counter)
            _IntVarImpl.counter = _IntVarImpl.counter + 1 # static counter

        super().__init__(int(lb), int(ub), name=name) # explicit cast: can be numpy

    # special casing for intvars (and boolvars)
    def __abs__(self):
        if self.lb >= 0:
            # no-op
            return self
        return super().__abs__()


class _BoolVarImpl(_IntVarImpl):
    """
    **Boolean** variable with given lowerbound and upperbound.

    Do not create this object directly, use `boolvar()` instead
    """
    counter = 0

    def __init__(self, lb=0, ub=1, name=None):
        assert(lb == 0 or lb == 1)
        assert(ub == 0 or ub == 1)

        if name is None:
            name = "BV{}".format(_BoolVarImpl.counter)
            _BoolVarImpl.counter = _BoolVarImpl.counter + 1 # static counter
        _IntVarImpl.__init__(self, lb, ub, name=name)

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return True

    def __invert__(self):
        return NegBoolView(self)

    def __abs__(self):
        return self

    # when redefining __eq__, must redefine custom__hash__
    # https://stackoverflow.com/questions/53518981/inheritance-hash-sets-to-none-in-a-subclass
    def __hash__(self):
        return hash(self.name)


class NegBoolView(_BoolVarImpl):
    """
        Represents not(`var`), not an actual variable implementation!

        It stores a link to `var`'s _BoolVarImpl

        Do not create this object directly, use the `~` operator instead: `~bv`
    """
    def __init__(self, bv):
        #assert(isinstance(bv, _BoolVarImpl))
        self._bv = bv
        # as it is always created using the ~ operator (only available for _BoolVarImpl)
        # it already comply with the asserts of the __init__ of _BoolVarImpl and can use 
        # __init__ from _IntVarImpl
        _IntVarImpl.__init__(self, 1-bv.ub, 1-bv.lb, name=str(self))

    def value(self):
        """ the negation of the value obtained in the last solve call by the viewed variable
            (or 'None')
        """
        v = self._bv.value()
        if v is None:
            return None
        return not v

    def clear(self):
        """ clear, for the viewed variable, the value obtained from the last solve call
        """
        self._bv.clear()

    def __repr__(self):
        return "~{}".format(self._bv.name)

    def __invert__(self):
        return self._bv


# subclass numericexpression for operators (first), ndarray for all the rest
class NDVarArray(Expression, np.ndarray):
    """
    N-dimensional numpy array of variables.

    Do not create this object directly, use one of the functions in this module
    """
    def __init__(self, shape, **kwargs):
        # TODO: global name?
        # this is nice and sneaky, 'self' is the list_of_arguments!
        Expression.__init__(self, "NDVarArray", self)
        # no need to call ndarray __init__ method as specified in the np.ndarray documentation:
        # "No ``__init__`` method is needed because the array is fully initialized
        #         after the ``__new__`` method."

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return False

    def value(self):
        """ the values, for each of the stored variables, obtained in the last solve call
            (or 'None')
        """
        return np.reshape([x.value() for x in self], self.shape)

    def clear(self):
        """ clear, for each of the stored variables, the value obtained from the last solve call
        """
        for e in self.flat:
            e.clear()

    def __repr__(self):
        """
            some ways in which np creates this object does not call
            the constructor, so the Expression does not have 'args'
            set..
        """
        if not hasattr(self, "args"):
            self.name = "NDVarArray"
            self.args = self
        return super().__repr__()

    def __getitem__(self, index):
        from .globalfunctions import Element # here to avoid circular
        # array access, check if variables are used in the indexing

        # index is single expression: direct element
        if isinstance(index, Expression):
            return Element(self, index)

        # multi-dimensional index
        if isinstance(index, tuple) and any(isinstance(el, Expression) for el in index):
            # find dimension of expression in index
            expr_dim = next(dim for dim,idx in enumerate(index) if isinstance(idx, Expression))
            arr = self[tuple(index[:expr_dim])] # select remaining dimensions
            index = index[expr_dim:]

            # calculate index for flat array
            flat_index = index[-1]
            for dim, idx in enumerate(index[:-1]):
                flat_index += idx * math.prod(arr.shape[dim+1:])
            # using index expression as single var for flat array
            return Element(arr.flatten(), flat_index)

        ret = super().__getitem__(index)
        # this is a bit ugly,
        # but np.int and np.bool do not play well with > overloading
        if isinstance(ret, np.integer):
            return int(ret)
        elif isinstance(ret, np.bool_):
            return bool(ret)
        return ret

    """
    make the given array the first dimension in the returned array
    """
    def __axis(self, axis):

        arr = self

        # correct type and value checks
        if not isinstance(axis,int):
            raise TypeError("Axis keyword argument in .sum() should always be an integer")
        if axis >= arr.ndim:
            raise ValueError("Axis out of range")

        if axis < 0:
            axis += arr.ndim

        # Change the array to make the selected axis the first dimension
        if axis > 0:
            iter_axis = list(range(arr.ndim))
            iter_axis.remove(axis)
            iter_axis.insert(0, axis)
            arr = arr.transpose(iter_axis)

        return arr

    def sum(self, axis=None, out=None):
        """
            overwrite np.sum(NDVarArray) as people might use it
        """
        if out is not None:
            raise NotImplementedError()

        if axis is None:    # simple case where we want the sum over the whole array
            arr = self.flatten()
            return Operator("sum", arr)

        arr = self.__axis(axis=axis)

        out = []
        for i in range(0, arr.shape[0]):
            out.append(Operator("sum", arr[i, ...]))

        # return the NDVarArray that contains the sum constraints
        return out


    def prod(self, axis=None, out=None):
        """
            overwrite np.prod(NDVarArray) as people might use it
        """
        if out is not None:
            raise NotImplementedError()

        if axis is None:    # simple case where we want the product over the whole array
            arr = self.flatten()
            return reduce(lambda a, b: a * b, arr)

        arr = self.__axis(axis=axis)

        out = []
        for i in range(0, arr.shape[0]):
            out.append(reduce(lambda a, b: a * b, arr[i, ...]))

        # return the NDVarArray that contains the sum constraints
        return out

    def max(self, axis=None, out=None):
        """
            overwrite np.max(NDVarArray) as people might use it
        """
        from .globalfunctions import Maximum
        if out is not None:
            raise NotImplementedError()

        if axis is None:    # simple case where we want the maximum over the whole array
            arr = self.flatten()
            return Maximum(arr)

        arr = self.__axis(axis=axis)

        out = []
        for i in range(0, arr.shape[0]):
            out.append(Maximum(arr[i, ...]))

        # return the NDVarArray that contains the Maximum global constraints
        return out

    def min(self, axis=None, out=None):
        """
            overwrite np.min(NDVarArray) as people might use it
        """
        from .globalfunctions import Minimum
        if out is not None:
            raise NotImplementedError()

        if axis is None:    # simple case where we want the Minimum over the whole array
            arr = self.flatten()
            return Minimum(arr)

        arr = self.__axis(axis=axis)

        out = []
        for i in range(0, arr.shape[0]):
            out.append(Minimum(arr[i, ...]))

        # return the NDVarArray that contains the Minimum global constraints
        return out

    def any(self, axis=None, out=None):
        """
            overwrite np.any(NDVarArray)
        """
        from .python_builtins import any

        if any(not is_boolexpr(x) for x in self.flatten()):
            raise TypeError("Cannot call .any() in an array not consisting only of bools")

        if out is not None:
            raise NotImplementedError()

        if axis is None:    # simple case where we want the .any() over the whole array
            arr = self.flatten()
            return any(arr)

        arr = self.__axis(axis=axis)

        out = []
        for i in range(0, arr.shape[0]):
            out.append(any(arr[i, ...]))

        # return the NDVarArray that contains the any() constraints
        return out

    def all(self, axis=None, out=None):
        """
            overwrite np.any(NDVarArray)
        """

        from .python_builtins import all
        if out is not None:
            raise NotImplementedError()

        if axis is None:    # simple case where we want the .all() over the whole array
            arr = self.flatten()
            return all(arr)

        arr = self.__axis(axis=axis)

        out = []
        for i in range(0, arr.shape[0]):
            out.append(all(arr[i, ...]))

        # return the NDVarArray that contains the all() constraints
        return out

    # VECTORIZED master function (delegate)
    def _vectorized(self, other, attr):
        if not isinstance(other, Iterable):
            other = [other]*len(self)
        # this is a bit cryptic, but it calls 'attr' on s with o as arg
        # s.__eq__(o) <-> getattr(s, '__eq__')(o)
        return cpm_array([getattr(s,attr)(o) for s,o in zip(self, other)])

    # VECTORIZED comparisons
    def __eq__(self, other):
        return self._vectorized(other, '__eq__')

    def __ne__(self, other):
        return self._vectorized(other, '__ne__')

    def __lt__(self, other):
        return self._vectorized(other, '__lt__')

    def __le__(self, other):
        return self._vectorized(other, '__le__')

    def __gt__(self, other):
        return self._vectorized(other, '__gt__')

    def __ge__(self, other):
        return self._vectorized(other, '__ge__')

    # VECTORIZED math operators
    # only 'abs' 'neg' and binary ones
    # '~' not needed, gets translated to ==0 and that is already handled
    def __abs__(self):
        return cpm_array([abs(s) for s in self])

    def __neg__(self):
        return cpm_array([-s for s in self])

    def __add__(self, other):
        return self._vectorized(other, '__add__')

    def __radd__(self, other):
        return self._vectorized(other, '__radd__')

    def __sub__(self, other):
        return self._vectorized(other, '__sub__')

    def __rsub__(self, other):
        return self._vectorized(other, '__rsub__')

    def __mul__(self, other):
        return self._vectorized(other, '__mul__')

    def __rmul__(self, other):
        return self._vectorized(other, '__rmul__')

    def __truediv__(self, other):
        return self._vectorized(other, '__truediv__')

    def __rtruediv__(self, other):
        return self._vectorized(other, '__rtruediv__')

    def __floordiv__(self, other):
        return self._vectorized(other, '__floordiv__')

    def __rfloordiv__(self, other):
        return self._vectorized(other, '__rfloordiv__')

    def __mod__(self, other):
        return self._vectorized(other, '__mod__')

    def __rmod__(self, other):
        return self._vectorized(other, '__rmod__')

    def __pow__(self, other, modulo=None):
        assert (modulo is None), "Power operator: modulo not supported"
        return self._vectorized(other, '__pow__')

    def __rpow__(self, other, modulo=None):
        assert (modulo is None), "Power operator: modulo not supported"
        return self._vectorized(other, '__rpow__')

    # VECTORIZED Bool operators
    def __invert__(self):
        return cpm_array([~s for s in self])

    def __and__(self, other):
        return self._vectorized(other, '__and__')

    def __rand__(self, other):
        return self._vectorized(other, '__rand__')

    def __or__(self, other):
        return self._vectorized(other, '__or__')

    def __ror__(self, other):
        return self._vectorized(other, '__ror__')

    def __xor__(self, other):
        return self._vectorized(other, '__xor__')

    def __rxor__(self, other):
        return self._vectorized(other, '__rxor__')

    def implies(self, other):
        return self._vectorized(other, 'implies')

    #in	  __contains__(self, value) 	Check membership
    # CANNOT meaningfully overwrite, python always returns True/False
    # regardless of what you return in the __contains__ function

    # TODO?
    #object.__matmul__(self, other)


def _genname(basename, idxs):
    """
    Helper function to 'name' array variables
    - idxs: list of indices, one for every dimension of the array
    - basename: base name to prepend

    if basename is 'None', then it returns None

    output: something like "basename[0,1]"
    """
    if basename == None:
        return None
    stridxs = ",".join(map(str, idxs))
    return f"{basename}[{stridxs}]" # "<name>[<idx0>,<idx1>,...]"

