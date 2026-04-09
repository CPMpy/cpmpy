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

    All variables in CPMpy are n-dimensional array objects and have defined dimensions. 
    Following the numpy library, the dimension sizes of an n-dimenionsal array is called its ``shape``. 
    In CPMpy all variables are considered an array with a given shape. For 'single' variables the shape 
    is '1'. For an array of length `n` the shape is 'n'. An `n*m` matrix has shape (n,m), and tensors 
    with more than 2 dimensions are all supported too. For the implementation of this, 
    CPMpy builts on numpy's n-dimensional `ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_ 
    and inherits many of its benefits (vectorized operators and advanced indexing).

    This module contains the cornerstone ``boolvar()`` and ``intvar()`` functions, which create (numpy arrays of) 
    variables. There is also a helper function ``cpm_array()`` for wrapping standard numpy arrays so they can be 
    indexed by a variable. Apart from these 3 functions, none of the classes in this module should be directly 
    instantiated; they are created by these 3 helper functions.


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
from __future__ import annotations

import math
from collections.abc import Iterable
import warnings # for deprecation warning
from functools import reduce
from typing import Any, Literal, Optional, overload

import numpy as np
import cpmpy as cp  # to avoid circular import
from .core import Expression, ExprLike, ListLike, BoolVal
from .utils import is_num, is_int, is_boolexpr, get_bounds

_BV_PREFIX = "BV"
_IV_PREFIX = "IV"
_VAR_ERR  = f"Variable names starting with {_IV_PREFIX} or {_BV_PREFIX} are reserved for internal use only, chose a different name"

def BoolVar(shape=1, name=None):
    """
    .. deprecated:: 0.9.0
          Please use :func:`~cpmpy.expressions.variables.boolvar` instead.
    """
    warnings.warn("Deprecated, use boolvar() instead, will be removed in stable version", DeprecationWarning)
    return boolvar(shape=shape, name=name)


@overload
def boolvar(shape: Literal[1] = 1,  # special case: a shape of =1 returns a single variable
            name: Optional[str] = None) -> _BoolVarImpl: ...  # implementation below
@overload
def boolvar(shape: int|np.integer|tuple[int|np.integer, ...] = 1,
            name: Optional[str|ListLike[str]] = None) -> NDVarArray: ...  # implementation below

def boolvar(shape: int|np.integer|tuple[int|np.integer, ...] = 1,
            name: Optional[str|ListLike[str]] = None) -> _BoolVarImpl|NDVarArray:  # the joint implementation
    """
    Create Boolean decision variables that take either the value `True` or `False`.

    Arguments:
        shape (int or tuple of int, optional) : The shape of the n-dimensional array of variables. Default is 1.
    
        name (str, list of str, tuple of str, or None, optional) : 
            Name(s) to assign to the variables. Default is None.

            - If `name` is None, a name of the form ``BV<unique number>`` will be assigned to the variables.
            - If `name` is a string, it will be used as the suffix of the variable names.
            - If `name` is a list/tuple/array of strings, it must match the shape of the variables,
                             and they will be assigned to the variable names accordingly.

    Notes:

        - If `shape` is not 1, each element of the array will have its specific location appended to its name.
        - Examples:
            - ``boolvar(shape=3, name="x")`` will create the variables ``[x[0], x[1], x[2]]``.
            - ``boolvar(shape=3, name=list("xyz"))`` will create the variables ``[x, y, z]``.

    Examples:

        Creating a single Boolean variable:
        
        .. code-block:: python

            x = boolvar()              # auto name BV<n> (unique counter)
            x = boolvar(name="x")      # user-chosen name
        
        Creating a vector of Boolean variables:
        
        .. code-block:: python

            x = boolvar(shape=3, name="x")

            # Assigning them to individual variables:
            e, x, a, m, p, l = boolvar(shape=6, name=list("exampl"))

        Creating a matrix or higher-order tensor of Boolean variables:
        
        .. code-block:: python

            matrix = boolvar(shape=(9, 9), name="matrix")
            matrix2 = boolvar(shape=(2, 2), name=[['a', 'b'], ['c', 'd']])
            tensor = boolvar(shape=(3, 8, 7), name="tensor")
    """
    if shape is None or shape == 0:
        raise NullShapeError(shape)
    if shape == 1:
        # special case: a shape of =1 returns a single variable
        if name is not None:
            assert isinstance(name, str), f"name must be a string, got {name}"
            if _is_invalid_name(name):
                raise ValueError(_VAR_ERR)
        return _BoolVarImpl(name=name)

    # collect the `names` of each individual decision variable
    names = _gen_var_names(name, shape)

    # create np.array 'data' representation of the decision variables
    data = np.array([_BoolVarImpl(name=n) for n in names])
    # insert into custom ndarray
    r = NDVarArray(shape, dtype=object, buffer=data)
    r._has_subexpr = False # A bit ugly (acces to private field) but otherwise np.ndarray constructor complains if we pass it as an argument to NDVarArray
    return r


def IntVar(lb, ub, shape=1, name=None):
    """
    .. deprecated:: 0.9.0
          Please use :func:`~cpmpy.expressions.variables.intvar` instead.
    """
    warnings.warn("Deprecated, use intvar() instead, will be removed in stable version", DeprecationWarning)
    return intvar(lb, ub, shape=shape, name=name)


@overload
def intvar(lb: int, ub: int, shape: Literal[1] = 1,  # special case: a shape of =1 returns a single variable
           name: Optional[str] = None) -> _IntVarImpl: ...  # implementation below
@overload
def intvar(lb: int, ub: int, shape: int|np.integer|tuple[int|np.integer, ...] = 1,
           name: Optional[str|ListLike[str]] = None) -> NDVarArray: ...  # implementation below

def intvar(lb: int, ub: int, shape: int|np.integer|tuple[int|np.integer, ...] = 1,
           name: Optional[str|ListLike[str]] = None) -> _IntVarImpl|NDVarArray:  # the joint implementation
    """
    Integer decision variables are constructed by specifying the lowest (lb) value
    the decision variable can take, as well as the highest value (ub).

    Arguments:
        lb (int) : Lower bound on the values the variable can take.
        ub (int) : Upper bound on the values the variable can take.
        shape (int or tuple of int, optional) :
            The shape of the n-dimensional array of variables. Default is 1.
        name (str, list of str, tuple of str, or None, optional) :
            Name(s) to assign to the variables. Default is None.

            - If `name` is None, a name of the form ``IV<unique number>`` will be assigned to the variables.
            - If `name` is a string, it will be used as the suffix of the variable names.
            - If `name` is a list/tuple/array of strings, they will be assigned to the variable names accordingly.

    Notes:

        The range of values between ``lb..ub`` is called the `domain` of the integer variable.
        All variables in an array start from the same domain.
        Specific values in the domain of individual variables can be forbidden with constraints.

        If `shape` is not 1, each element of the array will have its specific location appended to its name.

        - ``intvar(0, 2, shape=3, name="x")`` will create the variables ``[x[0], x[1], x[2]]``.

        If `shape` is not 1 and a list of names has been provided (with the same length as the array), each decision variable will receive its respective name.

        - ``intvar(0, 2, shape=3, name=list("xyz"))`` will create the variables ``[x, y, z]``.

    Examples:

        Creation of a single (unit-sized or scalar) integer variable with a given lower bound (**lb**) of 3 and upper bound (**ub**) of 8. Variable `x` can thus take values 3, 4, 5, 6, 7, 8 (upper bound included!).

        .. code-block:: python

            # creation of a unit integer variable with lowerbound of 3 and upperbound of 8 
            x = intvar(3, 8, name="x")

        Creation of a vector of integer variables with all having the same given lower bound and upper bound:

        .. code-block:: python

            # creation of a vector Boolean of 5 variables with lowerbound of 3 and upperbound of 8 
            x = intvar(3, 8, shape=5, name="x")

            # Python's unpacking can assign multiple intermediate variables at once
            e, x, a, m, p, l = intvar(3, 8, shape=6, name=list("exampl"))

        Creation of a 4D-array/tensor (of dimensions 100 x 100 x 100 x 100) of integer variables.
        
        .. code-block:: python

            arrx s= intvar(3, 8, shape=(100, 100, 100, 100), name="arrx")

    """
    if shape is None or shape == 0:
        raise NullShapeError(shape)
    if shape == 1:
        # special case: a shape of =1 returns a single variable
        if name is not None:
            assert isinstance(name, str), f"name must be a string, got {name}"
            if _is_invalid_name(name):
                raise ValueError(_VAR_ERR)
        return _IntVarImpl(lb, ub, name=name)

    # collect the `names` of each individual decision variable
    names = _gen_var_names(name, shape)

    # create np.array 'data' representation of the decision variables
    data = np.array([_IntVarImpl(lb, ub, name=n) for n in names]) # repeat new instances
    # insert into custom ndarray
    r = NDVarArray(shape, dtype=object, buffer=data)
    r._has_subexpr = False # A bit ugly (acces to private field) but otherwise np.ndarray constructor complains if we pass it as an argument to NDVarArray
    return r

def cparray(arr):
    """
    .. deprecated:: 0.9.0
          Please use :func:`~cpmpy.expressions.variables.cpm_array` instead.
    """
    warnings.warn("Deprecated, use cpm_array() instead, will be removed in stable version", DeprecationWarning)
    return cpm_array(arr)


def cpm_array(arr: ListLike[ExprLike]) -> NDVarArray:
    """
    N-dimensional wrapper, to wrap standard numpy arrays or lists.

    In CP modeling languages, indexing an array by an integer variable is common, e.g. `[1,2,3,4][var1] == var2`.
    This is called an `element` constraint. Python does not allow expressing it on standard arrays,
    but CPMpy-numpy arrays do allow it, so you first have to wrap the array.

    Note that `arr` will be transformed to vector and indexed as such, 2-dimensional indexing is not supported (yet?).

    .. code-block:: python

        # Transforming a given numpy-array **m** into a cparray

        iv1, iv2 = intvar(0, 9, shape=2)

        data = [1, 2, 3, 4]
        data = cpm_array(data)

        Model([ data[iv1] == iv2 ])

    As an alternative, you can also write the :class:`~cpmpy.expressions.globalfunctions.Element` constraint directly on `data`: 
    
    .. code-block:: python

        Element(data, iv1) == iv2
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    elif not arr.flags['FORC']:   # Ensure the array is contiguous
        arr = np.ascontiguousarray(arr)
    
    order = 'F' if arr.flags['F_CONTIGUOUS'] else 'C'
    return NDVarArray(shape=arr.shape, dtype=arr.dtype, buffer=arr, order=order)


class NullShapeError(Exception):
    """
    Error returned when providing an empty or size 0 shape for numpy arrays of variables
    """
    def __init__(self, shape: Optional[int|np.integer|tuple[int|np.integer, ...]], 
                 message: str = "Shape should be non-zero"
                ):
        self.shape = shape
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f'{self.shape}: {self.message}'


class _NumVarImpl(Expression):
    """
    Abstract base class for numerical variables in CPMpy.

    This class implements the common functionality for numerical variables,
    including bounds checking and value management. It should not be instantiated
    directly, but rather through the helper functions :func:`~cpmpy.expressions.variables.intvar` and :func:`~cpmpy.expressions.variables.boolvar`.
    """
    def __init__(self, lb: int, ub: int, name: str):
        assert (is_num(lb) and is_num(ub))
        assert (lb <= ub)
        self.lb = lb
        self.ub = ub
        self.name = name
        self._value: Optional[int] = None

    def has_subexpr(self) -> bool:
        """Does it contains nested Expressions?
           Is of importance when deciding whether transformation/decomposition is needed.
        """
        return False

    def is_bool(self) -> bool:
        """ is it a Boolean (return type) Operator?
        """
        return False

    def value(self) -> Optional[int]:
        """ the value obtained in the last solve call
            (or 'None')
        """
        return self._value

    def get_bounds(self) -> tuple[int, int]:
        """ the lower and upper bounds"""
        return self.lb, self.ub

    def clear(self) -> None:
        """ clear the value obtained from the last solve call
        """
        self._value = None

    def __repr__(self) -> str:
        return self.name

    # for sets/dicts. Because names are unique, so is the str repr
    def __hash__(self) -> int:
        return hash(self.name)


class _IntVarImpl(_NumVarImpl):
    """
    Implementation class for integer variables in CPMpy.

    This class represents integer decision variables with a given domain [lb, ub] (inclusive).
    It should not be instantiated directly, but rather through the :func:`~cpmpy.expressions.variables.intvar` helper function.
    """
    counter = 0

    def __init__(self, lb: int, ub: int, name: Optional[str] = None):
        assert is_int(lb), "IntVar lowerbound must be integer {} {}".format(type(lb), lb)
        assert is_int(ub), "IntVar upperbound must be integer {} {}".format(type(ub), ub)

        if name is None:
            name = f"{_IV_PREFIX}{_IntVarImpl.counter}"
            _IntVarImpl.counter = _IntVarImpl.counter + 1 # static counter

        super().__init__(int(lb), int(ub), name=name) # explicit cast: can be numpy

    # special casing for intvars (and boolvars)
    def __abs__(self) -> Expression:
        if self.lb >= 0:
            # no-op
            return self
        return super().__abs__()


class _BoolVarImpl(_IntVarImpl):
    """
    Implementation class for Boolean variables in CPMpy.

    This class represents Boolean decision variables that can take values True/False.
    It should not be instantiated directly, but rather through the :func:`~cpmpy.expressions.variables.boolvar` helper function.
    """
    counter = 0

    def __init__(self, lb:int=0, ub:int=1, name:Optional[str]=None):
        assert(lb == 0 or lb == 1)
        assert(ub == 0 or ub == 1)

        if name is None:
            name = f"{_BV_PREFIX}{_BoolVarImpl.counter}"
            _BoolVarImpl.counter = _BoolVarImpl.counter + 1 # static counter
        _IntVarImpl.__init__(self, lb, ub, name=name)

    def is_bool(self) -> bool:
        """ is it a Boolean (return type) Operator?
        """
        return True

    def __invert__(self) -> Expression:
        return NegBoolView(self)

    def __abs__(self) -> Expression:
        return self

    # when redefining __eq__, must redefine custom__hash__
    # https://stackoverflow.com/questions/53518981/inheritance-hash-sets-to-none-in-a-subclass
    def __hash__(self) -> int:
        return hash(self.name)


class NegBoolView(_BoolVarImpl):
    """
        Represents not(`var`), not an actual variable implementation!

        It stores a link to `var`'s _BoolVarImpl

        Do not create this object directly, use the `~` operator instead: `~bv`
    """
    def __init__(self, bv: _BoolVarImpl):
        #assert(isinstance(bv, _BoolVarImpl))
        self._bv = bv
        # as it is always created using the ~ operator (only available for _BoolVarImpl)
        # it already comply with the asserts of the __init__ of _BoolVarImpl and can use 
        # __init__ from _IntVarImpl
        _IntVarImpl.__init__(self, 1-bv.ub, 1-bv.lb, name=str(self))

    def value(self) -> Optional[bool]:
        """ the negation of the value obtained in the last solve call by the viewed variable
            (or 'None')
        """
        v = self._bv.value()
        if v is None:
            return None
        return not v

    def clear(self) -> None:
        """ clear, for the viewed variable, the value obtained from the last solve call
        """
        self._bv.clear()

    def __repr__(self) -> str:
        return "~{}".format(self._bv.name)

    def __invert__(self) -> Expression:
        return self._bv


class NDVarArray(np.ndarray):
    """
    N-dimensional numpy array of ExprLike's (the name of this class is historically misleading...).

    Do not create this object directly, use one of the functions in this module

    ``_has_subexpr`` caches :meth:`has_subexpr` (``None`` = not computed yet; ``True`` / ``False`` = cached).
    """
    _has_subexpr: Optional[bool] = None  # will be overwritten in instance, here for type hinting

    def __init__(self, shape: int|np.integer|tuple[int|np.integer, ...], **kwargs: Any) -> None:
        # bit ugly, but np.int and np.bool do not play well with > overloading
        if np.issubdtype(self.dtype, np.integer):
            self.astype(int)
        elif np.issubdtype(self.dtype, np.bool_):
            self.astype(bool)

        self._has_subexpr = None
        # no need to call ndarray __init__ method as specified in the np.ndarray documentation:
        # "No ``__init__`` method is needed because the array is fully initialized
        #         after the ``__new__`` method."

    def has_subexpr(self) -> bool:
        """True if :meth:`flat` has an :class:`Expression` that is not a variable (:class:`_NumVarImpl`) or :class:`~cpmpy.expressions.core.BoolVal`."""
        if self._has_subexpr is not None:
            return self._has_subexpr

        for e in self.flat:
            if isinstance(e, Expression) and not isinstance(e, (_NumVarImpl, BoolVal)):
                self._has_subexpr = True
                return True
        self._has_subexpr = False
        return False

    def value(self) -> np.ndarray:
        """ the values, for each of the stored variables, obtained in the last solve call
            (or 'None')
        """
        return np.reshape([x.value() for x in self], self.shape)

    def clear(self) -> None:
        """ clear, for each of the stored variables, the value obtained from the last solve call
        """
        for e in self.flat:
            e.clear()

    def __getitem__(self, index):  # TODO: any typing would have to be compatible with supertype "numpy.ndarray"
        # array access, check if variables are used in the indexing

        # index is single expression: direct element (1D only)
        if isinstance(index, Expression):
            if self.ndim != 1:
                raise NotImplementedError("CPMpy does not support returning an array from an Element constraint. Provide an index for each dimension (comma separated indices). If you really need this, please report on github.")
            return cp.Element(self, index)

        # multi-dimensional index
        if isinstance(index, tuple) and any(isinstance(el, Expression) for el in index):

            if len(index) != self.ndim:
                raise NotImplementedError("CPMpy does not support returning an array from an Element constraint. Provide an index for each dimension (comma separated indices). If you really need this, please report on github.")

            # find dimension of expression in index
            expr_dim = [dim for dim,idx in enumerate(index) if isinstance(idx, Expression)]
            arr = self[tuple(index[:expr_dim[0]])] # select remaining dimensions
            index = list(index[expr_dim[0]:])

            # eliminate constant indices to reduce dimensionality
            selector = []
            new_indices = []
            for idx in index:
                if isinstance(idx, Expression):
                    selector.append(slice(None))
                    new_indices.append(idx)
                else:
                    selector.append(idx)
            arr = arr[tuple(selector)]

            if len(new_indices) == 1:
                return cp.Element(arr, new_indices[0])
            return cp.MultiDimElement(arr, new_indices)

        return super().__getitem__(index)

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
            return cp.sum(self)

        return cpm_array(np.apply_along_axis(cp.sum, axis=axis, arr=self))


    def prod(self, axis=None, out=None):
        """
            overwrite np.prod(NDVarArray) as people might use it
        """

        if out is not None:
            raise NotImplementedError()

        if axis is None:  # simple case where we want the product over the whole array
            return reduce(lambda a, b: a * b, self.flatten())

        # TODO: is there a better way? This does pairwise multiplication still
        return cpm_array(np.multiply.reduce(self, axis=axis))

    def max(self, axis=None, out=None):
        """
            overwrite np.max(NDVarArray) as people might use it
        """
        if out is not None:
            raise NotImplementedError()

        if axis is None:    # simple case where we want the maximum over the whole array
            return cp.max(self)

        return cpm_array(np.apply_along_axis(cp.max, axis=axis, arr=self))

    def min(self, axis=None, out=None):
        """
            overwrite np.min(NDVarArray) as people might use it
        """
        if out is not None:
            raise NotImplementedError()

        if axis is None:    # simple case where we want the minimum over the whole array
            return cp.min(self)

        return cpm_array(np.apply_along_axis(cp.min, axis=axis, arr=self))

    def any(self, axis=None, out=None):
        """
            overwrite np.any(NDVarArray)
        """
        if any(not is_boolexpr(x) for x in self.flat):
            raise TypeError("Cannot call .any() in an array not consisting only of bools")

        if out is not None:
            raise NotImplementedError()

        if axis is None:    # simple case where we want a disjunction over the whole array
            return cp.any(self)

        return cpm_array(np.apply_along_axis(cp.any, axis=axis, arr=self))


    def all(self, axis=None, out=None):
        """
            overwrite np.any(NDVarArray)
        """
        if any(not is_boolexpr(x) for x in self.flat):
            raise TypeError("Cannot call .any() in an array not consisting only of bools")

        if out is not None:
            raise NotImplementedError()

        if axis is None:  # simple case where we want a conjunction over the whole array
            return cp.all(self)

        return cpm_array(np.apply_along_axis(cp.all, axis=axis, arr=self))

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0:  # believe it or not, this does happen... e.g. in test_int2bool and an exmaple
            z = np.empty(self.shape, dtype=np.int64)
            return z, z

        lbs, ubs = zip(*[get_bounds(e) for e in self.flat])
        return np.asarray(lbs).reshape(self.shape), \
               np.asarray(ubs).reshape(self.shape)

    # VECTORIZED master function (delegate)
    def _vectorized(self, other: ExprLike|Iterable|Any, attr: str) -> NDVarArray:
        """
        Vectorized implementation of the given attribute (e.g. __eq__, __add__, etc.)

        Args:
            other (ExprLike|Iterable|Any): The other operand.
                Typically an array/list of Expressions, or a single Expression, or a constant (or anything np compatible)
            attr (str): The attribute to vectorize (e.g. __eq__, __add__, etc.)

        Returns:
            NDVarArray: The vectorized result.
        """
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


def _gen_var_names(name: Optional[str|ListLike[str]],
                   shape: int|np.integer|tuple[int|np.integer, ...]
                  ) -> list[Optional[str]]:
    """
    Helper function to collect the name of all decision variables (in np.ndindex(shape) order)

    `name` can be None, str, or an enumerable with the same shape as `shape`.
    Raises errors if invalid name
    """
    if name is None or isinstance(name, str):
        return [_genname(name, idx) for idx in np.ndindex(shape)]
    elif isinstance(name, (list, tuple, np.ndarray)):
        # special case: should match shape of decision variables
        name_arr = np.array(name)
        if isinstance(shape, int):
            shape = (shape,)
        if name_arr.shape != shape:
            raise ValueError(f"The shape of name sequence {name_arr.shape} does not match {shape}.")
        if len(name_arr.flat) != len(np.unique(name_arr)):
            raise ValueError(f"Duplicated names in {name_arr}.")
        if any(_is_invalid_name(n) for n in name_arr.flat):
            raise ValueError(_VAR_ERR)
        # same order as np.ndindex(shape): C-order, last axis varies fastest
        return list(name_arr.flat)
    else:
        raise TypeError(f"Unsupported type for name: {type(name)}")

def _genname(basename: Optional[str], idxs: tuple[int|np.integer, ...]) -> Optional[str]:
    """
    Helper function to 'name' array variables
    - idxs: list of indices, one for every dimension of the array
    - basename: base name to prepend

    if basename is 'None', then it returns None

    output: something like "basename[0,1]"
    """
    if basename is None:
        return None
    if _is_invalid_name(basename):
        raise ValueError(_VAR_ERR)
    stridxs = ",".join(map(str, idxs))
    return f"{basename}[{stridxs}]" # "<name>[<idx0>,<idx1>,...]"

def _is_invalid_name(name: Any) -> bool:
    if isinstance(name, str):
        return name.startswith(_IV_PREFIX) or name.startswith(_BV_PREFIX)
    # rest invalid indeed
    return True

