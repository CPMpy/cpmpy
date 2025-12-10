#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## globalfunctions.py
##
"""
    Global functions conveniently express numerical global constraints.

    Using global functions
    ------------------------

    If a solver does not support such a global function (see solvers/), then it will be automatically
    decomposed by calling its `.decompose()` function.

    CPMpy GlobalFunctions do not exactly match what is implemented in the solvers.
    Solvers can have specialised implementations for global functions, when used in a comparison, as global constraints.
    These global functions will be treated as global constraints in such cases.

    For example solvers may implement the global constraint `Minimum(iv1, iv2, iv3) == iv4` through an API
    call `addMinimumEquals([iv1,iv2,iv3], iv4)`.

    However, CPMpy also wishes to support the expressions `Minimum(iv1, iv2, iv3) > iv4` as well as
    `iv4 + Minimum(iv1, iv2, iv3)`.

    Hence, the CPMpy global functions only capture the `Minimum(iv1, iv2, iv3)` part, whose return type
    is numeric and can be used in any other CPMpy expression. Only at the time of transforming the CPMpy
    model to the solver API, will the expressions be decomposed and auxiliary variables introduced as needed
    such that the solver only receives `Minimum(iv1, iv2, iv3) == ivX` expressions.
    This is the burden of the CPMpy framework, not of the user who wants to express a problem formulation.


    Subclassing GlobalFunction
    ----------------------------

    If you do wish to add a GlobalFunction, because it is supported by solvers or because you will do
    advanced analysis and rewriting on it, then impelement a decompose function that returns a tuple of two arguments:
        1. A single CPMpy expression representing the numerical value represented by the global function
                this is often an auxiliary variable, a sum of auxiliary variables or a sum over nested expressions.
        2. A list of CPMpy constraints that define the auxiliary variables used in the first argument.

    To make maximum use of simplification and common subexpression elimination, we recommend that you use
        nested expression as much as possible and avoid creating auxiliary variables unless really needed

    e.g.:

    .. code-block:: python

        class MySum(GlobalFunction):
            def __init__(self, args):
                super().__init__("my_sum", args)

            def decompose(self):
                return self.args[0] + self.args[1], [] # your decomposition

    Also, implement `.value()` accordingly.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        Minimum
        Maximum
        Abs
        Element
        Count
        Among
        NValue
        NValueExcept

"""
import warnings  # for deprecation warning

import numpy as np
import cpmpy as cp

from ..exceptions import CPMpyException, IncompleteFunctionError, TypeError
from .core import Expression, Operator
from .variables import boolvar, intvar, cpm_array
from .utils import flatlist, argval, is_num, eval_comparison, is_any_list, is_boolexpr, get_bounds, argvals, get_bounds, implies


class GlobalFunction(Expression):
    """
        Abstract superclass of GlobalFunction

        Like all expressions it has a `.name` and `.args` property.
        Overwrites the `.is_bool()` method.
    """

    def is_bool(self):
        """ is it a Boolean (return type) Operator? No
        """
        return False

    def decompose(self):
        """
            Returns a decomposition into smaller constraints.
            Returns a numerical expression and a list of defining constraints.

            The decomposition might create auxiliary variables
            and use other global constraints as long as
            it does not create a circular dependency.

            Returns
            1) a numerical value to replace the constraint, and
            2) a list of defining constraints, which should be enforced toplevel
        """
        raise NotImplementedError("Decomposition for", self, "not available")

    def decompose_comparison(self, cmp_op, cmp_rhs):
        """
            Returns a decomposition into smaller constraints.

            The decomposition might create auxiliary variables
            and use other global constraints as long as
            it does not create a circular dependency.
        """
        warnings.warn(f"Deprecated, use {self}.decompose() instead, will be removed in "
                      "stable version", DeprecationWarning)
        val, tl = self.decompose()
        return eval_comparison(cmp_op, val, cmp_rhs), tl


    def get_bounds(self):
        """
        Returns the bounds of the global function
        """
        return NotImplementedError("Bounds calculation for", self, "not available")

    def is_total(self):
        """
            Returns whether it is a total function.
            If true, its value is defined for all arguments
        """
        return True


class Minimum(GlobalFunction):
    """
        Computes the minimum value of the arguments
    """

    def __init__(self, arg_list):
        super().__init__("min", flatlist(arg_list))

    def value(self):
        argvals = [argval(a) for a in self.args]
        if any(val is None for val in argvals):
            return None
        else:
            return min(argvals)

    def decompose(self):
        """
        Decomposition of Minimum constraint

        Can only be decomposed by introducing an auxiliary variable and enforcing it to be larger than each variable,
         while at the same time not being larger then all (e.g. it needs to be (smaller or) equal to one of them)
        """
        _min = intvar(*self.get_bounds())
        return _min, [cp.all(x >= _min for x in self.args), cp.any(x <= _min for x in self.args)]

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        bnds = [get_bounds(x) for x in self.args]
        return min(lb for lb, ub in bnds), min(ub for lb, ub in bnds)


class Maximum(GlobalFunction):
    """
        Computes the maximum value of the arguments
    """

    def __init__(self, arg_list):
        super().__init__("max", flatlist(arg_list))

    def value(self):
        argvals = [argval(a) for a in self.args]
        if any(val is None for val in argvals):
            return None
        else:
            return max(argvals)

    def decompose(self):
        """
        Decomposition of Maximum constraint.

        Can only be decomposed by introducing an auxiliary variable and enforcing it to be smaller than each variable,
         while at the same time not being smaller then all (e.g. it needs to be (larger or) equal to one of them)
        """
        _max = intvar(*self.get_bounds())
        return _max, [cp.all(x <= _max for x in self.args), cp.any(x >= _max for x in self.args)]

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        bnds = [get_bounds(x) for x in self.args]
        return max(lb for lb, ub in bnds), max(ub for lb, ub in bnds)

class Abs(GlobalFunction):
    """
        Computes the absolute value of the argument

        Can only be decomposed by introducing an auxiliary variable and enforcing it's value to be positive,
            based on the value of the given argument to the global function. I.e., if the argument is negative,
            the auxiliary variable will take the negated value of the argument, and otherwise it will take the argument itself.
    """

    def __init__(self, expr):
        super().__init__("abs", [expr])

    def value(self):
        return abs(argval(self.args[0]))

    def decompose(self):
        """
        Decomposition of Abs constraint.
        """
        arg = self.args[0]
        lb, ub = get_bounds(arg)
        if lb >= 0: # always positive
            return arg, []
        if ub <= 0: # always negative
            return -arg, []

        _abs = intvar(*self.get_bounds())

        is_pos = arg >= 0 # CPMpy expression that checks whether the argument is positive
        return _abs, [is_pos.implies(arg == _abs), (~is_pos).implies(arg == -_abs)]

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        lb,ub = get_bounds(self.args[0])
        if lb >= 0:
            return lb, ub
        if ub <= 0:
            return -ub, -lb
        return 0, max(-lb, ub)


def element(arg_list):
    warnings.warn("Deprecated, use Element(arr,idx) instead, will be removed in stable version", DeprecationWarning)
    assert (len(arg_list) == 2), "Element expression takes 2 arguments: Arr, Idx"
    return Element(arg_list[0], arg_list[1])

class Element(GlobalFunction):
    """
        The `Element` global constraint enforces that the result equals `Arr[Idx]`
        with `Arr` an array of constants or variables (the first argument)
        and `Idx` an integer decision variable, representing the index into the array.

        Solvers implement it as `Arr[Idx] == Y`, but CPMpy will automatically derive or create
        an appropriate `Y`. Hence, you can write expressions like `Arr[Idx] + 3 <= Y`.

        Element is a CPMpy built-in global constraint, so the class implements a few more
        extra things for convenience (`.value()` and `.__repr__()`). It is also an example of
        a 'numeric' global constraint. Consequently, the return expression of the
        `Element` function is also numeric, even if `Arr` only contains Boolean variables.
    """

    def __init__(self, arr, idx):
        if is_boolexpr(idx):
            raise TypeError("index cannot be a boolean expression: {}".format(idx))
        if is_any_list(idx):
            raise TypeError("For using multiple dimensions in the Element constraint, use comma-separated indices")
        super().__init__("element", [arr, idx])

    def __getitem__(self, index):
        raise CPMpyException("For using multiple dimensions in the Element constraint use comma-separated indices")

    def value(self):
        arr, idx = self.args
        idxval = argval(idx)
        if idxval is not None:
            if idxval >= 0 and idxval < len(arr):
                return argval(arr[idxval])
            raise IncompleteFunctionError(f"Index {idxval} out of range for array of length {len(arr)} while calculating value for expression {self}"
                                          + "\n Use argval(expr) to get the value of expr with relational semantics.")
        return None # default

    def decompose(self):
        """
        Decomposition of Abs constraint.
        """
        arr, idx = self.args

        idx_lb, idx_ub = get_bounds(idx)
        assert idx_lb >= 0 and idx_ub < len(arr), "Element constraint is unsafe to decompose as it can be partial. Safen first using `cpmpy.transformations.safening.no_partial_functions`"

        _elem = intvar(*self.get_bounds())

        return _elem, [implies(idx == i, _elem == arr[i]) for i in range(len(arr))]


    def __repr__(self):
        return "{}[{}]".format(self.args[0], self.args[1])

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        arr, idx = self.args
        bnds = [get_bounds(x) for x in arr]
        return min(lb for lb,ub in bnds), max(ub for lb,ub in bnds)


class Count(GlobalFunction):
    """
    The Count (numerical) global constraint represents the number of occurrences of val in arr
    """

    def __init__(self,arr,val):
        if is_any_list(val) or not is_any_list(arr):
            raise TypeError("count takes an array and a value as input, not: {} and {}".format(arr,val))
        super().__init__("count", [arr,val])

    def decompose(self):
        """
        Decomposition of the Count constraint.
        Does not require the use of auxiliary variables, simply count the number of variables that take the given value.
        """
        arr, val = self.args
        return cp.sum(a == val for a in arr), []

    def value(self):
        arr, val = self.args
        val = argval(val)
        return sum([argval(a) == val for a in arr])

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        arr, val = self.args
        return 0, len(arr)



class Among(GlobalFunction):
    """
        The Among (numerical) global constraint represents the number of variable that take values among the values in arr
    """

    def __init__(self,arr,vals):
        if not is_any_list(arr) or not is_any_list(vals):
            raise TypeError("Among takes as input two arrays, not: {} and {}".format(arr,vals))
        if any(isinstance(val, Expression) for val in vals):
            raise TypeError(f"Among takes a set of values as input, not {vals}")
        super().__init__("among", [arr,vals])

    def decompose(self):
        """
         Decomposition of the Among constraint.
         Decomposed using several Count constraints, one for each value in values.
        """
        arr, values = self.args
        return cp.sum(Count(arr, val) for val in values), []


    def value(self):
        return int(sum(np.isin(argvals(self.args[0]), self.args[1])))

    def get_bounds(self):
        return 0, len(self.args[0])


class NValue(GlobalFunction):

    """
    The NValue constraint counts the number of distinct values in a given set of variables.
    """

    def __init__(self, arr):
        if not is_any_list(arr):
            raise ValueError("NValue takes an array as input")
        super().__init__("nvalue", arr)

    def decompose(self):
        """
        Decomposition of the Count constraint.

        Based on "simple decomposition" from:

            Bessiere, Christian, et al. "Decomposition of the NValue constraint."
            International Conference on Principles and Practice of Constraint Programming.
            Berlin, Heidelberg: Springer Berlin Heidelberg, 2010.
        """
        lbs, ubs = get_bounds(self.args)
        lb, ub = min(lbs), max(ubs)

        return cp.sum(cp.any(a == v for a in self.args) for v in range(lb,ub+1)), []

    def value(self):
        return len(set(argval(a) for a in self.args))

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        return 1, len(self.args)


class NValueExcept(GlobalFunction):

    """
        The NValueExceptN constraint counts the number of distinct values,
            not including value N, if any argument is assigned to it.
    """

    def __init__(self, arr, n):
        if not is_any_list(arr):
            raise ValueError("NValueExcept takes an array as input")
        if not is_num(n):
            raise ValueError(f"NValueExcept takes an integer as second argument, but got {n} of type {type(n)}")
        super().__init__("nvalue_except",[arr, n])

    def decompose(self):
        """
        Decomposition of the Count constraint.

        Based on "simple decomposition" from:

            Bessiere, Christian, et al. "Decomposition of the NValue constraint."
            International Conference on Principles and Practice of Constraint Programming.
            Berlin, Heidelberg: Springer Berlin Heidelberg, 2010.
        """
        arr, n = self.args
        assert is_num(n)

        lbs, ubs = get_bounds(arr)
        lb, ub = min(lbs), max(ubs)

        n_values = 0
        for v in range(lb, ub+1):
            if v == n:
                continue
            n_values += cp.any(a == v for a in arr)

        return n_values, []

    def value(self):
        return len(set(argval(a) for a in self.args[0]) - {self.args[1]})

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        return 0, len(self.args)
