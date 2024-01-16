#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## globalfunctions.py
##
"""

    Using global functions
    ------------------------

    If a solver does not support such a global function (see solvers/), then it will be automatically
    decomposed by calling its `.decompose_comparison()` function.

    CPMpy GlobalFunctions does not exactly match what is implemented in the solvers.
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
    advanced analysis and rewriting on it, then preferably define it with a standard comparison decomposition,
    e.g.:

    .. code-block:: python

        class my_global(GlobalFunction):
            def __init__(self, args):
                super().__init__("my_global", args)

            def decompose_comparison(self):
                return [self.args[0] + self.args[1]] # your decomposition

    Also, implement `.value()` accordingly.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        Minimum
        Maximum
        Element
        Count
        NValue
        Abs

"""
import copy
import warnings  # for deprecation warning
import numpy as np
from ..exceptions import CPMpyException, IncompleteFunctionError, TypeError
from .core import Expression, Operator, Comparison
from .variables import boolvar, intvar, cpm_array, _NumVarImpl
from .utils import flatlist, all_pairs, argval, is_num, eval_comparison, is_any_list, is_boolexpr, get_bounds


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

    def decompose_comparison(self, cmp_op, cmp_rhs):
        """
            Returns a decomposition into smaller constraints.

            The decomposition might create auxiliary variables
            and use other global constraints as long as
            it does not create a circular dependency.
        """
        raise NotImplementedError("Decomposition for", self, "not available")

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

    def decompose_comparison(self, cpm_op, cpm_rhs):
        """
        Decomposition if it's part of a comparison
        Returns two lists of constraints:
            1) constraints representing the comparison
            2) constraints that (totally) define new auxiliary variables needed in the decomposition,
               they should be enforced toplevel.
        """
        from .python_builtins import any, all
        if cpm_op == "==":  # can avoid creating aux var
            return [any(x <= cpm_rhs for x in self.args),
                    all(x >= cpm_rhs for x in self.args)], []

        lb, ub = self.get_bounds()
        _min = intvar(lb, ub)
        return [eval_comparison(cpm_op, _min, cpm_rhs)], \
               [any(x <= _min for x in self.args), all(x >= _min for x in self.args), ]

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

    def decompose_comparison(self, cpm_op, cpm_rhs):
        """
        Decomposition if it's part of a comparison
        Returns two lists of constraints:
            1) constraints representing the comparison
            2) constraints that (totally) define new auxiliary variables needed in the decomposition,
               they should be enforced toplevel.
        """
        from .python_builtins import any, all
        if cpm_op == "==":  # can avoid creating aux var here
            return [any(x >= cpm_rhs for x in self.args),
                    all(x <= cpm_rhs for x in self.args)], []

        lb, ub = self.get_bounds()
        _max = intvar(lb, ub)
        return [eval_comparison(cpm_op, _max, cpm_rhs)], \
               [any(x >= _max for x in self.args), all(x <= _max for x in self.args)]

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        bnds = [get_bounds(x) for x in self.args]
        return max(lb for lb, ub in bnds), max(ub for lb, ub in bnds)

class Abs(GlobalFunction):
    """
        Computes the maximum value of the arguments
    """

    def __init__(self, expr):
        super().__init__("abs", [expr])

    def value(self):
        return abs(argval(self.args[0]))

    def decompose_comparison(self, cpm_op, cpm_rhs):
        """
        Decomposition if it's part of a comparison
        Returns two lists of constraints:
            1) constraints representing the comparison
            2) constraints that (totally) define new auxiliary variables needed in the decomposition,
               they should be enforced toplevel.
        """
        arg = self.args[0]
        return ([Comparison(cpm_op, Maximum([arg, -arg]), cpm_rhs)],[])


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
        The 'Element' global constraint enforces that the result equals Arr[Idx]
        with 'Arr' an array of constants of variables (the first argument)
        and 'Idx' an integer decision variable, representing the index into the array.

        Solvers implement it as Arr[Idx] == Y, but CPMpy will automatically derive or create
        an appropriate Y. Hence, you can write expressions like Arr[Idx] + 3 <= Y

        Element is a CPMpy built-in global constraint, so the class implements a few more
        extra things for convenience (.value() and .__repr__()). It is also an example of
        a 'numeric' global constraint.
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
            raise IncompleteFunctionError(f"Index {idxval} out of range for array of length {len(arr)} while calculating value for expression {self}")
        return None # default

    def decompose_comparison(self, cpm_op, cpm_rhs):
        """
            `Element(arr,ix)` represents the array lookup itself (a numeric variable)
            When used in a comparison relation: Element(arr,idx) <CMP_OP> CMP_RHS
            it is a constraint, and that one can be decomposed.
            Returns two lists of constraints:
                1) constraints representing the comparison
                2) constraints that (totally) define new auxiliary variables needed in the decomposition,
                   they should be enforced toplevel.

        """
        from .python_builtins import any

        arr, idx = self.args
        return [(idx == i).implies(eval_comparison(cpm_op, arr[i], cpm_rhs)) for i in range(len(arr))] + \
               [idx >= 0, idx < len(arr)], []

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

    def decompose_comparison(self, cmp_op, cmp_rhs):
        """
        Count(arr,val) can only be decomposed if it's part of a comparison
        """
        arr, val = self.args
        return [eval_comparison(cmp_op, Operator('sum',[ai==val for ai in arr]), cmp_rhs)], []

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

class NValue(GlobalFunction):

    """
    The NValue constraint counts the number of distinct values in a given set of variables.
    """

    def __init__(self, arr):
        if not is_any_list(arr):
            raise ValueError("NValue takes an array as input")
        super().__init__("nvalue", arr)

    def decompose_comparison(self, cmp_op, cpm_rhs):
        """
        NValue(arr) can only be decomposed if it's part of a comparison

        Based on "simple decomposition" from:
            Bessiere, Christian, et al. "Decomposition of the NValue constraint."
            International Conference on Principles and Practice of Constraint Programming.
            Berlin, Heidelberg: Springer Berlin Heidelberg, 2010.
        """
        from .python_builtins import sum, any

        lbs, ubs = get_bounds(self.args)
        lb, ub = min(lbs), max(ubs)

        constraints = []

        # introduce boolvar for each possible value
        bvars = boolvar(shape=(ub+1-lb))

        args = cpm_array(self.args)
        # bvar is true if the value is taken by any variable
        for bv, val in zip(bvars, range(lb, ub+1)):
            constraints += [any(args == val) == bv]

        return [eval_comparison(cmp_op, sum(bvars), cpm_rhs)], constraints

    def value(self):
        return len(set(argval(a) for a in self.args))

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        return 1, len(self.args)