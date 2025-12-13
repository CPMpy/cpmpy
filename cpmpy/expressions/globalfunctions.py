#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## globalfunctions.py
##
"""
    Global functions conveniently express numerical global constraints in function form.

    For example `cp.Maximum(iv1, iv2, iv3) == iv4`, or `cp.Abs(iv1) > iv2` or `m.minimize(cp.Count(IVS, 0))`,
    or other nested numeric expressions.

    Global functions are implemented as classes that inherit from `GlobalFunction`.

    Solver perspective
    ------------------

    * Native support: some solvers natively support the function form of global functions,
    such as other CP languages, SMT solvers, Cplex, Hexaly etc.

    * Predicate form: CP solvers and MINLP solvers only support the predicate form of global functions.
    For example, `cp.Maximum(iv1, iv2, iv3) == iv4` has to be posted to the solver API
    as `addMaximumEquals([iv1,iv2,iv3], iv4)`, and expressions like `cp.Abs(iv1) > iv2` have to be flattened
    and posted as `addAbsEquals(iv1, aux)` and `aux > iv2`. This is automatically done by the flattening
    transformation in the `flatten_model` transformation.

    * Decomposition: if a solver does not support a global function, then it will be automatically
    decomposed by calling its `.decompose()` function in the `decompose_global` transformation.

    The `.decompose()` function returns two arguments:
        1. A single CPMpy expression representing the numerical value of the global function,
            this is often an auxiliary variable, a sum of auxiliary variables or a sum over
            nested expressions.
        2. If the decomposition introduces new *auxiliary variables*, then the second argument
            has to be a list of constraints that (totally) define those new variables.

    To make maximum use of simplification and common subexpression elimination, we recommend that decompositions
        use nested expression as much as possible and avoid creating auxiliary variables unless not expressible
        in a more direct way.


    Example:

    .. code-block:: python

        class MySum(GlobalFunction):
            def __init__(self, args):
                assert len(args) == 2, "MySum takes 2 arguments"
                super().__init__("my_sum", args)

            def decompose(self):
                return (self.args[0] + self.args[1]), []  # the decomposition

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
from typing import Optional, Union
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
        It overwrites the `.is_bool()` method to return False, as global functions are numeric.
    """

    def is_bool(self) -> bool:
        """
        Returns:
            bool: False, global functions are numeric
        """
        return False

    def decompose(self) -> tuple[Expression, list[Expression]]:
        """
            Returns a decomposition into smaller constraints as a tuple of
            (numerical expression, list of constraints defining auxiliary variables)

            The first one will replace the GlobalFunction expression in-place,
            the second one will be added to the list of top-level constraints.

            The decomposition might create auxiliary variables
            and use other global constraints as long as
            it does not create a circular dependency.

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the numerical expression and a list of constraints defining auxiliary variables
        """
        raise NotImplementedError("Decomposition for", self, "not available")

    def decompose_comparison(self, cmp_op: str, cmp_rhs: Expression) -> tuple[list[Expression], list[Expression]]:
        """
            DEPRECATED: returns a list of constraints representing the decomposed
            comparison of the global function (and any auxiliary variables intorduced).

        Arguments:
            cmp_op (str): Comparison operator
            cmp_rhs (Expression): Right-hand side expression for the comparison

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing two lists: constraints representing the comparison, and constraints defining auxiliary variables
        """
        warnings.warn(f"Deprecated, use {self}.decompose() instead, will be removed in "
                      "stable version", DeprecationWarning)
        valexpr, cons = self.decompose()
        return eval_comparison(cmp_op, valexpr, cmp_rhs), cons

    def get_bounds(self) -> tuple[int, int]:
        """
        Returns the bounds of the global function

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound)
        """
        return NotImplementedError("Bounds calculation for", self, "not available")

    def is_total(self) -> bool:
        """
            Returns whether it is a total function.
            If true, its value is defined for all arguments

            TODO: I do not find anywhere where we set it dynamically to False?
            TODO: REMOVE??
        """
        return True


class Minimum(GlobalFunction):
    """
        Computes the minimum value of the arguments
    """

    def __init__(self, arg_list: list[Expression]):
        """
        Arguments:
            arg_list (list[Expression]): List of expressions of which to compute the minimum
        """
        super().__init__("min", flatlist(arg_list))

    def value(self) -> Optional[int]:
        """
        Returns:
            Optional[int]: The minimum value of the arguments, or None if any argument is not assigned
        """
        argvals = [argval(a) for a in self.args]
        if any(val is None for val in argvals):
            return None
        else:
            return min(argvals)

    def decompose(self) -> tuple[Expression, list[Expression]]:
        """
        Decomposition of Minimum constraint

        Can only be decomposed by introducing an auxiliary variable and enforcing it to be larger than each variable,
         while at the same time not being larger then all (e.g. it needs to be (smaller or) equal to one of them)

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the auxiliary variable representing the minimum value, and a list of constraints defining it
        """
        _min = intvar(*self.get_bounds())
        return _min, [cp.all(_min <= a for a in self.args), cp.any(_min >= a for a in self.args)]

    def get_bounds(self) -> tuple[int, int]:
        """
        Returns the lowest and highest possible minimum value

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the minimum value
        """
        bnds = [get_bounds(x) for x in self.args]
        return min(lb for lb, ub in bnds), min(ub for lb, ub in bnds)


class Maximum(GlobalFunction):
    """
        Computes the maximum value of the arguments
    """

    def __init__(self, arg_list: list[Expression]):
        """
        Arguments:
            arg_list (list[Expression]): List of expressions of which to compute the maximum
        """
        super().__init__("max", flatlist(arg_list))

    def value(self) -> Optional[int]:
        """
        Returns:
            Optional[int]: The maximum value of the arguments, or None if any argument is not assigned
        """
        argvals = [argval(a) for a in self.args]
        if any(val is None for val in argvals):
            return None
        else:
            return max(argvals)

    def decompose(self) -> tuple[Expression, list[Expression]]:
        """
        Decomposition of Maximum constraint.

        Can only be decomposed by introducing an auxiliary variable and enforcing it to be larger than each variable,
         while at the same time not being larger then all (e.g. it needs to be (smaller or) equal to one of them)

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the auxiliary variable representing the maximum value, and a list of constraints defining it
        """
        _max = intvar(*self.get_bounds())
        return _max, [cp.all(_max >= a  for a in self.args), cp.any(_max <= a for a in self.args)]

    def get_bounds(self) -> tuple[int, int]:
        """
        Returns the lowest and highest possible maximum value

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the maximum value
        """
        bnds = [get_bounds(x) for x in self.args]
        return max(lb for lb, ub in bnds), max(ub for lb, ub in bnds)


class Abs(GlobalFunction):
    """
        Computes the absolute value of the argument
    """

    def __init__(self, expr: Expression):
        """
        Arguments:
            expr (Expression): Expression of which to compute the absolute value
        """
        super().__init__("abs", [expr])

    def value(self) -> Optional[int]:
        """
        Returns:
            Optional[int]: The absolute value of the argument, or None if the argument is not assigned
        """
        argval = argval(self.args[0])
        if argval is not None:
            return abs(argval)
        return None

    def decompose(self) -> tuple[Expression, list[Expression]]:
        """
        Decomposition of Abs constraint.

        Can only be decomposed by introducing an auxiliary variable and enforcing it's value to be positive,
            based on the value of the given argument to the global function. I.e., if the argument is negative,
            the auxiliary variable will take the negated value of the argument, and otherwise it will take the argument itself.

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the expression representing the absolute value (may be the argument itself, its negation, or an auxiliary variable), and a list of constraints defining it (empty if no auxiliary variable is needed)
        """
        arg = self.args[0]
        lb, ub = get_bounds(arg)
        if lb >= 0: # always positive
            return arg, []
        if ub <= 0: # always negative
            return -arg, []

        _abs = intvar(*self.get_bounds())
        return _abs, [(arg >= 0).implies(_abs == arg), (arg < 0).implies(_abs == -arg)]

    def get_bounds(self) -> tuple[int, int]:
        """
        Returns the lowest and highest possible absolute value

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the absolute value
        """
        lb,ub = get_bounds(self.args[0])
        if lb >= 0:
            return lb, ub
        if ub <= 0:
            return -ub, -lb
        return 0, max(-lb, ub)


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
        """
        Arguments:
            arr: Array of constants or variables to index into
            idx: Integer decision variable representing the index into the array
        """
        if is_boolexpr(idx):
            raise TypeError("index cannot be a boolean expression: {}".format(idx))
        if is_any_list(idx):
            raise TypeError("For using multiple dimensions in the Element constraint, use comma-separated indices")
        super().__init__("element", [arr, idx])

    def __getitem__(self, index):
        raise CPMpyException("For using multiple dimensions in the Element constraint use comma-separated indices")

    def value(self):
        """
        Returns:
            The value of the array element at the given index, or None if the index is not assigned
        """
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
        Decomposition of Element constraint.

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the expression representing the element value, and an empty list of constraints (no auxiliary variables needed)
        """
        arr, idx = self.args

        idx_lb, idx_ub = get_bounds(idx)
        assert idx_lb >= 0 and idx_ub < len(arr), "Element constraint is unsafe to decompose as it can be partial. Safen first using `cpmpy.transformations.safening.no_partial_functions`"

        return cp.sum(arr[i] * (i == idx) for i in range(len(arr))), []

    def __repr__(self):
        return "{}[{}]".format(self.args[0], self.args[1])

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the element value
        """
        arr, idx = self.args
        bnds = [get_bounds(x) for x in arr]
        return min(lb for lb,ub in bnds), max(ub for lb,ub in bnds)

def element(arg_list: list[Expression]) -> Element:
    warnings.warn("Deprecated, use Element(arr,idx) instead, will be removed in stable version", DeprecationWarning)
    assert (len(arg_list) == 2), "Element expression takes 2 arguments: Arr, Idx"
    return Element(arg_list[0], arg_list[1])


class Count(GlobalFunction):
    """
    The Count (numerical) global constraint represents the number of occurrences of val in arr
    """

    def __init__(self,arr,val):
        """
        Arguments:
            arr: Array of expressions to count in
            val: Value to count occurrences of
        """
        if is_any_list(val) or not is_any_list(arr):
            raise TypeError("count takes an array and a value as input, not: {} and {}".format(arr,val))
        super().__init__("count", [arr,val])

    def decompose(self):
        """
        Decomposition of the Count constraint.
        Does not require the use of auxiliary variables, simply count the number of variables that take the given value.

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the sum expression representing the count, and an empty list of constraints (no auxiliary variables needed)
        """
        arr, val = self.args
        return cp.sum(a == val for a in arr), []

    def value(self):
        """
        Returns:
            int: The number of occurrences of val in arr
        """
        arr, val = self.args
        val = argval(val)
        return sum([argval(a) == val for a in arr])

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the count value
        """
        arr, val = self.args
        return 0, len(arr)



class Among(GlobalFunction):
    """
        The Among (numerical) global constraint represents the number of variable that take values among the values in arr
    """

    def __init__(self,arr,vals):
        """
        Arguments:
            arr: Array of expressions to count in
            vals: Array of values to count occurrences of
        """
        if not is_any_list(arr) or not is_any_list(vals):
            raise TypeError("Among takes as input two arrays, not: {} and {}".format(arr,vals))
        if any(isinstance(val, Expression) for val in vals):
            raise TypeError(f"Among takes a set of values as input, not {vals}")
        super().__init__("among", [arr,vals])

    def decompose(self):
        """
         Decomposition of the Among constraint.
         Decomposed using several Count constraints, one for each value in values.

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the sum expression representing the count, and an empty list of constraints (no auxiliary variables needed)
        """
        arr, values = self.args
        return cp.sum(Count(arr, val) for val in values), []


    def value(self):
        """
        Returns:
            int: The number of variables in arr that take values among the values in vals
        """
        return int(sum(np.isin(argvals(self.args[0]), self.args[1])))

    def get_bounds(self):
        """
        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the among count value
        """
        return 0, len(self.args[0])


class NValue(GlobalFunction):

    """
    The NValue constraint counts the number of distinct values in a given set of variables.
    """

    def __init__(self, arr):
        """
        Arguments:
            arr: Array of expressions to count distinct values in
        """
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
        """
        Returns:
            int: The number of distinct values in the array
        """
        return len(set(argval(a) for a in self.args))

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the number of distinct values
        """
        return 1, len(self.args)


class NValueExcept(GlobalFunction):

    """
        The NValueExceptN constraint counts the number of distinct values,
            not including value N, if any argument is assigned to it.
    """

    def __init__(self, arr, n):
        """
        Arguments:
            arr: Array of expressions to count distinct values in
            n: Integer value to exclude from the count
        """
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
        """
        Returns:
            int: The number of distinct values in the array, excluding value n
        """
        return len(set(argval(a) for a in self.args[0]) - {self.args[1]})

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the number of distinct values (excluding n)
        """
        return 0, len(self.args)
