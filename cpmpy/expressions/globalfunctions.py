#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## globalfunctions.py
##
"""
    Global functions conveniently express numerical global constraints in function form.

    For example `cp.Maximum(iv1, iv2, iv3) == iv4`, or `cp.Abs(iv1) > iv2` or `m.minimize(cp.Count(IVS, 0))`,
    or other nested numeric expressions.

    Global functions are implemented as classes that inherit from :class:`GlobalFunction <cpmpy.expressions.globalfunctions.GlobalFunction>`.


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
    decomposed by the :func:`decompose_in_tree() <cpmpy.transformations.decompose_global.decompose_in_tree>` transformation, which will call its :meth:`decompose() <cpmpy.expressions.globalfunctions.GlobalFunction.decompose>` method.

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
from .utils import flatlist, argval, is_num, eval_comparison, is_any_list, is_boolexpr, get_bounds, argvals, implies


class GlobalFunction(Expression):
    """
    Abstract superclass of GlobalFunction

    Like all expressions it has a `.name` and `.args` property.
    It overwrites the :meth:`is_bool() <cpmpy.expressions.core.Expression.is_bool>` method to return False, as global functions are numeric.
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
            comparison of the global function (and any auxiliary variables introduced).

        Arguments:
            cmp_op (str): Comparison operator
            cmp_rhs (Expression): Right-hand side expression for the comparison

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing two lists: constraints representing the comparison, and constraints defining auxiliary variables
        """
        warnings.warn(f"Deprecated, use {self}.decompose() instead, will be removed in "
                      "stable version", DeprecationWarning)
        valexpr, cons = self.decompose()
        return [eval_comparison(cmp_op, valexpr, cmp_rhs)], cons

    def get_bounds(self) -> tuple[int, int]:
        """
        Returns the bounds of the global function

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound)
        """
        raise NotImplementedError("Bounds calculation for", self, "not available")

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
        vargs = [argval(a) for a in self.args]
        if any(val is None for val in vargs):
            return None

        return min(vargs)

    def decompose(self) -> tuple[Expression, list[Expression]]:
        """
        Decomposition of Minimum global function.

        Can only be decomposed by introducing an auxiliary variable and enforcing it to be smaller or equal than
        each variable, while at the same time not being smaller than all (e.g. it needs to be (larger or)
        equal to one of them)

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
        vargs = [argval(a) for a in self.args]
        if any(val is None for val in vargs):
            return None

        return max(vargs)

    def decompose(self) -> tuple[Expression, list[Expression]]:
        """
        Decomposition of Maximum global function.

        Can only be decomposed by introducing an auxiliary variable and enforcing it to be larger or equal than
        each variable, while at the same time not being larger than all (e.g. it needs to be (smaller or)
        equal to one of them)

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the auxiliary variable representing the maximum value, and a list of constraints defining it
        """
        _max = intvar(*self.get_bounds())
        return _max, [cp.all(_max >= a for a in self.args), cp.any(_max <= a for a in self.args)]

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
        varg = argval(self.args[0])
        if varg is None:
            return None

        return abs(varg)

    def decompose(self) -> tuple[Expression, list[Expression]]:
        """
        Decomposition of Abs global function.

        Can only be decomposed by introducing an auxiliary variable and enforcing its value to be positive,
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
        lb, ub = get_bounds(self.args[0])
        if lb >= 0:
            return lb, ub
        if ub <= 0:
            return -ub, -lb
        return 0, max(-lb, ub)


class Element(GlobalFunction):
    """
    The `Element(Arr, Idx)` global function allows indexing into an array with a decision variable.

    Its return value will be the value of the array element at the index specified by the decision
    variable's value.

    When you index into a :class:`NDVarArray <cpmpy.expressions.variables.NDVarArray>` (e.g. when creating a `Arr=boolvar(shape=...)` or
    `Arr=intvar(lb,ub, shape=...)` using :func:`boolvar() <cpmpy.expressions.variables.boolvar>` or :func:`intvar() <cpmpy.expressions.variables.intvar>`), or index into a list wrapped as `Arr = cpm_array(lst)` using :func:`cpm_array() <cpmpy.expressions.variables.cpm_array>`,
    then using standard Python indexing, e.g. `Arr[Idx]` with `Idx` an integer decision variable,
    will automatically create this `Element(Arr, Idx)` object.

    Note: because Element is a numeric global function, the return type of the `Element` function
    is always numeric, even if `Arr` only contains Boolean variables.
    """

    def __init__(self, arr: list[Union[int, Expression]], idx: Expression):
        """
        Arguments:
            arr (list[Union[int, Expression]]): list (including NDVarArray) of integers or expressions to index into
            idx (Expression): Integer decision variable or expression, representing the index into the array
        """
        if is_boolexpr(idx):
            raise TypeError(f"Element(arr, idx) takes an integer expression as second argument, not a boolean expression: {idx}")
        if is_any_list(idx):
            raise TypeError(f"Element(arr, idx) takes an integer expression as second argument, not a list: {idx}")
        super().__init__("element", [arr, idx])

    def __getitem__(self, index):
        raise CPMpyException("For using multi-dimensional Element, use comma-separated indices on the original array, e.g. instead of Arr[Idx1][Idx2], do Arr[Idx1, Idx2].")

    def value(self) -> Optional[int]:
        """
        Returns:
            Optional[int]: The value of the array element at the given index, or None if the index is not assigned or the array element is not assigned
        """
        arr, idx = self.args
        vidx = argval(idx)
        if vidx is None:
            return None

        if vidx < 0 or vidx >= len(arr):
            raise IncompleteFunctionError(f"Index {vidx} out of range for array of length {len(arr)} while calculating value for expression {self}"
                                            + "\n Use argval(expr) to get the value of expr with relational semantics.")
        return argval(arr[vidx])  # can be None

    def decompose(self) -> tuple[Expression, list[Expression]]:
        """
        Decomposition of Element global function.
        
        The index variable must be within the bounds of the array.

        This decomposition uses an auxiliary variable and implication constraints.

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the auxiliary variable representing the element value, and a list of constraints defining it
        """
        arr, idx = self.args

        idx_lb, idx_ub = get_bounds(idx)
        assert idx_lb >= 0 and idx_ub < len(arr), "Element constraint is unsafe to decompose as it can be partial. Safen first using `cpmpy.transformations.safening.no_partial_functions`"

        aux = intvar(*self.get_bounds())
        return aux, [implies(idx == i, aux == arr[i]) for i in range(idx_lb, idx_ub+1)]

    def decompose_linear(self) -> tuple[Expression, list[Expression]]:
        """
        Decomposition of Element global function.
        
        The index variable must be within the bounds of the array.

        This decomposition uses a weighted sum over the array elements times Boolean indicator for the index.

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the expression representing the element value, and an empty list of constraints (no auxiliary variables needed)
        """
        arr, idx = self.args

        idx_lb, idx_ub = get_bounds(idx)
        assert idx_lb >= 0 and idx_ub < len(arr), "Element constraint is unsafe to decompose as it can be partial. Safen first using `cpmpy.transformations.safening.no_partial_functions`"

        return cp.sum((idx == i)*arr[i] for i in range(len(arr))), []

    def get_bounds(self) -> tuple[int, int]:
        """
        Returns the bounds of the global function

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the element value
        """
        arr, idx = self.args
        bnds = [get_bounds(x) for x in arr]
        return min(lb for lb,ub in bnds), max(ub for lb,ub in bnds)

    def __repr__(self) -> str:
        """
        Custom string representation of the Element global function in 'Arr[Idx]' format.

        Returns:
            str: String representation of the Element global function.
        """
        return f"{self.args[0]}[{self.args[1]}]"

def element(arg_list: list[Expression]) -> Element:
    """
    DEPRECATED: Use Element(arr,idx) instead of element([arr,idx]).

    Arguments:
        arg_list (list[Expression]): List containing array and index (2 elements)

    Returns:
        Element: An Element global function instance
    """
    warnings.warn("Deprecated, use Element(arr,idx) instead, will be removed in stable version", DeprecationWarning)
    assert (len(arg_list) == 2), "Element expression takes 2 arguments: Arr, Idx"
    return Element(arg_list[0], arg_list[1])


class Count(GlobalFunction):
    """
    The Count global function represents the number of occurrences of a value in an array
    """

    def __init__(self, arr: list[Expression], val: Union[int, Expression]):
        """
        Arguments:
            arr (list[Expression]): Array of expressions to count in
            val (Union[int, Expression]): 'Value' to count occurrences of (can also be an expression)
        """
        if not is_any_list(arr):
            raise TypeError(f"Count(arr, val) takes an array of expressions as first argument, not: {arr}")
        if is_any_list(val):
            raise TypeError(f"Count(arr, val) takes a numeric expression as second argument, not a list: {val}")
        super().__init__("count", [arr, val])

    def decompose(self) -> tuple[Expression, list[Expression]]:
        """
        Decomposition of the Count global function.
        
        Does not require the use of auxiliary variables, simply count the number of variables that take the given value.

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the sum expression representing the count, and an empty list of constraints (no auxiliary variables needed)
        """
        arr, val = self.args
        return cp.sum((a == val) for a in arr), []

    def value(self) -> Optional[int]:
        """
        Returns:
            Optional[int]: The number of occurrences of val in arr, or None if val or any element in arr is not assigned
        """
        arr, val = self.args
        vval = argval(val)
        if vval is None:
            return None

        varr = [argval(a) for a in arr]
        if any(v is None for v in varr):
            return None

        return sum((a == vval) for a in varr)

    def get_bounds(self) -> tuple[int, int]:
        """
        Returns the bounds of the global function

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the count value
        """
        arr, val = self.args
        return 0, len(arr)



class Among(GlobalFunction):
    """
    The Among global function counts how many variables in an array take values that are in a given set of values.
    
    This is similar to :class:`Count <cpmpy.expressions.globalfunctions.Count>`, but instead of counting occurrences of a single value,
    it counts occurrences of any value in a set. For example, `Among([x1, x2, x3, x4], [1, 2])`
    returns the number of variables among x1, x2, x3, x4 that take the value 1 or 2.
    """

    def __init__(self, arr: list[Expression], vals: list[int]):
        """
        Arguments:
            arr (list[Expression]): Array of expressions to count occurrences in
            vals (list[int]): Array of integer values to count the total number of occurrences of
        """
        if not is_any_list(arr) or not is_any_list(vals):
            raise TypeError(f"Among takes as input two arrays, not: {arr} and {vals}")
        if any(isinstance(val, Expression) for val in vals):
            raise TypeError(f"Among takes a set of integer values as input, not {vals}")
        super().__init__("among", [arr, vals])

    def decompose(self) -> tuple[Expression, list[Expression]]:
        """
        Decomposition of the Among global function.
        
        Among is decomposed into a sum of :class:`Count <cpmpy.expressions.globalfunctions.Count>` global functions, one for each value in the set.
        For example, `Among(arr, [1, 2, 3])` is decomposed as `Count(arr, 1) + Count(arr, 2) + Count(arr, 3)`.

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the sum expression representing the total number of occurrences, and an empty list of constraints (no auxiliary variables needed)
        """
        arr, vals = self.args
        return cp.sum(Count(arr, val) for val in vals), []

    def value(self) -> Optional[int]:
        """
        Returns:
            Optional[int]: The number of variables in arr that take a value present in vals, or None if any element in arr is not assigned
        """
        arr, vals = self.args
        varr = argvals(arr)  # recursive handling of nested structures
        if any(v is None for v in varr):
            return None

        return int(sum(np.isin(varr, vals)))

    def get_bounds(self) -> tuple[int, int]:
        """
        Returns the bounds of the global function

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the among count value
        """
        arr, vals = self.args
        return 0, len(arr)


class NValue(GlobalFunction):
    """
    The NValue global function counts the number of distinct values in an array.
    
    For example, if variables [x1, x2, x3, x4] take values [1, 2, 1, 3] respectively,
    then `NValue([x1, x2, x3, x4])` returns 3 (the distinct values are 1, 2, and 3).
    """

    def __init__(self, arr: list[Expression]):
        """
        Arguments:
            arr (list[Expression]): Array of expressions to count distinct values in
        """
        if not is_any_list(arr):
            raise ValueError(f"NValue(arr) takes an array as input, not: {arr}")
        super().__init__("nvalue", arr)

    def decompose(self) -> tuple[Expression, list[Expression]]:
        """
        Decomposition of the NValue global function.

        NValue is decomposed by checking, for each possible value in the domain range,
        whether at least one variable takes that value. The sum of these Boolean checks
        gives the number of distinct values.

        Based on "simple decomposition" from:

            Bessiere, Christian, et al. "Decomposition of the NValue constraint."
            International Conference on Principles and Practice of Constraint Programming.
            Berlin, Heidelberg: Springer Berlin Heidelberg, 2010.

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the sum expression representing the number of distinct values, and an empty list of constraints (no auxiliary variables needed)
        """
        lbs, ubs = get_bounds(self.args)
        lb, ub = min(lbs), max(ubs)

        return cp.sum(cp.any(a == v for a in self.args) for v in range(lb, ub+1)), []

    def value(self) -> Optional[int]:
        """
        Returns:
            Optional[int]: The number of distinct values in the array, or None if any element in arr is not assigned
        """
        vargs = [argval(a) for a in self.args]
        if any(v is None for v in vargs):
            return None

        return len(set(vargs))

    def get_bounds(self) -> tuple[int, int]:
        """
        Returns the bounds of the global function

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the number of distinct values
        """
        return 1, len(self.args)


class NValueExcept(GlobalFunction):
    """
    The NValueExcept global function counts the number of distinct values in an array,
    excluding a specified value.
    
    For example, if variables [x1, x2, x3, x4] take values [1, 2, 1, 0] respectively,
    then `NValueExcept([x1, x2, x3, x4], 0)` returns 2 (the distinct values are 1 and 2,
    excluding 0).
    """

    def __init__(self, arr: list[Expression], n: int):
        """
        Arguments:
            arr (list[Expression]): Array of expressions to count distinct values in
            n (int): Integer value to exclude from the count
        """
        if not is_any_list(arr):
            raise ValueError("NValueExcept takes an array as input")
        if not is_num(n):
            raise ValueError(f"NValueExcept takes an integer as second argument, but got {n} of type {type(n)}")
        super().__init__("nvalue_except",[arr, n])

    def decompose(self) -> tuple[Expression, list[Expression]]:
        """
        Decomposition of the NValueExcept global function.

        NValueExcept is decomposed similarly to :class:`NValue <cpmpy.expressions.globalfunctions.NValue>`, by checking for each possible value
        in the domain range (except the excluded value n) whether at least one variable
        takes that value, and counting for how many values that was the case.

        Returns:
            tuple[Expression, list[Expression]]: A tuple containing the sum expression representing the number of distinct values (excluding n), and an empty list of constraints (no auxiliary variables needed)
        """
        arr, n = self.args

        lbs, ubs = get_bounds(arr)
        lb, ub = min(lbs), max(ubs)

        return cp.sum([cp.any(a == v for a in arr) for v in range(lb, ub+1) if v != n]), []

    def value(self) -> Optional[int]:
        """
        Returns:
            Optional[int]: The number of distinct values in the array, excluding value n, or None if any element in arr is not assigned
        """
        arr, n = self.args
        varr = [argval(a) for a in arr]
        if any(v is None for v in varr):
            return None

        if n in varr:
            return len(set(varr)) - 1  # don't count 'n'
        else:
            return len(set(varr))

    def get_bounds(self) -> tuple[int, int]:
        """
        Returns the bounds of the global function

        Returns:
            tuple[int, int]: A tuple of (lower bound, upper bound) for the number of distinct values (excluding n)
        """
        arr, n = self.args
        return 0, len(arr)
