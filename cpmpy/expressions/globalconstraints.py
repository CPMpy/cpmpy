#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## globalconstraints.py
##
"""
    Global constraints conveniently express non-primitive constraints.

    Using global constraints
    ------------------------

    Solvers can have specialised implementations for global constraints. CPMpy has :class:`~cpmpy.expressions.globalconstraints.GlobalConstraint`
    expressions so that they can be passed to the solver as is when supported.

    If a solver does not support a global constraint (see :ref:`Solver Interfaces <solver-interfaces>`) then it will be automatically
    decomposed by calling its :func:`~cpmpy.expressions.globalconstraints.GlobalConstraint.decompose()` function.
    The :func:`~cpmpy.expressions.globalconstraints.GlobalConstraint.decompose()` function returns two arguments:
        - a list of simpler constraints replacing the global constraint
        - if the decomposition introduces *new variables*, then the second argument has to be a list
            of constraints that (totally) define those new variables

    As a user you **should almost never subclass GlobalConstraint()** unless you know of a solver that
    supports that specific global constraint, and that you will update its solver interface to support it.

    For all other use cases, it sufficies to write your own helper function that immediately returns the
    decomposition, e.g.:

    .. code-block:: python

        def alldifferent_except0(args):
            return [ ((var1!= 0) & (var2 != 0)).implies(var1 != var2) for var1, var2 in all_pairs(args)]


    Numeric global constraints
    --------------------------

    CPMpy also implements `Numeric Global Constraints`. For these, the CPMpy :class:`~cpmpy.expressions.globalconstraints.GlobalConstraint` does not
    exactly match what is implemented in the solver, but for good reason!!

    For example solvers may implement the global constraint ``Minimum(iv1, iv2, iv3) == iv4`` through an API
    call ``addMinimumEquals([iv1,iv2,iv3], iv4)``.

    However, CPMpy also wishes to support the expressions ``Minimum(iv1, iv2, iv3) > iv4`` as well as
    ``iv4 + Minimum(iv1, iv2, iv3)``. 

    Hence, the CPMpy global constraint only captures the ``Minimum(iv1, iv2, iv3)`` part, whose return type
    is numeric and can be used in any other CPMpy expression. Only at the time of transforming the CPMpy
    model to the solver API, will the expressions be decomposed and auxiliary variables introduced as needed
    such that the solver only receives ``Minimum(iv1, iv2, iv3) == ivX`` expressions.
    This is the burden of the CPMpy framework, not of the user who wants to express a problem formulation.


    Subclassing GlobalConstraint
    ----------------------------
    
    If you do wish to add a :class:`~cpmpy.expressions.globalconstraints.GlobalConstraint`, because it is supported by solvers or because you will do
    advanced analysis and rewriting on it, then preferably define it with a standard decomposition, e.g.:

    .. code-block:: python

        class my_global(GlobalConstraint):
            def __init__(self, args):
                super().__init__("my_global", args)

            def decompose(self):
                return [self.args[0] != self.args[1]] # your decomposition

    ..

    You can also implement a `.negate()` method if the global constraint has a better way to negate it than negating the decomposition.

    If it is a :class:`~cpmpy.expressions.globalfunctions.GlobalFunction` meaning that its return type is numeric (see :class:`~cpmpy.expressions.globalfunctions.Minimum` and :class:`~cpmpy.expressions.globalfunctions.Element`)
    then set `is_bool=False` in the super() constructor and preferably implement `.value()` accordingly.


    Alternative decompositions
    --------------------------
    
    For advanced use cases where you want to use another decomposition than the standard decomposition
    of a :class:`~cpmpy.expressions.globalconstraints.GlobalConstraint` expression, you can overwrite the :func:`~cpmpy.expressions.globalconstraints.GlobalConstraint.decompose` function of the class, e.g.:

    .. code-block:: python

        def my_circuit_decomp(self):
            return [self.args[0] == 1], [] # does not actually enforce circuit

        circuit.decompose = my_circuit_decomp # attach it, no brackets!

        vars = intvar(1,9, shape=10)
        constr = circuit(vars)

        Model(constr).solve()

    The above will use ``my_circuit_decomp``, if the solver does not
    natively support :class:`~cpmpy.expressions.globalconstraints.Circuit`.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        AllDifferent
        AllDifferentExcept0
        AllDifferentExceptN
        AllEqual
        AllEqualExceptN
        Circuit
        Inverse
        Table
        ShortTable
        NegativeTable
        Regular
        IfThenElse
        InDomain
        Xor
        Cumulative
        Precedence
        NoOverlap
        GlobalCardinalityCount
        Increasing
        Decreasing
        IncreasingStrict
        DecreasingStrict
        LexLess
        LexLessEq
        LexChainLess
        LexChainLessEq
        DirectConstraint

"""
import copy
import warnings
from typing import cast, Literal, Union, Optional, Sequence, Any
import numpy as np

import cpmpy as cp

from .core import Expression, BoolVal
from .variables import cpm_array, intvar, boolvar
from .utils import all_pairs, is_int, is_bool, STAR, get_bounds, argvals, is_any_list, flatlist, is_num, is_boolexpr
from .globalfunctions import * # XXX make this file backwards compatible


# Base class GlobalConstraint
class GlobalConstraint(Expression):
    """
        Abstract superclass of GlobalConstraints

        Like all expressions it has a ``.name`` and ``.args`` property.
        Overwrites the ``.is_bool()`` method as all global constraints are Boolean.
    """

    def is_bool(self) -> bool:
        """ 
        Returns whether the global constraint is a Boolean (return type) Operator.

        Returns:
            bool: True, global constraints are Boolean
        """
        return True

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
            Returns a decomposition into (a conjunction of) smaller constraints.

            The decomposition might create auxiliary variables,
            it can also use other global constraints as long as
            it does not create a circular dependency.

            To ensure equivalence of decomposition, we split into constraints determining the value of the global constraint, and defining-constraints.
            Defining constraints (totally) define new auxiliary variables needed for the decomposition, and can always be enforced at top-level.

            Tip: avoid creating auxiliary variables and use nested expressions instead!
            (especially, don't create Booleans but use (iv == v) expressions instead, better for common subexpression elimination!)

            Returns:
                tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        raise NotImplementedError("Decomposition for", self, "not available")

    def get_bounds(self):
        """
        Returns the bounds of a Boolean global constraint.
        Numerical global constraints should reimplement this.
        """
        return 0, 1

    def negate(self):
        """
        Returns the negation of this global constraint.
        Defaults to ~self, but subclasses can implement a better version,
        > Fages, François, and Sylvain Soliman. Reifying global constraints. Diss. INRIA, 2012.
        """
        return ~self


# Global Constraints (with Boolean return type)
def alldifferent(args: Sequence[Expression]):
    """
    .. deprecated:: 0.9.0
          Please use :class:`AllDifferent` instead.
    """
    warnings.warn("Deprecated, use AllDifferent(v1,v2,...,vn) instead, will be removed in "
                  "stable version", DeprecationWarning)
    return AllDifferent(*args) # unfold list as individual arguments


class AllDifferent(GlobalConstraint):
    """
    Enforces that all arguments have a different (distinct) value
    """

    def __init__(self, *args: Expression):
        """
        Arguments:
            args (Sequence[Expression]): List of expressions to be different from each other
        """
        super().__init__("alldifferent", flatlist(args))

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the AllDifferent global constraint using pairwise disequality constraints.

        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        return [var1 != var2 for var1, var2 in all_pairs(self.args)], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        vals = argvals(self.args)
        if any(v is None for v in vals):
            return None
        return len(set(vals)) == len(self.args)


class AllDifferentExceptN(GlobalConstraint):
    """
    Enforces that all arguments, except those equal to a value in n, have a different (distinct) value.

    Arguments:
        arr (Sequence[Expression]): List of expressions to be different from each other, except those equal to a value in n
        n (int or list[int]): Value or list of values that are excluded from satisfying the alldifferent condition
    """

    def __init__(self, arr: Sequence[Expression], n: Union[int, list[int]]):
        """
        Arguments:
            arr (Sequence[Expression]): List of expressions to be different from each other, except those equal to a value in n
            n (int or list[int]): Value or list of values that are excluded from the distinctness constraint
        """
        flatarr = flatlist(arr)
        if not is_any_list(n):
            n = cast(int, n)
            n = [n] # ensure n is a list of ints
        super().__init__("alldifferent_except_n", [flatarr, n])

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the AllDifferentExceptN global constraint using pairwise constraints.

        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        cons = []
        arr, n = self.args
        for x,y in all_pairs(arr):
            cond = (x == y)
            if is_bool(cond):
                cond = cp.BoolVal(cond)
            cons.append(cond.implies(cp.any(x == a for a in n))) # equivalent to (x in n) | (y in n) | (x != y)
        return cons, []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        vals, exclude_vals = argvals(self.args)
        if any(v is None for v in vals):
            return None

        vals = [v for v in vals if v not in frozenset(exclude_vals)]
        return len(set(vals)) == len(vals)


class AllDifferentExcept0(AllDifferentExceptN):
    """
    Enforces that all arguments, except those equal to 0, have a different (distinct) value.
    """
    def __init__(self, *args: Expression):
        """
        Arguments:
            args (Sequence[Expression]): List of expressions to be different from each other, except those equal to 0
        """
        super().__init__(flatlist(args), 0)


def allequal(args):
    """
    .. deprecated:: 0.9.0
          Please use :class:`AllEqual` instead.
    """
    warnings.warn("Deprecated, use AllEqual(v1,v2,...,vn) instead, will be removed in stable version",
                  DeprecationWarning)
    return AllEqual(*args) # unfold list as individual arguments


class AllEqual(GlobalConstraint):
    """
    Enforces that all arguments have the same value
    """
    def __init__(self, *args: Expression):
        """
        Arguments:
            args (Sequence[Expression]): List of expressions to have the same value
        """
        super().__init__("allequal", flatlist(args))

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the AllEqual global constraint using cascaded equality constraints.

        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        # arg0 == arg1, arg1 == arg2, arg2 == arg3... no need to post n^2 equalities
        return [x == y for x, y in zip(self.args[:-1], self.args[1:])], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        vals = argvals(self.args)
        if any(v is None for v in vals):
            return None
        return len(set(vals)) == 1


class AllEqualExceptN(GlobalConstraint):
    """
    Enforces that all arguments, except those equal to a value in n, have the same value.
    """

    def __init__(self, arr: Sequence[Expression], n: Union[int, list[int]]):
        """
        Arguments:
            arr (Sequence[Expression]): List of expressions to have the same value, except those equal to a value in n
            n (int or list[int]): Value or list of values that are excluded from the equality constraint
        """
        flatarr = flatlist(arr)
        if not is_any_list(n):
            n = cast(int, n)
            n = [n] # ensure n is a list of ints
        super().__init__("allequal_except_n", [flatarr, n])

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the AllEqualExceptN global constraint using pairwise constraints.

        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """

        arr, n = self.args
        constraints = []
        for x, y in all_pairs(arr):
            # x and y are equal, or one of them is equal to an excluded value
            constraints += [cp.any(x == a for a in n) | (x == y) | cp.any(y == a for a in n)]
        return constraints, []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        vals, exclude_vals = argvals(self.args)
        if any(v is None for v in vals):
            return None
        vals = [v for v in vals if v not in frozenset(exclude_vals)]
        return len(set(vals)) <= 1


def circuit(args):
    """
    .. deprecated:: 0.9.0
          Please use :class:`Circuit` instead.
    """
    warnings.warn("Deprecated, use Circuit(v1,v2,...,vn) instead, will be removed in stable version",
                  DeprecationWarning)
    return Circuit(*args) # unfold list as individual arguments


class Circuit(GlobalConstraint):
    """
    Enforces that the sequence of variables form a circuit, where x[i] = j means that node j is the successor of node i.
    """
    def __init__(self, *args: Expression):
        """
        Arguments:
            args (Sequence[Expression]): List of expressions representing the successors of the nodes to form the circuit
        """
        flatargs = flatlist(args)
        if len(flatargs) < 2:
            raise ValueError('Circuit constraint must be given a minimum of 2 variables')
        super().__init__("circuit", flatargs)

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
            Decomposition of the Circuit global constraint using auxiliary variables to reprsent the order in which we visit all the nodes.
            Auxiliary variables are defined in the defining part of the decomposition, which is alwasy enforced top-level.

        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        succ = cpm_array(self.args)
        n = len(succ)
        order = intvar(0,n-1, shape=n)
        defining: list[Expression] = []
        value: list[Expression] = []

        # We define the auxiliary order variables to represent the order we visit all the nodes.
        # `order[i] == succ[order[i - 1]]`
        # These constraints need to be in the defining part, since they define our auxiliary vars
        # However, this would make it impossible for ~circuit to be satisfied in some cases,
        # because there does not always exist a valid ordering
        # This happens when the variables in succ don't take values in the domain of 'order',
        # i.e. for succ = [9,-1,0], there is no valid ordering, but we satisfy ~circuit(succ)
        # We explicitly deal with these cases by defining the variable 'a' that indicates if we can define an ordering.
        # TODO at some point: do not introduce these auxiliary variables ourselves, rely on cse instead. 
        # Blocking factor: need safening and decomposing of global constraints to be integrated.
        # accumulator = 0
        # for i = 0..n-1:
        #    accumulator = succ[accumulator]  # creates an element global function
        # return [0 == accumulator, cp.AllDiff(succ), cp.all(succ >= 0), cp.all(succ <= n)], []


        lbs, ubs = get_bounds(succ)
        if min(lbs) > 0 or max(ubs) < n - 1:
            # no way this can be a circuit
            return [BoolVal(False)], []
        elif min(lbs) >= 0 and max(ubs) < n:
            # there always exists a valid ordering, since our bounds are tight
            a = BoolVal(True)
        else:
            # we may get values in succ that are outside the bounds of it's array length (making the ordering undefined)
            a = boolvar()
            defining += [a == ((cp.Minimum(succ) >= 0) & (cp.Maximum(succ) < n))]
            for i in range(n):
                defining += [(~a).implies(order[i] == 0)]  # assign arbitrary value, so a is totally defined.

        value += [AllDifferent(succ)]  # different successors
        value += [AllDifferent(order)]  # different orders
        value += [order[n - 1] == 0]  # symmetry breaking, last one is '0'
        defining += [a.implies(order[0] == succ[0])]
        for i in range(1, n):
            defining += [a.implies(
                order[i] == succ[order[i - 1]])]  # first one is successor of '0', ith one is successor of i-1
        
        return value, defining

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        idx = 0
        visited = set()
        arr = argvals(self.args)

        while idx not in visited:
            if idx is None:
                return None # not assigned
            if not (0 <= idx < len(arr)):
                return False # out of bounds
            visited.add(idx)
            idx = arr[idx]

        return len(visited) == len(self.args) and idx == 0


class Inverse(GlobalConstraint):
    """
    Enforces that the forward and reverse arrays represent the inverse function of one another.
    I.e., fwd[i] == x <==> rev[x] == i

    Also known as channeling / assignment constraint.
    """
    def __init__(self, fwd: Sequence[Expression], rev: Sequence[Expression]):
        """
        Arguments:
            fwd (Sequence[Expression]): List of expressions representing the forward function
            rev (Sequence[Expression]): List of expressions representing the reverse function
        """
        if len(fwd) != len(rev):
            raise ValueError("Length of fwd and rev must be equal for Inverse constraint")
        super().__init__("inverse", [fwd, rev])

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the Inverse global constraint using Element global function constraints, and explicit safening.

        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """

        fwd, rev = self.args
        rev = cpm_array(rev)

        constraining, defining = [], []
        for i,x in enumerate(fwd):
            if is_num(x) and not 0 <= x < len(rev): 
                return [cp.BoolVal(False)], [] # can never satisfy the Inverse constraint
           
            lb, ub = get_bounds(x)
            if lb >= 0 and ub < len(rev): # safe, index is within bounds
                constraining.append(rev[x] == i)
            else: # partial! need safening here
                is_defined, total_expr, toplevel = cp.transformations.safening._safen_range(rev[x], (0, len(rev)-1), 1)
                constraining += [is_defined, total_expr == i]
                defining += toplevel
        
        return constraining, defining

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        fwd, rev = argvals(self.args)
        if any(x is None for x in fwd) or any(y is None for y in rev):
            return None
        # explicit check for partial element constraints
        if any(not (0 <= x < len(rev)) for x in fwd):
            return False
        if any(not (0 <= y < len(fwd)) for y in rev):
            return False
        return all(rev[x] == i for i, x in enumerate(fwd))


class Table(GlobalConstraint):
    """
    Enforces that the values of the variables in 'array' correspond to a row in 'table'.
    """
    def __init__(self, array: Sequence[Expression], table: Union[list[list[int]], np.ndarray]):
        """
        Arguments:
            array (Sequence[Expression]): List of expressions representing the array of variables
            table (list[list[int]] | np.ndarray): List of lists of integers or 2D ndarray of ints representing the table.
        """
        array = flatlist(array)
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError(f"the first argument of a Table constraint should only contain variables/expressions: {array}")
        if isinstance(table, np.ndarray):  # Ensure it is a list
            assert table.ndim == 2, "Table's table must be a 2D array"
            assert table.dtype != object, "Table's table must have primitive type, not 'object'/expressions"
            table = table.tolist()
        else:
            tmp = np.array(table)
            assert tmp.ndim == 2, "Table's table must be a 2D array"
            assert tmp.dtype != object, "Table's table must have primitive type, not 'object'/expressions"
            
        super().__init__("table", [array, table])

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the Table global constraint. Enforces at least one row of the table is assigned to the array.
        "
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        arr, tab = self.args
        return [cp.any(cp.all(ai == ri for ai, ri in zip(arr, row)) for row in tab)], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        arr, tab = self.args
        arrval = argvals(arr)
        if any(x is None for x in arrval):
            return None
        return arrval in tab

    def negate(self):
        return NegativeTable(self.args[0], self.args[1])

    # specialisation to avoid recursing over big tables
    def has_subexpr(self):
        if not hasattr(self, '_has_subexpr'): # if _has_subexpr has not been computed before or has been reset
            arr, tab = self.args  # the table 'tab' is asserted to only hold constants
            self._has_subexpr = any(a.has_subexpr() for a in arr)
        return self._has_subexpr

class ShortTable(GlobalConstraint):
    """
    Extension of the `Table` constraint where the `table` matrix may contain wildcards (STAR), meaning there are
    no restrictions for the corresponding variable in that tuple.
    """
    def __init__(self, array: Sequence[Expression], table: Union[list[list[int|Literal["*"]]], np.ndarray]):
        """
        Arguments:
            array (Sequence[Expression]): List of expressions representing the array of variables
            table (list[list[int | '*']] | np.ndarray): List of lists or 2D ndarray; entries are integers or STAR ('*')
                STAR represents a wildcard (corresponding variable can take any value).
        """
        array = flatlist(array)
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError("The first argument of a Table constraint should only contain variables/expressions")
        if isinstance(table, np.ndarray):
            assert table.ndim == 2, "ShortTable's table must be a 2D array"
            table = table.tolist()
        if not all(is_int(x) or x == STAR for row in table for x in row):
            raise TypeError(f"elements in argument `table` should be integer or {STAR}")
        super().__init__("short_table", [array, table])

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the ShortTable global constraint. Enforces at least one row of the table is assigned to the array.
        "
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        arr, tab = self.args
        return [cp.any(cp.all(ai == ri for ai, ri in zip(arr, row) if ri != STAR) for row in tab)], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        arr, tab = self.args
        arrval = argvals(arr)
        if any(x is None for x in arrval):
            return None
        arrval = np.asarray(arrval, dtype=int)
        tab = np.asarray(tab)
        for row in tab:
            mask = (row != STAR)
            if (row[mask].astype(int) == arrval[mask]).all():
                return True
        return False

    # specialisation to avoid recursing over big tables
    def has_subexpr(self):
        if not hasattr(self, '_has_subexpr'): # if _has_subexpr has not been computed before or has been reset
            arr, tab = self.args # the table 'tab' can only hold constants, never a nested expression
            self._has_subexpr = any(a.has_subexpr() for a in arr)
        return self._has_subexpr

class NegativeTable(GlobalConstraint):
    """
    The values of the variables in 'array' do not correspond to any row in 'table'.
    """
    def __init__(self, array: Sequence[Expression], table: Union[list[list[int]], np.ndarray]):
        """
        Arguments:
            array (Sequence[Expression]): List of expressions representing the array of variables
            table (list[list[int]] | np.ndarray): List of lists of integers or 2D ndarray of ints representing the table.
        """
        array = flatlist(array)
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError(f"the first argument of a NegativeTable constraint should only contain variables/expressions: {array}")
        if isinstance(table, np.ndarray):  # Ensure it is a list
            assert table.ndim == 2, "NegativeTable's table must be a 2D array"
            assert table.dtype != object, "NegativeTable's table must have primitive type, not 'object'/expressions"
            table = table.tolist()
        else:
            tmp = np.array(table)
            assert tmp.ndim == 2, "NegativeTable's table must be a 2D array"
            assert tmp.dtype != object, "NegativeTable's table must have primitive type, not 'object'/expressions"
            
        super().__init__("negative_table", [array, table])

    def decompose(self):
        """
        Decomposition of the NegativeTable global constraint. 
        Enforces that the values of the variables in 'array' do not correspond to any row in 'table'.
        
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        arr, tab = self.args
        return [cp.all(cp.any(ai != ri for ai, ri in zip(arr, row)) for row in tab)], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        arr, tab = self.args
        arrval = argvals(arr)
        if any(x is None for x in arrval):
            return None
        return arrval not in tab

    # specialisation to avoid recursing over big tables
    def has_subexpr(self):
        if not hasattr(self, '_has_subexpr'): # if _has_subexpr has not been computed before or has been reset
            arr, tab = self.args # the table 'tab' can only hold constants, never a nested expression
            self._has_subexpr = any(a.has_subexpr() for a in arr)
        return self._has_subexpr

    def negate(self):
        return Table(self.args[0], self.args[1])
    

class Regular(GlobalConstraint):
    """
    Regular-constraint (or Automaton-constraint)
    Takes as input a sequence of variables and a automaton representation using a transition table.
    The constraint is satisfied if the sequence of variables corresponds to an accepting path in the automaton.

    The automaton is defined by a list of transitions, a starting node and a list of accepting nodes.
    The transitions are represented as a list of tuples, where each tuple is of the form (id1, value, id2).
    An id is an integer or string representing a state in the automaton, and value is an integer representing the value of the variable in the sequence.
    The starting node is an integer or string representing the starting state of the automaton.
    The accepting nodes are a list of integers or strings representing the accepting states of the automaton.

    Example: an automaton that accepts the language 0*10* (exactly 1 variable taking value 1) is defined as:
        cp.Regular(array = cp.intvar(0,1, shape=4),
                   transitions = [("A",0,"A"), ("A",1,"B"), ("B",0,"C"), ("C",0,"C")],
                   start = "A",
                   accepting = ["C"])
    """
    def __init__(self, array: Sequence[Expression], transitions: list[tuple[int|str, int, int|str]], start: int|str, accepting: list[int|str]):
        array = flatlist(array)
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError("The first argument of a regular constraint should only contain variables/expressions")
        
        if not is_any_list(transitions):
            raise TypeError("The second argument of a regular constraint should be a list of transitions")
        _node_type = type(transitions[0][0])
        for s,v,e in transitions:
            if not isinstance(s, _node_type) or not isinstance(e, _node_type) or not isinstance(v, int):
                raise TypeError(f"The second argument of a regular constraint should be a list of transitions ({_node_type}, int, {_node_type})")
        if not isinstance(start, _node_type):
            raise TypeError("The third argument of a regular constraint should be a node id")
        if not (is_any_list(accepting) and all(isinstance(e, _node_type) for e in accepting)):
            raise TypeError("The fourth argument of a regular constraint should be a list of node ids")
        super().__init__("regular", [list(array), list(transitions), start, list(accepting)])

        node_set = set()
        self.trans_dict = {}
        for s, v, e in transitions:
            node_set.update([s,e])
            self.trans_dict[(s, v)] = e
        self.nodes = sorted(node_set)
        # normalize node_ids to be 0..n-1, allows for smaller domains
        self.node_map = {n: i for i, n in enumerate(self.nodes)}

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the Regular global constraint. 
        Encodes the automaton by encoding the transition table into `class:cpmpy.expressions.globalconstraints.Table` constraints.
        Then enforces that the last state is accepting.
        
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        # Decompose to transition table using Table constraints
        
        arr, transitions, start, accepting = self.args
        lbs, ubs = get_bounds(arr)
        lb, ub = min(lbs), max(ubs)
        
        transitions = [[self.node_map[n_in], v, self.node_map[n_out]] for n_in, v, n_out in transitions]

        # add a sink node for transitions that are not defined
        sink = len(self.nodes)
        transitions += [[self.node_map[n], v, sink] for n in self.nodes for v in range(lb, ub + 1) if (n, v) not in self.trans_dict]
        transitions += [[sink, v, sink] for v in range(lb, ub + 1)]

        # keep track of current state when traversing the array
        state_vars = intvar(0, sink, shape=len(arr))
        id_start = self.node_map[start]
        # optimization: we know the entry node of the automaton, results in smaller table
        defining = [Table([arr[0], state_vars[0]], [[v,e] for s,v,e in transitions if s == id_start])]        
        # define the rest of the automaton using transition table
        defining += [Table([state_vars[i - 1], arr[i], state_vars[i]], transitions) for i in range(1, len(arr))]
        
        # constraint is satisfied iff last state is accepting
        return [InDomain(state_vars[-1], [self.node_map[e] for e in accepting])], defining

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        arr, transitions, start, accepting = self.args
        arrval = argvals(arr)
        if any(x is None for x in arrval):
            return None
        curr_node = start
        for v in arrval:
            if (curr_node, v) in self.trans_dict:
                curr_node = self.trans_dict[curr_node, v]
            else:
                return False
        return curr_node in accepting

# syntax of the form 'if b then x == 9 else x == 0' is not supported (no override possible)
# same semantic as CPLEX IfThenElse constraint
# https://www.ibm.com/docs/en/icos/12.9.0?topic=methods-ifthenelse-method
class IfThenElse(GlobalConstraint):
    """
    Enforces a conditional expression of the form: if condition then if_true else if_false.
    `condition`, `if_true` and `if_false` are be boolean expressions.
    """
    def __init__(self, condition: Expression, if_true: Expression, if_false: Expression):
        if not is_boolexpr(condition) or not is_boolexpr(if_true) or not is_boolexpr(if_false):
            raise TypeError(f"only boolean expression allowed in IfThenElse: Instead got "
                            f"{condition, if_true, if_false}")
        super().__init__("ite", [condition, if_true, if_false])

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        condition, if_true, if_false = argvals(self.args)
        if condition is None or if_true is None or if_false is None:
            return None
        if condition:
            return if_true
        else:
            return if_false

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the IfThenElse global constraint.
        Enforces that the condition is satisfied.
        "
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        condition, if_true, if_false = self.args
        if is_bool(condition):
            condition = cp.BoolVal(condition) # ensure it is a CPMpy expression
        return [condition.implies(if_true), (~condition).implies(if_false)], []

    def __repr__(self):
        condition, if_true, if_false = self.args
        return "If {} Then {} Else {}".format(condition, if_true, if_false)

    def negate(self):
        return IfThenElse(self.args[0], self.args[2], self.args[1])



class InDomain(GlobalConstraint):
    """
    Enforces the expression is assigned to a value in the given domain.
    """

    def __init__(self, expr: Expression, arr: list[int]):
        """
        Arguments:
            expr (Expression): Expression to be assigned to a value in the given domain
            arr (list[int]): List of integers representing the domain
        """
        if not all(is_int(x) for x in arr):
            raise TypeError("The second argument of an InDomain constraint should be a list of integer constants")
        super().__init__("InDomain", [expr, arr])

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the InDomain global constraint.
        Enforces that the expression is assigned to a value in the given domain.
        
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        expr, arr = self.args
        lb, ub = get_bounds(expr)
        
        defining = []
        #if expr is not a var
        if not isinstance(expr,Expression):
            aux = intvar(lb, ub)
            defining.append(aux == expr)
            expr = aux

        return [expr != val for val in range(lb, ub + 1) if val not in arr], defining

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        expr, arr = self.args
        exprval = argvals(expr)
        if exprval is None:
            return None
        return exprval in arr

    def __repr__(self):
        return "{} in {}".format(self.args[0], self.args[1])

    def negate(self):
        lb, ub = get_bounds(self.args[0])
        return InDomain(self.args[0],
                        [v for v in range(lb,ub+1) if v not in set(self.args[1])])


class Xor(GlobalConstraint):
    """
    Enforces the exclusive-or relation of the arguments.
    Supports n-ary xor-constraints, which are treated as cascaed binary xor-constraints.
    Equivalent to `sum(args) % 2 == 1`
    """

    def __init__(self, arg_list: Sequence[Expression]):
        """
        Arguments:
            arg_list (Sequence[Expression]): List of Boolean expressions to be xor'ed
        """
        if not all(is_boolexpr(arg) for arg in arg_list):
            raise TypeError("Only Boolean arguments allowed in Xor global constraint: {}".format(arg_list))
        # convention for commutative binary operators:
        # swap if right is constant and left is not
        arg_list = list(arg_list)
        if len(arg_list) == 2 and is_num(arg_list[1]):
            arg_list[0], arg_list[1] = arg_list[1], arg_list[0]
        super().__init__("xor", list(arg_list))

    def decompose(self):
        """
        Decomposition of the Xor global constraint.
        Recursively decomposes the constraint into a chain of binary xor-constraints, represented using a sum.
        
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        # there are multiple decompositions possible, Recursively using sum allows it to be efficient for all solvers.
        decomp = [sum(self.args[:2]) == 1]
        if len(self.args) > 2:
            decomp = Xor(decomp + self.args[2:]).decompose()[0]
        return decomp, []

    def value(self):
        return sum(argvals(self.args)) % 2 == 1

    def __repr__(self):
        if len(self.args) == 2:
            return "{} xor {}".format(*self.args)
        return "xor({})".format(self.args)

    def negate(self):
        # negate one of the arguments, ideally a variable
        new_args = None
        for i, a in enumerate(self.args):
            if isinstance(a, _BoolVarImpl):
                new_args = self.args[:i] + [~a] + self.args[i+1:]
                break

        if new_args is None:# did not find a Boolean variable to negate
            # pick first arg, and push down negation
            new_args = list(self.args)
            new_args[0] = cp.transformations.negation.recurse_negation(self.args[0])

        return Xor(new_args)


class Cumulative(GlobalConstraint):
    """
    Enforces that a set of tasks is scheduled such that the capacity of the resource is never exceeded and enforces:
        - duration >= 0
        - demand >= 0
        - start + duration == end

    Equivalent to :class:`~cpmpy.expressions.globalconstraints.NoOverlap` when demand and capacity are equal to 1.
    Supports both varying demand across tasks or equal demand for all jobs.
    """
    def __init__(self, start: Sequence[Expression], duration: Sequence[Expression], end: Optional[Sequence[Expression]] = None, demand: Optional[Union[Sequence[Expression],Expression]] = None, capacity: Optional[Expression] = None):
        """
            Arguments:
                start (Sequence[Expression]): List of Expression objects representing the start times of the tasks
                duration (Sequence[Expression]): List of Expression objects representing the durations of the tasks
                end (Sequence[Expression] | None): optional, list of Expression objects representing the end times of the tasks
                demand (Sequence[Expression] | Expression | None): List of Expression objects or single Expression to indicate constant demand for all tasks
                capacity (Expression | None): Expression object representing the capacity of the resource
        """

        if not is_any_list(start):
            raise TypeError("start should be a list")
        if not is_any_list(duration):
            raise TypeError("duration should be a list")
        if end is not None and not is_any_list(end):
            raise TypeError("end should be a list if it is provided")
        if demand is None:
            raise TypeError("demand should be provided but was None")
        if capacity is None:
            raise TypeError("capacity should be provided but was None")
        
        if len(start) != len(duration):
            raise ValueError("Start and duration should have equal length")
        if end is not None and len(start) != len(end):
            raise ValueError(f"Start and end should have equal length, but got {len(start)} and {len(end)}")

        if is_any_list(demand):
            if len(demand) != len(start):
                raise ValueError(f"Demand should be supplied for each task or be single constant, but got {len(demand)} and {len(start)}")
        else: # constant demand
            demand = [demand] * len(start)

        super(Cumulative, self).__init__("cumulative", [list(start), list(duration), list(end) if end is not None else None, list(demand), capacity])

    
    def decompose(self, how:str="auto") -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decompose the Cumulative constraint
        Support time-based decomposition or task-based decomposition.
        By default, we heuristically select the best decomposition based on the number of tasks and the horizon.

        Arguments:
            how (str): how the cumulative constraint should be decomposed, can be "time", "task", or "auto" (default)

        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """

        if how not in ["time", "task", "auto"]:
            raise ValueError(f"how can only be time, task, or auto (default), but got {how}")

        start, duration, end, demand, capacity = self.args

        lbs, ubs = get_bounds(start)
        horizon = max(ubs) - min(lbs)
        if (how == "time") or (how == "auto" and len(start) <= horizon):
            return self._time_decomposition()
        elif (how == "task") or (how == "auto" and len(start) > horizon):
            return self._task_decomposition()
        raise Exception

    def _task_decomposition(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Task-based decomposition of the cumulative constraint.
        Schutt, Andreas, et al. "Why cumulative decomposition is not as bad as it sounds."
        International Conference on Principles and Practice of Constraint Programming. Springer, Berlin, Heidelberg, 2009.
        
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        start, duration, end, demand, capacity = self.args

        cons = [d >= 0 for d in duration]  # enforce non-negative durations
        cons += [h >= 0 for h in demand]  # enforce non-negative demand

        # set duration of tasks, only if end is user-provided
        if end is None:
            end = [start[i] + duration[i] for i in range(len(start))]
        else:
            cons += [start[i] + duration[i] == end[i] for i in range(len(start))]

        # demand doesn't exceed capacity
        # tasks are uninterruptible, so we only need to check each starting point of each task
        # I.e., for each task, we check if it can be started, given the tasks that are already running.
        for t in range(len(start)):
            demand_at_start_of_t = []
            for j in range(len(start)):
                if t != j:
                    demand_at_start_of_t += [demand[j] * ((start[j] <= start[t]) & (end[j] > start[t]))]

            cons += [(demand[t] + sum(demand_at_start_of_t)) <= capacity]

        return cons, []

    def _time_decomposition(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Time-resource decomposition of the cumulative constraint.
        Schutt, Andreas, et al. "Why cumulative decomposition is not as bad as it sounds."
        International Conference on Principles and Practice of Constraint Programming. Springer, Berlin, Heidelberg, 2009.
        
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        start, duration, end, demand, capacity = self.args

        cons = [d >= 0 for d in duration] # enforce non-negative durations
        cons += [h >= 0 for h in demand] # enforce non-negative demand

        # set duration of tasks, only if end is user-provided
        if end is None:
            end = [start[i] + duration[i] for i in range(len(start))]
        else:
            cons += [start[i] + duration[i] == end[i] for i in range(len(start))]

        # demand doesn't exceed capacity
        # for each time-step, we check if the running demand does not exceed the capacity
        lbs, ubs = get_bounds(start)
        lb, ub = min(lbs), max(ubs)
        for t in range(lb,ub+1):
            cons += [cp.sum(d * ((s <= t) & (e > t)) for s,e,d in zip(start, end, demand)) <= capacity]

        return cons, []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        start, dur, end, demand, capacity = self.args
        
        start, dur, demand, capacity = argvals([start, dur, demand, capacity])
        if end is None:
            end = [s + d for s,d in zip(start, dur)]
        else:
            end = argvals(end)

        if any(a is None for a in flatlist([start, dur, end, demand, capacity])):
            return None
                
        if any(d < 0 for d in dur):
            return False
        if any(s + d != e for s,d,e in zip(start, dur, end)):
            return False

        if any(d < 0 for d in demand):
            return False

        # ensure demand doesn't exceed capacity
        lb, ub = min(start), max(end)
        start, end = np.array(start), np.array(end) # eases check below
        for t in range(lb, ub+1):
            if capacity < sum(demand * ((start <= t) & (end > t))):
                return False

        return True
    
    def __repr__(self) -> str:
        """
        Returns:
            str: String representation of the cumulative constraint
        """
        start, dur, end, demand, capacity = self.args
        if end is None:
            return f"Cumulative({start}, {dur}, {demand}, {capacity})"
        else:
            return f"Cumulative({start}, {dur}, {end}, {demand}, {capacity})"

class NoOverlap(GlobalConstraint):
    """
    Enforces that a set of tasks are scheduled without overlapping, and enforces:
        - duration >= 0
        - start + duration == end
    """

    def __init__(self, start: Sequence[Expression], duration: Sequence[Expression], end: Optional[Sequence[Expression]] = None):
        """
        Arguments:
            start (Sequence[Expression]): List of Expression objects representing the start times of the tasks
            duration (Sequence[Expression]): List of Expression objects representing the durations of the tasks
            end (Sequence[Expression] | None): optional, list of Expression objects representing the end times of the tasks
        """
       
        if not is_any_list(start):
            raise TypeError("start should be a list")
        if not is_any_list(duration):
            raise TypeError("duration should be a list")
        if end is not None and not is_any_list(end):
            raise TypeError("end should be a list if it is provided")
        
        if len(start) != len(duration):
            raise ValueError("Start and duration should have equal length")
        if end is not None and len(start) != len(end):
            raise ValueError(f"Start and end should have equal length, but got {len(start)} and {len(end)}")
        
        super().__init__("no_overlap", [list(start), list(duration), list(end) if end is not None else None])

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the NoOverlap constraint, using pairwise no-overlap constraints.
        
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        start, dur, end = self.args
        cons = [d >= 0 for d in dur]
        
        if end is None:
            end = [s+d for s,d in zip(start, dur)]
        else: # can use the expression directly below
            cons += [s + d == e for s,d,e in zip(start, dur, end)]
            
        for (s1, e1), (s2, e2) in all_pairs(zip(start, end)):
            cons += [(e1 <= s2) | (e2 <= s1)]
        return cons, []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        start, dur, end = argvals(self.args)
        if end is None:
            if any(s is None for s in start) or any(d is None for d in dur):
                return None
            end = [s + d for s,d in zip(start, dur)]
        else:
            if any(s is None for s in start) or any(d is None for d in dur) or any(e is None for e in end):
                return None
       
        if any(d < 0 for d in dur):
            return False
        if any(s + d != e for s,d,e in zip(start, dur, end)):
            return False
        for (s1,d1), (s2,d2) in all_pairs(zip(start,dur)):
            if s1 + d1 > s2 and s2 + d2 > s1:
                return False
        return True
    
    def __repr__(self) -> str:
        """
        Returns:
            str: String representation of the NoOverlap constraint
        """
        start, dur, end = self.args
        if end is None:
            return f"NoOverlap({start}, {dur})"
        else:
            return f"NoOverlap({start}, {dur}, {end})"

class Precedence(GlobalConstraint):
    """
    Enforces a precedence relationship between a set of variables.
    Given an array of variables X and a list of values P, values in P must appear in X in the order specified by P.
    I.e., if X[i] = P[j+1], then there exists a X[i'] = P[j] with i' < i

    Examples:
        - X = [1,2,1,3] satisfies the precedence [1,2,3].
        - X = [4,1,2,1,3] also satisfies the precedence, as values not appearing in P can appear in any order.
        - X = [2,1,3] does not satisfy the precedence, as 1 does not appear before 2.
    """
    def __init__(self, vars: Sequence[Expression], precedence: list[int]):
        """
        Arguments:
            vars (Sequence[Expression]): List of Expression objects representing the variables
            precedence (list[int]): List of integers representing the precedence
        """
        if not is_any_list(vars):
            raise TypeError("Precedence expects a list of variables as first argument, but got", vars)
        if not is_any_list(precedence) or not all(is_num(p) for p in precedence):
            raise TypeError("Precedence expects a list of values as second argument, but got", precedence)
        super().__init__("precedence", [list(vars), list(precedence)])

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition based on:
        Law, Yat Chiu, and Jimmy HM Lee. "Global constraints for integer and set value precedence."
        Principles and Practice of Constraint Programming–CP 2004: 10th International Conference, CP 2004

        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """

        args, precedence = self.args
        args = cpm_array(args)
        constraints = []
        for s,t in zip(precedence[:-1], precedence[1:]):
            # constraint 1 from paper
            constraints.append(args[0] != t) 
            # constraint 2 from paper
            for j in range(1,len(args)):
                lhs = args[j] == t
                if is_bool(lhs):  # args[j] and t could both be constants
                    lhs = BoolVal(lhs)
                constraints += [lhs.implies(cp.any(args[:j] == s))]
        return constraints, []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        args, precedence = self.args
        vals = argvals(args)
        if any(v is None for v in vals):
            return None
        vals = np.array(vals)
        for s,t in zip(precedence[:-1], precedence[1:]):
            if vals[0] == t:
                return False
            for j in range(len(args)):
                if vals[j] == t and sum(vals[:j] == s) == 0:
                    return False
        return True

class GlobalCardinalityCount(GlobalConstraint):
    """
    Enforces that the number of occurrences of each value `vals[i]` in the list of variables `vars` is equal to `occ[i]`.
    """

    def __init__(self, vars: Sequence[Expression], vals: list[int], occ: Sequence[Expression], closed: bool = False):
        """
        Arguments:
            vars (Sequence[Expression]): List of Expression objects representing the variables
            vals (list[int]): List of integers representing the values
            occ (Sequence[Expression]): List of Expression objects representing the number of occurrences of each value
            closed (bool): Whether the constraint is closed, if true, `vars` can only take values in `vals`
        """
        if not is_any_list(vars):
            raise TypeError("GlobalCardinalityCount expects a list of variables, but got", vars)
        if not is_any_list(vals) or not all(is_num(v) for v in vals):
            raise TypeError("GlobalCardinalityCount expects a list of values, but got", vals)
        if not is_any_list(occ):
            raise TypeError("GlobalCardinalityCount expects a list of variables as occurrences, but got", occ)
        if len(vals) != len(occ):
            raise ValueError(f"Number of values and occurrences must be equal, but got {len(vals)} and {len(occ)}")
        super().__init__("gcc", [list(vars), list(vals), list(occ)])
        self.closed = closed

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the GlobalCardinalityCount constraint.
        Uses a conjunction of Count global function constraints.
        """
        vars, vals, occ = self.args
        constraints = [cp.Count(vars, i) == v for i, v in zip(vals, occ)]
        if self.closed:
            constraints += [InDomain(v, vals) for v in vars]
        return constraints, []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        vars, vals, occ = self.args
        vars, occ = argvals([vars, occ])
        if any(x is None for x in vars + occ):
            return None

        vals = np.array(vals)
        for val, cnt in zip(vals, occ):
            if sum(vars == val) != cnt:
                return False
        
        if self.closed and any(v not in frozenset(vals) for v in vars):
            return False

        return True

class Increasing(GlobalConstraint):
    """
    Enforces that the expressions are assigned to (non-strictly) increasing values.
    """

    def __init__(self, *args: Expression):
        """
        Arguments:
            args (Sequence[Expression]): List of expressions to be assigned to increasing values
        """
        super().__init__("increasing", flatlist(args))

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the Increasing constraint.

        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        args = self.args
        return [args[i] <= args[i+1] for i in range(len(args)-1)], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        args = argvals(self.args)
        if any(x is None for x in args):
            return None
        return all(args[i] <= args[i+1] for i in range(len(args)-1))


class Decreasing(GlobalConstraint):
    """
    Enforces that the expressions are assigned to (non-strictly) decreasing values.
    """

    def __init__(self, *args: Expression):
        """
        Arguments:
            args (Sequence[Expression]): List of expressions to be assigned to decreasing values
        """
        super().__init__("decreasing", flatlist(args))

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the Decreasing constraint.
        
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        args = self.args
        return [args[i] >= args[i+1] for i in range(len(args)-1)], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        args = argvals(self.args)
        if any(x is None for x in args):
            return None
        return all(args[i] >= args[i+1] for i in range(len(args)-1))


class IncreasingStrict(GlobalConstraint):
    """
    Enforces that the expressions are assigned to strictly increasing values.
    """

    def __init__(self, *args: Expression):
        """
        Arguments:
            args (Sequence[Expression]): List of expressions to be assigned to strictly increasing values
        """
        super().__init__("strictly_increasing", flatlist(args))

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the IncreasingStrict constraint.
        
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        args = self.args
        return [args[i] < args[i+1] for i in range(len(args)-1)], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        args = argvals(self.args)
        if any(x is None for x in args):
            return None
        args = argvals(self.args)
        return all(args[i] < args[i+1] for i in range(len(args)-1))


class DecreasingStrict(GlobalConstraint):
    """
    Enforces that the expressions are assigned to strictly decreasing values.
    """

    def __init__(self, *args: Expression):
        """
        Arguments:
            args (Sequence[Expression]): List of expressions to be assigned to strictly decreasing values
        """
        super().__init__("strictly_decreasing", flatlist(args))

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the DecreasingStrict constraint.
        
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        args = self.args
        return [(args[i] > args[i+1]) for i in range(len(args)-1)], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        args = argvals(self.args)
        if any(x is None for x in args):
            return None
        args = argvals(self.args)
        return all(args[i] > args[i+1] for i in range(len(args)-1))


class LexLess(GlobalConstraint):
    """ 
    Enforces that the first list is lexicographically smaller than the second list.
    """
    def __init__(self, list1: Sequence[Expression], list2: Sequence[Expression]):
        """
        Arguments:
            list1 (Sequence[Expression]): First list of expressions to be compared lexicographically
            list2 (Sequence[Expression]): Second list of expressions to be compared lexicographically
        """ 
        if len(list1) != len(list2):
            raise ValueError(f"The 2 lists given in LexLess must have the same size: list1 length is {len(list1)} and list2 length is {len(list2)}")
        super().__init__("lex_less", [list1, list2])

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Implementation inspired by Hakan Kjellerstrand (http://hakank.org/cpmpy/cpmpy_hakank.py)

        The decomposition creates auxiliary Boolean variables and constraints that
        collectively ensure X is lexicographically less than Y
        The auxiliary boolean vars are defined to represent if the given lists are lexicographically ordered
        (less or equal) up to the given index.
        Decomposition enforces through the constraining part that the first boolean variable needs to be true, and thus
        through the defining part it is enforced that if it is not strictly lexicographically less in a given index,
        then next index must be lexicographically less or equal. It needs to be strictly less in at least one index.

        The use of auxiliary Boolean variables bvar ensures that the constraints propagate immediately,
        maintaining arc-consistency. Each bvar[i] enforces the lexicographic ordering at each position, ensuring that
        every value in the domain of X[i] can be extended to a consistent value in the domain of $Y_i$ for all
        subsequent positions.

        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        X, Y = cpm_array(self.args)

        if len(X) == 0 == len(Y):
            return [cp.BoolVal(False)], [] # based on the decomp, it's false...

        bvar = boolvar(shape=(len(X) + 1))

        # Constraint ensuring that each element in X is less than or equal to the corresponding element in Y,
        # until a strict inequality is encountered.
        defining = []
        defining.extend(bvar == ((X <= Y) & ((X < Y) | bvar[1:])))  # vectorized expression, treat as list
        # enforce the last element to be true iff (X[-1] < Y[-1]), enforcing strict lexicographic order
        defining.append(bvar[-1] == (X[-1] < Y[-1]))
        constraining = [bvar[0]]

        return constraining, defining

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        X, Y = argvals(self.args)
        if any(val is None for val in X + Y):
            return None
        return any((X[i] < Y[i]) & all(X[j] <= Y[j] for j in range(i)) for i in range(len(X)))


class LexLessEq(GlobalConstraint):
    """
    Enforces that the first list is lexicographically smaller than or equal to the second list.
    """
    def __init__(self, list1: Sequence[Expression], list2: Sequence[Expression]):
        """
        Arguments:
            list1 (Sequence[Expression]): First list of expressions to be compared lexicographically
            list2 (Sequence[Expression]): Second list of expressions to be compared lexicographically
        """
        if len(list1) != len(list2):
            raise ValueError(f"The 2 lists given in LexLessEq must have the same size: list1 length is {len(list1)} and list2 length is {len(list2)}")
        super().__init__("lex_lesseq", [list1, list2])

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Implementation inspired by Hakan Kjellerstrand (http://hakank.org/cpmpy/cpmpy_hakank.py)

        The decomposition creates auxiliary Boolean variables and constraints that
        collectively ensure X is lexicographically less than Y
        The auxiliary boolean vars are defined to represent if the given lists are lexicographically ordered
        (less or equal) up to the given index.
        Decomposition enforces through the constraining part that the first boolean variable needs to be true, and thus
        through the defining part it is enforced that if it is not strictly lexicographically less in a given index,
        then next index must be lexicographically less or equal.

        The use of auxiliary Boolean variables bvar ensures that the constraints propagate immediately,
        maintaining arc-consistency. Each bvar[i] enforces the lexicographic ordering at each position, ensuring that
        every value in the domain of X[i] can be extended to a consistent value in the domain of $Y_i$ for all
        subsequent positions.

        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        X, Y = cpm_array(self.args)

        if len(X) == 0 == len(Y):
            return [cp.BoolVal(False)], [] # based on the decomp, it's false...

        bvar = boolvar(shape=(len(X) + 1))
        defining = []
        defining.extend(bvar == ((X <= Y) & ((X < Y) | bvar[1:])))  # vectorized expression, treat as list
        defining.append(bvar[-1] == (X[-1] <= Y[-1]))
        constraining = [bvar[0]]

        return constraining, defining

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        X, Y = argvals(self.args)
        if any(val is None for val in X + Y):
            return None
        return any((X[i] < Y[i]) & all(X[j] <= Y[j] for j in range(i)) for i in range(len(X))) | all(X[i] == Y[i] for i in range(len(X)))


class LexChainLess(GlobalConstraint):
    """
    Enforces that all rows of the matrix are lexicographically ordered.
    """
    def __init__(self, X: Sequence[Sequence[Expression]]):
        """
        Arguments:
            X (Sequence[Sequence[Expression]]): Matrix of expressions to be compared lexicographically
        """
        Xarr = np.array(X) # also checks length of each row is equal
        if Xarr.ndim != 2:
            raise ValueError(f"The matrix given in LexChainLess must be 2D, but got {Xarr.ndim} dimensions")
        super().__init__("lex_chain_less", Xarr.tolist())

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """
        Decomposition of the LexChainLess constraint.
        
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        X = self.args
        return [LexLess(prev_row, curr_row) for prev_row, curr_row in zip(X, X[1:])], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        X = argvals(self.args)
        if any(val is None for val in flatlist(X)):
            return None
        return all(LexLess(prev_row, curr_row).value() for prev_row, curr_row in zip(X, X[1:]))


class LexChainLessEq(GlobalConstraint):
    """ 
    Enforces that all rows of the matrix are lexicographically ordered (less or equal)
    """
    def __init__(self, X: Sequence[Sequence[Expression]]):
        """
        Arguments:
            X (Sequence[Sequence[Expression]]): Matrix of expressions to be compared lexicographically
        """
        Xarr = np.array(X) # also checks length of each row is equal
        if Xarr.ndim != 2:
            raise ValueError(f"The matrix given in LexChainLessEq must be 2D, but got {Xarr.ndim} dimensions")
        super().__init__("lex_chain_lesseq", Xarr.tolist())

    def decompose(self) -> tuple[Sequence[Expression], Sequence[Expression]]:
        """ Decompose to a series of LexLessEq constraints between subsequent rows
        """
        X = self.args
        return [LexLessEq(prev_row, curr_row) for prev_row, curr_row in zip(X, X[1:])], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        X = argvals(self.args)
        if any(val is None for val in flatlist(X)):
            return None
        return all(LexLessEq(prev_row, curr_row).value() for prev_row, curr_row in zip(X, X[1:]))


class DirectConstraint(Expression):
    """
        A ``DirectConstraint`` will directly call a function of the underlying solver when added to a CPMpy solver

        It can not be reified, it is not flattened, it can not contain other CPMpy expressions than variables.
        When added to a CPMpy solver, it will literally just directly call a function on the underlying solver,
        replacing CPMpy variables by solver variables along the way.

        See the documentation of the solver (constructor) for details on how that solver handles them.

        If you want/need to use what the solver returns (e.g. an identifier for use in other constraints),
        then use :func:`~cpmpy.expressions.variables.directvar` instead, or access the solver object from the solver interface directly.
    """
    def __init__(self, name:str, arguments:tuple[Expression, ...], novar:Optional[Sequence[int]]=None):
        """
            name: name of the solver function that you wish to call
            arguments: tuple of arguments to pass to the solver function with name 'name'
            novar: list of indices (offset 0) of arguments in `arguments` that contain no variables,
                   that can be passed 'as is' without scanning for variables
        """
        if not isinstance(arguments, tuple):
            arguments = (arguments,)  # force tuple
        super().__init__(name, arguments)
        self.novar = novar

    def is_bool(self) -> bool:
        """ is it a Boolean (return type) Operator?
        """
        return True

    def callSolver(self, CPMpy_solver:"SolverInterface", Native_solver:Any):
        """
            Call the `directname()` function of the native solver,
            with stored arguments replacing CPMpy variables with solver variables as needed.

            SolverInterfaces will call this function when this constraint is added.

        :param CPMpy_solver: a CPM_solver object, that has a `solver_vars()` function
        :param Native_solver: the python interface to some specific solver
        :return: the response of the solver when calling the function
        """
        # get the solver function, will raise an AttributeError if it does not exist
        solver_function = getattr(Native_solver, self.name)
        solver_args = copy.copy(self.args)
        for i in range(len(solver_args)):
            if self.novar is None or i not in self.novar:
                # it may contain variables, replace
                solver_args[i] = CPMpy_solver.solver_vars(solver_args[i])
        # len(native_args) should match nr of arguments of `native_function`
        return solver_function(*solver_args)

