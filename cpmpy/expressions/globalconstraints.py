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
        CumulativeOptional
        NoOverlap
        NoOverlapOptional
        Precedence
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
import warnings
from typing import cast, Literal, Optional, Iterable, Any, TYPE_CHECKING
import numpy as np

import cpmpy as cp

from ..exceptions import TypeError
from .core import Expression, BoolVal, ExprLike, ListLike
from .variables import cpm_array, intvar, boolvar, _BoolVarImpl, _IntVarImpl, NDVarArray
from .utils import all_pairs, is_int, is_bool, STAR, get_bounds, argvals, is_any_list, flatlist, is_num, is_boolexpr, implies

if TYPE_CHECKING:
    from cpmpy.solvers.solver_interface import SolverInterface


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

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
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
                tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        raise NotImplementedError("Decomposition for", self, "not available")

    def get_bounds(self) -> tuple[int, int]:
        """
        Returns the bounds of a Boolean global constraint.
        Numerical global constraints should reimplement this.
        """
        return 0, 1

    def negate(self) -> Expression:
        """
        Returns the negation of this global constraint.
        Defaults to ~self, but subclasses can implement a better version,
        > Fages, François, and Sylvain Soliman. Reifying global constraints. Diss. INRIA, 2012.
        """
        return ~self


# Global Constraints (with Boolean return type)
def alldifferent(args):
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

    def __init__(self, *args: ExprLike|ListLike[ExprLike]):
        """
        Arguments:
            args (ExprLike|ListLike[ExprLike]): List of expressions or constants to be different from each other
        """
        super().__init__("alldifferent", tuple(flatlist(args)))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the AllDifferent global constraint using pairwise disequality constraints.

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        return [var1 != var2 for var1, var2 in all_pairs(self.args)], []

    def decompose_linear(self) -> tuple[list[Expression], list[Expression]]:
        """
        Linear-friendly decomposition using sums over (arg[i] == val) expressions (which will become Boolean variables):
        at most one integer variable can take each value in the domain.
        
        For use with integer linear programming and pb/sat solvers.
        """
        lbs, ubs = get_bounds(self.args)
        lb, ub = min(lbs), max(ubs)
        return [cp.sum((arg_i == val) for arg_i in self.args) <= 1 for val in range(lb, ub + 1)], []

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

    def __init__(self, arr: ListLike[ExprLike], n: int|np.integer|list[int|np.integer]):
        """
        Arguments:
            arr (ListLike[ExprLike]): List of expressions or constants to be different from each other, except those equal to a value in n
            n (int | np.integer | list[int | np.integer]): Value or list of values that are excluded from the distinctness constraint
        """
        flatarr = flatlist(arr)
        if not is_any_list(n):
            n = cast(int, n)
            n = [n] # ensure n is a list of ints
        super().__init__("alldifferent_except_n", (flatarr, n))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the AllDifferentExceptN global constraint using pairwise constraints.

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        cons = []
        arr, n = self.args
        for x,y in all_pairs(arr):
            cond = (x == y)
            if is_bool(cond):
                cond = cp.BoolVal(cond)
            cons.append(cond.implies(cp.any([x == a for a in n]))) # equivalent to (x in n) | (y in n) | (x != y)
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
    def __init__(self, *args: ExprLike | ListLike[ExprLike]):
        """
        Arguments:
            args (ListLike[ExprLike]): List of expressions or constants to be different from each other, except those equal to 0
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
    def __init__(self, *args: ExprLike | ListLike[ExprLike]):
        """
        Arguments:
            args (ListLike[ExprLike]): List of expressions or constants to have the same value
        """
        super().__init__("allequal", tuple(flatlist(args)))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the AllEqual global constraint using cascaded equality constraints.

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
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

    def __init__(self, arr: ListLike[ExprLike], n: int|np.integer|list[int|np.integer]):
        """
        Arguments:
            arr (ListLike[ExprLike]): List of expressions or constants to have the same value, except those equal to a value in n
            n (int | np.integer | list[int | np.integer]): Value or list of values that are excluded from the equality constraint
        """
        flatarr = flatlist(arr)
        if not is_any_list(n):
            n = cast(int, n)
            n = [n] # ensure n is a list of ints
        super().__init__("allequal_except_n", (flatarr, n))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the AllEqualExceptN global constraint using pairwise constraints.

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
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
    def __init__(self, *args: ExprLike | ListLike[ExprLike]):
        """
        Arguments:
            args (ListLike[ExprLike]): List of expressions or constants representing the successors of the nodes to form the circuit
        """
        flatargs = flatlist(args)
        if len(flatargs) < 2:
            raise ValueError('Circuit constraint must be given a minimum of 2 variables')
        super().__init__("circuit", tuple(flatargs))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
            Decomposition of the Circuit global constraint using auxiliary variables to reprsent the order in which we visit all the nodes.
            Auxiliary variables are defined in the defining part of the decomposition, which is alwasy enforced top-level.

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """

        # construct the chain of neighbors
        succ = cp.cpm_array(self.args)
        order = [succ[0]]
        for i in range(1, len(succ)):
            order.append(succ[order[i - 1]])

        # element constraints can be partial
        from cpmpy.transformations.safening import _no_partial_functions
        changed, safe_order, toplevel, nbc = _no_partial_functions(order, safen_toplevel=frozenset(), is_toplevel=False)
        if changed:
            order = safe_order # operate on the safened order expressions

        value = [order[-1] == 0, # return to start node
                 AllDifferent(order),  # ensure no subcircuits
                 AllDifferent(succ)    # redundant constraint, strengthens decomposition
                ]
        return value + nbc, toplevel

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        idx = 0
        visited = set()
        arr = argvals(self.args)
        if any(a is None for a in arr):
            return None

        while idx not in visited:
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
    def __init__(self, fwd: ListLike[ExprLike], rev: ListLike[ExprLike]):
        """
        Arguments:
            fwd (ListLike[ExprLike]): List of expressions or constants representing the forward function
            rev (ListLike[ExprLike]): List of expressions or constants representing the reverse function
        """
        if len(fwd) != len(rev):
            raise ValueError("Length of fwd and rev must be equal for Inverse constraint")
        super().__init__("inverse", (fwd, rev))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the Inverse global constraint using Element global function constraints, and explicit safening.

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """

        # we try to avoid in-function imports (needed when cyclic dependency),
        # but decompose is typically only called once anyway, so here it is acceptable
        from cpmpy.transformations.safening import _no_partial_functions

        fwd, rev = self.args
        rev = cpm_array(rev)

        constraining = [rev[x] == i for i,x in enumerate(fwd)]
        # Element constraints can be partial, so run safening transformation
        changed, safe_constraining, toplevel, nbc = _no_partial_functions(constraining, is_toplevel=False,
                                                              safen_toplevel=frozenset())
        if changed:
            constraining = safe_constraining + nbc
        
        return constraining, toplevel

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
    def __init__(self, array: ListLike[Expression], table: ListLike[ListLike[int]] | np.ndarray):
        """
        Arguments:
            array (ListLike[Expression]): List of expressions representing the array of variables
            table (ListLike[ListLike[int]] | np.ndarray): List of lists of integers or 2D ndarray of ints representing the table.
        """
        if isinstance(array, NDVarArray):
            has_subexpr = array.has_subexpr()  # fast shortcut
            if array.ndim != 1:  # reshape to 1D
                array = array.reshape(-1)
        else:
            has_subexpr = False
            for x in array:  # C-style python
                if x.has_subexpr():
                    has_subexpr = True
                    break

        if not isinstance(table, np.ndarray):  # Ensure it is a numpy array with integers
            table = np.array(table, dtype=int)
        elif table.dtype.kind != 'i':  # dtype int
            table = table.astype(int, copy=False)
        assert table.ndim == 2, "Table's table must be a 2D array"
        assert table.shape[1] == len(array), f"Table width {table.shape[1]} != array length {len(array)}"

        # args: tuple[ListLike[Expression], np.ndarray]
        super().__init__("table", (array, table), has_subexpr=has_subexpr)

    @property
    def args(self) -> tuple[ListLike[Expression], np.ndarray]:
        """ READ-ONLY, the well-tuped arguments of this global constraint
        """
        return self._args


    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the Table global constraint. Enforces at least one row of the table is assigned to the array.

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        arr, tab = self.args
        return [cp.any([cp.all([ai == ri for ai, ri in zip(arr, row)]) for row in tab])], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        arr, tab = self.args
        arrval = np.asarray(argvals(arr))
        if arrval.dtype == object and any(x is None for x in arrval.flat):  # if not object, there is no None
            return None
        return bool(np.any(np.all(tab == arrval, axis=1)))

    def negate(self) -> Expression:
        arr, tab = self.args
        return NegativeTable(arr, tab)


class ShortTable(GlobalConstraint):
    """
    Extension of the `Table` constraint where the `table` matrix may contain wildcards (STAR), meaning there are
    no restrictions for the corresponding variable in that tuple.
    """

    def __init__(self, array: ListLike[Expression], table: ListLike[ListLike[int|Literal["*"]]] | np.ndarray):
        """
        Arguments:
            array (ListLike[Expression]): List of expressions representing the array of variables
            table (ListLike[ListLike[int | '*']] | np.ndarray): List of lists or 2D ndarray; entries are integers or STAR ('*')
                STAR represents a wildcard (corresponding variable can take any value).
        """
        if isinstance(array, NDVarArray):
            has_subexpr = array.has_subexpr()  # fast shortcut
            if array.ndim != 1:  # reshape to 1D
                array = array.reshape(-1)
        else:
            has_subexpr = False
            for x in array:  # C-style python
                if x.has_subexpr():
                    has_subexpr = True
                    break

        if not isinstance(table, np.ndarray):
            table = np.array(table, dtype=object)  # object, otherwise np makes it all string
        assert table.ndim == 2, "ShortTable's table must be a 2D array"
        assert table.shape[1] == len(array), f"ShortTable width {table.shape[1]} != array length {len(array)}"

        # args: tuple[ListLike[Expression], np.ndarray]
        super().__init__("short_table", (array, table), has_subexpr=has_subexpr)

    @property
    def args(self) -> tuple[ListLike[Expression], np.ndarray]:
        """ READ-ONLY, the well-tuped arguments of this global constraint
        """
        return self._args

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the ShortTable global constraint. Enforces at least one row of the table is assigned to the array.

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        arr, tab = self.args
        return [cp.any([cp.all([ai == ri for ai, ri in zip(arr, row) if ri != STAR]) for row in tab])], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        arr, tab = self.args
        arrval = np.asarray(argvals(arr))
        if arrval.dtype == object and any(x is None for x in arrval.flat):  # if not object, there is no None
            return None
        arrval = arrval.astype(int, copy=False)

        non_star = (tab != STAR)
        for row, mask in zip(tab, non_star):
            if (row[mask].astype(int, copy=False) == arrval[mask]).all():
                return True
        return False

class NegativeTable(GlobalConstraint):
    """
    The values of the variables in 'array' do not correspond to any row in 'table'.
    """

    def __init__(self, array: ListLike[Expression], table: ListLike[ListLike[int]] | np.ndarray):
        """
        Arguments:
            array (ListLike[Expression]): List of expressions representing the array of variables
            table (ListLike[ListLike[int]] | np.ndarray): List of lists of integers or 2D ndarray of ints representing the table.
        """
        if isinstance(array, NDVarArray):
            has_subexpr = array.has_subexpr()  # fast shortcut
            if array.ndim != 1:  # reshape to 1D
                array = array.reshape(-1)
        else:
            has_subexpr = False
            for x in array:  # C-style python
                if x.has_subexpr():
                    has_subexpr = True
                    break

        if not isinstance(table, np.ndarray):  # Ensure it is a numpy array
            table = np.array(table, dtype=int)
        elif table.dtype.kind != 'i':  # dtype int
            table = table.astype(int, copy=False)
        assert table.ndim == 2, "NegativeTable's table must be a 2D array"
        assert table.shape[1] == len(array), f"NegativeTable width {table.shape[1]} != array length {len(array)}"

        # args: tuple[ListLike[Expression], np.ndarray]
        super().__init__("negative_table", (array, table), has_subexpr=has_subexpr)

    @property
    def args(self) -> tuple[ListLike[Expression], np.ndarray]:
        """ READ-ONLY, the well-tuped arguments of this global constraint
        """
        return self._args

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the NegativeTable global constraint. 
        Enforces that the values of the variables in 'array' do not correspond to any row in 'table'.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        arr, tab = self.args
        return [cp.all([cp.any([ai != ri for ai, ri in zip(arr, row)]) for row in tab])], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        arr, tab = self.args
        arrval = np.asarray(argvals(arr))
        if arrval.dtype == object and any(x is None for x in arrval.flat):  # if not object, there is no None
            return None
        return not bool(np.any(np.all(tab == arrval, axis=1)))

    def negate(self) -> Expression:
        arr, tab = self.args
        return Table(arr, tab)
    

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
    def __init__(self, array: ListLike[Expression], transitions: ListLike[tuple[int|str, int, int|str]], start: int|str, accepting: ListLike[int|str]):
        """
        Arguments:
            array (ListLike[Expression]): List of expressions representing the input sequence
            transitions (ListLike[tuple[int | str, int, int | str]]): List of transition triples (source, value, destination)
            start (int | str): Starting node id
            accepting (ListLike[int | str]): List of accepting node ids
        """
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
        super().__init__("regular", (list(array), list(transitions), start, list(accepting)))

        node_set = set()
        self.trans_dict = {}
        for s, v, e in transitions:
            node_set.update([s,e])
            self.trans_dict[(s, v)] = e
        self.nodes = sorted(node_set)
        # normalize node_ids to be 0..n-1, allows for smaller domains
        self.node_map = {n: i for i, n in enumerate(self.nodes)}

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the Regular global constraint. 
        Encodes the automaton by encoding the transition table into `class:cpmpy.expressions.globalconstraints.Table` constraints.
        Then enforces that the last state is accepting.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
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
        defining: list[Expression] = [Table([arr[0], state_vars[0]], [[v,e] for s,v,e in transitions if s == id_start])]
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
    def __init__(self, condition: ExprLike, if_true: ExprLike, if_false: ExprLike):
        """
        Arguments:
            condition (ExprLike): Boolean expression or constant
            if_true (ExprLike): Boolean expression or constant
            if_false (ExprLike): Boolean expression or constant
        """
        if not is_boolexpr(condition) or not is_boolexpr(if_true) or not is_boolexpr(if_false):
            raise TypeError(f"only boolean expression allowed in IfThenElse: Instead got "
                            f"{condition, if_true, if_false}")
        super().__init__("ite", (condition, if_true, if_false))

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

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the IfThenElse global constraint.
        Enforces that the condition is satisfied.
        "
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        condition, if_true, if_false = self.args
        if is_bool(condition):
            condition = cp.BoolVal(condition) # ensure it is a CPMpy expression
        return [condition.implies(if_true), (~condition).implies(if_false)], []

    def __repr__(self) -> str:
        condition, if_true, if_false = self.args
        return "If {} Then {} Else {}".format(condition, if_true, if_false)

    def negate(self) -> Expression:
        return IfThenElse(self.args[0], self.args[2], self.args[1])



class InDomain(GlobalConstraint):
    """
    Enforces the expression is assigned to a value in the given domain.
    """

    def __init__(self, expr: Expression, arr: Iterable[int|np.integer]):
        """
        Arguments:
            expr (Expression): Expression to be assigned to a value in the given domain
            arr (Iterable[int | np.integer]): Iterable of integer constants representing the domain
        """
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=int)
        assert arr.ndim == 1, "The second argument of an InDomain constraint should be a 1D array of integer constants"

        has_subexpr = expr.has_subexpr()
        # args: tuple[Expression, np.ndarray]
        super().__init__("InDomain", (expr, arr), has_subexpr=has_subexpr)

    @property
    def args(self) -> tuple[Expression, np.ndarray]:
        """ READ-ONLY, the well-tuped arguments of this global constraint
        """
        return self._args

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the InDomain global constraint.
        Enforces that the expression is assigned to a value in the given domain.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        expr, arr = self.args
        lb, ub = expr.get_bounds()
        
        return [expr != val for val in range(lb, ub + 1) if val not in arr], []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        expr, arr = self.args
        exprval = expr.value()
        if exprval is None:
            return None
        return bool(np.any(arr == exprval))

    def __repr__(self) -> str:
        expr, arr = self.args
        return "{} in {}".format(expr, arr)

    def negate(self) -> Expression:
        expr, arr = self.args
        lb, ub = expr.get_bounds()

        # complement of arr
        return InDomain(expr, [v for v in range(lb,ub+1) if v not in arr])


class Xor(GlobalConstraint):
    """
    Enforces the exclusive-or relation of the arguments.
    Supports n-ary xor-constraints, which are treated as cascaed binary xor-constraints.
    Equivalent to `sum(args) % 2 == 1`
    """

    def __init__(self, arg_list: ListLike[ExprLike]):
        """
        Arguments:
            arg_list (ListLike[ExprLike]): List of expressions or constants, to be xor'ed
        """
        if not all(is_boolexpr(arg) for arg in arg_list):
            raise TypeError("Only Boolean arguments allowed in Xor global constraint: {}".format(arg_list))
        # convention for commutative binary operators:
        # swap if right is constant and left is not
        arg_list = list(arg_list)
        if len(arg_list) == 2 and is_num(arg_list[1]):
            arg_list[0], arg_list[1] = arg_list[1], arg_list[0]
        super().__init__("xor", tuple(arg_list))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the Xor global constraint.
        Recursively decomposes the constraint into a chain of binary xor-constraints, represented using a sum.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        # there are multiple decompositions possible, Recursively using sum allows it to be efficient for all solvers.
        decomp = [sum(self.args[:2]) == 1]
        if len(self.args) > 2:
            decomp = Xor(decomp + list(self.args[2:])).decompose()[0]
        return decomp, []

    def value(self) -> Optional[bool]:
        arrvals = argvals(self.args)
        if any(a is None for a in arrvals):
            return None
        return sum(arrvals) % 2 == 1

    def __repr__(self) -> str:
        if len(self.args) == 2:
            return "{} xor {}".format(*self.args)
        return "xor({})".format(self.args)

    def negate(self) -> Expression:
        # negate one of the arguments, ideally a variable
        new_args = list(self.args)  # takes shallow copy
        changed = False
        for i, a in enumerate(self.args):
            if isinstance(a, _BoolVarImpl):
                new_args[i] = ~a
                changed = True
                break

        if not changed:  # did not find a Boolean variable to negate
            # pick first arg, and push down negation
            from cpmpy.transformations.negation import recurse_negation
            new_args[0] = recurse_negation(self.args[0])

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
    def __init__(self, start: ListLike[ExprLike], duration: ListLike[ExprLike], end: Optional[ListLike[ExprLike]] = None, demand: Optional[ListLike[ExprLike]|ExprLike] = None, capacity: Optional[ExprLike] = None):
        """
            Arguments:
                start (ListLike[ExprLike]): Start times of the tasks
                duration (ListLike[ExprLike]): Durations of the tasks
                end (ListLike[ExprLike] | None): Optional end times of the tasks
                demand (ListLike[ExprLike] | ExprLike): Per-task demands or a single constant demand, required
                capacity (ExprLike): Capacity of the resource, required
            
            Technical note: demand/capacity marked as Optional because it comes after an Optional argument
        """

        if not is_any_list(start):
            raise TypeError("start should be a list")
        if not is_any_list(duration):
            raise TypeError("duration should be a list")
        if end is not None and not is_any_list(end):
            raise TypeError("end should be a list if it is provided")
        if demand is None:  # marked optional due to 'end' being optional and parameters after that must be optional too
            raise TypeError("demand should be provided but was None")
        if capacity is None:  # marked optional due to 'end' being optional and parameters after that must be optional too
            raise TypeError("capacity should be provided but was None")
        
        if len(start) != len(duration):
            raise ValueError("Start and duration should have equal length")
        if end is not None and len(start) != len(end):
            raise ValueError(f"Start and end should have equal length, but got {len(start)} and {len(end)}")

        demand_list = []
        if is_any_list(demand):
            demand_list = list(demand)
            if len(demand_list) != len(start):
                raise ValueError(f"Demand should be supplied for each task or be single constant, but got {len(demand_list)} and {len(start)}")
        else: # constant demand
            demand_list = [demand] * len(start)

        super(Cumulative, self).__init__("cumulative", (list(start), list(duration), list(end) if end is not None else None, demand_list, capacity))

    
    def decompose(self, how:str="auto") -> tuple[list[Expression], list[Expression]]:
        """
        Decompose the Cumulative constraint
        Support time-based decomposition or task-based decomposition.
        By default, we heuristically select the best decomposition based on the number of tasks and the horizon.

        Arguments:
            how (str): how the cumulative constraint should be decomposed, can be "time", "task", or "auto" (default)

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """

        if how not in ["time", "task", "auto"]:
            raise ValueError(f"how can only be time, task, or auto (default), but got {how}")

        start= self.args[0]

        lbs, ubs = get_bounds(start)
        horizon = max(ubs) - min(lbs)
        if (how == "time") or (how == "auto" and len(start) <= horizon):
            return self._time_decomposition()
        elif (how == "task") or (how == "auto" and len(start) > horizon):
            return self._task_decomposition()
        raise Exception # should not be reached

    def _consistency_constraints(self) -> list[Expression]:
        """
        Helper function to enforce consistency constraints, used in the decomposition.

        Consistency constraints enforce that:
        - duration >= 0
        - demand >= 0
        - start + duration == end
        """
        start, duration, end, demand, capacity = self.args
        cons = [d >= 0 for d in duration]  # enforce non-negative durations
        cons += [h >= 0 for h in demand]  # enforce non-negative demand

        if end is not None:
            cons += [start[i] + duration[i] == end[i] for i in range(len(start))]

        return cons

    def _task_decomposition(self) -> tuple[list[Expression], list[Expression]]:
        """
        Task-based decomposition of the cumulative constraint.
        Schutt, Andreas, et al. "Why cumulative decomposition is not as bad as it sounds."
        International Conference on Principles and Practice of Constraint Programming. Springer, Berlin, Heidelberg, 2009.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        start, duration, end, demand, capacity = self.args

        cons = self._consistency_constraints()
        if end is None:
            end = [start[i] + duration[i] for i in range(len(start))]

        # demand doesn't exceed capacity
        # tasks are uninterruptible, so we only need to check each starting point of each task
        # I.e., for each task, we check if it can be started, given the tasks that are already running.
        for t in range(len(start)):
            st = start[t]
            demand_at_start_of_t = []
            for j in range(len(start)):
                if t != j:
                    demand_at_start_of_t.append(demand[j] * ((start[j] <= st) & (end[j] > st)))

            cons.append((demand[t] + sum(demand_at_start_of_t)) <= capacity)

        return cons, []

    def _time_decomposition(self) -> tuple[list[Expression], list[Expression]]:
        """
        Time-resource decomposition of the cumulative constraint.
        Schutt, Andreas, et al. "Why cumulative decomposition is not as bad as it sounds."
        International Conference on Principles and Practice of Constraint Programming. Springer, Berlin, Heidelberg, 2009.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        start, duration, end, demand, capacity = self.args

        cons = self._consistency_constraints()
        if end is None:
            end = [start[i] + duration[i] for i in range(len(start))]
            
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
        if any(a is None for a in flatlist([start, dur, demand, capacity])):
            return None
        if end is None:
            end = [s + d for s,d in zip(start, dur)]
        else:
            end = argvals(end)
            if any(a is None for a in end):
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

class CumulativeOptional(GlobalConstraint):
    """
        Generalization of the Cumulative constraint which allows for optional tasks.
        A task is only scheduled if the corresponing is_present variable is set to True.

        If the task is present, the constraint enforces that:
        - duration >= 0
        - demand >= 0
        - start + duration == end

        If the task is not present, the constraint does not enforce any of the above.

        Equivalent to :class:`~cpmpy.expressions.globalconstraints.NoOverlapOptional` when demand and capacity are equal to 1.
        Supports both varying demand across tasks or equal demand for all jobs.
    """

    def __init__(self, start: ListLike[ExprLike], 
                       duration: ListLike[ExprLike], 
                       end: Optional[ListLike[ExprLike]] = None, 
                       demand: Optional[ListLike[ExprLike]|ExprLike] = None, 
                       capacity: Optional[ExprLike] = None, 
                       is_present: Optional[ListLike[ExprLike]] = None):
        """
            Arguments:
                start (ListLike[ExprLike]): Start times of the tasks
                duration (ListLike[ExprLike]): Durations of the tasks
                end (ListLike[ExprLike] | None): Optional end times of the tasks
                demand (ListLike[ExprLike] | ExprLike): Per-task demands or a single constant demand, required
                capacity (ExprLike): Capacity of the resource, required
                is_present (ListLike[ExprLike]): Presence of the tasks
            
            Technical note: demand/capacity marked as Optional because it comes after an Optional argument
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
        if is_present is None:
            raise TypeError("is_present should be provided but was None")
        
        if len(start) != len(duration):
            raise ValueError("Start and duration should have equal length")
        if len(start) != len(is_present):
            raise ValueError("Start and is_present should have equal length")
        if end is not None and len(start) != len(end):
            raise ValueError(f"Start and end should have equal length, but got {len(start)} and {len(end)}")

        demand_list = []
        if is_any_list(demand):
            demand_list = list(demand)
            if len(demand_list) != len(start):
                raise ValueError(f"Demand should be supplied for each task or be single constant, but got {len(demand_list)} and {len(start)}")
        else: # constant demand
            demand_list = [demand] * len(start)

        super().__init__("cumulative_optional", (list(start), list(duration), list(end) if end is not None else None,
                                                 demand_list, capacity, list(is_present)))

    def decompose(self, how:str="auto") -> tuple[list[Expression], list[Expression]]:
        """
        Decompose the Cumulative constraint
        Support time-based decomposition or task-based decomposition.
        By default, we heuristically select the best decomposition based on the number of tasks and the horizon.

        Arguments:
            how (str): how the cumulative constraint should be decomposed, can be "time", "task", or "auto" (default)

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """

        if how not in ["time", "task", "auto"]:
            raise ValueError(f"how can only be time, task, or auto (default), but got {how}")

        start, *args = self.args

        lbs, ubs = get_bounds(start)
        horizon = max(ubs) - min(lbs)
        if (how == "time") or (how == "auto" and len(start) <= horizon):
            return self._time_decomposition()
        elif (how == "task") or (how == "auto" and len(start) > horizon):
            return self._task_decomposition()
        raise Exception # should not be reached

    def _consistency_constraints(self) -> list[Expression]:
        """
        Helper function to enforce concistency constraints, used in the decomposition.
        
        Consistency constraints enforce that:
        - duration >= 0 if the task is present
        - demand >= 0 if the task is present
        - start + duration == end if the task is present
        """

        start, duration, end, demand, capacity, is_present = self.args
        cons = [implies(p,d >= 0) for d, p in zip(duration, is_present)]  # enforce non-negative durations when present
        cons += [implies(p,h >= 0) for h, p in zip(demand, is_present)]  # enforce non-negative demand when present

        # set duration of tasks, only if end is user-provided and the task is present
        if end is not None:
            cons += [implies(is_present[i], start[i] + duration[i] == end[i]) for i in range(len(start))]

        return cons

    
    def _task_decomposition(self) -> tuple[list[Expression], list[Expression]]:
        """
        Task-based decomposition of the cumulative constraint.
        Schutt, Andreas, et al. "Why cumulative decomposition is not as bad as it sounds."
        International Conference on Principles and Practice of Constraint Programming. Springer, Berlin, Heidelberg, 2009.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        start, duration, end, demand, capacity, is_present = self.args
        
        cons = self._consistency_constraints()
        if end is None:
            end = [start[i] + duration[i] for i in range(len(start))]

        # demand of tasks that are present doesn't exceed capacity
        # tasks are uninterruptible, so we only need to check each starting point of each task
        # I.e., for each task, we check if it can be started, given the tasks that are already running.
        for t in range(len(start)):
            st = start[t]
            demand_at_start_of_t = []
            for j in range(len(start)):
                if t != j:
                    demand_at_start_of_t.append(demand[j] * (is_present[j] & (start[j] <= st) & (end[j] > st)))

            cons.append(implies(is_present[t], (demand[t] + sum(demand_at_start_of_t)) <= capacity))

        return cons, []

    def _time_decomposition(self) -> tuple[list[Expression], list[Expression]]:
        """
        Time-resource decomposition of the cumulative constraint.
        Schutt, Andreas, et al. "Why cumulative decomposition is not as bad as it sounds."
        International Conference on Principles and Practice of Constraint Programming. Springer, Berlin, Heidelberg, 2009.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        start, duration, end, demand, capacity, is_present = self.args

        cons = self._consistency_constraints()
        if end is None:
            end = [start[i] + duration[i] for i in range(len(start))]

        # demand of tasks that are presentdoesn't exceed capacity
        # for each time-step, we check if the running demand does not exceed the capacity
        lbs, ubs = get_bounds(start)
        lb, ub = min(lbs), max(ubs)
        for t in range(lb,ub+1):
            cons += [cp.sum(d * (present & (s <= t) & (e > t)) for s,e,d,present in zip(start, end, demand, is_present)) <= capacity]

        return cons, []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """        
        start, dur, end, demand, capacity, is_present = argvals(self.args)
        if end is None:
            end = [s + d for s,d in zip(start, dur)]
        else:
            end = argvals(end)

        if any(a is None for a in flatlist([start, dur, end, demand, capacity, is_present])):
            return None
                
        if any(p and d < 0 for d,p in zip(dur, is_present)):
            return False
        if any(p and s + d != e for s,d,e,p in zip(start, dur, end, is_present)):
            return False

        if any(p and d < 0 for d,p in zip(demand, is_present)):
            return False

        # ensure demand doesn't exceed capacity
        lb, ub = min(start), max(end)
        start, end, present = np.array(start), np.array(end), np.array(is_present) # eases check below
        for t in range(lb, ub+1):
            if capacity < sum(demand * (present & (start <= t) & (end > t))):
                return False

        return True


class NoOverlap(GlobalConstraint):
    """
    Enforces that a set of tasks are scheduled without overlapping, and enforces:
        - duration >= 0
        - start + duration == end
    """

    def __init__(self, start: ListLike[ExprLike], duration: ListLike[ExprLike], end: Optional[ListLike[ExprLike]] = None):
        """
        Arguments:
            start (ListLike[ExprLike]): Start times of the tasks
            duration (ListLike[ExprLike]): Durations of the tasks
            end (ListLike[ExprLike] | None): Optional end times of the tasks
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
        
        super().__init__("no_overlap", (list(start), list(duration), list(end) if end is not None else None))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the NoOverlap constraint, using pairwise no-overlap constraints.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
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

class NoOverlapOptional(GlobalConstraint):
    """
        Generalization of the NoOverlap constraint which allows for optional tasks.
        A task is only scheduled if the corresponing is_present variable is set to True.

        The constraint enforces that all present tasks are scheduled without overlapping, and for each present task, the constraint enforces that:
        - duration >= 0
        - demand >= 0
        - start + duration == end

        if the task is not present, it does not enforce any of the above.
    """
    
    def __init__(self, start: ListLike[ExprLike], duration: ListLike[ExprLike], end: Optional[ListLike[ExprLike]] = None, is_present: Optional[ListLike[ExprLike]] = None):
        """
        Arguments:
            start (ListLike[Expression]): List of Expression objects representing the start times of the tasks
            duration (ListLike[Expression]): List of Expression objects representing the durations of the tasks
            end (ListLike[Expression] | None): optional, list of Expression objects representing the end times of the tasks
            is_present (ListLike[Expression]): List of Boolean Expression objects representing the presence of the tasks
        """
       
        if not is_any_list(start):
            raise TypeError("start should be a list")
        if not is_any_list(duration):
            raise TypeError("duration should be a list")
        if end is not None and not is_any_list(end):
            raise TypeError("end should be a list if it is provided")
        if is_present is None or not is_any_list(is_present):
            raise ValueError("is_present should be provided and should be a list")
        
        if len(start) != len(duration):
            raise ValueError("Start and duration should have equal length")
        if len(start) != len(is_present):
            raise ValueError("Start and is_present should have equal length")
        if end is not None and len(start) != len(end):
            raise ValueError(f"Start and end should have equal length, but got {len(start)} and {len(end)}")
        
        super().__init__("no_overlap_optional", (start, duration, end, is_present))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the NoOverlap constraint, using pairwise no-overlap constraints.
        
        Returns:
            tuple[Sequence[Expression], Sequence[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
        """
        start, dur, end, is_present = self.args
        cons = [implies(p, d >= 0) for d, p in zip(dur, is_present)]
        
        if end is None:
            end = [s+d for s,d in zip(start, dur)]
        else: # can use the expression directly below
            cons += [implies(p, s + d == e) for s,d,e,p in zip(start, dur, end, is_present)]
            
        for (s1, e1, p1), (s2, e2, p2) in all_pairs(zip(start, end, is_present)):
            cons += [implies(p1 & p2, (e1 <= s2) | (e2 <= s1))]
        return cons, []

    def value(self) -> Optional[bool]:
        """
        Returns:
            Optional[bool]: True if the global constraint is satisfied, False otherwise, or None if any argument is not assigned
        """
        start, dur, end, is_present = argvals(self.args)
        if end is None:
            if any(s is None for s in start) or any(d is None for d in dur):
                return None
            end = [s + d for s,d in zip(start, dur)]
        else:
            if any(s is None for s in start) or any(d is None for d in dur) or any(e is None for e in end) or any(p is None for p in is_present):
                return None
       
        if any(p and d < 0 for d,p in zip(dur, is_present)):
            return False
        if any(p and s + d != e for s,d,e,p in zip(start, dur, end, is_present)):
            return False
        for (s1,d1,p1), (s2,d2,p2) in all_pairs(zip(start,dur,is_present)):
            if p1 and p2 and (s1 + d1 > s2) and (s2 + d2 > s1):
                return False
        return True
    
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
    def __init__(self, vars: ListLike[ExprLike], precedence: ListLike[int|np.integer]):
        """
        Arguments:
            vars (ListLike[ExprLike]): List of expressions or constants representing the variables
            precedence (ListLike[int | np.integer]): List of integer precedence values
        """
        if not is_any_list(vars):
            raise TypeError("Precedence expects a list of variables as first argument, but got", vars)
        if not is_any_list(precedence) or not all(is_num(p) for p in precedence):
            raise TypeError("Precedence expects a list of values as second argument, but got", precedence)
        super().__init__("precedence", (list(vars), list(precedence)))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition based on:
        Law, Yat Chiu, and Jimmy HM Lee. "Global constraints for integer and set value precedence."
        Principles and Practice of Constraint Programming–CP 2004: 10th International Conference, CP 2004

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
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

    def __init__(self, vars: ListLike[ExprLike], vals: ListLike[int|np.integer], occ: ListLike[ExprLike], closed: bool = False):
        """
        Arguments:
            vars (ListLike[ExprLike]): List of expressions or constants representing the variables
            vals (ListLike[int | np.integer]): List of integer values
            occ (ListLike[ExprLike]): List of expressions or constants representing the number of occurrences of each value
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
        super().__init__("gcc", (list(vars), list(vals), list(occ)))
        self.closed = closed

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
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

    def __init__(self, *args: ExprLike | ListLike[ExprLike]):
        """
        Arguments:
            args (ListLike[ExprLike]): List of expressions or constants to be assigned to increasing values
        """
        super().__init__("increasing", tuple(flatlist(args)))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the Increasing constraint.

        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
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

    def __init__(self, *args: ExprLike | ListLike[ExprLike]):
        """
        Arguments:
            args (ListLike[ExprLike]): List of expressions or constants to be assigned to decreasing values
        """
        super().__init__("decreasing", tuple(flatlist(args)))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the Decreasing constraint.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
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

    def __init__(self, *args: ExprLike | ListLike[ExprLike]):
        """
        Arguments:
            args (ListLike[ExprLike]): List of expressions or constants to be assigned to strictly increasing values
        """
        super().__init__("strictly_increasing", tuple(flatlist(args)))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the IncreasingStrict constraint.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
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

    def __init__(self, *args: ExprLike | ListLike[ExprLike]):
        """
        Arguments:
            args (ListLike[ExprLike]): List of expressions or constants to be assigned to strictly decreasing values
        """
        super().__init__("strictly_decreasing", tuple(flatlist(args)))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the DecreasingStrict constraint.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
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
    def __init__(self, list1: ListLike[ExprLike], list2: ListLike[ExprLike]):
        """
        Arguments:
            list1 (ListLike[ExprLike]): First List of expressions or constants to be compared lexicographically
            list2 (ListLike[ExprLike]): Second List of expressions or constants to be compared lexicographically
        """ 
        if len(list1) != len(list2):
            raise ValueError(f"The 2 lists given in LexLess must have the same size: list1 length is {len(list1)} and list2 length is {len(list2)}")
        super().__init__("lex_less", (list1, list2))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
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
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
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
    def __init__(self, list1: ListLike[ExprLike], list2: ListLike[ExprLike]):
        """
        Arguments:
            list1 (ListLike[ExprLike]): First List of expressions or constants to be compared lexicographically
            list2 (ListLike[ExprLike]): Second List of expressions or constants to be compared lexicographically
        """
        if len(list1) != len(list2):
            raise ValueError(f"The 2 lists given in LexLessEq must have the same size: list1 length is {len(list1)} and list2 length is {len(list2)}")
        super().__init__("lex_lesseq", (list1, list2))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
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
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
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
    def __init__(self, X: ListLike[ListLike[ExprLike]]):
        """
        Arguments:
            X (ListLike[ListLike[ExprLike]]): Matrix (List of lists) of expressions or constants to be compared lexicographically
        """
        Xarr = np.array(X) # also checks length of each row is equal
        if Xarr.ndim != 2:
            raise ValueError(f"The matrix given in LexChainLess must be 2D, but got {Xarr.ndim} dimensions")
        super().__init__("lex_chain_less", tuple(Xarr.tolist()))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
        """
        Decomposition of the LexChainLess constraint.
        
        Returns:
            tuple[list[Expression], list[Expression]]: A tuple containing the constraints representing the constraint value and the defining constraints
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
    def __init__(self, X: ListLike[ListLike[ExprLike]]):
        """
        Arguments:
            X (ListLike[ListLike[ExprLike]]): Matrix (List of lists) of expressions or constants to be compared lexicographically
        """
        Xarr = np.array(X) # also checks length of each row is equal
        if Xarr.ndim != 2:
            raise ValueError(f"The matrix given in LexChainLessEq must be 2D, but got {Xarr.ndim} dimensions")
        super().__init__("lex_chain_lesseq", tuple(Xarr.tolist()))

    def decompose(self) -> tuple[list[Expression], list[Expression]]:
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
    def __init__(self, name: str, arguments: tuple[Any, ...], novar: Optional[ListLike[int]] = None):
        """
            name (str): Name of the solver function that you wish to call
            arguments (tuple[Any, ...]): Tuple of arguments to pass to the solver function with name `name`
            novar (Optional[ListLike[int]]): Optional List of indices (offset 0) of arguments in `arguments` that contain no variables,
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

    def callSolver(self, CPMpy_solver: "SolverInterface", Native_solver: Any) -> Any:
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
        solver_args = list(self.args)  # takes a copy
        for i in range(len(solver_args)):
            if self.novar is None or i not in self.novar:
                # it may contain variables, replace
                solver_args[i] = CPMpy_solver.solver_vars(solver_args[i])
        # len(native_args) should match nr of arguments of `native_function`
        return solver_function(*solver_args)

