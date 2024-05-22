#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## globalconstraints.py
##
"""
    Global constraints conveniently express non-primitive constraints.

    Using global constraints
    ------------------------

    Solvers can have specialised implementations for global constraints. CPMpy has GlobalConstraint
    expressions so that they can be passed to the solver as is when supported.

    If a solver does not support a global constraint (see solvers/) then it will be automatically
    decomposed by calling its `.decompose()` function.
    The `.decompose()` function returns two arguments:
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

    CPMpy also implements __Numeric Global Constraints__. For these, the CPMpy GlobalConstraint does not
    exactly match what is implemented in the solver, but for good reason!!

    For example solvers may implement the global constraint `Minimum(iv1, iv2, iv3) == iv4` through an API
    call `addMinimumEquals([iv1,iv2,iv3], iv4)`.

    However, CPMpy also wishes to support the expressions `Minimum(iv1, iv2, iv3) > iv4` as well as
    `iv4 + Minimum(iv1, iv2, iv3)`. 

    Hence, the CPMpy global constraint only captures the `Minimum(iv1, iv2, iv3)` part, whose return type
    is numeric and can be used in any other CPMpy expression. Only at the time of transforming the CPMpy
    model to the solver API, will the expressions be decomposed and auxiliary variables introduced as needed
    such that the solver only receives `Minimum(iv1, iv2, iv3) == ivX` expressions.
    This is the burden of the CPMpy framework, not of the user who wants to express a problem formulation.


    Subclassing GlobalConstraint
    ----------------------------
    
    If you do wish to add a GlobalConstraint, because it is supported by solvers or because you will do
    advanced analysis and rewriting on it, then preferably define it with a standard decomposition, e.g.:

    .. code-block:: python

        class my_global(GlobalConstraint):
            def __init__(self, args):
                super().__init__("my_global", args)

            def decompose(self):
                return [self.args[0] != self.args[1]] # your decomposition

    If it is a __numeric global constraint__ meaning that its return type is numeric (see `Minimum` and `Element`)
    then set `is_bool=False` in the super() constructor and preferably implement `.value()` accordingly.


    Alternative decompositions
    --------------------------
    
    For advanced use cases where you want to use another decomposition than the standard decomposition
    of a GlobalConstraint expression, you can overwrite the 'decompose' function of the class, e.g.:

    .. code-block:: python

        def my_circuit_decomp(self):
            return [self.args[0] == 1], [] # does not actually enforce circuit
        circuit.decompose = my_circuit_decomp # attach it, no brackets!

        vars = intvar(1,9, shape=10)
        constr = circuit(vars)

        Model(constr).solve()

    The above will use 'my_circuit_decomp', if the solver does not
    natively support 'circuit'.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        AllDifferent
        AllDifferentExcept0
        AllEqual
        Circuit
        SubCircuit
        SubCircuitWithStart
        Inverse
        Table
        Xor
        Cumulative
        IfThenElse
        GlobalCardinalityCount
        DirectConstraint
        InDomain
        Increasing
        Decreasing
        IncreasingStrict
        DecreasingStrict

"""
import copy
import warnings # for deprecation warning
import numpy as np
from ..exceptions import CPMpyException, IncompleteFunctionError, TypeError
from .core import Expression, Operator, Comparison
from .variables import boolvar, intvar, cpm_array, _NumVarImpl, _IntVarImpl
from .utils import flatlist, all_pairs, argval, is_num, eval_comparison, is_any_list, is_boolexpr, get_bounds, argvals
from .globalfunctions import * # XXX make this file backwards compatible


# Base class GlobalConstraint
class GlobalConstraint(Expression):
    """
        Abstract superclass of GlobalConstraints

        Like all expressions it has a `.name` and `.args` property.
        Overwrites the `.is_bool()` method.
    """

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return True

    def decompose(self):
        """
            Returns a decomposition into smaller constraints.

            The decomposition might create auxiliary variables
            and use other global constraints as long as
            it does not create a circular dependency.

            To ensure equivalence of decomposition, we split into contraining and defining constraints.
            Defining constraints (totally) define new auxiliary variables needed for the decomposition,
            they can always be enforced top-level.
        """
        raise NotImplementedError("Decomposition for", self, "not available")

    def get_bounds(self):
        """
        Returns the bounds of a Boolean global constraint.
        Numerical global constraints should reimplement this.
        """
        return 0, 1


# Global Constraints (with Boolean return type)
def alldifferent(args):
    warnings.warn("Deprecated, use AllDifferent(v1,v2,...,vn) instead, will be removed in stable version", DeprecationWarning)
    return AllDifferent(*args) # unfold list as individual arguments


class AllDifferent(GlobalConstraint):
    """All arguments have a different (distinct) value
    """
    def __init__(self, *args):
        super().__init__("alldifferent", flatlist(args))

    def decompose(self):
        """Returns the decomposition
        """
        return [var1 != var2 for var1, var2 in all_pairs(self.args)], []

    def value(self):
        return len(set(argvals(self.args))) == len(self.args)


class AllDifferentExcept0(GlobalConstraint):
    """
    All nonzero arguments have a distinct value
    """
    def __init__(self, *args):
        super().__init__("alldifferent_except0", flatlist(args))

    def decompose(self):
        # equivalent to (var1 == 0) | (var2 == 0) | (var1 != var2)
        return [(var1 == var2).implies(var1 == 0) for var1, var2 in all_pairs(self.args)], []

    def value(self):
        vals = [argval(a) for a in self.args if argval(a) != 0]
        return len(set(vals)) == len(vals)

def allequal(args):
    warnings.warn("Deprecated, use AllEqual(v1,v2,...,vn) instead, will be removed in stable version", DeprecationWarning)
    return AllEqual(*args) # unfold list as individual arguments


class AllEqual(GlobalConstraint):
    """All arguments have the same value
    """
    def __init__(self, *args):
        super().__init__("allequal", flatlist(args))

    def decompose(self):
        """Returns the decomposition
        """
        # arg0 == arg1, arg1 == arg2, arg2 == arg3... no need to post n^2 equalities
        return [var1 == var2 for var1, var2 in zip(self.args[:-1], self.args[1:])], []

    def value(self):
        return len(set(argvals(self.args))) == 1

def circuit(args):
    warnings.warn("Deprecated, use Circuit(v1,v2,...,vn) instead, will be removed in stable version", DeprecationWarning)
    return Circuit(*args) # unfold list as individual arguments


class Circuit(GlobalConstraint):
    """The sequence of variables form a circuit, where x[i] = j means that j is the successor of i.
    """
    def __init__(self, *args):
        flatargs = flatlist(args)
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("Circuit global constraint only takes arithmetic arguments: {}".format(flatargs))
        super().__init__("circuit", flatargs)
        if len(flatargs) < 2:
            raise CPMpyException('Circuit constraint must be given a minimum of 2 variables')

    def decompose(self):
        """
            Decomposition for Circuit

            Not sure where we got it from,
            MiniZinc has slightly different one:
            https://github.com/MiniZinc/libminizinc/blob/master/share/minizinc/std/fzn_circuit.mzn
        """
        succ = cpm_array(self.args)
        n = len(succ)
        order = intvar(0,n-1, shape=n)
        constraining = []
        constraining += [AllDifferent(succ)] # different successors
        constraining += [AllDifferent(order)] # different orders
        constraining += [order[n-1] == 0] # symmetry breaking, last one is '0'

        defining = [order[0] == succ[0]]
        defining += [order[i] == succ[order[i-1]] for i in range(1,n)] # first one is successor of '0', ith one is successor of i-1

        return constraining, defining

    def value(self):
        pathlen = 0
        idx = 0
        visited = set()
        arr = argvals(self.args)

        while idx not in visited:
            if idx is None:
                return False
            if not (0 <= idx < len(arr)):
                break
            visited.add(idx)
            pathlen += 1
            idx = arr[idx]

        return pathlen == len(self.args) and idx == 0

class SubCircuit(GlobalConstraint):
    """
        The sequence of variables form a subcircuit, where x[i] = j means that j is the successor of i.
        Contrary to Circuit, there is no requirement on all nodes needing to be part of the circuit.
        Nodes which aren't part of the subcircuit, should self loop i.e. x[i] = i.
        The subcircuit can be empty (all stops self-loop).
        A length 1 subcircuit is treated as an empty subcircuit.

        Global Constraint Catalog:
        https://sofdem.github.io/gccat/gccat/Cproper_circuit.html
    """

    def __init__(self, *args):
        flatargs = flatlist(args)

        # Ensure all args are integer successor values
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("SubCircuit global constraint only takes arithmetic arguments: {}".format(flatargs))
        # Ensure there are at least two stops to create a circuit with
        if len(flatargs) < 2:
            raise CPMpyException("SubCircuitWithStart constraint must be given a minimum of 2 variables for field 'args' as stops to route between.")
        
        # Create the object
        super().__init__("subcircuit", flatargs)

    def decompose(self):
        """
            Decomposition for SubCircuit

            A mix of the above circuit implementation, with elements from the Minizinc implementation for the support of optional visits:
            https://github.com/MiniZinc/minizinc-old/blob/master/lib/minizinc/std/subcircuit.mzn
        """
        from .python_builtins import min as cpm_min
        from .python_builtins import all as cpm_all
        
        # Input arguments
        succ = cpm_array(self.args) # Successor variables
        n = len(succ)

        # Decision variables
        order = intvar(0, n-1, shape=n) # Order variables for the stops within the subcircuit (if part of it, otherwise a remaining value in the domain as to satisfy AllDiff).
        is_part_of_circuit = boolvar(shape=n) # Whether a stop is part of the subcircuit.
        end_index = intvar(0, n-1) # Index of the last stop in the subcircuit, before looping back to the start_index.
        start_index = intvar(0, n-1) # An additional auxilary start_index variable, helps with ensuring only one subcircuit is present
        empty = boolvar() # To detect when a subcircuit is completely empty

        constraining = []
        constraining += [AllDifferent(succ)] # All stops should have a unique successor.
        constraining += list( is_part_of_circuit.implies(succ < len(succ)) ) # Successor values should remain within domain.

        defining = []
        defining += [ empty == cpm_all(succ == np.arange(n)) ] # Definition of empty subcircuit
        defining += [ empty.implies(cpm_all(order == cpm_array([0]*n))) ] # If the subcircuit is empty, default all order values to 0
        defining += [ empty.implies(start_index == 0) ] # If the subcircuit is empty, any stop could be a start of a 0-length circuit. Default to stop 0 as symmetry breaking.
        defining += [ order[start_index] == 0 ] # The ordering starts at the start_index stop.   
        for i in range(0, n):
            # If a stop is on the subcircuit and it is not the last one, than its successor should have +1 as order.
            defining += [(is_part_of_circuit[i] & (i != end_index)).implies(
                order[succ[i]] == (order[i] + 1)
            )]
        # Either a stop is part of the subcircuit, with an order value with the range [0, order[end_index]] (so never greater than the order of the last stop within the circuit),
        # or it is not part of the subcircuit and thus must have an order value outside of this range (thus strictly greater that the order of the last stop within the circuit).
        # All stops in the subcircuit will thus be the ones with an order value between 0 and order[end_index], whilst the rest is by definition outside of the subcircuit.
        # The distribution of the order values within the range gets decided by the successor values (see previous constraint). Stops not placed on the subcircuit get the 
        # the order[end_index] + 1 assigned to them. This is needed for removing the degeneracy caused by unconstrained assignment of the remaining order values to the
        # non-subcircuit stops. This seems like a better approach than an AllDiff over 'order' and trying to add an ordering over the 'order' values of the non-subcircuit stops,
        # i.e. (~is_part_of_circuit[i] and ~is_part_of_circuit[j]) -> ( ( (i > j) and (order[i] > order[j]) ) or ( (i < j) and (order[i] < order[j]) ) ).
        # Since we don't know which nodes will be part of the subcircuit, this would have had to be formulated over all pairs of nodes.
        # We additionally have to check for the empty case. Otherwise, forcing all 'order' values to be 0 for symmetry-breaking reasons would force `is_part_of_circuit` 
        # to be true for all.
        defining += [ (is_part_of_circuit[i] == ((~empty) & (order[end_index] + 1 != order[i]))) for i in range(n)] 
        defining += [ ( empty | (is_part_of_circuit[start_index] == True) ) ] # The start index can only not belong to the subcircuit when the subcircuit is empty.
        # These two should be added if we assume a subcircuit can never be empty.
        #   It was not enforced in the end as to follow the convention of the solvers.
        # defining += [is_part_of_circuit[start_index] == True, is_part_of_circuit[end_index] == True] # Both the start and end stops should be part of the circuit.
        # defining += [start_index != end_index] # The end stop cannot be the start stop, thus the subcircuit length should be longer than 1.
        defining += [succ[end_index] == start_index] # Definition of the last stop. As the successor we should cycle back to the start.
        defining += list( is_part_of_circuit == (succ != np.arange(n)) ) # When a node is part of the subcircuit it should not self loop, if it is not part it should self loop.
        
        
        # The start_index decision variable can freely take the value of any of the stops inside the subcircuit, 
        # whilst still resulting in the same subcircuit. The start_index is a completely internal component, not externally visible.
        # Degenerate solutions will thus be created, resulting in a much higher solution count when doing solveAll()
        # Thus some symmetry breaking is needed. By assuming that the start_index will take the smallest index
        # of all stops included in the subcircuit, the degeneracy is removed.

        # Part of the formulation from the following is used:
        # https://sofdem.github.io/gccat/gccat/Ccycle.html#uid18336

        # Auxiliary variable which will hold the indices of the subcircuit in a looping fashion.
        # If we have 6 stops [0, .., 5] and the subcircuit is [0, 2, 1], then s = [0, 2, 1, 0, 2, 1].
        # Thus s is a way to collect all the indices of stops which get visited.
        s = intvar(0, n-1, shape=n)
        defining += [s[0] == start_index]
        defining += [s[i+1] == succ[s[i]] for i in range(n-1)]

        # The free start_index could be any of the values of s, resulting in degenerate solutions.
        # By enforcing start_index to take the smallest value, symmetry breaking is ensured.
        defining += [start_index == cpm_min(s)]

        return constraining, defining
    
    def value(self):

        succ = [argval(a) for a in self.args]
        n = len(succ)

        # Find a start_index
        start_index = None
        for i,s in enumerate(succ):
            if i != s:
                # first non self-loop found is taken as start
                start_index = i
                break
        # No valid start found, thus empty subcircuit
        if start_index is None:
            return True # Change to False if empty subcircuits not allowed

        # Collect subcircuit
        visited = set([start_index])
        idx = succ[start_index]
        while idx != start_index:
            # Something is wrong, certain variables didn't get values
            if idx is None: return False
            # Check bounds on successor value
            if not (0 <= idx < n): return False
            # Collect
            visited.add(idx)
            idx = succ[idx]

        # Check subcircuit
        for i in range(n):
            # A stop is either visited or self-loops
            if not ( (i in visited) or (succ[i] == i) ):
                return False

        # Check that subcircuit has length of at least 1.
        return succ[start_index] != start_index

class SubCircuitWithStart(GlobalConstraint):

    """
        The sequence of variables form a subcircuits, where x[i] = j means that j is the successor of i.
        Contrary to Circuit, there is no requirement on all nodes needing to be part of the circuit.
        Nodes which aren't part of the subcircuit, should self loop i.e. x[i] = i.
        The size of the subcircuit should be strictly greater than 1, so not all stops can selfloop 
        (as otherwise the start_index will never get visited).
        start_index will be treated as the start of the subcircuit. 
        The only impact of this is that start_index will be garuanteed to be inside the subcircuit.

        Global Constraint Catalog:
        https://sofdem.github.io/gccat/gccat/Cproper_circuit.html
    """

    def __init__(self, *args, start_index:int=0):
        flatargs = flatlist(args)

        # Ensure all args are integer successor values
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("SubCircuitWithStart global constraint only takes arithmetic arguments: {}".format(flatargs))
        # Ensure start_index is an integer
        if not isinstance(start_index, int):
            raise TypeError("SubCircuitWithStart global constraint's start_index argument must be an integer: {}".format(start_index))
        # Ensure that the start_index is within range
        if not ((start_index >= 0) and (start_index < len(flatargs))):
            raise ValueError("SubCircuitWithStart's start_index must be within the range [0, #stops-1] and thus refer to an actual stop as provided through 'args'.")
        # Ensure there are at least two stops to create a circuit with
        if len(flatargs) < 2:
            raise CPMpyException("SubCircuitWithStart constraint must be given a minimum of 2 variables for field 'args' as stops to route between.")
        
        # Create the object
        super().__init__("subcircuitwithstart", flatargs)
        self.start_index = start_index

    def decompose(self):
        """
            Decomposition for SubCircuitWithStart.

            SubCircuitWithStart simply gets decomposed into SubCircuit and a constraint
            enforcing the start_index to be part of the subcircuit.
        """
        # Get the arguments
        start_index = self.start_index
        succ = cpm_array(self.args) # Successor variables

        constraining = []

        # Optional improvement by passing start_index to SubCircuit.
        #   Whilst SubCircuit does not assume a start_index (and does not acdept one as an argument),
        #   it still uses one internally as the enforce that only one subcircuit is present.
        #   start_index takes form as a free decision variable, but by enforming SubCircuit of the
        #   known start_index, we can improve its solving performance.
        #   Unfortunately, directly accessing and modifying another constraint's private fields is 
        #   a quite ugly approach.

        # sc = SubCircuit(succ)
        # sc.symmetry_breaking = False
        # sc.start_index = start_index

        constraining += [SubCircuit(succ)] # The successor variables should form a subcircuit.
        constraining += [succ[start_index] != start_index] # The start_index should be inside the subcircuit.

        defining = []

        return constraining, defining
    
    def value(self):
        start_index = self.start_index
        succ = cpm_array(self.args) # Successor variables

        # Check if we have a valid subcicircuit and that the start_index is part of it.
        return SubCircuit(succ).value() and (succ[start_index] != start_index)
class Inverse(GlobalConstraint):
    """
       Inverse (aka channeling / assignment) constraint. 'fwd' and
       'rev' represent inverse functions; that is,

           fwd[i] == x  <==>  rev[x] == i

    """
    def __init__(self, fwd, rev):
        flatargs = flatlist([fwd,rev])
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("Only integer arguments allowed for global constraint Inverse: {}".format(flatargs))
        assert len(fwd) == len(rev)
        super().__init__("inverse", [fwd, rev])

    def decompose(self):
        from .python_builtins import all
        fwd, rev = self.args
        rev = cpm_array(rev)
        return [all(rev[x] == i for i, x in enumerate(fwd))], []

    def value(self):
        fwd = argvals(self.args[0])
        rev = argvals(self.args[1])
        # args are fine, now evaluate actual inverse cons
        try:
            return all(rev[x] == i for i, x in enumerate(fwd))
        except IndexError: # partiality of Element constraint
            return False


class Table(GlobalConstraint):
    """The values of the variables in 'array' correspond to a row in 'table'
    """
    def __init__(self, array, table):
        array = flatlist(array)
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError("the first argument of a Table constraint should only contain variables/expressions")
        super().__init__("table", [array, table])

    def decompose(self):
        from .python_builtins import any, all
        arr, tab = self.args
        return [any(all(ai == ri for ai, ri in zip(arr, row)) for row in tab)], []

    def value(self):
        arr, tab = self.args
        arrval = argvals(arr)
        return arrval in tab



# syntax of the form 'if b then x == 9 else x == 0' is not supported (no override possible)
# same semantic as CPLEX IfThenElse constraint
# https://www.ibm.com/docs/en/icos/12.9.0?topic=methods-ifthenelse-method
class IfThenElse(GlobalConstraint):
    def __init__(self, condition, if_true, if_false):
        if not is_boolexpr(condition) or not is_boolexpr(if_true) or not is_boolexpr(if_false):
            raise TypeError("only boolean expression allowed in IfThenElse")
        super().__init__("ite", [condition, if_true, if_false])

    def value(self):
        condition, if_true, if_false = self.args
        if argval(condition):
            return argval(if_true)
        else:
            return argval(if_false)

    def decompose(self):
        condition, if_true, if_false = self.args
        return [condition.implies(if_true), (~condition).implies(if_false)], []

    def __repr__(self):
        condition, if_true, if_false = self.args
        return "If {} Then {} Else {}".format(condition, if_true, if_false)



class InDomain(GlobalConstraint):
    """
        The "InDomain" constraint, defining non-interval domains for an expression
    """

    def __init__(self, expr, arr):
        super().__init__("InDomain", [expr, arr])

    def decompose(self):
        """
        Returns two lists of constraints:
            1) constraints representing the comparison
            2) constraints that (totally) define new auxiliary variables needed in the decomposition,
               they should be enforced toplevel.
        """
        from .python_builtins import any
        expr, arr = self.args
        lb, ub = expr.get_bounds()

        defining = []
        #if expr is not a var
        if not isinstance(expr,_IntVarImpl):
            aux = intvar(lb, ub)
            defining.append(aux == expr)
            expr = aux

        expressions = any(isinstance(a, Expression) for a in arr)
        if expressions:
            return [any(expr == a for a in arr)], defining
        else:
            return [expr != val for val in range(lb, ub + 1) if val not in arr], defining


    def value(self):
        return argval(self.args[0]) in argvals(self.args[1])

    def __repr__(self):
        return "{} in {}".format(self.args[0], self.args[1])


class Xor(GlobalConstraint):
    """
        The 'xor' exclusive-or constraint
    """

    def __init__(self, arg_list):
        flatargs = flatlist(arg_list)
        if not (all(is_boolexpr(arg) for arg in flatargs)):
            raise TypeError("Only Boolean arguments allowed in Xor global constraint: {}".format(flatargs))
        # convention for commutative binary operators:
        # swap if right is constant and left is not
        if len(arg_list) == 2 and is_num(arg_list[1]):
            arg_list[0], arg_list[1] = arg_list[1], arg_list[0]
            flatargs = arg_list
        super().__init__("xor", flatargs)

    def decompose(self):
        # there are multiple decompositions possible, Recursively using sum allows it to be efficient for all solvers.
        decomp = [sum(self.args[:2]) == 1]
        if len(self.args) > 2:
            decomp = Xor([decomp,self.args[2:]]).decompose()[0]
        return decomp, []

    def value(self):
        return sum(argvals(self.args)) % 2 == 1

    def __repr__(self):
        if len(self.args) == 2:
            return "{} xor {}".format(*self.args)
        return "xor({})".format(self.args)


class Cumulative(GlobalConstraint):
    """
        Global cumulative constraint. Used for resource aware scheduling.
        Ensures no overlap between tasks and never exceeding the capacity of the resource
        Supports both varying demand across tasks or equal demand for all jobs
    """
    def __init__(self, start, duration, end, demand, capacity):
        assert is_any_list(start), "start should be a list"
        assert is_any_list(duration), "duration should be a list"
        assert is_any_list(end), "end should be a list"

        start = flatlist(start)
        duration = flatlist(duration)
        end = flatlist(end)
        assert len(start) == len(duration) == len(end), "Start, duration and end should have equal length"
        n_jobs = len(start)

        for lb in get_bounds(duration)[0]:
            if lb < 0:
                raise TypeError("Durations should be non-negative")

        if is_any_list(demand):
            demand = flatlist(demand)
            assert len(demand) == n_jobs, "Demand should be supplied for each task or be single constant"
        else: # constant demand
            demand = [demand] * n_jobs

        super(Cumulative, self).__init__("cumulative", [start, duration, end, demand, capacity])

    def decompose(self):
        """
            Time-resource decomposition from:
            Schutt, Andreas, et al. "Why cumulative decomposition is not as bad as it sounds."
            International Conference on Principles and Practice of Constraint Programming. Springer, Berlin, Heidelberg, 2009.
        """
        from ..expressions.python_builtins import sum

        arr_args = (cpm_array(arg) if is_any_list(arg) else arg for arg in self.args)
        start, duration, end, demand, capacity = arr_args

        cons = []

        # set duration of tasks
        for t in range(len(start)):
            cons += [start[t] + duration[t] == end[t]]

        # demand doesn't exceed capacity
        lb, ub = min(s.lb for s in start), max(s.ub for s in end)
        for t in range(lb,ub+1):
            demand_at_t = 0
            for job in range(len(start)):
                if is_num(demand):
                    demand_at_t += demand * ((start[job] <= t) & (t < end[job]))
                else:
                    demand_at_t += demand[job] * ((start[job] <= t) & (t < end[job]))

            cons += [demand_at_t <= capacity]

        return cons, []

    def value(self):
        arg_vals = [np.array(argvals(arg)) if is_any_list(arg)
                   else argval(arg) for arg in self.args]

        if any(a is None for a in arg_vals):
            return None

        # start, dur, end are np arrays
        start, dur, end, demand, capacity = arg_vals
        # start and end seperated by duration
        if not (start + dur == end).all():
            return False

        # demand doesn't exceed capacity
        lb, ub = min(start), max(end)
        for t in range(lb, ub+1):
            if capacity < sum(demand * ((start <= t) & (t < end))):
                return False

        return True


class GlobalCardinalityCount(GlobalConstraint):
    """
    GlobalCardinalityCount(vars,vals,occ): The number of occurrences of each value vals[i] in the list of variables vars
    must be equal to occ[i].
    """

    def __init__(self, vars, vals, occ):
        flatargs = flatlist([vars, vals, occ])
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("Only numerical arguments allowed for gcc global constraint: {}".format(flatargs))
        super().__init__("gcc", [vars,vals,occ])

    def decompose(self):
        from .globalfunctions import Count
        vars, vals, occ = self.args
        return [Count(vars, i) == v for i, v in zip(vals, occ)], []

    def value(self):
        from .python_builtins import all
        decomposed, _ = self.decompose()
        return all(decomposed).value()


class Increasing(GlobalConstraint):
    """
        The "Increasing" constraint, the expressions will have increasing (not strictly) values
    """

    def __init__(self, *args):
        super().__init__("increasing", flatlist(args))

    def decompose(self):
        """
        Returns two lists of constraints:
            1) the decomposition of the Increasing constraint
            2) empty list of defining constraints
        """
        args = self.args
        return [args[i] <= args[i+1] for i in range(len(args)-1)], []

    def value(self):
        from .python_builtins import all
        args = self.args
        return all(args[i].value() <= args[i+1].value() for i in range(len(args)-1))


class Decreasing(GlobalConstraint):
    """
        The "Decreasing" constraint, the expressions will have decreasing (not strictly) values
    """

    def __init__(self, *args):
        super().__init__("decreasing", flatlist(args))

    def decompose(self):
        """
        Returns two lists of constraints:
            1) the decomposition of the Decreasing constraint
            2) empty list of defining constraints
        """
        args = self.args
        return [args[i] >= args[i+1] for i in range(len(args)-1)], []

    def value(self):
        from .python_builtins import all
        args = self.args
        return all(args[i].value() >= args[i+1].value() for i in range(len(args)-1))


class IncreasingStrict(GlobalConstraint):
    """
        The "IncreasingStrict" constraint, the expressions will have increasing (strictly) values
    """

    def __init__(self, *args):
        super().__init__("strictly_increasing", flatlist(args))

    def decompose(self):
        """
        Returns two lists of constraints:
            1) the decomposition of the IncreasingStrict constraint
            2) empty list of defining constraints
        """
        args = self.args
        return [args[i] < args[i+1] for i in range(len(args)-1)], []

    def value(self):
        from .python_builtins import all
        args = self.args
        return all((args[i].value() < args[i+1].value()) for i in range(len(args)-1))


class DecreasingStrict(GlobalConstraint):
    """
        The "DecreasingStrict" constraint, the expressions will have decreasing (strictly) values
    """

    def __init__(self, *args):
        super().__init__("strictly_decreasing", flatlist(args))

    def decompose(self):
        """
        Returns two lists of constraints:
            1) the decomposition of the DecreasingStrict constraint
            2) empty list of defining constraints
        """
        args = self.args
        return [(args[i] > args[i+1]) for i in range(len(args)-1)], []

    def value(self):
        from .python_builtins import all
        args = self.args
        return all((args[i].value() > args[i+1].value()) for i in range(len(args)-1))


class DirectConstraint(Expression):
    """
        A DirectConstraint will directly call a function of the underlying solver when added to a CPMpy solver

        It can not be reified, it is not flattened, it can not contain other CPMpy expressions than variables.
        When added to a CPMpy solver, it will literally just directly call a function on the underlying solver,
        replacing CPMpy variables by solver variables along the way.

        See the documentation of the solver (constructor) for details on how that solver handles them.

        If you want/need to use what the solver returns (e.g. an identifier for use in other constraints),
        then use `directvar()` instead, or access the solver object from the solver interface directly.
    """
    def __init__(self, name, arguments, novar=None):
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

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return True

    def callSolver(self, CPMpy_solver, Native_solver):
        """
            Call the `directname`() function of the native solver,
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

