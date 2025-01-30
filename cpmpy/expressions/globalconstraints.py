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
        AllDifferentExceptN
        AllEqual
        AllEqualExceptN
        Circuit
        SubCircuit
        SubCircuitWithStart
        Inverse
        Table
        NegativeTable
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


import cpmpy as cp

from .core import BoolVal
from .utils import all_pairs, is_transition
from .variables import _IntVarImpl
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
    warnings.warn("Deprecated, use AllDifferent(v1,v2,...,vn) instead, will be removed in "
                  "stable version", DeprecationWarning)
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

class AllDifferentExceptN(GlobalConstraint):
    """
        All arguments except those equal to a value in n have a distinct value.
    """
    def __init__(self, arr, n):
        flatarr = flatlist(arr)
        if not is_any_list(n):
            n = [n]
        super().__init__("alldifferent_except_n", [flatarr, n])

    def decompose(self):
        # equivalent to (var1 == n) | (var2 == n) | (var1 != var2)
        return [(var1 == var2).implies(cp.any(var1 == a for a in self.args[1])) for var1, var2 in all_pairs(self.args[0])], []

    def value(self):
        vals = [argval(a) for a in self.args[0] if argval(a) not in argvals(self.args[1])]
        return len(set(vals)) == len(vals)


class AllDifferentExcept0(AllDifferentExceptN):
    """
        All nonzero arguments have a distinct value
    """
    def __init__(self, *arr):
        flatarr = flatlist(arr)
        super().__init__(arr, 0)


def allequal(args):
    warnings.warn("Deprecated, use AllEqual(v1,v2,...,vn) instead, will be removed in stable version",
                  DeprecationWarning)
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


class AllEqualExceptN(GlobalConstraint):
    """
    All arguments except those equal to a value in n have the same value.
    """

    def __init__(self, arr, n):
        flatarr = flatlist(arr)
        if not is_any_list(n):
            n = [n]
        super().__init__("allequal_except_n", [flatarr, n])

    def decompose(self):
        return [(cp.any(var1 == a for a in self.args[1]) | (var1 == var2) | cp.any(var2 == a for a in self.args[1]))
                for var1, var2 in all_pairs(self.args[0])], []

    def value(self):
        vals = [argval(a) for a in self.args[0] if argval(a) not in argvals(self.args[1])]
        return len(set(vals)) == 1 or len(set(vals)) == 0


def circuit(args):
    warnings.warn("Deprecated, use Circuit(v1,v2,...,vn) instead, will be removed in stable version",
                  DeprecationWarning)
    return Circuit(*args) # unfold list as individual arguments


class Circuit(GlobalConstraint):
    """The sequence of variables form a circuit, where x[i] = j means that j is the successor of i.
    """
    def __init__(self, *args):
        flatargs = flatlist(args)
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("Circuit global constraint only takes arithmetic arguments: {}".format(flatargs))
        if len(flatargs) < 2:
            raise CPMpyException('Circuit constraint must be given a minimum of 2 variables')
        super().__init__("circuit", flatargs)

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
        defining = []
        constraining = []

        # We define the auxiliary order variables to represent the order we visit all the nodes.
        # `order[i] == succ[order[i - 1]]`
        # These constraints need to be in the defining part, since they define our auxiliary vars
        # However, this would make it impossible for ~circuit to be satisfied in some cases,
        # because there does not always exist a valid ordering
        # This happens when the variables in succ don't take values in the domain of 'order',
        # i.e. for succ = [9,-1,0], there is no valid ordering, but we satisfy ~circuit(succ)
        # We explicitly deal with these cases by defining the variable 'a' that indicates if we can define an ordering.

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
            defining += [a == ((Minimum(succ) >= 0) & (Maximum(succ) < n))]
            for i in range(n):
                defining += [(~a).implies(order[i] == 0)]  # assign arbitrary value, so a is totally defined.

        constraining += [AllDifferent(succ)]  # different successors
        constraining += [AllDifferent(order)]  # different orders
        constraining += [order[n - 1] == 0]  # symmetry breaking, last one is '0'
        defining += [a.implies(order[0] == succ[0])]
        for i in range(1, n):
            defining += [a.implies(
                order[i] == succ[order[i - 1]])]  # first one is successor of '0', ith one is successor of i-1
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

            A mix of the above Circuit decomposition, with elements from the Minizinc implementation for the support of optional visits:
            https://github.com/MiniZinc/minizinc-old/blob/master/lib/minizinc/std/subcircuit.mzn
        """
        from .python_builtins import min as cpm_min
        from .python_builtins import all as cpm_all

        # Input arguments
        succ = cpm_array(self.args) # Successor variables
        n = len(succ)

        # Decision variables
        start_node = intvar(0, n-1) # The first stop in the subcircuit.
        end_node = intvar(0, n-1) # The last stop in the subcircuit, before looping back to the "start_node".
        index_within_subcircuit = intvar(0, n-1, shape=n) # The position each stop takes within the subcircuit, with the assumption that the stop "start_node" gets index 0.
        is_part_of_circuit = boolvar(shape=n) # Whether a stop is part of the subcircuit.
        empty = boolvar() # To detect when a subcircuit is completely empty

        # Constraining
        constraining = []
        constraining += [AllDifferent(succ)] # All stops should have a unique successor.
        constraining += list( is_part_of_circuit.implies(succ < len(succ)) ) # Successor values should remain within domain.
        for i in range(0, n):
            # If a stop is on the subcircuit and it is not the last one, than its successor should have +1 as index.
            constraining += [(is_part_of_circuit[i] & (i != end_node)).implies(
                index_within_subcircuit[succ[i]] == (index_within_subcircuit[i] + 1)
            )]
        constraining += list( is_part_of_circuit == (succ != np.arange(n)) ) # When a node is part of the subcircuit it should not self loop, if it is not part it should self loop.

        # Defining
        defining = []
        defining += [ empty == cpm_all(succ == np.arange(n)) ] # Definition of empty subcircuit (all nodes self-loop)
        defining += [ empty.implies(cpm_all(index_within_subcircuit == cpm_array([0]*n))) ] # If the subcircuit is empty, default all index values to 0
        defining += [ empty.implies(start_node == 0) ] # If the subcircuit is empty, any node could be a start of a 0-length circuit. Default to node 0 as symmetry breaking.
        defining += [succ[end_node] == start_node] # Definition of the last node. As the successor we should cycle back to the start.
        defining += [ index_within_subcircuit[start_node] == 0 ] # The ordering starts at the start_node.
        defining += [ ( empty | (is_part_of_circuit[start_node] == True) ) ] # The start node can only NOT belong to the subcircuit when the subcircuit is empty.
        # Nodes which are not part of the subcircuit get an index fixed to +1 the index of "end_node", which equals the length of the subcircuit.
        # Nodes part of the subcircuit must have an index <= index_within_subcircuit[end_node].
        # The case of an empty subcircuit is an exception, since "end_node" itself is not part of the subcircuit
        defining += [ (is_part_of_circuit[i] == ((~empty) & (index_within_subcircuit[end_node] + 1 != index_within_subcircuit[i]))) for i in range(n)]
        # In a subcircuit any of the visited nodes can be the "start node", resulting in symmetrical solutions -> Symmetry breaking
        # Part of the formulation from the following is used: https://sofdem.github.io/gccat/gccat/Ccycle.html#uid18336
        subcircuit_visits = intvar(0, n-1, shape=n) # The visited nodes in sequence of length n, with possible repeated stops. e.g. subcircuit [0, 2, 1] -> [0, 2, 1, 0, 2, 1]
        defining += [subcircuit_visits[0] == start_node] # The start nodes is the first stop
        defining += [subcircuit_visits[i+1] == succ[subcircuit_visits[i]] for i in range(n-1)] # We follow the successor values
        # The free "start_node" could be any of the values of aux_subcircuit_visits (the actually visited nodes), resulting in degenerate solutions.
        # By enforcing "start_node" to take the smallest value, symmetry breaking is ensured.
        defining += [start_node == cpm_min(subcircuit_visits)]

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

        # Check AllDiff
        if not AllDifferent(s).value():
            return False

        # Collect subcircuit
        visited = set([start_index])
        idx = succ[start_index]
        for i in range(len(succ)):
            if idx ==start_index:
                break
            else:
                if idx in visited:
                    return False
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
        The sequence of variables form a subcircuit, where x[i] = j means that j is the successor of i.
        Contrary to Circuit, there is no requirement on all nodes needing to be part of the circuit.
        Nodes which aren't part of the subcircuit, should self loop i.e. x[i] = i.
        The size of the subcircuit should be strictly greater than 1, so not all stops can self loop
        (as otherwise the start_index will never get visited).
        start_index will be treated as the start of the subcircuit.
        The only impact of start_index is that it will be guaranteed to be inside the subcircuit.

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
        constraining += [SubCircuit(self.args)] # The successor variables should form a subcircuit.
        constraining += [succ[start_index] != start_index] # The start_index should be inside the subcircuit.

        defining = []

        return constraining, defining

    def value(self):
        start_index = self.start_index
        succ = [argval(a) for a in self.args] # Successor variables

        # Check if we have a valid subcircuit and that the start_index is part of it.
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
        fwd, rev = self.args
        rev = cpm_array(rev)
        return [cp.all(rev[x] == i for i, x in enumerate(fwd))], []

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
        if isinstance(table, np.ndarray): # Ensure it is a list
            table = table.tolist()
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError(f"the first argument of a Table constraint should only contain variables/expressions: "
                            f"{array}")
        super().__init__("table", [array, table])

    def decompose(self):
        arr, tab = self.args
        return [cp.any(cp.all(ai == ri for ai, ri in zip(arr, row)) for row in tab)], []

    def value(self):
        arr, tab = self.args
        arrval = argvals(arr)
        return arrval in tab


class NegativeTable(GlobalConstraint):
    """The values of the variables in 'array' do not correspond to any row in 'table'
    """
    def __init__(self, array, table):
        array = flatlist(array)
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError(f"the first argument of a Table constraint should only contain variables/expressions: "
                            f"{array}")
        super().__init__("negative_table", [array, table])

    def decompose(self):
        arr, tab = self.args
        return [cp.all(cp.any(ai != ri for ai, ri in zip(arr, row)) for row in tab)], []

    def value(self):
        arr, tab = self.args
        arrval = argvals(arr)
        tabval = argvals(tab)
        return arrval not in tabval

class MDD(GlobalConstraint):
    """
    MDD-constraint: an MDD (Multi-valued Decision Diagram) is an acyclic layerd graph starting from a single node and
    ending in one. Each edge layer corresponds to a variables and each path corresponds to a solution

    The values of the variables in 'array' correspond to a path in the mdd formed by the transitions in 'transitions'.
    Root node is the first node used as a start in the first transition (i.e. transitions[0][0])

    spec:
        - array: an array of CPMpy expressions (integer variable, global functions,...)
        - transitions: an array of tuples (nodeID, int, nodeID) where nodeID is some unique identifiers for the nodes
        (int or str are fine)

    Example:
        The following transitions depict a 3 layer MDD, starting at 'r' and ending in 't'
        ("r", 0, "n1"), ("r", 1, "n2"), ("r", 2, "n3"), ("n1", 2, "n4"), ("n2", 2, "n4"), ("n3", 0, "n5"),
        ("n4", 0, "t"), ("n5", 1, "t")
        Its graphical representation is:
                  r
              0/ |1  \2     X
            n1   n2   n3
            2| /2    /O     Y
             n4     n5
              0\   /1       Z
                 t
        It has 3 paths, corresponding to 3 solution for (X,Y,Z): (0,2,0), (1,2,0) and (2,0,1)
    """

    def __init__(self, array, transitions):
        array = flatlist(array)
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError("The first argument of an MDD constraint should only contain variables/expressions")
        if not all(is_transition(transition) for transition in transitions):
            raise TypeError("The second argument of an MDD constraint should be collection of transitions")
        super().__init__("mdd", [array, transitions])
        self.root_node = transitions[0][0]
        self.mapping = {}
        for s, v, e in transitions:
            self.mapping[(s, v)] = e

    def _transition_to_layer_representation(self):
        """ auxiliary function to compute which nodes belongs to which node-layer and which transition belongs to which
        edge-layer of the MDD, needed to compute decomposition
        """
        arr, transitions = self.args
        nodes_by_level = [[self.root_node]]
        transitions_by_level = []
        tran = transitions
        for i in range(len(arr)): # go through each layer
            nodes_by_level.append([])
            transitions_by_level.append([])
            remaining_tran = []
            for t in tran: # test each transition
                ns, _, ne = t
                if ns in nodes_by_level[i]: # add to the current layer if start node belongs to the node-layer
                    if ne not in nodes_by_level[i + 1]:
                        nodes_by_level[i + 1].append(ne)
                    transitions_by_level[i].append(t)
                else:
                    remaining_tran.append(t)
            tran = remaining_tran
        return nodes_by_level, transitions_by_level

    # auxillary method to transform into layered representation (gather all the node by node-layers)
    def _normalize_layer_representation(self, nodes_by_level, transitions_by_level):
        """ auxiliary function to normalize the names of the nodes in layer by layer representation. Node ID in
        normalized representation goes from 0 to n-1 for each layer. Used by the decomposition of the constraint.
        """
        nb_nodes_by_level = [len(x) for x in nodes_by_level]
        num_mapping = {}
        for lvl in nodes_by_level:
            for i in range(len(lvl)):
                num_mapping[lvl[i]] = i
        transitions_by_level_normalized = [[[num_mapping[n_in], v, num_mapping[n_out]]
                                            for n_in, v, n_out in lvl]
                                           for lvl in transitions_by_level]
        return nb_nodes_by_level, num_mapping, transitions_by_level_normalized


    def decompose(self):
        # Table decomposition (not by decomposition of the mdd into one big table, but by having transitions tables for
        # each layer and auxiliary variables for the nodes. Similar to decomposition of regular into table,
        # but with one table for each layer
        arr, _ = self.args
        lb = [x.lb for x in arr]
        ub = [x.ub for x in arr]
        # transform to layer representation
        nbl, tbl = self._transition_to_layer_representation()
        # normalize the naming of the nodes so it can be use as value for aux variables
        nb_nodes_by_level, num_mapping, transitions_by_level_normalized = self._normalize_layer_representation(nbl, tbl)
        # choose the best decomposition depending on number of levels
        if len(transitions_by_level_normalized) > 2:
            # decomposition with multiple transitions table and aux variables for the nodes
            aux = [intvar(0, nb_nodes) for nb_nodes in nb_nodes_by_level[1:]]
            # complete the MDD with additional dummy transitions to get the false end node also represented,
            # needed so the negation works.
            # I.E., now any assignment have a path in the MDD, some, the solutions, ending in an accepting state
            # (end node of the initial MDD), other, the non-solutions, ending in a rejecting state (dummy end node)
            for i in range(len(arr)):
                # add for each state the missing transition to a dummy node on the next level
                transition_dummy = [[num_mapping[n], v, nb_nodes_by_level[i+1]] for n in nbl[i] for v in range(lb[i], ub[i] + 1) if
                            (n, v) not in self.mapping]
                if i != 0:
                    # add transition from one dummy node to the other (not needed for initial layer as no dummy there)
                    transition_dummy += [[nb_nodes_by_level[i], v, nb_nodes_by_level[i+1]] for v in range(lb[i], ub[i] + 1)]
                # add the new transitions
                transitions_by_level_normalized[i] = transitions_by_level_normalized[i] + transition_dummy
            # optimization for first level (only one node, allows to deal with smaller table on first layer)
            tab_first = [x[1:] for x in transitions_by_level_normalized[0]]
            # defining constraints: aux and arr variables define a path in the augmented-with-negative-path-MDD
            defining = [Table([arr[0], aux[0]], tab_first)] \
                   + [Table([aux[i - 1], arr[i], aux[i]], transitions_by_level_normalized[i]) for i in
                      range(1, len(arr))]
            # constraining constraint: end of the path in accepting node
            constraining = [aux[-1] == 0]
            return constraining, defining
        elif len(transitions_by_level_normalized) == 2:
            # decomposition by unfolding into a table (i.e., extract all paths and list them as table entries),
            # avoid auxiliary variables
            tab = [[t_a[1], t_b[1]] for t_a in transitions_by_level_normalized[0] for t_b in
                   transitions_by_level_normalized[1] if t_a[2] == t_b[0]]
            return [Table(arr, tab)], []

        elif len(transitions_by_level_normalized) == 1:
            # decomposition to inDomain, avoid auxiliary variables and tables
            return [InDomain(arr[0], [t[1] for t in transitions_by_level_normalized[0]])], []

    def value(self):
        arr, transitions = self.args
        arrval = [argval(a) for a in arr]
        curr_node = self.root_node
        for v in arrval:
            if (curr_node, v) in self.mapping:
                curr_node = self.mapping[curr_node, v]
            else:
                return False
        return True # can only have reached end node

class Regular(GlobalConstraint):
    """
    Regular-constraint (or Automaton-constraint): An automaton is a directed graph. Each node correspond to a state.
    Each edge correspond to a transition from one state to the other given a value. A given node serves as start
    node. A path a size N is a solution if, by following the transitions given by the values of the variables we end up
    in one of the defined end nodes.

    The values of the variables in 'array' correspond to a path in the automaton formed by the transitions in
    'transitions'. The path starts in 'start' and ends in one of the ending states ('ends')

    spec:
        - array: an array of CPMpy expressions (integer variable, global functions,...)
        - transitions: an array of tuples (nodeID, int, nodeID) where nodeID is some unique identifiers for the nodes
        (int or str)
        - start: a singular nodeID node start of the automaton
        - ends: a list of nodeID corresponding to the accepting end nodes

    Example:
        The following transitions depict an automaton, starting at 'a' and ending in ['c']
        ("a", 1, "b"), ("b", 1, "c"), ("b", 0, "b"), ("c", 1, "c"), ("c", 0, "b")
        Its graphical representation is:
                |--0----|
                v       |
        a -1->  b  -1-> c --
               ^  \     ^  |
              |-0-|     |-1-
        It has 2 solution for (X,Y,Z): (1,1,1) and (1,0,1)
        It has 4 solutions for (W,X,Y,Z): (1,1,1,1), (1,1,0,1), (1,0,0,1) and (1,0,1,1)
    """
    def __init__(self, array, transitions, start, ends):
        array = flatlist(array)
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError("The first argument of a regular constraint should only contain variables/expressions")
        if not isinstance(start, (str, int)):
            raise TypeError("The third argument of a regular constraint should be a nodeID")
        node_type = type(start) # all nodes must be of the same type
        if not all(is_transition(transition, type=node_type) for transition in transitions):
            raise TypeError("The second argument of a regular constraint should be a collection of transitions")
        if not (isinstance(ends, list) and all(isinstance(e, node_type)for e in ends)):
            raise TypeError("The fourth argument of a regular constraint should be a list of nodeID")

        nodes = set(arg[0] for arg in transitions) | set(arg[2] for arg in transitions)
        if node_type == str: # map strings to integers
            node_map = dict(zip(sorted(nodes), range(len(nodes))))  # map nodes to integers
            transitions = [(node_map[s1], v, node_map[s2]) for s1,v,s2 in transitions]
            start = node_map[start]
            ends = [node_map[e] for e in ends]

        super().__init__("regular", [array, transitions, start, ends])

        self.mapping = {}
        for s, v, e in transitions:
            self.mapping[(s, v)] = e

    def decompose(self):
        arr, transitions, start, ends = self.args
        # get the range of possible transition value
        lb = min([x.lb for x in arr])
        ub = max([x.ub for x in arr])
        # Table decomposition with aux variables for the states
        nodes = list(set([t[0] for t in transitions] + [t[-1] for t in transitions]))  # get all nodes used
        # normalization of the id of the node (from 0 to n-1)
        num_mapping = dict(zip(nodes, range(len(nodes))))  # map node to integer ids for the nodes
        num_transitions = [[num_mapping[n_in], v, num_mapping[n_out]] for n_in, v, n_out in
                           transitions]  # apply mapping to transition
        # compute missing transition with an additionnal never-accepting sink node (dummy default node)
        id_dummy = len(nodes)  # default node id
        transition_dummy = [[num_mapping[n], v, id_dummy] for n in nodes for v in range(lb, ub + 1) if
                            (n, v) not in self.mapping] + [[id_dummy, v, id_dummy] for v in range(lb, ub + 1)]
        num_transitions = num_transitions + transition_dummy
        # auxiliary variable representing the sequence of state node in the path
        aux_vars = intvar(0, id_dummy, shape=len(arr))
        id_start = num_mapping[start]
        # optimization for first level (only one node, allows to deal with smaller table on first layer)
        tab_first = [t[1:] for t in num_transitions if t[0] == id_start]
        id_ends = [num_mapping[e] for e in ends]
        # defining constraints: aux and arr variables define a path in the augmented-with-negative-path-Automaton
        defining = [Table([arr[0], aux_vars[0]], tab_first)] + \
                                                  [Table([aux_vars[i - 1], arr[i], aux_vars[i]], num_transitions) for i
                                                   in range(1, len(arr))]
        # constraining constraint: end of the path in accepting node
        constraining = [InDomain(aux_vars[-1], id_ends)]
        return constraining, defining

    def value(self):
        arr, transitions, start, ends = self.args
        arrval = [argval(a) for a in arr]
        curr_node = start
        for v in arrval:
            if (curr_node, v) in self.mapping:
                curr_node = self.mapping[curr_node, v]
            else:
                return False
        return curr_node in ends


# syntax of the form 'if b then x == 9 else x == 0' is not supported (no override possible)
# same semantic as CPLEX IfThenElse constraint
# https://www.ibm.com/docs/en/icos/12.9.0?topic=methods-ifthenelse-method
class IfThenElse(GlobalConstraint):
    def __init__(self, condition, if_true, if_false):
        if not is_boolexpr(condition):
            raise TypeError("only Boolean expression allowed as condition")
        if not is_boolexpr(if_true) or not is_boolexpr(if_false):
            raise TypeError("only Boolean expressions allowed as result, use cpmpy.expressions.globalfunctions.IfThenElseNum instead")
        super().__init__("ite", [condition, if_true, if_false])

    def value(self):
        condition, if_true, if_false = self.args
        try:
            if argval(condition):
                return argval(if_true)
            else:
                return argval(if_false)
        except IncompleteFunctionError:
            return False

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
            return [cp.any(expr == a for a in arr)], defining
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
        Ensures that the capacity of the resource is never exceeded
        Equivalent to noOverlap when demand and capacity are equal to 1
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

        arr_args = (cpm_array(arg) if is_any_list(arg) else arg for arg in self.args)
        start, duration, end, demand, capacity = arr_args

        cons = []

        # set duration of tasks
        for t in range(len(start)):
            cons += [start[t] + duration[t] == end[t]]

        # demand doesn't exceed capacity
        lb, ub = min(get_bounds(start)[0]), max(get_bounds(end)[1])
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


class Precedence(GlobalConstraint):
    """
        Constraint enforcing some values have precedence over others.
        Given an array of variables X and a list of precedences P:
        Then in order to satisfy the constraint, if X[i] = P[j+1], then there exists a X[i'] = P[j] with i' < i
    """
    def __init__(self, vars, precedence):
        if not is_any_list(vars):
            raise TypeError("Precedence expects a list of variables, but got", vars)
        if not is_any_list(precedence) or any(isinstance(x, Expression) for x in precedence):
            raise TypeError("Precedence expects a list of values as precedence, but got", precedence)
        super().__init__("precedence", [vars, precedence])

    def decompose(self):
        """
        Decomposition based on:
        Law, Yat Chiu, and Jimmy HM Lee. "Global constraints for integer and set value precedence."
        Principles and Practice of Constraint Programming–CP 2004: 10th International Conference, CP 2004
        """

        args, precedence = self.args
        constraints = []
        for s,t in zip(precedence[:-1], precedence[1:]):
            for j in range(len(args)):
                constraints += [(args[j] == t).implies(cp.any(args[:j] == s))]
        return constraints, []

    def value(self):

        args, precedence = self.args
        vals = np.array(argvals(args))
        for s,t in zip(precedence[:-1], precedence[1:]):
            if vals[0] == t: return False
            for j in range(len(args)):
                if vals[j] == t and sum(vals[:j] == s) == 0:
                    return False
        return True


class NoOverlap(GlobalConstraint):

    def __init__(self, start, dur, end):
        assert is_any_list(start), "start should be a list"
        assert is_any_list(dur), "duration should be a list"
        assert is_any_list(end), "end should be a list"

        start = flatlist(start)
        dur = flatlist(dur)
        end = flatlist(end)
        assert len(start) == len(dur) == len(end), "Start, duration and end should have equal length " \
                                                   "in NoOverlap constraint"

        super().__init__("no_overlap", [start, dur, end])

    def decompose(self):
        start, dur, end = self.args
        cons = [s + d == e for s,d,e in zip(start, dur, end)]
        for (s1, e1), (s2, e2) in all_pairs(zip(start, end)):
            cons += [(e1 <= s2) | (e2 <= s1)]
        return cons, []
    def value(self):
        start, dur, end = argvals(self.args)
        if any(s + d != e for s,d,e in zip(start, dur, end)):
            return False
        for (s1,d1, e1), (s2,d2, e2) in all_pairs(zip(start,dur, end)):
            if e1 > s2 and e2 > s1:
                return False
        return True


class GlobalCardinalityCount(GlobalConstraint):
    """
    GlobalCardinalityCount(vars,vals,occ): The number of occurrences of each value vals[i] in the list of variables vars
    must be equal to occ[i].
    """

    def __init__(self, vars, vals, occ, closed=False):
        flatargs = flatlist([vars, vals, occ])
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("Only numerical arguments allowed for gcc global constraint: {}".format(flatargs))
        super().__init__("gcc", [vars,vals,occ])
        self.closed = closed

    def decompose(self):
        vars, vals, occ = self.args
        constraints = [Count(vars, i) == v for i, v in zip(vals, occ)]
        if self.closed:
            constraints += [InDomain(v, vals) for v in vars]
        return constraints, []

    def value(self):
        decomposed, _ = self.decompose()
        return cp.all(decomposed).value()


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
        args = argvals(self.args)
        return all(args[i] <= args[i+1] for i in range(len(args)-1))


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
        args = argvals(self.args)
        return all(args[i] >= args[i+1] for i in range(len(args)-1))


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
        args = argvals(self.args)
        return all(args[i] < args[i+1] for i in range(len(args)-1))


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
        args = argvals(self.args)
        return all(args[i] > args[i+1] for i in range(len(args)-1))


class LexLess(GlobalConstraint):
    """ Given lists X,Y, enforcing that X is lexicographically less than Y.
    """
    def __init__(self, list1, list2):
        X = flatlist(list1)
        Y = flatlist(list2)
        if len(X) != len(Y):
            raise CPMpyException(f"The 2 lists given in LexLess must have the same size: X length is {len(X)} "
                                 f"and Y length is {len(Y)}")
        super().__init__("lex_less", [X, Y])

    def decompose(self):
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
        """
        X, Y = cpm_array(self.args)

        bvar = boolvar(shape=(len(X) + 1))

        # Constraint ensuring that each element in X is less than or equal to the corresponding element in Y,
        # until a strict inequality is encountered.
        defining = [bvar == ((X <= Y) & ((X < Y) | bvar[1:]))]
        # enforce the last element to be true iff (X[-1] < Y[-1]), enforcing strict lexicographic order
        defining.append(bvar[-1] == (X[-1] < Y[-1]))
        constraining = [bvar[0]]

        return constraining, defining

    def value(self):
        X, Y = argvals(self.args)
        return any((X[i] < Y[i]) & all(X[j] <= Y[j] for j in range(i)) for i in range(len(X)))


class LexLessEq(GlobalConstraint):
    """ Given lists X,Y, enforcing that X is lexicographically less than Y (or equal).
    """
    def __init__(self, list1, list2):
        X = flatlist(list1)
        Y = flatlist(list2)
        if len(X) != len(Y):
            raise CPMpyException(f"The 2 lists given in LexLessEq must have the same size: X length is "
                                 f"{len(X)} and Y length is {len(Y)}")
        super().__init__("lex_lesseq", [X, Y])

    def decompose(self):
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
        """
        X, Y = cpm_array(self.args)

        bvar = boolvar(shape=(len(X) + 1))
        defining = [bvar == ((X <= Y) & ((X < Y) | bvar[1:]))]
        defining.append(bvar[-1] == (X[-1] <= Y[-1]))
        constraining = [bvar[0]]

        return constraining, defining

    def value(self):
        X, Y = argvals(self.args)
        return any((X[i] < Y[i]) & all(X[j] <= Y[j] for j in range(i)) for i in range(len(X))) | all(X[i] == Y[i] for i in range(len(X)))


class LexChainLess(GlobalConstraint):
    """ Given a matrix X, LexChainLess enforces that all rows are lexicographically ordered.
    """
    def __init__(self, X):
        # Ensure the numpy array is 2D
        X = cpm_array(X)
        assert X.ndim == 2, "Input must be a 2D array or a list of lists"
        super().__init__("lex_chain_less", X.tolist())

    def decompose(self):
        """ Decompose to a series of LexLess constraints between subsequent rows
        """
        X = self.args
        return [LexLess(prev_row, curr_row) for prev_row, curr_row in zip(X, X[1:])], []

    def value(self):
        X = argvals(self.args)
        return all(LexLess(prev_row, curr_row).value() for prev_row, curr_row in zip(X, X[1:]))


class LexChainLessEq(GlobalConstraint):
    """ Given a matrix X, LexChainLessEq enforces that all rows are lexicographically ordered.
    """
    def __init__(self, X):
        # Ensure the numpy array is 2D
        X = cpm_array(X)
        assert X.ndim == 2, "Input must be a 2D array or a list of lists"
        super().__init__("lex_chain_lesseq", X.tolist())

    def decompose(self):
        """ Decompose to a series of LexLessEq constraints between subsequent rows
        """
        X = self.args
        return [LexLessEq(prev_row, curr_row) for prev_row, curr_row in zip(X, X[1:])], []

    def value(self):
        X = argvals(self.args)
        return all(LexLessEq(prev_row, curr_row).value() for prev_row, curr_row in zip(X, X[1:]))


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

