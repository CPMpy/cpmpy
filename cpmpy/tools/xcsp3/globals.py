"""
Additional global constraints which are not (yet) part of the standard CPMpy collection.

This file contains all the missing global constraints in order to support XCSP3-core, 
which is a restricted scope of the complete XCSP3 specification (as used for the competitions).

Currently, version 3.2 is supported.


===============
List of classes
===============

.. autosummary::
    :nosignatures:

    AllDifferentLists
    AllDifferentListsExceptN
    SubCircuit
    SubCircuitWithStart
    SafeOnlyInverse
    InverseOne
    Channel
    NonReifiedTable
    RowSelectingShortTable
    NegativeShortTable
    MDD
    Regular
    NotInDomain
    NoOverlap2d
    IfThenElseNum
    DynamicCumulative

=============================
List of Solver-native classes
=============================

.. autosummary::
    :nosignatures:

    OrtNoOverlap2D
    OrtSubcircuit
    OrtSubcircuitWithStart
    ChocoSubcircuit
    MinizincSubcircuit
    MinizincSubcircuitWithStart
"""

import numpy as np
import cpmpy as cp
from cpmpy import cpm_array, intvar, boolvar
from cpmpy.exceptions import CPMpyException
from cpmpy.expressions.core import Expression, Operator
from cpmpy.expressions.globalconstraints import GlobalConstraint, GlobalFunction, AllDifferent, InDomain, DirectConstraint
from cpmpy.expressions.utils import STAR, is_any_list, is_num, all_pairs, argvals, flatlist, is_boolexpr, argval, is_int, \
    get_bounds, eval_comparison
from cpmpy.expressions.variables import _IntVarImpl


# ---------------------------------------------------------------------------- #
#                                Generic Globals                               #
# ---------------------------------------------------------------------------- #
class AllDifferentLists(GlobalConstraint):
    """
        Ensures none of the lists given are exactly the same.
        Called 'lex_alldifferent' in the global constraint catalog:
        https://sofdem.github.io/gccat/gccat/Clex_alldifferent.html#uid24923
    """
    def __init__(self, lists):
        if any(not is_any_list(lst) for lst in lists):
            raise TypeError(f"AllDifferentLists expects a list of lists, but got {lists}")
        if any(len(lst) != len(lists[0]) for lst in lists):
            raise ValueError("Lists should have equal length, but got these lengths:", list(map(len, lists)))
        super().__init__("alldifferent_lists", [flatlist(lst) for lst in lists])

    def decompose(self):
        """Returns the decomposition
        """
        from cpmpy.expressions.python_builtins import any as cpm_any
        constraints = []
        for lst1, lst2 in all_pairs(self.args):
            constraints += [cpm_any(var1 != var2 for var1, var2 in zip(lst1, lst2))]
        return constraints, []

    def value(self):
        lst_vals = [tuple(argvals(a)) for a in self.args]
        return len(set(lst_vals)) == len(self.args)
class AllDifferentListsExceptN(GlobalConstraint):
    """
        Ensures none of the lists given are exactly the same. Excluding the tuples given in N
        Called 'lex_alldifferent' in the global constraint catalog:
        https://sofdem.github.io/gccat/gccat/Clex_alldifferent.html#uid24923
    """
    def __init__(self, lists, n):
        if not is_any_list(n):
            raise TypeError(f"AllDifferentListsExceptN expects a (list of) lists to exclude but got {n}")
        if any(not is_any_list(x) for x in n): #only one list given, not a list of lists
            n = [n]
        for lst in n:
            if not all(is_num(x) for x in lst):
                raise TypeError("Can only use constants as excepting argument")
        if any(not is_any_list(lst) for lst in lists):
            raise TypeError(f"AllDifferentListsExceptN expects a list of lists, but got {lists}")
        if any(len(lst) != len(lists[0]) for lst in lists + n):
            raise ValueError("Lists should have equal length, but got these lengths:", list(map(len, lists)))
        super().__init__("alldifferent_lists_except_n", [[flatlist(lst) for lst in lists], [flatlist(x) for x in n]])

    def decompose(self):
        """Returns the decomposition
        """
        from cpmpy.expressions.python_builtins import all as cpm_all
        constraints = []
        for lst1, lst2 in all_pairs(self.args[0]):
            constraints += [cpm_all(var1 == var2 for var1, var2 in zip(lst1, lst2)).implies(NonReifiedTable(lst1, self.args[1]))]
        return constraints, []

    def value(self):
        lst_vals = [tuple(argvals(a)) for a in self.args[0]]
        except_vals = [tuple(argvals(a)) for a in self.args[1]]
        return len(set(lst_vals) - set(except_vals)) == len([x for x in lst_vals if x not in except_vals])

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
        from cpmpy.expressions.python_builtins import min as cpm_min
        from cpmpy.expressions.python_builtins import all as cpm_all

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
        super().__init__("subcircuitwithstart", flatargs + [start_index])

    def decompose(self):
        """
            Decomposition for SubCircuitWithStart.
            SubCircuitWithStart simply gets decomposed into SubCircuit and a constraint
            enforcing the start_index to be part of the subcircuit.
        """
        # Get the arguments
        start_index = self.args[-1]
        succ = cpm_array(self.args[:-1]) # Successor variables

        constraining = []
        constraining += [SubCircuit(succ)] # The successor variables should form a subcircuit.
        constraining += [succ[start_index] != start_index] # The start_index should be inside the subcircuit.

        defining = []

        return constraining, defining

    def value(self):
        start_index = self.args[-1]
        succ = [argval(a) for a in self.args[:-1]] # Successor variables

        # Check if we have a valid subcircuit and that the start_index is part of it.
        return SubCircuit(succ).value() and (succ[start_index] != start_index)

class SafeOnlyInverse(GlobalConstraint):
    """
       Inverse (aka channeling / assignment) constraint. 'fwd' and
       'rev' represent inverse functions; that is,

            The symmetric version (where len(fwd) == len(rev)) is defined as:
                fwd[i] == x  <==>  rev[x] == i
            The asymmetric version (where len(fwd) < len(rev)) is defined as:
                fwd[i] == x   =>   rev[x] == i

        The included decomposition only covers the "safe" variant, meaning that the expressions in fwd are assumed to have a domain within [0, len(rev)[.
        To also handle the "unsafe" variant, have a look at :class:`globalconstraints.Inverse <cpmpy.expressions.globalconstraints.Inverse>`.

    """
    def __init__(self, fwd, rev):
        flatargs = flatlist([fwd, rev])
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("Only integer arguments allowed for global constraint Inverse: {}".format(flatargs))
        if len(fwd) > len(rev):
            raise TypeError("len(fwd) should be equal to len(rev) for the symmetric inverse, or smaller than len(rev) for the asymmetric inverse")
        if len(fwd) == len(rev):
            name = "inverse"
        else:
            name = "inverseAsym"
        super().__init__(name, [fwd, rev])

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

class InverseOne(GlobalConstraint):
    """
       Inverse (aka channeling / assignment) constraint but with only one array.
       Equivalent to Inverse(x,x)
                arr[i] == j  <==>  arr[j] == i
    """
    def __init__(self, arr):
        flatargs = flatlist([arr])
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("Only integer arguments allowed for global constraint Inverse: {}".format(flatargs))
        super().__init__("inverseOne", [arr])

    def decompose(self):
        from cpmpy.expressions.python_builtins import all
        arr = self.args[0]
        arr = cpm_array(arr)
        return [all(arr[x] == i for i, x in enumerate(arr))], []

    def value(self):
        valsx = argvals(self.args[0])
        try:
            return all(valsx[x] == i for i, x in enumerate(valsx))
        except IndexError: # partiality of Element constraint
            return False


class Channel(GlobalConstraint):
    """
        Channeling constraint. Channeling integer representation of a variable into a representation with boolean
        indicators
            for all 0<=i<len(arr) : arr[i] = 1 <=> value = i
            exists 0<=i<len(arr) s.t. arr[i] = 1
    """
    def __init__(self, arr, v):
        flatargs = flatlist([arr])
        if not all(x.lb >= 0 and x.ub <= 1 for x in flatargs):
            raise TypeError(
                "the first argument of a Channel constraint should only contain 0-1 variables/expressions (i.e., " +
                "intvars/intexprs with domain {0,1} or boolvars/boolexprs)")
        super().__init__("channelValue", [arr, v])

    def decompose(self):
        arr, v = self.args
        return [(arr[i] == 1) == (v == i) for i in range(len(arr))] + [v >= 0, v < len(arr)], []

    def value(self):
        arr, v = self.args
        return sum(argvals(x) for x in arr) == 1 and 0 <= argval(v) < len(arr) and arr[argval(v)] == 1


class NonReifiedTable(GlobalConstraint):
    """
    The values of the variables in 'array' correspond to a row in 'table'.
    This global represents the non-reified version, meaning that it does not support occuring reified when decomposing.
    Look at :class:`globalconstraints.Table <cpmpy.expressions.globalconstraints.Table` for a formulation that does support reification.
    """

    def __init__(self, array, table):
        array = flatlist(array)
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError("the first argument of a Table constraint should only contain variables/expressions")
        super().__init__("table", [array, table])

    def decompose(self):
        """
            This decomposition is only valid in a non-reified setting.
        """
        from cpmpy.expressions.python_builtins import any, all
        arr, tab = self.args
        row_selected = boolvar(shape=len(tab))
        if len(tab) == 1:
            return [all(t == a for (t, a) in zip(tab[0], arr))], []
        
        cons = []
        for i, row in enumerate(tab):
            subexpr = Operator("and", [x == v for x,v in zip(arr, row)])
            cons.append(Operator("->", [row_selected[i], subexpr]))

        return [Operator("or", row_selected)]+cons,[]

    def value(self):
        arr, tab = self.args
        arrval = argvals(arr)
        return arrval in tab

    @property
    def vars(self):
        return self._args[0]

    # specialisation to avoid recursing over big tables
    def has_subexpr(self):
        if not hasattr(self, '_has_subexpr'):  # if _has_subexpr has not been computed before or has been reset
            arr, tab = self.args  # the table 'tab' can only hold constants, never a nested expression
            self._has_subexpr = any(a.has_subexpr() for a in arr)
        return self._has_subexpr

class RowSelectingShortTable(GlobalConstraint):
    """
        Extension of the `Table` constraint where the `table` matrix may contain wildcards (STAR), meaning there are
        no restrictions for the corresponding variable in that tuple.
        This global skips typechecks on the table and has a different decomposition than the standard 
        :class:`globalconstraints.ShortTable <cpmpy.expressions.globalconstraints.ShortTable>`.
    """
    def __init__(self, array, table):
        array = flatlist(array)
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError("The first argument of a Table constraint should only contain variables/expressions")
        if isinstance(table, np.ndarray): # Ensure it is a list
            table = table.tolist()
        super().__init__("short_table", [array, table])

    def decompose(self):
        """
        Alternative decomposition, similar to `element` from Gleb's paper: "Improved Linearization of Constraint
        Programming Models"
        """
        from cpmpy.expressions.python_builtins import any, all

        arr, tab = self.args
        row_selected = boolvar(shape=(len(tab),))

        cons = []
        for i, row in enumerate(tab):
            subexpr = Operator("and", [ai == ri for ai, ri in zip(arr, row) if ri != STAR])
            cons.append(row_selected[i].implies(subexpr))

        return [any(row_selected)]+cons,[]

    def value(self):
        arr, tab = self.args
        tab = np.array(tab)
        arrval = np.array(argvals(arr))
        for row in tab:
            num_row = row[row != STAR].astype(int)
            num_vals = arrval[row != STAR].astype(int)
            if (num_row == num_vals).all():
                return True
        return False

class NegativeShortTable(GlobalConstraint):
    """The values of the variables in 'array' do not correspond to any row in 'table'
    """
    def __init__(self, array, table):
        array = flatlist(array)
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError("the first argument of a Table constraint should only contain variables/expressions")
        super().__init__("negative_shorttable", [array, table])

    def decompose(self):
        from cpmpy.expressions.python_builtins import all as cpm_all
        from cpmpy.expressions.python_builtins import any as cpm_any
        arr, tab = self.args
        return [cpm_all(cpm_any(ai != ri for ai, ri in zip(arr, row) if ri != "*") for row in tab)], []

    def value(self):
        arr, tab = self.args
        arrval = [argval(a) for a in arr]
        for tup in tab:
            thistup = True
            for aval, tval in zip(arrval, tup):
                if tval != '*':
                    if aval != tval:
                        thistup = False
                        break
            if thistup:
                # found tuple that matches
                return False
        # didn't find tuple that matches
        return True


class MDD(GlobalConstraint):
    """
    MDD-constraint: an MDD (Multi-valued Decision Diagram) is an acyclic layerd graph starting from a single node and
    ending in one. Each edge layer corresponds to a variables and each path corresponds to a solution
    The values of the variables in 'array' correspond to a path in the mdd formed by the transitions in 'transitions'.
    Root node is the first node used as a start in the first transition (i.e. transitions[0][0])

    Arguments:
        array: an array of CPMpy expressions (integer variable, global functions,...)
        transitions: an array of tuples (nodeID, int, nodeID) where nodeID is some unique identifiers for the nodes
            (int or str are fine)
    
    Example:
        The following transitions depict a 3 layer MDD, starting at 'r' and ending in 't'
        ("r", 0, "n1"), ("r", 1, "n2"), ("r", 2, "n3"), ("n1", 2, "n4"), ("n2", 2, "n4"), ("n3", 0, "n5"),
        ("n4", 0, "t"), ("n5", 1, "t")
        Its graphical representation is:

        .. code-block:: text
            
                  r
              0/ |1  \\2     X
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
            defining = [NonReifiedTable([arr[0], aux[0]], tab_first)] \
                   + [NonReifiedTable([aux[i - 1], arr[i], aux[i]], transitions_by_level_normalized[i]) for i in
                      range(1, len(arr))]
            # constraining constraint: end of the path in accepting node
            constraining = [aux[-1] == 0]
            return constraining, defining
        elif len(transitions_by_level_normalized) == 2:
            # decomposition by unfolding into a table (i.e., extract all paths and list them as table entries),
            # avoid auxiliary variables
            tab = [[t_a[1], t_b[1]] for t_a in transitions_by_level_normalized[0] for t_b in
                   transitions_by_level_normalized[1] if t_a[2] == t_b[0]]
            return [NonReifiedTable(arr, tab)], []

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
    Regular-constraint (or Automaton-constraint)
    Takes as input a sequence of variables and a automaton representation using a transition table.
    The constraint is satisfied if the sequence of variables corresponds to an accepting path in the automaton.

    The automaton is defined by a list of transitions, a starting node and a list of accepting nodes.
    The transitions are represented as a list of tuples, where each tuple is of the form (id1, value, id2).
    An id is an integer or string representing a state in the automaton, and value is an integer representing the value of the variable in the sequence.
    The starting node is an integer or string representing the starting state of the automaton.
    The accepting nodes are a list of integers or strings representing the accepting states of the automaton.

    Example: 
        an automaton that accepts the language 0*10* (exactly 1 variable taking value 1) is defined as:

        .. code-block:: python

            cp.Regular(array = cp.intvar(0,1, shape=4),
                    transitions = [("A",0,"A"), ("A",1,"B"), ("B",0,"C"), ("C",0,"C")],
                    start = "A",
                    accepting = ["C"])
    """
    def __init__(self, array, transitions, start, accepting):
        array = flatlist(array)
        # skip all typechecks for comp
        # if not all(isinstance(x, Expression) for x in array):
        #     raise TypeError("The first argument of a regular constraint should only contain variables/expressions")
        
        # if not is_any_list(transitions):
        #     raise TypeError("The second argument of a regular constraint should be a list of transitions")
        # _node_type = type(transitions[0][0])
        # for s,v,e in transitions:
        #     if not isinstance(s, _node_type) or not isinstance(e, _node_type) or not isinstance(v, int):
        #         raise TypeError(f"The second argument of a regular constraint should be a list of transitions ({_node_type}, int, {_node_type})")
        # if not isinstance(start, _node_type):
        #     raise TypeError("The third argument of a regular constraint should be a node id")
        # if not (is_any_list(accepting) and all(isinstance(e, _node_type) for e in accepting)):
        #     raise TypeError("The fourth argument of a regular constraint should be a list of node ids")
        super().__init__("regular", [array, transitions, start, list(accepting)])

        self.nodes = set()
        self.trans_dict = {}
        for s, v, e in transitions:
            self.nodes.update([s,e])
            self.trans_dict[(s, v)] = e
        self.nodes = sorted(self.nodes)
        # normalize node_ids to be 0..n-1, allows for smaller domains
        self.node_map = {n: i for i, n in enumerate(self.nodes)}

    def decompose(self):
        # Decompose to transition table using Table constraints
        
        arr, transitions, start, accepting = self.args
        lbs, ubs = get_bounds(arr)
        lb, ub = min(lbs), max(ubs)
        
        transitions = [[self.node_map[n_in], v, self.node_map[n_out]] for n_in, v, n_out in transitions]

        # add a sink node for transitions that are not defined
        # --> not necessary for comp, because positive context
        # sink = len(self.nodes)
        # transitions += [[self.node_map[n], v, sink] for n in self.nodes for v in range(lb, ub + 1) if (n, v) not in self.trans_dict]
        # transitions += [[sink, v, sink] for v in range(lb, ub + 1)]

        # keep track of current state when traversing the array
        state_vars = intvar(0, len(self.nodes)-1, shape=len(arr))
        id_start = self.node_map[start]
        # optimization: we know the entry node of the automaton, results in smaller table
        defining = [NonReifiedTable([arr[0], state_vars[0]], [[v,e] for s,v,e in transitions if s == id_start])]        
        # define the rest of the automaton using transition table
        defining += [NonReifiedTable([state_vars[i - 1], arr[i], state_vars[i]], transitions) for i in range(1, len(arr))]
        
        # constraint is satisfied iff last state is accepting
        return [InDomain(state_vars[-1], [self.node_map[e] for e in accepting])], defining

    def value(self):
        arr, transitions, start, accepting = self.args
        arrval = [argval(a) for a in arr]
        curr_node = start
        for v in arrval:
            if (curr_node, v) in self.trans_dict:
                curr_node = self.trans_dict[curr_node, v]
            else:
                return False
        return curr_node in accepting



class NotInDomain(GlobalConstraint):
    """
        The "NotInDomain" constraint, defining non-interval domains for an expression
    """

    def __init__(self, expr, arr):
        super().__init__("NotInDomain", [expr, arr])

    def decompose(self):
        """
        This decomp only works in positive context
        """
        from cpmpy.expressions.python_builtins import any, all
        expr, arr = self.args
        lb, ub = expr.get_bounds()

        defining = []
        #if expr is not a var
        if not isinstance(expr, _IntVarImpl):
            aux = intvar(lb, ub)
            defining.append(aux == expr)
            expr = aux

        if not any(isinstance(a, Expression) for a in arr):
            given = len(set(arr))
            missing = ub + 1 - lb - given
            if missing < 2 * given:  # != leads to double the amount of constraints
                # use == if there is less than twice as many gaps in the domain.
                row_selected = boolvar(shape=missing)
                return [any(row_selected)] + [rs.implies(expr == val) for val,rs in zip(range(lb, ub + 1), row_selected) if val not in arr], defining
        return [all([(expr != a) for a in arr])], defining


    def value(self):
        return argval(self.args[0]) not in argvals(self.args[1])

    def __repr__(self):
        return "{} not in {}".format(self.args[0], self.args[1])


class NoOverlap2d(GlobalConstraint):
    """
        2D-version of the NoOverlap constraint.
        Ensures a set of rectangles is placed on a grid such that they do not overlap.
    """
    def __init__(self, start_x, dur_x, end_x,  start_y, dur_y, end_y):
        assert len(start_x) == len(dur_x) == len(end_x) == len(start_y) == len(dur_y) == len(end_y)
        super().__init__("no_overlap2d", [start_x, dur_x, end_x,  start_y, dur_y, end_y])

    def decompose(self):
        from cpmpy.expressions.python_builtins import any as cpm_any

        start_x, dur_x, end_x,  start_y, dur_y, end_y = self.args
        n = len(start_x)
        cons =  [s + d == e for s,d,e in zip(start_x, dur_x, end_x)]
        cons += [s + d == e for s,d,e in zip(start_y, dur_y, end_y)]

        for i,j in all_pairs(list(range(n))):
            cons += [cpm_any([end_x[i] <= start_x[j], end_x[j] <= start_x[i],
                              end_y[i] <= start_y[j], end_y[j] <= start_y[i]])]
        return cons,[]
    def value(self):
        start_x, dur_x, end_x,  start_y, dur_y, end_y = argvals(self.args)
        n = len(start_x)
        if any(s + d != e for s, d, e in zip(start_x, dur_x, end_x)):
            return False
        if any(s + d != e for s, d, e in zip(start_y, dur_y, end_y)):
            return False
        for i,j in all_pairs(list(range(n))):
            if  end_x[i] > start_x[j] and end_x[j] > start_x[i] and \
                 end_y[i] > start_y[j] and end_y[j] > start_y[i]:
                return False
        return True


class IfThenElseNum(GlobalFunction):
    """
        Function returning x if b is True and otherwise y
    """
    def __init__(self, b, x,y):
        super().__init__("IfThenElseNum",[b,x,y])

    def decompose_comparison(self, cmp_op, cpm_rhs):
        b,x,y = self.args

        lbx,ubx = get_bounds(x)
        lby,uby = get_bounds(y)
        iv = intvar(min(lbx,lby), max(ubx,uby))
        defining = [b.implies(x == iv), (~b).implies(y == iv)]

        return [eval_comparison(cmp_op, iv, cpm_rhs)], defining

    def get_bounds(self):
        b,x,y = self.args
        lbs,ubs = get_bounds([x,y])
        return min(lbs), max(ubs)
    def value(self):
        b,x,y = self.args
        if argval(b):
            return argval(x)
        else:
            return argval(y)


class DynamicCumulative(GlobalConstraint):
    """
        Global cumulative constraint. Used for resource aware scheduling. Ensures that the capacity of the resource is never exceeded.
        When decomposed, dynamically switches between time-resource and task-resource decompositions depending on size of domains.
        Equivalent to :class:`~cpmpy.expressions.globalconstraints.NoOverlap` when demand and capacity are equal to 1.
        Supports both varying demand across tasks or equal demand for all jobs.
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

        super(DynamicCumulative, self).__init__("cumulative", [start, duration, end, demand, capacity])

    def decompose(self):
        """
            Decomposition from:
            Schutt, Andreas, et al. "Why cumulative decomposition is not as bad as it sounds."
            International Conference on Principles and Practice of Constraint Programming. Springer, Berlin, Heidelberg, 2009.

            Heuristically switches between time-resource and task-resource decomposition depending on the relative size of the time horizon and the number of tasks.
            
            If 
                n = number of tasks
                t = size of time horizon
            then    
                time-resource decomposition scales with n*t 
                task-resource decomposition scales with 3n(n-1)
            thus
                switch when t > 3*n
        """

        arr_args = (cpm_array(arg) if is_any_list(arg) else arg for arg in self.args)
        start, duration, end, demand, capacity = arr_args

        num_tasks = len(demand) # number of tasks
        lb, ub = min(get_bounds(start)[0]), max(get_bounds(end)[1])
        time_horizon =  ub - lb

        cons = []

        if time_horizon > 3 * num_tasks:
            version = "task"
        else:
            version = "time"

        if version == "time":
            # set duration of tasks
            for t in range(len(start)):
                cons += [start[t] + duration[t] == end[t]]

            # demand doesn't exceed capacity
            for t in range(lb,ub+1):
                demand_at_t = 0
                for job in range(len(start)):
                    if is_num(demand):
                        demand_at_t += demand * ((start[job] <= t) & (t < end[job]))
                    else:
                        demand_at_t += demand[job] * ((start[job] <= t) & (t < end[job]))

                cons += [demand_at_t <= capacity]
                
        elif version == "task":

            # set duration of tasks
            for t in range(num_tasks):
                cons += [start[t] + duration[t] == end[t]]

            for j in range(num_tasks):
                cons += [capacity >= demand[j] + cp.sum([(start[i] <= start[j]) & (start[j] < start[i] + duration[i]) for i in range(num_tasks) if i != j])]

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


# helper function
def is_transition(arg):
    """ test if the argument is a transition, i.e. a 3-elements-tuple specifying a starting state,
    a transition value and an ending node"""
    return len(arg) == 3 and \
        isinstance(arg[0], (int, str)) and is_int(arg[1]) and isinstance(arg[2], (int, str))



# ---------------------------------------------------------------------------- #
#                            Solver-specific Globals                           #
# ---------------------------------------------------------------------------- #


"""
A collection of XCSP3 solver-native global constraints.
"""


# --------------------------------- OR-Tools --------------------------------- #

class OrtNoOverlap2D(DirectConstraint):
    """
    OR-Tools native `NoOverlap2D` global constraint.

    Ensures that all provided rectangles are positioned within a plane whilst not overlapping.
    The rectangles have their sides aligned with the perpendicular x- and y-axis.
    """
    def __init__(self, arguments):
        super().__init__("ortnooverlap2d", arguments)

    def callSolver(self, CPMpy_solver, Native_solver):
        start_x, dur_x, end_x, start_y, dur_y, end_y = CPMpy_solver.solver_vars(self.args[0])
        intervals_x = [Native_solver.NewIntervalVar(s,d,e, f"xinterval_{s}-{d}-{d}") for s,d,e in zip(start_x,dur_x,end_x)]
        intervals_y = [Native_solver.NewIntervalVar(s,d,e, f"yinterval_{s}-{d}-{d}") for s,d,e in zip(start_y,dur_y,end_y)]
        return Native_solver.add_no_overlap_2d(intervals_x, intervals_y)

class OrtSubcircuit(DirectConstraint):
    """
    OR-Tools native `SubCircuit` global constraint.

    A subcircuit is a Hamiltonian path between a subset of the nodes of a graph.
    When a node `i` is not part of the circuit, it should self-loop with an arc `i -> i`.
    """
    def __init__(self, arguments):
        super().__init__("ortsubcircuit", arguments)

    def callSolver(self, CPMpy_solver, Native_solver):
        N = len(self.args[0])
        arcvars = cp.boolvar(shape=(N,N))
        # post channeling constraints from int to bool
        CPMpy_solver.add([b == (self.args[0][i] == j) for (i,j),b in np.ndenumerate(arcvars)])
        # post the global constraint
        #   posting arcs on diagonal (i==j) allows for subcircuits
        ort_arcs = [(i,j, CPMpy_solver.solver_var(b)) for (i,j),b in np.ndenumerate(arcvars)] # Allows for empty subcircuits

        return Native_solver.AddCircuit(ort_arcs)

class OrtSubcircuitWithStart(DirectConstraint):
    """
    OR-Tools native `SubCircuitWithStart` global constraint.

    This global is the same as `SubCircuit`, only can a additional `start_index` be provided 
    which is ensure to be part of the subcircuit.
    """
    def __init__(self, arguments, start_index:int=0):
        super().__init__("ortsubcircuitwithstart", (arguments, start_index))

    def callSolver(self, CPMpy_solver, Native_solver):
        N = len(self.args[0])
        arcvars = cp.boolvar(shape=(N,N))
        # post channeling constraints from int to bool
        CPMpy_solver.add([b == (self.args[0][i] == j) for (i,j),b in np.ndenumerate(arcvars)])
        # post the global constraint
        # posting arcs on diagonal (i==j) allows for subcircuits
        ort_arcs = [(i,j,CPMpy_solver.solver_var(b)) for (i,j),b in np.ndenumerate(arcvars) if not ((i == j) and (i == self.args[1]))] # The start index cannot self loop and thus must be part of the subcircuit.

        return Native_solver.AddCircuit(ort_arcs)
        

# ----------------------------------- Choco ---------------------------------- #

class ChocoSubcircuit(DirectConstraint):
    """
    Choco's native `SubCircuit` global constraint.

    A subcircuit is a Hamiltonian path between a subset of the nodes of a graph.
    When a node `i` is not part of the circuit, it should self-loop with an arc `i -> i`.
    """
    def __init__(self, arguments):
        super().__init__("chocosubcircuit", arguments)

    def callSolver(self, CPMpy_solver, Native_solver):
        # Successor variables
        succ = CPMpy_solver.solver_vars(self.args[0])
        # Add an unused variable for the subcircuit length.
        subcircuit_length = CPMpy_solver.solver_var(cp.intvar(0, len(succ)))
        return Native_solver.sub_circuit(succ, 0, subcircuit_length)

# --------------------------------- Minizinc --------------------------------- #

class MinizincSubcircuit(DirectConstraint):
    """
    Minizinc's native `SubCircuit` global constraint.

    A subcircuit is a Hamiltonian path between a subset of the nodes of a graph.
    When a node `i` is not part of the circuit, it should self-loop with an arc `i -> i`.
    """
    def __init__(self, arguments):
        super().__init__("minizincsubcircuit", arguments)

    def callSolver(self, CPMpy_solver, Native_solver):
        # minizinc is offset 1, which can be problematic here...
        args_str = ["{}+1".format(CPMpy_solver._convert_expression(e)) for e in self.args[0]]
        return "{}([{}])".format("subcircuit", ",".join(args_str))

class MinizincSubcircuitWithStart(DirectConstraint):
    """
    Minizinc's native `SubCircuitWithStart` global constraint.

    This global is the same as `SubCircuit`, only can a additional `start_index` be provided 
    which is ensure to be part of the subcircuit.
    """
    def __init__(self, arguments):
        super().__init__("minizincsubcircuitwithstart", arguments)

    def callSolver(self, CPMpy_solver, Native_solver):
        # minizinc is offset 1, which can be problematic here...
        start_index = self.args[0][-1]
        succ = cpm_array(self.args[0][:-1]) # Successor variables

        CPMpy_solver += (succ[start_index] != start_index)
        args_str = ["{}+1".format(CPMpy_solver._convert_expression(e)) for e in succ]
        return "{}([{}])".format("subcircuit", ",".join(args_str))
        