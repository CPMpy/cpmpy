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


import cpmpy as cp

from .core import BoolVal
from .utils import all_pairs, is_int, is_bool, STAR
from .variables import _IntVarImpl
from .globalfunctions import * # XXX make this file backwards compatible


# Base class GlobalConstraint
class GlobalConstraint(Expression):
    """
        Abstract superclass of GlobalConstraints

        Like all expressions it has a ``.name`` and ``.args`` property.
        Overwrites the ``.is_bool()`` method.
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
    """
    .. deprecated:: 0.9.0
          Please use :class:`AllDifferent` instead.
    """
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
        cons = []
        arr, n = self.args
        for x,y in all_pairs(arr):
            cond = x == y
            if is_bool(cond):
                cond = cp.BoolVal(cond)
            cons.append(cond.implies(cp.any(x == a for a in n))) # equivalent to (var1 in n) | (var2 in n) | (var1 != var2)
        return cons, []

    def value(self):
        vals = [argval(a) for a in self.args[0] if argval(a) not in argvals(self.args[1])]
        return len(set(vals)) == len(vals)


class AllDifferentExcept0(AllDifferentExceptN):
    """
        All nonzero arguments have a distinct value
    """
    def __init__(self, *arr):
        super().__init__(arr, 0)


def allequal(args):
    """
    .. deprecated:: 0.9.0
          Please use :class:`AllEqual` instead.
    """
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
    """
    .. deprecated:: 0.9.0
          Please use :class:`Circuit` instead.
    """
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
        super().__init__("circuit", flatargs)
        if len(flatargs) < 2:
            raise CPMpyException('Circuit constraint must be given a minimum of 2 variables')

    def decompose(self):
        """
            Decomposition for Circuit

            ..
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

class ShortTable(GlobalConstraint):
    """
        Extension of the `Table` constraint where the `table` matrix may contain wildcards (STAR), meaning there are
        no restrictions for the corresponding variable in that tuple.
    """
    def __init__(self, array, table):
        array = flatlist(array)
        if not all(isinstance(x, Expression) for x in array):
            raise TypeError("The first argument of a Table constraint should only contain variables/expressions")
        if not all(is_int(x) or x == STAR for row in table for x in row):
            raise TypeError(f"elements in argument `table` should be integer or {STAR}")
        if isinstance(table, np.ndarray): # Ensure it is a list
            table = table.tolist()
        super().__init__("short_table", [array, table])

    def decompose(self):
        arr, tab = self.args
        return [cp.any(cp.all(ai == ri for ai, ri in zip(arr, row) if ri != STAR) for row in tab)], []

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
    def __init__(self, array, transitions, start, accepting):
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

# syntax of the form 'if b then x == 9 else x == 0' is not supported (no override possible)
# same semantic as CPLEX IfThenElse constraint
# https://www.ibm.com/docs/en/icos/12.9.0?topic=methods-ifthenelse-method
class IfThenElse(GlobalConstraint):
    """
        The IfThenElse constraint, defining a conditional expression
        of the form: if condition then if_true else if_false
        where condition, if_true and if_false are all boolean expressions.
    """
    def __init__(self, condition, if_true, if_false):
        if not is_boolexpr(condition) or not is_boolexpr(if_true) or not is_boolexpr(if_false):
            raise TypeError(f"only boolean expression allowed in IfThenElse: Instead got "
                            f"{condition, if_true, if_false}")
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
        if is_bool(condition):
            condition = cp.BoolVal(condition) # ensure it is a CPMpy expression
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
        lb, ub = get_bounds(expr)
        
        defining = []
        #if expr is not a var
        if not isinstance(expr,Expression):
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
        The :class:`Xor` exclusive-or constraint
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
        Ensures that the capacity of the resource is never exceeded.
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
        super().__init__("precedence", [cpm_array(vars), precedence])

    def decompose(self):
        """
        Decomposition based on:
        Law, Yat Chiu, and Jimmy HM Lee. "Global constraints for integer and set value precedence."
        Principles and Practice of Constraint Programming–CP 2004: 10th International Conference, CP 2004
        """

        args, precedence = self.args
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
    """
    NoOverlap constraint, enforcing that the intervals defined by start, duration and end do not overlap.
    """

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
    The number of occurrences of each value `vals[i]` in the list of variables `vars`
    must be equal to `occ[i]`.
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

        if len(X) == 0 == len(Y):
            return [cp.BoolVal(False)], [] # based on the decomp, it's false...

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

        if len(X) == 0 == len(Y):
            return [cp.BoolVal(False)], [] # based on the decomp, it's false...

        bvar = boolvar(shape=(len(X) + 1))
        defining = [bvar == ((X <= Y) & ((X < Y) | bvar[1:]))]
        defining.append(bvar[-1] == (X[-1] <= Y[-1]))
        constraining = [bvar[0]]

        return constraining, defining

    def value(self):
        X, Y = argvals(self.args)
        return any((X[i] < Y[i]) & all(X[j] <= Y[j] for j in range(i)) for i in range(len(X))) | all(X[i] == Y[i] for i in range(len(X)))


class LexChainLess(GlobalConstraint):
    """ Given a matrix X, :class:`LexChainLess` enforces that all rows are lexicographically ordered.
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
        A ``DirectConstraint`` will directly call a function of the underlying solver when added to a CPMpy solver

        It can not be reified, it is not flattened, it can not contain other CPMpy expressions than variables.
        When added to a CPMpy solver, it will literally just directly call a function on the underlying solver,
        replacing CPMpy variables by solver variables along the way.

        See the documentation of the solver (constructor) for details on how that solver handles them.

        If you want/need to use what the solver returns (e.g. an identifier for use in other constraints),
        then use :func:`~cpmpy.expressions.variables.directvar` instead, or access the solver object from the solver interface directly.
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

