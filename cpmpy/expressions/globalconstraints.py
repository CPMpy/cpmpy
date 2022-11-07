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
            return [self.args[0] == 1] # does not actually enforce circuit
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
        AllEqual
        Circuit
        Table
        Minimum
        Maximum
        Element

"""
import warnings # for deprecation warning
from .core import Expression, Operator, Comparison
from .variables import boolvar, intvar, cpm_array
from .utils import flatlist, all_pairs, argval, is_num, eval_comparison, is_any_list
from ..transformations.flatten_model import get_or_make_var

# Base class GlobalConstraint
class GlobalConstraint(Expression):
    """
        Abstract superclass of GlobalConstraints

        Like all expressions it has a `.name` and `.args` property.
        Overwrites the `.is_bool()` method. You can indicate
        in the constructer whether it has Boolean return type or not.
    """
    # is_bool: whether this is normal constraint (True or False)
    #   not is_bool: it computes a numeric value (ex: Minimum, Element)
    def __init__(self, name, arg_list, is_bool=True):
        super().__init__(name, arg_list)
        self._is_bool = is_bool

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return self._is_bool

    def decompose(self):
        """
            Returns a decomposition into smaller constraints.

            The decomposition might create auxiliary variables
            and use other other global constraints as long as
            it does not create a circular dependency.
        """
        raise NotImplementedError("Decomposition for",self,"not available")

    def deepcopy(self, memodict={}):
        copied_args = self._deepcopy_args(memodict)
        return type(self)(self.name, copied_args, self._is_bool)


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
        return [var1 != var2 for var1, var2 in all_pairs(self.args)]

    def deepcopy(self, memodict={}):
        """
            Return a deep copy of the Alldifferent global constraint
            :param: memodict: dictionary with already copied objects, similar to copy.deepcopy()
        """
        copied_args = self._deepcopy_args(memodict)
        return AllDifferent(*copied_args)

    def value(self):
        return len(set(a.value() for a in self.args)) == len(self.args)

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
        return [var1 == var2 for var1, var2 in all_pairs(self.args)]

    def deepcopy(self, memdict={}):
        """
            Return a deep copy of the AllEqual global constraint
            :param: memodict: dictionary with already copied objects, similar to copy.deepcopy()
        """
        copied_args = self._deepcopy_args(memdict)
        return AllEqual(*copied_args)

    def value(self):
        return len(set(a.value() for a in self.args)) == 1


def circuit(args):
    warnings.warn("Deprecated, use Circuit(v1,v2,...,vn) instead, will be removed in stable version", DeprecationWarning)
    return Circuit(*args) # unfold list as individual arguments
class Circuit(GlobalConstraint):
    """The sequence of variables form a circuit, where x[i] = j means that j is the successor of i.
    """
    def __init__(self, *args):
        super().__init__("circuit", flatlist(args))

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
        return [
            # different successors
            AllDifferent(succ),
            # different orders
            AllDifferent(order),
            # last one is '0'
            order[n-1] == 0,
            # loop: first one is successor of '0'
            order[0] == succ[0],
            # others: ith one is successor of i-1
        ] + [order[i] == succ[order[i-1]] for i in range(1,n)]


    def deepcopy(self, memdict={}):
        """
            Return a deep copy of the Circuit global constraint
           :param: memodict: dictionary with already copied objects, similar to copy.deepcopy()
        """
        copied_args = self._deepcopy_args(memdict)
        return Circuit(*copied_args)


    # TODO: value()


class Table(GlobalConstraint):
    """The values of the variables in 'array' correspond to a row in 'table'
    """
    def __init__(self, array, table):
        super().__init__("table", [array, table])

    def decompose(self):
        raise NotImplementedError("TODO: table decomposition")


    def deepcopy(self, memodict={}):
        """
            Return a deep copy of the Table global constraint
            :param: memodict: dictionary with already copied objects, similar to copy.deepcopy()
        """
        array, table = self._deepcopy_args(memodict)
        return Table(array, table)

    # TODO: value()

# Numeric Global Constraints (with integer-valued return type)


class Minimum(GlobalConstraint):
    """
        Computes the minimum value of the arguments

        It is a 'functional' global constraint which implicitly returns a numeric variable
    """
    def __init__(self, arg_list):
        super().__init__("min", flatlist(arg_list), is_bool=False)

    def value(self):
        argvals = [argval(a) for a in self.args]
        if any(val is None for val in argvals):
            return None
        else:
            return min(argvals)

    def deepcopy(self, memodict={}):
        """
            Return a deep copy of the Minimum global constraint
            :param: memodict: dictionary with already copied objects, similar to copy.deepcopy()
        """
        copied_args = self._deepcopy_args(self.args)
        return Minimum(copied_args)

class Maximum(GlobalConstraint):
    """
        Computes the maximum value of the arguments

        It is a 'functional' global constraint which implicitly returns a numeric variable
    """
    def __init__(self, arg_list):
        super().__init__("max", flatlist(arg_list), is_bool=False)

    def value(self):
        argvals = [argval(a) for a in self.args]
        if any(val is None for val in argvals):
            return None
        else:
            return max(argvals)

    def deepcopy(self, memodict={}):
        """
            Return a deep copy of the Maximum global constraint
            :param: memodict: dictionary with already copied objects, similar to copy.deepcopy()
        """
        copied_args = self._deepcopy_args(memodict)
        return Maximum(copied_args)

def element(arg_list):
    warnings.warn("Deprecated, use Circuit(v1,v2,...,vn) instead, will be removed in stable version", DeprecationWarning)
    assert (len(arg_list) == 2), "Element expression takes 2 arguments: Arr, Idx"
    return Element(arg_list[0], arg_list[1])
class Element(GlobalConstraint):
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
        super().__init__("element", [arr, idx], is_bool=False)

    def value(self):
        arr, idx = self.args
        idxval = argval(idx)
        if idxval is not None:
            return argval(arr[idxval])
        return None # default

    def decompose_comparison(self, cmp_op, cmp_rhs):
        """
            `Element(arr,ix)` represents the array lookup itself (a numeric variable)
            It is not a constraint itself, so it can not have a decompose().
            However, when used in a comparison relation: Element(arr,idx) <CMP_OP> CMP_RHS
            it is a constraint, and that one can be decomposed.
            That is what this function does
            (for now only used in transformations/reification.py)
        """
        from .python_builtins import any

        arr,idx = self.args
        return [any(eval_comparison(cmp_op, arr[j], cmp_rhs) & (idx == j) for j in range(len(arr)))]

    def __repr__(self):
        return "{}[{}]".format(self.args[0], self.args[1])

    def deepcopy(self, memodict={}):
        """
            Return a deep copy of the Element global constraint
            :param: memodict: dictionary with already copied objects, similar to copy.deepcopy()
        """
        arr, idx = self._deepcopy_args(memodict)
        return Element(arr, idx)


class Xor(GlobalConstraint):
    """
        The 'xor' constraint for more then 2 arguments.
        Acts like cascaded xor operators with two inputs
    """

    def __init__(self, arg_list):
        # convention for commutative binary operators:
        # swap if right is constant and left is not
        if len(arg_list) == 2 and is_num(arg_list[1]):
            arg_list[0], arg_list[1] = arg_list[1], arg_list[0]
        i = 0  # length can change
        while i < len(arg_list):
            if isinstance(arg_list[i], Xor):
                # merge args in at this position
                arg_list[i:i + 1] = arg_list[i].args
            else:
                i += 1
        super().__init__("xor", arg_list)

    def decompose(self):
        if len(self.args) == 2:
            return [(self.args[0] + self.args[1]) == 1]
        prev_var, cons = get_or_make_var(self.args[0] ^ self.args[1])
        for arg in self.args[2:]:
            prev_var, new_cons = get_or_make_var(prev_var ^ arg)
            cons += new_cons
        return cons + [prev_var]

    def value(self):
        return sum(argval(a) for a in self.args) % 2 == 1

    def __repr__(self):
        if len(self.args) == 2:
            return "{} xor {}".format(*self.args)
        return "xor({})".format(self.args)

    def deepcopy(self, memodict={}):
        """
           Return a deep copy of the xor global constraint
           :param: memodict: dictionary with already copied objects, similar to copy.deepcopy()
       """
        copied_args = self._deepcopy_args(memodict)
        return Xor(copied_args)

class Cumulative(GlobalConstraint):
    """
        Global cumulative constraint. Used for resource aware scheduling.
        Ensures no overlap between tasks and never exceeding the capacity of the resource
        Supports both varying demand across tasks or equal demand for all jobs
    """
    def __init__(self, start, duration, end, demand, capacity):
        super(Cumulative, self).__init__("cumulative",[start,
                                                       duration,
                                                       end,
                                                       demand,
                                                       capacity])

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
            cons += [capacity >= demand_at_t]

        return cons

    def value(self):
        start, dur, end, demand, cap = [argval(a) for a in self.args]
        # start and end seperated by duration
        if not (start + dur == end).all():
            return False

        # demand doesn't exceed capacity
        lb, ub = min(start), max(end)
        for t in range(lb, ub+1):
            if cap < sum(demand * ((start <= t) & (t < end))):
                return False

        return True




