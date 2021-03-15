#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## globalconstraints.py
##
"""
    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        alldifferent
        allequal
        circuit
        GlobalConstraint
        Element

    ================
    List of functions
    =================

    .. autosummary::
        :nosignatures:

        min
        max

    ==================
    Module description
    ==================

    Global constraint definitions

    A global constraint is nothing special in CPMpy. It is just an
    expression of type `GlobalConstraint` with a name and arguments.


    You can define a new global constraint as simply as:

    .. code-block:: python

        def my_global(args):
            return GlobalConstraint("my_global", args)


    Of course, solvers may not support a global constraint
    (if it does, it should be mapped to the API call in its SolverInterface)

    You can provide a decomposition for your global constraint through
    the decompose() function.
    To overwrite it, you should define your global constraint as a
    subclass of GlobalConstraint, rather then as a function above.

    Your decomposition function can use any standard CPMpy expression.
    For example:

    .. code-block:: python

        class my_global(GlobalConstraint):
            def __init__(self, args):
                super().__init__("my_global", args)

            def decompose(self):
                return [self.args[0] != self.args[1]] # your decomposition

    If you are modeling a problem and you want to use another decomposition,
    simply overwrite the 'decompose' function of the class, e.g.:

    .. code-block:: python

        def my_circuit_decomp(self):
            return [self.args[0] == 1] # does not actually enforce circuit
        circuit.decompose = my_circuit_decomp # attach it, no brackets!

        vars = IntVars(1,9, shape=(10,))
        constr = circuit(vars)

        Model(constr).solve()

    The above will use 'my_circuit_decomp', if the solver does not
    natively support 'circuit'.
"""
from .variables import *
from .expressions import *
from itertools import chain, combinations


class GlobalConstraint(Expression):
    # is_bool: whether this is normal constraint (True or False)
    #   not is_bool: it computes a numeric value (ex: Element)
    def __init__(self, name, arg_list, is_bool=True):
        super().__init__(name, arg_list)
        self._is_bool = is_bool

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return self._is_bool

    def decompose(self):
        """
            if a global constraint has a default decomposition,
            then it should monkey-patch this function, e.g.:
            def my_decomp_function(self):
                return []
            g = GlobalConstraint("g", args)
            g.decompose = my_decom_function
        """
        return None


# min: listwise 'min'
def min(iterable):
    """
        min() overwrites python built-in,
        checks if all constants and computes np.min() in that case
    """
    if not any(isinstance(elem, Expression) for elem in iterable):
        return np.min(iterable)
    return Minimum(iterable)

def max(iterable):
    """
        max() overwrites python built-in,
        checks if all constants and computes np.map() in that case
    """
    if not any(isinstance(elem, Expression) for elem in iterable):
        return np.max(iterable)
    return Maximum(iterable)



class alldifferent(GlobalConstraint):
    """
    all arguments have a different (distinct) value
    """
    def __init__(self, args):
        super().__init__("alldifferent", args)

    def decompose(self):
        return [var1 != var2 for var1, var2 in _all_pairs(self.args)]


class allequal(GlobalConstraint):
    """
    all arguments have the same value
    """
    def __init__(self, args):
        super().__init__("allequal", args)

    def decompose(self):
        return [var1 == var2 for var1, var2 in _all_pairs(self.args)]


class circuit(GlobalConstraint):
    """
    Variables of the constraint form a circuit ex: 0 -> 3 -> 2 -> 0
    """
    def __init__(self, args):
        super().__init__("circuit", args)

    def decompose(self):
        """
            TODO needs explanation/reference
        """
        n = len(self.args)
        a = self.args
        z = IntVar(0, n-1, n)
        constraints = [alldifferent(z),
                       alldifferent(a),
                       z[0]==a[0],
                       z[n-1]==0]
        for i in range(1,n-1):
            constraints += [z[i] != 0,
                            z[i] == a[z[i-1]]]
        return constraints


class Minimum(GlobalConstraint):
    """
        Computes the minimum value of the arguments

        It is a 'functional' global constraint which implicitly returns a numeric variable
    """
    def __init__(self, arg_list):
        super().__init__("min", arg_list, is_bool=False)

    def value(self):
        return min([_argval(a) for a in self.args])

class Maximum(GlobalConstraint):
    """
        Computes the maximum value of the arguments

        It is a 'functional' global constraint which implicitly returns a numeric variable
    """
    def __init__(self, arg_list):
        super().__init__("max", arg_list, is_bool=False)

    def value(self):
        return max([_argval(a) for a in self.args])

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

    def __init__(self, arg_list):
        assert (len(arg_list) == 2), "Element expression takes 2 arguments: Arr, Idx"
        super().__init__("element", arg_list, is_bool=False)

    def value(self):
        idxval = _argval(self.args[1])
        if not idxval is None:
            return _argval(self.args[0][idxval])
        return None # default

    def __repr__(self):
        return "{}[{}]".format(self.args[0], self.args[1])



def _all_pairs(args):
    """ internal helper function
    """
    pairs = list(combinations(args, 2))
    return pairs

# XXX, make argval shared util function?
def _argval(a):
    """ returns .value() of Expression, otherwise the variable itself
    """
    return a.value() if isinstance(a, Expression) else a
