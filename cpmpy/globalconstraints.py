from .variables import *
from .expressions import *
from itertools import chain, combinations

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

    ```
    def my_global(args):
        return GlobalConstraint("my_global", args)
    ```


    Of course, solvers may not support a global constraint
    (if it does, it should be mapped to the API call in its SolverInterface)

    You can provide a decomposition for your global constraint through
    the decompose() function.
    To overwrite it, you should define your global constraint as a
    subclass of GlobalConstraint, rather then as a function above.
    
    Your decomposition function can use any standard CPMpy expression.
    For example:

    ```
    class my_global(GlobalConstraint):
        def __init__(self, args):
            super().__init__("my_global", args)
    
        def decompose(self):
            return [self.args[0] != self.args[1]] # your decomposition
    ```


    If you are modeling a problem and you want to use another decomposition,
    simply overwrite the 'decompose' function of the class, e.g.:

    ```
    def my_circuit_decomp(self):
        return [self.args[0] == 1] # does not actually enforce circuit
    circuit.decompose = my_circuit_decomp # attach it, no brackets!

    vars = IntVars(1,9, shape=(10,))
    constr = circuit(vars)

    Model(constr).solve()
    ```

    The above will use 'my_circuit_decomp', if the solver does not
    natively support 'circuit'.
"""


# min: listwise 'min'
def min(iterable):
    """
        min() overwrites python built-in,
        checks if all constants and computes np.min() in that case
    """
    if not any(isinstance(elem, Expression) for elem in iterable):
        return np.min(iterable)
    return GlobalConstraint("min", list(iterable))

def max(iterable):
    """
        max() overwrites python built-in,
        checks if all constants and computes np.map() in that case
    """
    if not any(isinstance(elem, Expression) for elem in iterable):
        return np.max(iterable)
    return GlobalConstraint("max", list(iterable))



class alldifferent(GlobalConstraint):
    """ all arguments have a different (distinct) value
    """
    def __init__(self, args):
        super().__init__("alldifferent", args)
    
    def decompose(self):
        return [var1 != var2 for var1, var2 in _all_pairs(self.args)]


class allequal(GlobalConstraint):
    """ all arguments have the same value
    """
    def __init__(self, args):
        super().__init__("allequal", args)
    
    def decompose(self):
        return [var1 == var2 for var1, var2 in _all_pairs(self.args)]


class circuit(GlobalConstraint):
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
                       z[n-1]==a[0]]
        for i in range(1,n-1):
            constraints += [z[i] != 0,
                            z[i] == a[z[i-1]]]
        return constraints


def _all_pairs(args):
    """ internal helper function
    """
    pairs = list(combinations(args, 2))
    return pairs

