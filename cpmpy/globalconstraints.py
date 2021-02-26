from .expressions import *
import types # for overloading decompose()

# in one file for easy overview, does not include interpretation
# TODO: docstrings, generic decomposition method
"""
    Global constraint definitions

    A global constraint is nothing special in CPMpy. It is just an
    expression of type `GlobalConstraint` with a name and arguments.

    You can define a new global constraint as simply as:

    ```
    def my_global(vars):
        return GlobalConstraint("my_global", vars)
    ```


    Of course, solvers may not support a global constraint
    (if it does, it should be mapped to the API call in its SolverInterface)

    You can provide a decomposition for your global constraint through
    a separate function. This function should then be 'monkey patched'
    to the object, by overwriting the 'decompose' function.

    Your decomposition function can use any standard CPMpy expression.
    For example:

    ```
    def my_global_decomposition(self):
        return any(self.args)
    ```

    Attaching it to the object (monkey patching) should be done just
    after creating the GlobalConstraint object, e.g.:
    
    ```
    def my_global(vars):
        expr = GlobalConstraint("my_global", vars)
        expr.decompose = my_global_decomposition # no brackets!
        return expr
    ```


    If you are modeling a problem and you want to use another decomposition,
    simply overwrite the 'decompose' as above, e.g.:

    ```
    import types
    def my_circuit_decomp(self):
        return any(self.args) # does not actually enforce circuit

    vars = IntVars(1,9, shape=(10,))
    constr = circuit(vars)
    constr.decompose = types.MethodType(my_circuit_decomp, constr) # attach it

    Model(constr).solve()
    ```

    The above will use 'my_circuit_decomp', if the solver does not
    natively support 'circuit'.
"""


def alldifferent(variables):
    expr = GlobalConstraint("alldifferent", variables)

    # a generic decomposition
    def decomp(self):
        raise NotImplementedError()

    expr.decompose = types.MethodType(decomp, expr) # attaches function to object
    return expr

def circuit(variables):
    expr = GlobalConstraint("circuit", variables)

    # a generic decomposition
    def decomp(self):
        raise NotImplementedError()

    expr.decompose = types.MethodType(decomp, expr) # attaches function to object
    return expr
