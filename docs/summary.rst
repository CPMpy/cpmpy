Summary sheet
=============

More extensive user documentation in :doc:`Modeling and solving with CPMpy <modeling>`.

``import cpmpy as cp``

Model class
~~~~~~~~~~~

- :meth:`model = cp.Model() <cpmpy.model.Model.__init__>` -- Create a :class:`Model <cpmpy.model.Model>`.
- :meth:`model += constraint <cpmpy.model.Model.__add__>` -- Add a constraint (an :mod:`Expression <cpmpy.expressions>`) to the model.
- :meth:`model.maximize(obj) <cpmpy.model.Model.maximize>` or :meth:`model.minimize(obj) <cpmpy.model.Model.minimize>` -- Set the objective (an :mod:`Expression <cpmpy.expressions>`).
- :meth:`model.solve() <cpmpy.model.Model.solve>` -- Solve the model with the default solver, returns True/False.
- :meth:`model.solveAll() <cpmpy.model.Model.solveAll>` -- Solve and enumerate all solutions, returns number of solutions.
- :meth:`model.status() <cpmpy.model.Model.status>` -- Get the status of the last solver run.
- :meth:`model.objective_value() <cpmpy.model.Model.objective_value>` -- Get the objective value obtained during the last solver run.

Solvers
~~~~~~~

:mod:`Solvers <cpmpy.solvers>` have the same API as :class:`Model <cpmpy.model.Model>`. Solvers are instantiated throught the static :class:`cp.SolverLookup <cpmpy.solvers.utils.SolverLookup>` class:

- :meth:`cp.SolverLookup.solvernames() <cpmpy.solvers.utils.SolverLookup.solvernames>` -- List all installed solvers (including subsolvers).
- :meth:`cp.SolverLookup.get(solvername, model=None) <cpmpy.solvers.utils.SolverLookup.get>` -- Initialize a specific solver.


Decision Variables
~~~~~~~~~~~~~~~~~~

:mod:`Decision variables <cpmpy.expressions.variables>` are NumPy-like objects: ``shape=None|1`` creates one variable, ``shape=4`` creates a vector of 4 variables, ``shape=(2,3)`` creates a matrix of 2x3 variables, etc.
Name is optional too, indices are automatically added to the name so each variable has a unique name.

- :meth:`x = cp.boolvar(shape=4, name="x") <cpmpy.expressions.variables.boolvar>` -- Create four Boolean decision variables.
- :meth:`x = cp.intvar(lb, ub) <cpmpy.expressions.variables.intvar>` -- Create one integer decision variable with domain ``[lb, ub]`` (inclusive).
- :meth:`x.value() <cpmpy.expressions.variables._NumVarImpl.value>` -- Get the value of ``x`` obtained during the last solver run.

Core Expressions
~~~~~~~~~~~~~~~~

You can apply the following standard Python operators on CPMpy expressions, which creates the corresponding :class:`Core Expression <cpmpy.expressions.core>` object:

- Comparison: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``
- Arithmetic: ``+``, ``-``, ``*``, ``//`` (integer division), ``%`` (modulo), ``**`` (power)
- Logical: ``&`` (and), ``|`` (or), ``~`` (not), ``^`` (xor)
- Logical implication: :meth:`x.implies(y) <cpmpy.expressions.core.Expression.implies>`

Logical operators only work on Boolean variables/constraints, numeric operators work on both integer and Boolean variables/expressions.

CPMpy overwrites the following :mod:`Python built-ins <cpmpy.expressions.python_builtins>`, they allow vectorized operations:

- :meth:`cp.sum <cpmpy.expressions.python_builtins.sum>`, :meth:`cp.abs <cpmpy.expressions.python_builtins.abs>`, :meth:`cp.max <cpmpy.expressions.python_builtins.max>`, :meth:`cp.min <cpmpy.expressions.python_builtins.min>`
- :meth:`cp.all <cpmpy.expressions.python_builtins.all>`, :meth:`cp.any <cpmpy.expressions.python_builtins.any>`

You can **index** CPMpy expressions with an integer decision variable: ``x[y]``, which will create an :class:`Element <cpmpy.expressions.globalfunctions.Element>` expression object.
To index non-CPMpy arrays, wrap them with :func:`~cpmpy.expressions.variables.cpm_array`: ``cpm_array([1,2,3])[y]``.

Global Functions
~~~~~~~~~~~~~~~~

:mod:`Global functions <cpmpy.expressions.globalfunctions>` are numeric functions that some solvers support natively (through a solver-specific global constraint). CPMpy automatically rewrites the global function as needed to work with any solver.

.. currentmodule:: cpmpy.expressions.globalfunctions
.. autosummary::
    :nosignatures:

        Minimum
        Maximum
        Abs
        Element
        Count
        Among
        NValue
        NValueExcept


Global Constraints
~~~~~~~~~~~~~~~~~~

:mod:`Global constraints <cpmpy.expressions.globalconstraints>` are constraints (Boolean functions) that some solvers support natively. All global constraints can be reified (implication, equivalence) and used in other expressions, which CPMpy will handle.

.. currentmodule:: cpmpy.expressions.globalconstraints
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


Guidelines and tips
~~~~~~~~~~~~~~~~~~~

- Do not ``from cpmpy import *``, the implicit overloading of any/all and sum may break or slow down other libraries.
- Explicitly use CPMpy versions of built-in functions (``cp.sum``, ``cp.all``, etc.).
- Use global constraints/global functions where possible, some solvers will be much faster.
- Stick to integer constants; floats and fractional numbers are not supported.
- For maintainability, use logical code organization and comments to explain your constraints.


Toy example
~~~~~~~~~~~

.. code-block:: python

    import cpmpy as cp

    # Decision Variables
    b = cp.boolvar()
    x1, x2, x3 = x = cp.intvar(1, 10, shape=3)

    # Constraints
    model = cp.Model()

    model += (x[0] == 1)
    model += cp.AllDifferent(x)
    model += cp.Count(x, 9) == 1
    model += b.implies(x[1] + x[2] > 5)

    # Objective
    model.maximize(cp.sum(x) + 100 * b)

    # Solving
    solved = model.solve()
    if solved:
        print("Solution found:")
        print('b:', b.value(), ' x:', x.value().tolist())
    else:
        print("No solution found.")

