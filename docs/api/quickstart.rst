Quickstart
==========

This page gives a (very) brief overview of CPMpy. For more details, please refer to the full documentation.

Example: A simple model
------------------------------

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

Brief API Summary
----------------

Model Class
~~~~~~~~~~~

- ``model = cp.Model()`` -- Create a model.
- ``model += constraint`` -- Add a constraint to the model.
- ``model.maximize(obj)`` or ``model.minimize(obj)`` -- Set the objective.
- ``model.solve()`` -- Solve the model.

Decision Variables
~~~~~~~~~~~~~~~~~~

- ``x = cp.boolvar()`` -- x is a boolean decision variable, possible values (domain) are 0 and 1).
- ``x = cp.intvar(lb, ub)`` -- x is an integer decision variable, its domain is ``[lb, ub]`` (inclusive).
- ``x.value()`` -- Get the value of ``x`` after solving.


Global Constraints
~~~~~~~~~~~~~~~~~~

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


Global Functions
~~~~~~~~~~~~~~~~


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


Core Expressions
~~~~~~~~~~~~~~~~

- Python built-in overwrites: ``sum``, ``max``, ``min``, ``all``, ``any``, ``abs``
- Arithmetic: ``+``, ``-``, ``*``, ``//`` (integer division), ``%`` (modulo)
- Logical: ``&`` (and), ``|`` (or), ``~`` (not), ``^`` (xor)
- Comparison: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``
- Implication: ``x.implies(y)``

