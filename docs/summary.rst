Summary sheet
=============

More extensive user documentation in `Modeling and solving with CPMpy <modeling.md>`_.

``import cpmpy as cp``

Model class (:mod:`cpmpy.model`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``model = cp.Model()`` -- Create a :class:`Model <cpmpy.model.Model>`.
- ``model += constraint`` -- Add a constraint (an :mod:`Expression <cpmpy.expressions>`) to the model.
- :meth:`model.maximize(obj) <cpmpy.model.Model.maximize>` or :meth:`model.minimize(obj) <cpmpy.model.Model.minimize>` -- Set the objective (an :mod:`Expression <cpmpy.expressions>`).
- :meth:`model.solve() <cpmpy.model.Model.solve>` -- Solve the model with the default solver, returns True/False.
- :meth:`model.solveAll() <cpmpy.model.Model.solveAll>` -- Solve and enumerate all solutions, returns number of solutions.


Solvers (:mod:`cpmpy.solvers`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solvers have the same API as :class:`Model <cpmpy.model.Model>`. Solvers are instantiated throught the static class :class:`cp.SolverLookup <cpmpy.solvers.utils.SolverLookup>`:

- :meth:`cp.SolverLookup.solvernames() <cpmpy.solvers.utils.SolverLookup.solvernames>` -- List all installed solvers (including subsolvers).
- :meth:`cp.SolverLookup.get(solvername, model=None) <cpmpy.solvers.utils.SolverLookup.get>` -- Initialize a specific solver.


Decision Variables
~~~~~~~~~~~~~~~~~~

- ``x = cp.boolvar()`` -- x is a boolean decision variable, possible values (domain) are 0 and 1).
- ``x = cp.intvar(lb, ub)`` -- x is an integer decision variable, its domain is ``[lb, ub]`` (inclusive).
- ``x.value()`` -- Get the value of ``x`` after solving.


Core Expressions
~~~~~~~~~~~~~~~~

- Python built-in overwrites: ``sum``, ``max``, ``min``, ``all``, ``any``, ``abs``
- Arithmetic: ``+``, ``-``, ``*``, ``//`` (integer division), ``%`` (modulo)
- Logical: ``&`` (and), ``|`` (or), ``~`` (not), ``^`` (xor)
- Comparison: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``
- Implication: ``x.implies(y)``


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


Toy example:
~~~~~~~~~~~~

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

