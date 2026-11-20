<!--
GENERATED FILE — do not edit by hand.
Source: docs/_build/html/summary.html.md
(built by the sphinx_llm Sphinx extension, see docs/conf.py)
Regenerate: python skills/generate_references.py --build
See skills/README.md#keeping-reference-files-in-sync
-->

# Summary sheet

More extensive user documentation in [Modeling and solving with CPMpy](../../../docs/_build/html/modeling.html.md).

`import cpmpy as cp`

## Model class

- [`model = cp.Model()`](../../../docs/_build/html/api/model.html.md#cpmpy.model.Model.__init__) – Create a [`Model`](../../../docs/_build/html/api/model.html.md#cpmpy.model.Model).
- [`model.add(constraint)`](../../../docs/_build/html/api/model.html.md#cpmpy.model.Model.add) – Add a constraint (an [`Expression`](../../../docs/_build/html/api/expressions.html.md#module-cpmpy.expressions)) to the model, also allowed: model += constraint.
- [`model.maximize(obj)`](../../../docs/_build/html/api/model.html.md#cpmpy.model.Model.maximize) or [`model.minimize(obj)`](../../../docs/_build/html/api/model.html.md#cpmpy.model.Model.minimize) – Set the objective (an [`Expression`](../../../docs/_build/html/api/expressions.html.md#module-cpmpy.expressions)).
- [`model.solve()`](../../../docs/_build/html/api/model.html.md#cpmpy.model.Model.solve) – Solve the model with the default solver, returns True/False.
- [`model.solveAll()`](../../../docs/_build/html/api/model.html.md#cpmpy.model.Model.solveAll) – Solve and enumerate all solutions, returns number of solutions.
- [`model.status()`](../../../docs/_build/html/api/model.html.md#cpmpy.model.Model.status) – Get the status of the last solver run.
- [`model.objective_value()`](../../../docs/_build/html/api/model.html.md#cpmpy.model.Model.objective_value) – Get the objective value obtained during the last solver run.

## Solvers

[`Solvers`](../../../docs/_build/html/api/solvers.html.md#module-cpmpy.solvers) have the same API as [`Model`](../../../docs/_build/html/api/model.html.md#cpmpy.model.Model). Solvers are instantiated throught the static [`cp.SolverLookup`](../../../docs/_build/html/api/solvers/utils.html.md#cpmpy.solvers.utils.SolverLookup) class:

- [`cp.SolverLookup.solvernames()`](../../../docs/_build/html/api/solvers/utils.html.md#cpmpy.solvers.utils.SolverLookup.solvernames) – List all installed solvers (including subsolvers).
- [`cp.SolverLookup.get(solvername, model=None)`](../../../docs/_build/html/api/solvers/utils.html.md#cpmpy.solvers.utils.SolverLookup.get) – Initialize a specific solver.

## Decision Variables

[`Decision variables`](../../../docs/_build/html/api/expressions/variables.html.md#module-cpmpy.expressions.variables) are NumPy-like objects: `shape=None|1` creates one variable, `shape=4` creates a vector of 4 variables, `shape=(2,3)` creates a matrix of 2x3 variables, etc.
Name is optional too, indices are automatically added to the name so each variable has a unique name.

- [`x = cp.boolvar(shape=4, name="x")`](../../../docs/_build/html/api/expressions/variables.html.md#cpmpy.expressions.variables.boolvar) – Create four Boolean decision variables.
- [`x = cp.intvar(lb, ub)`](../../../docs/_build/html/api/expressions/variables.html.md#cpmpy.expressions.variables.intvar) – Create one integer decision variable with domain `[lb, ub]` (inclusive).
- [`x.value()`](../../../docs/_build/html/api/expressions/variables.html.md#cpmpy.expressions.variables._NumVarImpl.value) – Get the value of `x` obtained during the last solver run.

## Core Expressions

You can apply the following standard Python operators on CPMpy expressions, which creates the corresponding [`Core Expression`](../../../docs/_build/html/api/expressions/core.html.md#module-cpmpy.expressions.core) object:

- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Arithmetic: `+`, `-`, `*`, `//` (integer division), `%` (modulo), `**` (power)
- Logical: `&` (and), `|` (or), `~` (not), `^` (xor)
- Logical implication: [`x.implies(y)`](../../../docs/_build/html/api/expressions/core.html.md#cpmpy.expressions.core.Expression.implies)

Logical operators only work on Boolean variables/constraints, numeric operators work on both integer and Boolean variables/expressions.

CPMpy overwrites the following [`Python built-ins`](../../../docs/_build/html/api/expressions/python_builtins.html.md#module-cpmpy.expressions.python_builtins), they allow vectorized operations:

- [`cp.sum`](../../../docs/_build/html/api/expressions/python_builtins.html.md#cpmpy.expressions.python_builtins.sum), [`cp.abs`](../../../docs/_build/html/api/expressions/python_builtins.html.md#cpmpy.expressions.python_builtins.abs), [`cp.max`](../../../docs/_build/html/api/expressions/python_builtins.html.md#cpmpy.expressions.python_builtins.max), [`cp.min`](../../../docs/_build/html/api/expressions/python_builtins.html.md#cpmpy.expressions.python_builtins.min)
- [`cp.all`](../../../docs/_build/html/api/expressions/python_builtins.html.md#cpmpy.expressions.python_builtins.all), [`cp.any`](../../../docs/_build/html/api/expressions/python_builtins.html.md#cpmpy.expressions.python_builtins.any)

You can **index** CPMpy expressions with an integer decision variable: `x[y]`, which will create an [`Element`](../../../docs/_build/html/api/expressions/globalfunctions.html.md#cpmpy.expressions.globalfunctions.Element) expression object.
To index non-CPMpy arrays, wrap them with [`cpm_array()`](../../../docs/_build/html/api/expressions/variables.html.md#cpmpy.expressions.variables.cpm_array): `cpm_array([1,2,3])[y]`.

## Global Functions

[`Global functions`](../../../docs/_build/html/api/expressions/globalfunctions.html.md#module-cpmpy.expressions.globalfunctions) are numeric functions that some solvers support natively (through a solver-specific global constraint). CPMpy automatically rewrites the global function as needed to work with any solver.

| [`Minimum`](../../../docs/_build/html/api/expressions/globalfunctions.html.md#cpmpy.expressions.globalfunctions.Minimum)           | Computes the minimum value of the arguments                                                                     |
|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [`Maximum`](../../../docs/_build/html/api/expressions/globalfunctions.html.md#cpmpy.expressions.globalfunctions.Maximum)           | Computes the maximum value of the arguments                                                                     |
| [`Abs`](../../../docs/_build/html/api/expressions/globalfunctions.html.md#cpmpy.expressions.globalfunctions.Abs)                   | Computes the absolute value of the argument                                                                     |
| [`Element`](../../../docs/_build/html/api/expressions/globalfunctions.html.md#cpmpy.expressions.globalfunctions.Element)           | The Element(Arr, Idx) global function allows indexing into an array with a decision variable.                   |
| [`Count`](../../../docs/_build/html/api/expressions/globalfunctions.html.md#cpmpy.expressions.globalfunctions.Count)               | The Count global function represents the number of occurrences of a value in an array                           |
| [`Among`](../../../docs/_build/html/api/expressions/globalfunctions.html.md#cpmpy.expressions.globalfunctions.Among)               | The Among global function counts how many variables in an array take values that are in a given set of values.  |
| [`NValue`](../../../docs/_build/html/api/expressions/globalfunctions.html.md#cpmpy.expressions.globalfunctions.NValue)             | The NValue global function counts the number of distinct values in an array.                                    |
| [`NValueExcept`](../../../docs/_build/html/api/expressions/globalfunctions.html.md#cpmpy.expressions.globalfunctions.NValueExcept) | The NValueExcept global function counts the number of distinct values in an array, excluding a specified value. |

## Global Constraints

[`Global constraints`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#module-cpmpy.expressions.globalconstraints) are constraints (Boolean functions) that some solvers support natively. All global constraints can be reified (implication, equivalence) and used in other expressions, which CPMpy will handle.

| [`AllDifferent`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.AllDifferent)                     | Enforces that all arguments have a different (distinct) value                                                                                                          |
|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`AllDifferentExcept0`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.AllDifferentExcept0)       | Enforces that all arguments, except those equal to 0, have a different (distinct) value.                                                                               |
| [`AllDifferentExceptN`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.AllDifferentExceptN)       | Enforces that all arguments, except those equal to a value in n, have a different (distinct) value.                                                                    |
| [`AllEqual`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.AllEqual)                             | Enforces that all arguments have the same value                                                                                                                        |
| [`AllEqualExceptN`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.AllEqualExceptN)               | Enforces that all arguments, except those equal to a value in n, have the same value.                                                                                  |
| [`Circuit`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.Circuit)                               | Enforces that the sequence of variables form a circuit, where x[i] = j means that node j is the successor of node i.                                                   |
| [`Inverse`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.Inverse)                               | Enforces that the forward and reverse arrays represent the inverse function of one another.                                                                            |
| [`Table`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.Table)                                   | Enforces that the values of the variables in 'array' correspond to a row in 'table'.                                                                                   |
| [`ShortTable`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.ShortTable)                         | Extension of the Table constraint where the table matrix may contain wildcards (STAR), meaning there are no restrictions for the corresponding variable in that tuple. |
| [`NegativeTable`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.NegativeTable)                   | The values of the variables in 'array' do not correspond to any row in 'table'.                                                                                        |
| [`IfThenElse`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.IfThenElse)                         | Enforces a conditional expression of the form: if condition then if_true else if_false.                                                                                |
| [`InDomain`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.InDomain)                             | Enforces the expression is assigned to a value in the given domain.                                                                                                    |
| [`Xor`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.Xor)                                       | Enforces the exclusive-or relation of the arguments.                                                                                                                   |
| [`Cumulative`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.Cumulative)                         | Enforces that a set of tasks is scheduled such that the capacity of the resource is never exceeded and enforces:                                                       |
| [`Precedence`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.Precedence)                         | Enforces a precedence relationship between a set of variables.                                                                                                         |
| [`NoOverlap`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.NoOverlap)                           | Enforces that a set of tasks are scheduled without overlapping, and enforces:                                                                                          |
| [`GlobalCardinalityCount`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.GlobalCardinalityCount) | Enforces that the number of occurrences of each value vals[i] in the list of variables vars is equal to occ[i].                                                        |
| [`Increasing`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.Increasing)                         | Enforces that the expressions are assigned to (non-strictly) increasing values.                                                                                        |
| [`Decreasing`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.Decreasing)                         | Enforces that the expressions are assigned to (non-strictly) decreasing values.                                                                                        |
| [`IncreasingStrict`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.IncreasingStrict)             | Enforces that the expressions are assigned to strictly increasing values.                                                                                              |
| [`DecreasingStrict`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.DecreasingStrict)             | Enforces that the expressions are assigned to strictly decreasing values.                                                                                              |
| [`LexLess`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.LexLess)                               | Enforces that the first list is lexicographically smaller than the second list.                                                                                        |
| [`LexLessEq`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.LexLessEq)                           | Enforces that the first list is lexicographically smaller than or equal to the second list.                                                                            |
| [`LexChainLess`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.LexChainLess)                     | Enforces that all rows of the matrix are lexicographically ordered.                                                                                                    |
| [`LexChainLessEq`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.LexChainLessEq)                 | Enforces that all rows of the matrix are lexicographically ordered (less or equal)                                                                                     |
| [`DirectConstraint`](../../../docs/_build/html/api/expressions/globalconstraints.html.md#cpmpy.expressions.globalconstraints.DirectConstraint)             | A `DirectConstraint` will directly call a function of the underlying solver when added to a CPMpy solver                                                               |

## Guidelines and tips

- Do not `from cpmpy import *`, the implicit overloading of any/all and sum may break or slow down other libraries.
- Explicitly use CPMpy versions of built-in functions (`cp.sum`, `cp.all`, etc.).
- Use global constraints/global functions where possible, some solvers will be much faster.
- Stick to integer constants in constraints and model objectives (some solvers support FloatSum but its very limited)
- For maintainability, use logical code organization and comments to explain your constraints.

## Toy example

```python
import cpmpy as cp

# Decision Variables
b = cp.boolvar()
x1, x2, x3 = x = cp.intvar(1, 10, shape=3)

# Constraints
model = cp.Model()

model.add(x[0] == 1)
model.add(cp.AllDifferent(x))
model.add(cp.Count(x, 9) == 1)
model.add(b.implies(x[1] + x[2] > 5))

# Objective
model.maximize(cp.sum(x) + 100 * b)

# Solving
solved = model.solve()
if solved:
    print("Solution found:")
    print('b:', b.value(), ' x:', x.value().tolist())
else:
    print("No solution found.")
```
