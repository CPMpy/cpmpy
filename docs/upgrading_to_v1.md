# Upgrading to CPMpy 1.0

CPMpy v1.0.0 is a major release: many transformations were rewritten, expressions are typed, and a few
modeling constructs changed in ways that may require small updates to your code.

This page shows the most common changes with small examples. For the complete list — including all
internal changes, deprecations and widened APIs — see the
[full changelog on GitHub](https://github.com/CPMpy/cpmpy/blob/master/changelog.md#100).

```{note}
Models saved to a `.pickle` file with CPMpy < 1.0.0 will no longer load (or will fail when used) in
v1.0.0. Re-run your model-building code under v1.0.0 and save again. For longer-term storage, consider
writing models to a solver-independent text format with the new {mod}`cpmpy.tools.io` module.
```

## Float objectives: use `FloatSum`

Float coefficients are no longer allowed inside regular expressions, and `Model.minimize()`/`Model.maximize()`
only accept integer-valued expressions. Float objectives are now expressed with the objective-only
{class}`~cpmpy.expressions.globalfunctions.FloatSum`, passed directly to a solver object:

```python
# before (<= 0.10)
model.minimize(0.5*x + 1.5*y)
model.solve()
print(model.objective_value())

# now (1.0)
obj = cp.FloatSum([0.5, 1.5], [x, y])
s = cp.SolverLookup.get("ortools", model)   # model contains only the constraints
s.minimize(obj)
s.solve()
print(obj.value())   # s.objective_value() stays None when the optimum is not integral
```

Float coefficients in *constraints* should be rescaled to integers, e.g. replace `0.5*x + 1.5*y <= 2`
by `x + 3*y <= 4`.

## Lowercase constraint aliases are gone

The functions `cp.alldifferent()`, `cp.allequal()` and `cp.circuit()` (deprecated since 0.9.0) are no
longer exported. Use the global constraint classes:

```python
cp.alldifferent(x)      # before, AttributeError in 1.0
cp.AllDifferent(x)      # now
```

## Inspecting expressions

If your code walks or matches expression trees, three changes matter:

```python
x, y = cp.intvar(0, 10, shape=2)
arr = cp.intvar(0, 10, shape=(3, 3))

# 1. multiplication is a global function now, no longer an Operator
isinstance(x*y, cp.expressions.core.Operator)              # False (was True)
isinstance(x*y, cp.Multiplication)                         # True

# 2. multi-dimensional indexing creates NDElement, and Element is 1D-only
isinstance(arr[x, y], cp.Element)                          # False (was True)
isinstance(arr[x, y], cp.NDElement)                        # True

# 3. .args is a read-only tuple instead of a list
(x + y).args                                               # (IV0, IV1) — a tuple
(x + y).update_args([x, 3])                                # explicit in-place update
```

Also note that variable arrays ({class}`~cpmpy.expressions.variables.NDVarArray`, as returned by
`intvar(..., shape=...)`) are no longer `Expression` subclasses: all modeling functionality is
unchanged, but `isinstance(arr, Expression)` is now `False` and `arr.args`/`arr.name` no longer exist.

## Relocated tooling

The DIMACS and XCSP3 readers moved to the new {mod}`cpmpy.tools.io` module (the old names still work
but raise a `DeprecationWarning`):

```python
from cpmpy.tools.dimacs import read_dimacs   # before
from cpmpy.tools.io import load_dimacs       # now (also accepts strings and file objects)

from cpmpy.tools.xcsp3 import read_xcsp3     # before
from cpmpy.tools.io import load_xcsp3        # now
```

Or use the generic entry points `cpmpy.tools.io.load()` and `cpmpy.tools.io.write()`, which select the
format based on the file extension.

## More

The [changelog on GitHub](https://github.com/CPMpy/cpmpy/blob/master/changelog.md#100) additionally covers:

- stricter global constraint constructors (e.g. `Table` requires a rectangular integer table, and
  `Table`/`Minimum`/`Maximum` no longer flatten nested lists),
- minor behavior changes from bug fixes (numpy broadcasting, `value()` on partially-assigned globals),
- changes to the transformation interfaces for advanced users,
- and the new, widened APIs (intermediate-solution `display=` callbacks, iterable assumptions, ...).
