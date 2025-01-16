# Obtaining multiple solutions

CPMpy models and solvers support the `solveAll()` function. It efficiently computes all solutions and optionally displays them. Alternatively, you can manually add blocking clauses as explained in the second half of this page.

When using `solveAll()`, a solver will use an optimized native implementation behind the scenes when that exists.

It has two special named optional arguments:

  - `display=...` : accepts a CPMpy expression, a list of CPMpy expressions or a callback function that will be called on every solution found (default: None)
  - `solution_limit=...` : stop after this many solutions (default: None)

It also accepts named argument `time_limit=...` and any other keyword argument is passed on to the solver just like `solve()` does.

It returns the number of solutions found.

## `solveAll()` examples

In the following examples, we assume:

```python
from cpmpy import *
x = intvar(0, 3, shape=2)
m = Model(x[0] > x[1])
```

Just return the number of solutions (here: 6)
```python
n = m.solveAll()
print("Nr of solutions:", n)
```

With a solution limit: e.g. find up to 2 solutions
```python
n = m.solveAll(solution_limit=2)
print("Nr of solutions:", n)
```

Find all solutions, and print the value of `x` for each solution found.
```python
n = m.solveAll(display=x)
```

`display` can also take lists of arbitrary CPMpy expressions:
```python
n = m.solveAll(display=[x,sum(x)])
```

Perhaps most powerful is the use of __callbacks__, which allows for rich printing, solution storing, dynamic stopping and more. You can use any variable name from the outer scope here (it is a closure). That does mean that you have to call `var.value()` each time to get the value(s) of this particular solution.

Rich printing with a callback function:
```python
def myprint():
    xval = x.value()
    print(f"x={xval}, sum(x)={sum(xval)}")
n = m.solveAll(display=myprint) # callback function without brackets 
```

Also callback with an anonymous lambda function possible:
```python
n = m.solveAll(display=lambda: print(f"x={x.value()} sum(x)={sum(x.value())}") 
```

See the [set_game.ipynb](https://github.com/CPMpy/cpmpy/blob/master/examples/set_game.ipynb) for an example of how we use it as a callback to call a plotting function, to plot all the solutions as they are found.

A callback is also the (only) way to go if you want to store information about all the found solutions (only recommended for small numbers of solutions).
```python
solutions = []
def collect():
    print(x.value())
    solutions.append(list(x.value()))
n = m.solveAll(display=collect, solution_limit=1000) # callback function without brackets
print(f"Stored {len(solutions)} solutions.")
```


## Solution enumeration with blocking clauses
The MiniSearch[1] paper promoted a small, high-level domain-specific language for specifying the search for multiple solutions with blocking clauses.

This approach makes use of the incremental nature of the solver interfaces. It is hence much more efficient (less overhead) to do this on a solver object rather then a generic model object.

Here is an example of standard solution enumeration, note that this will be much slower than `solveAll()`.

```python
from cpmpy import *

x = intvar(0,3, shape=2)
m = Model(x[0] > x[1])
s = SolverLookup.get("ortools", m) # faster on a solver interface directly

while s.solve():
    print(x.value())
    # block this solution from being valid
    s += ~all(x == x.value())
```

In case of multiple variables you should put them in one long Python-native list, as such:
```python
x = intvar(0,3, shape=2)
b = boolvar()
m = Model(b.implies(x[0] > x[1]))
s = SolverLookup.get("ortools", m) # faster on a solver interface directly

while s.solve():
    print(x.value(), b.value())
    allvars = list(x)+[b]
    # block this solution from being valid
    s += ~all(v == v.value() for v in allvars)
```


## Diverse solution search
A better, more complex example of repeated solving is when searching for diverse solutions.

The goal is to iteratively find solutions that are as diverse as possible with the previous solutions. Many definitions of diversity between solutions exist. We can for example measure the difference between two solutions with the [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance#:~:text=In%20information%20theory%2C%20the%20Hamming,the%20corresponding%20symbols%20are%20different.) (comparing the number of different values) or the [Euclidian distance](https://en.wikipedia.org/wiki/Euclidean_distance) (compare the absolute difference in value for the variables).

Here is the example code for enumerating K diverse solutions with Hamming distance, which overwrites the objective function in each iteration:

```python
# Diverse solutions, Hamming distance (inequality)
x = boolvar(shape=6)
m = Model(sum(x) == 2)
s = SolverLookup.get("ortools", m) # faster on a solver interface directly

K = 3
store = []
while len(store) < 3 and s.solve():
    print(len(store), ":", x.value())
    store.append(x.value())
    # Hamming dist: nr of different elements
    s.maximize(sum([sum(x != sol) for sol in store]))
```

As a fun fact, observe how `x != sol` works, even though one is a vector of Boolean variables and sol is a numpy array. However, both have the same length, so this is automatically translated into a pairwise comparison constraint by CPMpy. These numpy-style vectorized operations mean we have to write fewer loops while modeling.

To use the Euclidian distance, only the last line needs to be changed. We again use vectorized operations and obtain succinct models. The creation of intermediate variables (with appropriate domains) is done by CPMpy behind the scenes.

```python
    # Euclidian distance: absolute difference in value
    s.maximize(sum([sum( abs(np.add(x, -sol)) ) for sol in store]))
```

## Mixing native callbacks with CPMpy

CPMpy passes arguments to `solve()` directly to the underlying solver object, so you can actually define your own native callbacks and pass them to the solve call.

The following is an example of that, which is actually how the native `solveAll()` for OR-Tools is implemented. You could give it your own custom implemented callback `cb` too.
```python
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from cpmpy.solvers.ortools import OrtSolutionPrinter

x = intvar(0,3, shape=2)
m = Model(x[0] > x[1])

s = SolverLookup.get('ortools', m)
cb = OrtSolutionPrinter()
s.solve(enumerate_all_solutions=True, solution_callback=cb)
print("Nr of solutions:",cb.solution_count())
```
Have a look at `OrtSolutionPrinter`'s [implementation](https://github.com/CPMpy/cpmpy/blob/master/cpmpy/solvers/ortools.py#L650).
