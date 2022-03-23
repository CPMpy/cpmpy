# Obtaining multiple solutions

The MiniSearch[1] paper promoted a small, high-level domain-specific language for specifying the search for multiple solutions. The CPMpy solver interface follows the same guiding principles. 

If these examples and the CPMpy API insufficient for your needs, remember that you can access the underlying solver API directly too (see bottom of this page).

## Solution enumeration with blocking clauses
If you are just interested in counting all solutions, or in enumerating large numbers of solutions very efficiently, then you should use the specialized capabilities of a solver for this. See "Counting all solutions efficiently" lower in this document.

For all other use cases, the following MiniSearch-style approach is adviced as it is solver-independent and makes the intermediate solutions easy to manipulate.

```python
from cpmpy import *
from cpmpy.solvers import CPM_ortools

x = intvar(0,3, shape=2)
m = Model(x[0] > x[1])

# we can make repeated solving faster by working on a solver directly
m = CPM_ortools(m) # optional, use any CPMpy solver you like
while m.solve():
    print(x.value())
    # block this solution from being valid
    m += ~all(x == x.value())
```

Technical side-note: if your solution consists of more then one variable, you should first collect them in a single list (wrap arrays in list() and single vars in [] so you can concatenate them as pure python lists), for example:
```python
x = intvar(0,3, shape=2)
b = boolvar()
m = CPM_ortools(Model(b.implies(x[0] > x[1])))
while m.solve():
    print(x.value(), b.value())
    allvars = list(x)+[b]
    # block this solution from being valid
    m += ~all(v == v.value() for v in allvars)
```


## Diverse solution search
Another example of repeated solving is when searching for diverse solutions.

The goal is to iteratively find solutions that are as diverse as possible with the previous solutions. Many definitions of diversity between solutions exist. We can for example measure the difference between two solutions with the Hamming distance (comparing the number of different values) or the Euclidian distance (compare the absolute difference in value for the variables).

Here is the example code for enumerating K diverse solutions with Hamming distance:

```python
# Diverse solutions, Hamming distance (inequality)
x = boolvar(shape=6)
m = Model(sum(x) == 2)
m = CPM_ortools(m) # optional but faster

K = 3
store = []
while len(store) < 3 and m.solve():
    print(len(store), ":", x.value())
    store.append(x.value())
    # Hamming dist: nr of different elements
    m.maximize(sum([sum(x != sol) for sol in store]))
```

As a fun fact, observe how `x != sol` works, even though one is a vector of Boolean variables and sol is Numpy array. However, both have the same length, so this is automatically translated into a pairwise comparison constraint by CPMpy. These numpy-style vectorized operations mean we have to write much less loops while modelling.

Here is the code for Euclidian distance. We again use vectorized operations and obtain succinct models. The creation of intermediate variables (with appropriate domains) is done by CPMpy behind the scenes.

```python
import numpy as np
from cpmpy import *
from cpmpy.solvers import CPM_ortools

# Diverse solutions, Euclidian distance (absolute difference)
x = intvar(0,4, shape=6)
m = Model(sum(x) > 10, sum(x) < 20)
m = CPM_ortools(m) # optional but faster

K = 3
store = []
while len(store) < K and m.solve() is not False:
    print(len(store), ":", x.value())
    store.append(x.value())
    # Euclidian distance: absolute difference in value
    m.maximize(sum([sum( abs(np.add(x, -sol)) ) for sol in store]))
```

## Counting all solutions efficiently

CP solvers typically natively support enumerating all solutions. This will be much than adding blocking clauses from CPMpy, when we are dealing with hundreds of solutions.

Or-tools has an 'on solution' callback mechanism that can be used to efficiently count the number of solutions.

To better support the use case of _counting_ solutions (without printing them), CPMpy includes a native ortools callback that does counting and only optionally also prints something on every solution. The callback is called `OrtSolutionCounter`, defined in cpmpy.solvers.ortools.

We can use it to efficiently count solutions as follows (dont forget the 'enumerate_all_solutions=True' argument):
```python
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from cpmpy.solvers.ortools import OrtSolutionCounter

x = intvar(0,3, shape=2)
m = Model(x[0] > x[1])

s = CPM_ortools(m)
cb = OrtSolutionCounter()
s.solve(enumerate_all_solutions=True, solution_callback=cb)
print("Nr of solutions:",cb.solution_count())
```

## Using the native solver API directly

In the following, we will use the __or-tools CP-SAT Python interface__. To use some of its advanced features, it is recommended to read the [corresponding documentation](https://developers.google.com/optimization/reference/python/sat/python/cp_model).

This very advanced examples shows how to **print all solutions with CPMpy using a custom or-tools native callback**.

To do this, you need to pass the hidden 'varmap' mapping from CPMpy variables to the callback, so that we can populate the CPMpy variable`._value` property from the or-tools variableor-tools variables. As such:

```python
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from ortools.sat.python import cp_model as ort

x = intvar(0,3, shape=2)
m = Model(x[0] > x[1])
s = CPM_ortools(m)

# native or-tools callback, with CPMpy variables and printing
class ORT_myprint(ort.CpSolverSolutionCallback):
    def __init__(self, varmap, x):
        super().__init__()
        self.solcount = 0
        self.varmap = varmap
        self.x = x

    def on_solution_callback(self):
        # populate values before printing
        for cpm_var in self.x.flat: # flatten the vararray (or cpm_array())
            cpm_var._value = self.Value(self.varmap[cpm_var])

        self.solcount += 1
        print("x:",self.x.value())
cb = ORT_myprint(s.varmap, x)

ort_status = s.ort_solver.SearchForAllSolutions(s.ort_model, cb)
print(s._after_solve(ort_status)) # post-process after solve() call...
print(s.status())
print(x.value()) # will be the last found one
print("Nr solutions:", cb.solcount)
```

Actually `s.solve(enumerate_all_solutions=True, solution_callback=cb)` would also have worked, but this shows that if you want you can do your own calls and `_after_solve` and forward and reverse mapping between variables as you wish...
