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

Note that you can set `cb = OrtSolutionCounter(verbose=True)` if you want to see intermediate runtimes.

## Printing all solutions efficiently

The key part is that the solver does not understand CPMpy variables. Hence, before printing, we must first map our CPMpy variable to the corresponding solver variable, and then ask the solver what the value of that solver-native variable is. For this, every CPMpy solver interface maintains a `varmap` variable map.

We created an or-tools native `OrtSolutionPrinter` that does this mapping and value fetching for you. Check the source code to learn more, or to adapt it for special printing or other actions.

Here is example usage code:
```python
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from cpmpy.solvers.ortools import OrtSolutionPrinter

x = intvar(0,3, shape=2)
m = Model(x[0] > x[1])

s = CPM_ortools(m)
cb = OrtSolutionPrinter(s, x)
s.solve(enumerate_all_solutions=True, solution_callback=cb)
print("Nr of solutions:",cb.solution_count())
```

### Custom print functions

We also support custom print functions. The 'printer' argument is a callback that will be used to print the variables. By default, it is just the Python built-in `print` function, but you can give your own print function that will be called on each solution.

For example:

```python
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from cpmpy.solvers.ortools import OrtSolutionPrinter

x = intvar(0,3, shape=2)
m = Model(x[0] > x[1])
s = CPM_ortools(m)

def myprint(variables):
    x0, x1 = variables # we know we will give it 'x' as variables
    print(f"x0={x0.value()}, x1={x1.value()}")
cb = OrtSolutionPrinter(s, x, printer=myprint)

s.solve(enumerate_all_solutions=True, solution_callback=cb)
print("Nr of solutions:",cb.solution_count())
```
