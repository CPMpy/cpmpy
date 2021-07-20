# Obtaining multiple solutions

CPMpy is meant to be a light-weight modeling layer on top of Python-based solver interfaces. This makes it possible to model a problem using CPMpy expressions, while still using advances features of the solver.

Here is the standard, solver-agnostic, way of solving a model in CPMpy:

```python
from cpmpy import *

x = intvar(0,3, shape=2)
m = Model(x[0] > x[1])

print(m.solve())
print(m.status())
print(x.value())
```

In the following, we will use the __or-tools CP-SAT Python interface__. To use some of its advanced features, it is recommended to read the [corresponding documentation](https://developers.google.com/optimization/reference/python/sat/python/cp_model).



## Counting all solutions
Or-tools uses a callback mechanism to process solutions found, and it uses that to count all solutions effectively.

We can use the native callback system, and feed it directly to the created or-tools object created by the CPM_ortools constructor.

The status of or-tools, as well as variables etc, can then be translated back to CPMpy in the same way as is done when call s.solve(), by calling s._after_solve() directly.

We first demonstate this with a native or-tools callback that simply counts the number of solutions:

```python
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from ortools.sat.python import cp_model as ort

# native or-tools callback
class ORT_solcount(ort.CpSolverSolutionCallback):
    def __init__(self):
        super().__init__()
        self.solcount = 0

    def on_solution_callback(self):
        self.solcount += 1
cb = ORT_solcount()

# direct manipulation of the ort_solver instance created by CPMpy:
s = CPM_ortools(m)
ort_status = s.ort_solver.SearchForAllSolutions(s.ort_model, cb)
print(s._after_solve(ort_status)) # post-process after solve() call...
print(s.status())
print(x.value()) # will be the last found one
print("Nr solutions:", cb.solcount)
```

## Displaying intermediate solutions during solving
And a final example, where we show how to use or-tools' solve-with-solution-callback mechanism to display intermediate solutions during solving.

It uses the exact same solution callback as in the previous example where we print and count 'all' solutions. Note that this problem is too simple to have intermediate solutions.

```python
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from ortools.sat.python import cp_model as ort

# native or-tools callback, with CPMpy variables and printing
class ORT_myprint(ort.CpSolverSolutionCallback):
    def __init__(self, varmap, x):
        super().__init__()
        self.solcount = 0
        self.varmap = varmap
        self.x = x

    def on_solution_callback(self):
        # populate values before printing
        for cpm_var in self.x: 
            cpm_var._value = self.Value(self.varmap[cpm_var])

        self.solcount += 1
        print("x:",self.x.value())
cb = ORT_myprint(s.varmap, x)

m_opt = Model(x[0] > x[1], maximize=sum(x))
s = CPM_ortools(m_opt)
ort_status = s.ort_solver.SolveWithSolutionCallback(s.ort_model, cb)
print(s._after_solve(ort_status)) # post-process after solve() call...
print(s.status())
print(x.value())
print("Nr intermediate solutions:", cb.solcount)
```

## Printing all solutions, the efficient way
It is also possible to print the value at the level of CPMpy variables in the callback. To do this, you need to pass the 'varmap' mapping from CPMpy variables to or-tools variables, and populate the var.\_value property first, as such:

from ortools.sat.python import cp_model as ort

```python
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from ortools.sat.python import cp_model as ort

# native or-tools callback, with CPMpy variables and printing
class ORT_myprint(ort.CpSolverSolutionCallback):
    def __init__(self, varmap, x):
        super().__init__()
        self.solcount = 0
        self.varmap = varmap
        self.x = x

    def on_solution_callback(self):
        # populate values before printing
        for cpm_var in self.x: 
            cpm_var._value = self.Value(self.varmap[cpm_var])

        self.solcount += 1
        print("x:",self.x.value())
cb = ORT_myprint(s.varmap, x)

s = CPM_ortools(m)
ort_status = s.ort_solver.SearchForAllSolutions(s.ort_model, cb)
print(s._after_solve(ort_status)) # post-process after solve() call...
print(s.status())
print(x.value()) # will be the last found one
print("Nr solutions:", cb.solcount)
```

## Solution enumeration with blocking clauses
Another way to do solution enumeration is to manually add blocking clauses (clauses forbidding the current solution). In case you just want to enumerate all solutions, this will be less efficient then using or-tools callbacks.

However, in case you have custom blocking clauses, or don't care too much by some additional overhead caused by the non-incrementality of or-tools, then you can do the following:

```python
from cpmpy import *
from cpmpy.solvers import CPM_ortools

x = intvar(0,3, shape=2)
m = Model(x[0] > x[1])
s = CPM_ortools(m)
solcount = 0
while(s.solve()):
    solcount += 1
    print("x:",x.value())
    # add blocking clause, to CPMpy solver directly
    s += [ any(x != x.value()) ]
print(s.status())
print(x.value()) # will be the last found one
print("Nr solutions:", solcount)
```
