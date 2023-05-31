
   Youtube tutorial video <https://www.youtube.com/watch?v=A4mmmDAdusQ>

   installation_instructions
   Quickstart sudoku notebook <https://github.com/CPMpy/cpmpy/blob/master/examples/quickstart_sudoku.ipynb>
   More examples <https://github.com/CPMpy/cpmpy/blob/master/examples/>


# Modeling

CPMpy is a library for modeling and solving constrained satisfaction and optimisation problems (CSPs and COPs in the AI literature).

A constraint model consists of 3 key parts:

  * _Decision variables_ with their domain (allowed values)
  * _Constraints_ over decision variables
  * Optionally an _objective function_

```python
from cpmpy import *
m = Model()

# Variables
b = boolvar(name="b")
x = intvar(1,10, shape=3, name="x")

# Constraints
m += (x[0] == 1)
m += AllDifferent(x)
m += b.implies(x[1] + x[2] > 5)

# Objective function (optional)
m.maximize(sum(x) + 100*b)

print(m)
print(m.solve(), x.value(), b.value())
```

See <https://github.com/CPMpy/cpmpy/blob/master/examples/quickstart_sudoku.ipynb> for a more realistic step-by-step example.


## Variables

CPMpy supports discrete decision variables. All variables are numpy arrays, so creating 1 of them or an array/tensor of them is similar:

  * Boolean variables: `boolvar(shape=1, name=None)`
  * Integer variables: `intvar(lb, ub, shape=1, name=None)`

See [the API documentation on variables](api/expressions/variables.html) for more information.


## Constraints
Constraints have to be _added_ to a model, which is done with the `+=` operator, e.g. `model += constraint`. You can also add lists of constraints and other nested expressions.

Using Python's built-in comparison operators `==,!=,<,<=,>,>=` and logical operators `&,|,~,^` on CPMpy variables will automatically create constraint expressions that can be added to a model.

You can also use the built-in arithmetic operators `+,-,*,//,%` and we overwrite the built-in `abs,sum,min,max,all,any` functions so that you can use them in the construction of expressions.

CP languages like CPMpy also offer what is called **global constraints**. Convenient expressions that capture part of the problem structure and that the solvers can typically use efficiently too. A non-exhaustive list of global constraints is: `AllDifferent(), AllEqual(), Circuit(), Table(), Element()`.

See [the API documentation on expressions](api/expressions.html) for more information.


## Objective function

If a model has no objective function specified, then it is a satisfaction problem: the goal is to find out whether a solution, any solution, exists. When an objective function is added, this function needs to be minimized or maximized.

Any CPMpy expression can be added as objective function. Solvers are especially good in optimizing linear functions or the minimum/maximum of a set of expressions. Other (non-linear) expressions are supported too, just give it a try.


## Other uses of the `Model()` object

The `Model()` object has a number of other helpful functions, such as `status()` to print the status of the last `solve()` call, `solveAll()` to find all solutions, `to_file()` to store the model (you can print a model too, for debugging) and `copy` for creating a copy.

See [the API documentation on Model](api/model.html) for more information.

# Solvers

CPMpy can be used as a declarative modeling language: you create a `Model()`, add constraints and call `solve()` on it.

The default solver is ortools CP-SAT, an award winning constraint solver. But CPMpy supports multiple other solvers: a MIP solver (gurobi), SAT solvers (those in PySAT), the Z3 SMT solver, even a knowledge compiler (PySDD) and any CP solver supported by the text-based MiniZinc language.

See the list of solvers known by CPMpy with:

```python
SolverLookup.solvernames()
```

Note that many require additional packages to be installed. For example, try `SolverLookup.get("gurobi")` to see if the commercial gurobi solver is available on your system. See [the API documentation](api/solvers.html) of the solver for installation instructions.

You can specify a solvername when calling `solve()` on a model:

```python
from cpmpy import *
x = intvar(0,10, shape=3)
m = Model()
m += sum(x) <= 5
# use named solver
m.solve(solver="ortools")
```

In this case, a model is a **lazy container**. It simply stores the constraints. Only when `solve()` is called will it instantiate a solver, and send the entire model to it at once. The last line above is equivalent to:
```python
s = SolverLookup.get("ortools", m)
s.solve()
```

## Model versus solver interface
Solver interfaces allow more than the generic model interface, because, well, they can support solver-specific features. Such as solver-specific parameters, passing a previous solution to start from, incremental solving, unsat core extraction, solver-specific callbacks etc.

Importantly, the solver interface supports the same functions as the `Model()` object (for adding constraints, an objective, solve, solveAll, status, ...). So if you want to make use of some features of a solver, simply replace `m = Model()` by `m = SolverLookup.get("your-preferred-solvername")` and your code remains valid. Below, we replace `m` by `s` for readability.

```python
from cpmpy import *
x = intvar(0,10, shape=3)
s = SolverLookup.get("ortools")
s += sum(x) <= 5
# we are operating on the ortools interface here
s.solve()
```

Creating a solver object using an initialized `Model` instance will not alter the `Model` in any way during or after solving. This is especially important when querying the _status_ to get the result of a solve call. For example, in the following, `m.status()` and `s.status()` will not yield the same result!

```python
s = SolverLookup.get("ortools",m)
s.solve()
```
## Setting solver parameters

Now lets use our solver-specific powers: ortools has a parameter _log_search_progress_ that make it show information during solving for example:

```python
# we are operating on the ortools interface here
s.solve(log_search_progress=True)
```

Modern CP-solvers support a variety of hyperparameters. ([OR-tools parameters](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto) for example).
Using the solver interface, any solver parameter can be passed using the `.solve()` call.
These parameters will then be posted to the native solver object before solving the model.

```python
s.solve(cp_model_probing_level = 2,
        linearization_level = 0,
        symmetry_level = 1)
```

See [the API documentation of the solvers](api/solvers.html) for information and links on the parameters supported. See our documentation page on [solver parameters](solver_parameters.html) if you want to tune your hyperparameters automatically. 

## Using solver-specific CPMpy functions

We sometimes add solver-specific features to the CPMpy interface, for convenient access. Two examples of this are `solution_hint()` and `get_core()` which is supported by the OrTools and PySAT solvers and interfaces. Other solvers work very different and do not have these concepts.

`solution_hint()` tells the solver that it could use these variable-values first during search, e.g. typically from a previous solution:
```python
from cpmpy import *
x = intvar(0,10, shape=3)
s = SolverLookup.get("ortools")
s += sum(x) <= 5
# we are operating on a ortools' interface here
s.solution_hint(x, [1,2,3])
s.solve()
print(x.value())
```

`get_core()` asks the solver for an unsatisfiable core, in case a solution did not exist and assumption variables were used. See the documentation on [Unsat core extraction](unsat_core_extraction.html).

See [the API documentation of the solvers](api/solvers.html) to learn about their special functions.


## Incremental solving
It is important to realize that a CPMpy solver interface is _eager_. That means that when a CPMpy constraint is added to a solver object, CPMpy _immediately_ translates it and posts the constraints to the underlying solver.

This has two potential benefits for incremental solving, whereby you add more constraints and variables inbetween solve calls:

  1) CPMpy only translates and posts each constraint once, even if the model is solved multiple times; and 
  2) if the solver itself is incremental then it can reuse any information from call to call, as the state of the native solver object is kept between solver calls and can therefore rely on information derived during a previous `solve` call.

```python
gs = SolverLookup.get("gurobi")

gs += sum(ivar) <= 5 
gs.solve()

gs += sum(ivar) == 3
# the underlying gurobi instance is reused, only the new constraint is added to it.
# gurobi is an incremental solver and will look for solutions starting from the previous one.
gs.solve()
```
 
_Technical note_: ortools its model representation is incremental but its solving itself is not (yet?). Gurobi and the PySAT solvers are fully incremental, as is Z3. The text-based MiniZinc language is not incremental.

## Direct solver access
Some solvers implement more constraints then available in CPMpy. But CPMpy offers direct access to the underlying solver, so there are two ways to post such solver-specific constraints.

### DirectConstraint
The `DirectConstraint` will directly call a function of the underlying solver when added to a CPMpy solver. 

You provide it with the name of the function you want to call, as well as the arguments:

```python
from cpmpy import *
iv = intvar(1,9, shape=3)

s = SolverLookup.get("ortools")

s += AllDifferent(iv)
s += DirectConstraint("AddAllDifferent", iv)  # a DirectConstraint equivalent to the above for OrTools
```

This requires knowledge of the API of the underlying solver, as any function name that you give to it will be called. The only special thing that the DirectConstraint does, is automatically translate any CPMpy variable in the argument to the native solver variable.

Note that any argument given will be checked for whether it needs to be mapped to a native solver variable. This may give errors on complex arguments, or be inefficient. You can tell the `DirectConstraint` not to scan for variables with `noarg` argument, for example:

```python
from cpmpy import *
trans_vars = boolvar(shape=4, name="trans")

s = SolverLookup.get("ortools")

trans_tabl = [ # corresponds to regex 0* 1+ 0+
    (0, 0, 0),
    (0, 1, 1),
    (1, 1, 1),
    (1, 0, 2),
    (2, 0, 2)
]
s += DirectConstraint("AddAutomaton", (trans_vars, 0, [2], trans_tabl),
                      novar=[1, 2, 3])  # optional, what not to scan for vars
```

A minimal example of the DirectConstraint for every supported solver is [in the test suite](https://github.com/CPMpy/cpmpy/tree/master/tests/test_direct.py).

The `DirectConstraint` is a very powerful primitive to get the most out of specific solvers. See the following examples: [nonogram_ortools.ipynb](https://github.com/CPMpy/cpmpy/tree/master/examples/nonogram_ortools.ipynb) using of a helper function that generates automatons with DirectConstraints; [vrp_ortools.py](https://github.com/CPMpy/cpmpy/tree/master/examples/vrp_ortools.ipynb) demonstrating ortools' newly introduced multi-circuit global constraint through DirectConstraint; and [pctsp_ortools.py](https://github.com/CPMpy/cpmpy/tree/master/examples/pctsp_ortools.ipynb) that uses a DirectConstraint to use ortools circuit to post a sub-circuit constraint as needed for this price-collecting TSP variant.

### Directly accessing the underlying solver

The `DirectConstraint("AddAllDifferent", iv)` is equivalent to the following code, which demonstrates that you can mix the use of CPMpy with calling the underlying solver directly: 

```python
from cpmpy import *
iv = intvar(1,9, shape=3)

s = SolverLookup.get("ortools")

s += AllDifferent(iv)  # the traditional way, equivalent to:
s.ort_model.AddAllDifferent(s.solver_vars(iv))  # directly calling the API, has to be with native variables
```

observe how we first map the CPMpy variables to native variables by calling `s.solver_vars()`, and then give these to the native solver API directly.  This is in fact what happens behind the scenes when posting a DirectConstraint, or any CPMpy constraint.

While directly calling the solver offers a lot of freedom, it is a bit more cumbersome as you have to map the variables manually each time. Also, you no longer have a declarative model that you can pass along, print or inspect. In contrast, a `DirectConstraint` is a CPMpy expression so its use is identical to other constraints.

# Setting solver parameters and hyperparameter search

## Calling a solver by name

You can see the list of available solvers (and subsolvers) as follows:

```python
from cpmpy import *

print(SolverLookup.solvernames())
```

On my system, with pysat and minizinc installed, this gives `['ortools', 'minizinc', 'minizinc:chuffed', 'minizinc:coin-bc', ..., 'pysat:minicard', 'pysat:minisat22', 'pysat:minisat-gh']

You can use any of these solvers by passing its name to the `Model.solve()` parameter 'solver' as such:

```python
a,b = boolvar(2)
Model(a|b).solve(solver='minizinc:chuffed')
```

## Setting solver parameters
OR-tools has many solver parameters, [documented here](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto).

CPMpy's interface to ortools accepts keyword arguments to `solve()`, and will set the corresponding or-tools parameters if the name matches. We documented some of the frequent once in our [CPM_ortools API](cpmpy/solvers/ortools.py).

For example, with `model` a CPMpy Model(), you can do the following to make or-tools use 8 parallel cores and print search progress:

```python
from cpmpy import *
from cpmpy.solvers import CPM_ortools

s = CPM_ortools(model)
s.solve(num_search_workers=8, log_search_progress=True)
```


## Hyperparameter search across different parameters
Because CPMpy offers programmatic access to the solver API, hyperparameter search can be straightforwardly done with little overhead between the calls.

The tools directory contains a utility to efficiently search through the hyperparameter space defined by the solvers `tunable_params`.
This utlity is based on the SMBO framework and speeds up the search by implementing adaptive capping.

The parameter tuner is based on the following publication: 
>Ignace Bleukx, Senne Berden, Lize Coenen, Nicholas Decleyre, Tias Guns (2022). Model-Based Algorithm
>Configuration with Adaptive Capping and Prior Distributions. In: Schaus, P. (eds) Integration of Constraint
>Programming, Artificial Intelligence, and Operations Research. CPAIOR 2022. Lecture Notes in Computer Science,
>vol 13292. Springer, Cham. https://doi.org/10.1007/978-3-031-08011-1_6

Solver interfaces not providing the set of tunable parameters can still be tuned by using this utility and providing the parameter (values) yourself.

```python
from cpmpy import *
from cpmpy.tools import ParameterTuner

model = Model(...)

tunables = {
    "search_branching":[0,1,2],
    "linearization_level":[0,1],
    'symmetry_level': [0,1,2]}
defaults = {
    "search_branching": 0,
    "linearization_level": 1,
    'symmetry_level': 2
}

tuner = ParameterTuner("ortools", model, tunables, defaults)
best_params = tuner.tune(max_tries=100)
best_runtime = tuner.best_runtime
```

# Obtaining multiple solutions

CPMpy models and solvers support the `solveAll()` function. It efficiently computes all solutions and optionally displays them. Alternatively, you can manually add blocking clauses as explained in the second half of this page.

When using `solveAll()`, a solver will use an optimized native implementation behind the scenes when that exists.

It has two special named optional arguments:

  * `display=...`: accepts a CPMpy expression, a list of CPMpy expressions or a callback function that will be called on every solution found (default: None)
  * `solution_limit=...`: stop after this many solutions (default: None)

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

`display` Can also take lists of arbitrary CPMpy expressions:
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

In case of multiple variables you should put them in one long python-native list, as such:
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

The goal is to iteratively find solutions that are as diverse as possible with the previous solutions. Many definitions of diversity between solutions exist. We can for example measure the difference between two solutions with the Hamming distance (comparing the number of different values) or the Euclidian distance (compare the absolute difference in value for the variables).

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

As a fun fact, observe how `x != sol` works, even though one is a vector of Boolean variables and sol is Numpy array. However, both have the same length, so this is automatically translated into a pairwise comparison constraint by CPMpy. These numpy-style vectorized operations mean we have to write much less loops while modelling.

To use the Euclidian distance, only the last line needs to be changed. We again use vectorized operations and obtain succinct models. The creation of intermediate variables (with appropriate domains) is done by CPMpy behind the scenes.

```python
    # Euclidian distance: absolute difference in value
    s.maximize(sum([sum( abs(np.add(x, -sol)) ) for sol in store]))
```

## Mixing native callbacks with CPMpy

CPMpy passes arguments to `solve()` directly to the underlying solver object, so you can actually define your own native callbacks and pass them to the solve call.

The following is an example of that, which is actually how the native `solveAll()` for ortools is implemented. You could give it your own custom implemented callback `cb` too.
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



