# Solvers

CPMpy can be used as a declarative modeling language: you create a `Model()`, add constraints and call `solve()` on it.

The default solver is OR-Tools CP-SAT, an award winning constraint solver. But CPMpy supports multiple other solvers: a MIP solver (gurobi), SAT solvers (those in PySAT), the Z3 SMT solver, a conflict-driven cutting-planes solver (Exact), even a knowledge compiler (PySDD) and any CP solver supported by the text-based MiniZinc language.

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

`get_core()` asks the solver for an unsatisfiable core, in case a solution did not exist and assumptions were used. See the documentation on [Unsat core extraction](unsat_core_extraction.html).

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





