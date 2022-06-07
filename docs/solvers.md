# Solvers

CPMpy can be used as a declarative modeling language: you create a `Model()`, add constraints and call `solve()` on it.

The default solver is ortools CP-SAT, an award winning constraint solver. But CPMpy supports multiple other solvers: a MIP solver (gurobi), SAT solvers (those in PySAT) and any CP solver supported by the text-based MiniZinc language. 

See the list of solvers known by CPMpy with:

```python
SolverLookup.solvernames()
```

Note that many require additional packages to be installed. For example, try `SolverLookup.get("gurobi")` to see if the commercial gurobi solver is available on your system. See [the API documentation](api/solvers.rst) of the solver for installation instructions.

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
s = Solverlookup.get("ortools", m)
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

See [the API documentation of the solvers](api/solvers.rst) for information and links on the parameters supported. See our documentation page on [solver parameters](solver_parameters.md) if you want to tune your hyperparameters automatically. 

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

`get_core()` asks the solver for an unsatisfiable core, in case a solution did not exist and assumption variables were used. See the documentation on [Unsat core extraction](unsat_core_extraction.md).

See [the API documentation of the solvers](api/solvers.rst) to learn about their special functions.


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
# underlying solver instance is reused, only the new constraint is added to it
# gurobi can start looking for solutions at previous solution
gs.solve()
```
 
_Technical note_: ortools its model representation is incremental but its solving itself is not (yet?). Gurobi and the PySAT solvers are fully incremental. The text-based MiniZinc language is not incremental.

## Native solver access and constraints
Another benefit of using a solver interface directly is access to low level solver features not implemented in CPMpy.
The solver interface implemented by CPMpy encapsulates the native solver object and allows users to access these objects directly.

That means that you can mix posting CPMpy expressions as constraints, and posting __solver-specific global constraints__ directly.

To get you started, the following simple model:
```python
ffrom cpmpy import *
x = intvar(0,10, shape=3)
s = SolverLookup.get("ortools")

s += sum(x) > 10
s += AllDifferent(x)
s += x[1] == 5

s.solve()
print(x.value())
```

can equivalently be written by posting the native `AddAllDifferent()` directly on the underlying ortools object:
```python
from cpmpy import *
x = intvar(0,10, shape=3)
s = SolverLookup.get("ortools")

s += sum(x) > 10
s.ort_model.AddAllDifferent(s.solver_vars(x))
s += x[1] == 5

s.solve()
print(x.value())
```

observe how we first map the CPMpy variables to native variables by calling `s.solver_vars()`, and then give these to the native solver API directly.  This is in fact what happens behind the scenes when posting a constraint.




