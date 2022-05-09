# Accessing a solver

CPMpy provides interfaces to constraint solver using subclasses of the `SolverInterface`.
These classes encapsulate the native solver object and provide the necessary translations of constraints and variables.
A solver can either be called implicitly using the `solver` parameter of the `Model.solve()` method, or explicitely by constructing the CPMpy solverinterface directly.
The solver interface is constructed by invoking the `SolverLookup` helper object.

```python
from cpmpy import *

ivar = intvar(0,10, shape=3)

m = Model()
m += sum(ivar) <= 5

# access solver implicitly
m.solve(solver="minizinc")

# create solver object directly
s = SolverLookup.get("minizinc", m)
s.solve()
```

To get a list of all solver supported by your system, use the `SolverLookup.solvernames()` method.
```python
SolverLookup.solvernames()
>> ['ortools', 'minizinc', 'minizinc:chuffed', 'minizinc:coin-bc', 'minizinc:cplex', 'minizinc:gecode', 'minizinc:gurobi', 'minizinc:or-tools', 'minizinc:scip', 'minizinc:xpress']
```
Any of these solvernames can be passed to the `SolverLookup.get()` method to instantiate a solver interface.

## Using a solver as model
Once a solver interface is instantiated, it can be used just like a model. I.e, add constraints to the solver, post an objective...

Using a solver interface as a model has some benefits in terms of performance. This is certainly the case when using an incremental solver like Gurobi.
When solving a model, the solverinterface will always be rebuild from scratch, however, when using the solver interface directly as a model, this is not necessary.
The state of the native solver object is kept between solver calls and can therefore rely on information derived during a previous `solve` call.

```python
gs=  SolverLookup.get("gurobi")

gs += sum(ivar) <= 5 
gs.solve()

gs += sum(ivar) == 3
# internal state has not changed
# gurobi can start looking for solutions at previous solution
gs.solve()
```

## Setting solver parameters
Modern CP-solvers support a variety of hyperparameters. ([OR-tools parameters](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto) for example).
Using the solver interface, any solver parameter can be passed using the `.solve()` call.
These parameters will then be posted to the native solver object before solving the model.

```python
s = SolverLookup.get("ortools")
s += sum(ivar) <= 5

s.solve(cp_model_probing_level = 2,
        linearization_level = 0,
        symmetry_level = 1)
```


## Native solver access
Another benefit of using a solver interface directly is access to low level solver features not implemented in CPMpy.
The solver interface implemented by CPMpy encapsulates (a) native solver object(s) and allows users to access these objects directly.

For a full example take a look at [the advanced examples](/examples/advanced/ortools_presolve_propagate.py).




