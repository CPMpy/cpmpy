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


## Direct solver access

CPMpy also offers direct access to its solver API, as well as to the underlying native solver API. For most cases, including setting solver parameters, access to CPMpy's solver API will be sufficient.

In the following, we will use the [or-tools CP-SAT solver](cpmpy/solvers/ortools.py). The corresponding CPMpy class is `CPM_ortools` and can be included as follows:

```python
   from cpmpy.solvers import CPM_ortools
```

The same principles will apply to the other solver interfaces too.


## Setting solver parameters
or-tools has many solver parameters, [documented here](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto).

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

The cpmpy.solvers module has a helper function `param_combinations` that generates all parameter combinations of an input, which can then be looped over.

The example is in [examples/advanced/hyperparameter_search.py](examples/advanced/hyperparameter_search.py), the key part is:

```python
    from cpmpy.solvers import CPM_ortools, param_combinations

    params = {'cp_model_probing_level': [0,1,2,3],
              'linearization_level': [0,1,2],
              'symmetry_level': [0,1,2]}

    for params in param_combinations(all_params):
        s = CPM_ortools(model)
        s.solve(**params)
        print(s.status().runtime, "seconds for config", params)
```
