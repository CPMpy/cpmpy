# Setting solver parameters and hyperparameter search

CPMpy offers direct solver access. For most cases, including setting solver parameters, access to CPMpy's solver API will be sufficient.

In the following, we will use the [or-tools CP-SAT solver](cpmpy/solvers/ortools.py>). The corresponding CPMpy class is `CPM_ortools` and can be included as follows:

```python
   from cpmpy.solvers import CPM_ortools
```

The same principles will apply to the other solver interfaces too.


## Setting solver parameters
or-tools has many solver parameters, [documented here](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto]).

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

The full example is in [examples/advanced/hyperparameter_search.py](examples/advanced/hyperparameter_search.py), here is a relevant excrept:

```python
    params = {'cp_model_probing_level': [0,1,2,3],
              'linearization_level': [0,1,2],
              'symmetry_level': [0,1,2]}

    configs = gridsearch(model, CPM_ortools, params)

    best = configs[0]
    print("Best config:", best[1])
    print("    with runtime:", round(best[0],2))
```
