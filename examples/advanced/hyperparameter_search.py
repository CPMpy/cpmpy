"""
Example of gridsearch over solver parameters

Generally applicable, and demonstrated on n-queens with ortools
"""
import sys
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from cpmpy.transformations.flatten_model import flatten_model

def main():
    model = nqueens(n=20)

    # a selection of parameters, see docs of cpmpy.solvers.ortools
    params = {'cp_model_probing_level': [0,1,2,3],
              'linearization_level': [0,1,2],
              'symmetry_level': [0,1,2]}

    configs = gridsearch(model, CPM_ortools, params, verbose=True)

    print()
    best = configs[0]
    print("Best config:", best[1])
    print("\t with runtime:", round(best[0],2))
    print("Comparing best -- worst:", round(configs[0][0],2), "--", round(configs[-1][0],2))

    s = CPM_ortools(model); s.solve()
    print("With default parameters:", round(s.status().runtime,2))

    # Outputs:
    # Beste config: {'cp_model_probing_level': 0, 'linearization_level': 0, 'symmetry_level': 1}
    #     with runtime: 0.03
    # Comparing best -- worst: 0.03 -- 0.19
    # With default parameters: 0.12


def gridsearch(model, solver_class, all_params, verbose=False):
    """
        Perform gridsearch with `solver_class` and parameter choices `all_params` on `model`

        Arguments:
        - model: a `Model()`, it will be flattened once to save time
        - solver_class: a __class__ object of a CPMpy solver, e.g. CPM_ortools, without `()`!
        - all_params: a dict with keys = parameters, values = list of possible values for that parameter
        - verbose: prints status of every solve if True

        It will try all combinations of parameter values (full grid search) and
        returns an __ordered__ list of [(runtime, param)] so that the first element was the best one
    """
    # flatten once upfront, reduces CPMpy's overhead
    model = flatten_model(model)
    
    remaining_keys = list(all_params.keys())
    cur_params = dict()
    allresults = _do_gridsearch(model, solver_class, all_params,
                                remaining_keys=remaining_keys, cur_params=cur_params,
                                verbose=verbose)
    return sorted(allresults)
    

def _do_gridsearch(model, solverc, all_params, remaining_keys, cur_params, verbose=False):
    """
        Recursive gridsearch function, instantiates and calls solver at most inner level

        Arguments:
        - model: a `Model()`, it will be flattened once to save time
        - solver_class: a __class__ object of a CPMpy solver, e.g. CPM_ortools, without `()`!
        - all_params: a dict with keys = parameters, values = list of possible values for that parameter
        - remaining_keys: list of keys of `all_params` that have not been explored yet
        - cur_params: dict with some parameters already set, will be extended
        - verbose: prints status of every solve if True
    """
    if len(remaining_keys) == 0:
        # end of recursive stack, run solver
        s = solverc(model)
        if verbose:
            print("Running",s.name,"with",cur_params)
        s.solve(**cur_params)
        if verbose:
            print(s.status())
        return [(s.status().runtime, dict(cur_params))] # copies the params
    
    cur_key = remaining_keys[0]
    myresults = [] # (runtime, cur_params)
    for cur_value in all_params[cur_key]:
        cur_params[cur_key] = cur_value
        myresults += _do_gridsearch(model, solverc, all_params,
                                    remaining_keys=remaining_keys[1:],
                                    cur_params=cur_params,
                                    verbose=verbose)

    return myresults


def nqueens(n=8):
    """ N-queens problem
    """
    queens = intvar(1,n, shape=n)
    return Model(
             AllDifferent(queens),
             AllDifferent([queens[i] + i for i in range(n)]),
             AllDifferent([queens[i] - i for i in range(n)]),
           )


if __name__ == '__main__':
    main()
