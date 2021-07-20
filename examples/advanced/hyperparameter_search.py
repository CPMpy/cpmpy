"""
Example of gridsearch over solver parameters

Generally applicable, and demonstrated on n-queens with ortools
"""
import sys
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from cpmpy.transformations.flatten_model import flatten_model

def main():
    model = nqueens(n=25)
    # flatten once upfront, reduces overhead of multiple solves
    model = flatten_model(model)

    # a selection of parameters, see docs of cpmpy.solvers.ortools
    all_params = {'cp_model_probing_level': [0,1,2,3],
                  'linearization_level': [0,1,2],
                  'symmetry_level': [0,1,2]
                  }
    
    configs = [] # (runtime, param)
    for params in param_combinations(all_params):
        print("Running with", params)
        s = CPM_ortools(model)
        s.solve(**params)
        print(s.status())

        # store
        configs.append( (s.status().runtime, params) )

    configs = sorted(configs) # sort by runtime

    print()
    best = configs[0]
    print("Fastest in", round(best[0],2), "seconds, config:", best[1])
    print("Comparing best -- worst:", round(configs[0][0],2), "--", round(configs[-1][0],2))

    s = CPM_ortools(model); s.solve()
    print("With default parameters:", round(s.status().runtime,2))

    # Outputs:
    # Fastest in 0.02 seconds, config: {'cp_model_probing_level': 0, 'linearization_level': 0, 'symmetry_level': 1}
    # Comparing best -- worst: 0.02 -- 0.2
    # With default parameters: 0.13


def param_combinations(all_params, remaining_keys=None, cur_params=None):
    """
        Recursively yield all combinations of param values

        - all_params is a dict of {key: list} items, e.g.:
            {'val': [1,2], 'opt': [True,False]}

        - output is an generator over all {key:value} combinations
          of the keys and values. For the example above:
          generator([{'val':1,'opt':True},{'val':1,'opt':False},{'val':2,'opt':True},{'val':2,'opt':False}])
    """
    if remaining_keys is None or cur_params is None:
        # init
        remaining_keys = list(all_params.keys())
        cur_params = dict()

    cur_key = remaining_keys[0]
    myresults = [] # (runtime, cur_params)
    for cur_value in all_params[cur_key]:
        cur_params[cur_key] = cur_value
        if len(remaining_keys) == 1:
            # terminal, return copy
            yield dict(cur_params)
        else:
            # recursive call
            yield from param_combinations(all_params, 
                            remaining_keys=remaining_keys[1:],
                            cur_params=cur_params)



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
