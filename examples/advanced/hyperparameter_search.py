"""
Example of gridsearch over solver parameters

Generally applicable, and demonstrated on n-queens with ortools
"""
import sys
from cpmpy import *
from cpmpy.solvers import CPM_ortools, param_combinations
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
