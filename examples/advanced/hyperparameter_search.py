"""
Example of gridsearch over solver parameters

Generally applicable, and demonstrated on n-queens with ortools
"""
# import cpmpy as cp
# from cpmpy.solvers import param_combinations
# from cpmpy.transformations.flatten_model import flatten_model

# def main():
#     model = nqueens(n=50)
#     # flatten once upfront, reduces overhead of multiple solves
#     model = flatten_model(model)

#     # a selection of parameters, see docs of cpmpy.solvers.ortools
#     all_params = {'cp_model_probing_level': [0,1,2,3],
#                   'linearization_level': [0,1,2],
#                   'symmetry_level': [0,1,2]
#                   }
    
#     configs = [] # (runtime, param)
#     for params in param_combinations(all_params):
#         print("Running with", params)
#         s = cp.SolverLookup.get("ortools", model)
#         s.solve(**params)
#         print(s.status())

#         # store
#         configs.append( (s.status().runtime, params) )

#     configs = sorted(configs) # sort by runtime

#     print()
#     best = configs[0]
#     print("Fastest in", round(best[0],2), "seconds, config:", best[1])
#     print("Comparing best -- worst:", round(configs[0][0],2), "--", round(configs[-1][0],2))

#     s = cp.SolverLookup.get("ortools", model)
#     s.solve()
#     print("With default parameters:", round(s.status().runtime,2))

#     # Outputs:
#     # Fastest in 0.01 seconds, config: {'cp_model_probing_level': 0, 'linearization_level': 2, 'symmetry_level': 0}
#     # Comparing best -- worst: 0.05 -- 0.24
#     # With default parameters: 0.16



# def nqueens(n=8):
#     """ N-queens problem
#     """
#     queens = cp.intvar(1,n, shape=n)
#     return cp.Model(
#              cp.AllDifferent(queens),
#              cp.AllDifferent([queens[i] + i for i in range(n)]),
#              cp.AllDifferent([queens[i] - i for i in range(n)]),
#            )


# if __name__ == '__main__':
#     main()


import cpmpy as cp
from cpmpy.transformations.flatten_model import flatten_model
from cpmpy.tools.tune_solver import ParameterTuner, GridSearchTuner

def main():
    model = nqueens(n=50)

    # a selection of parameters, see docs of cpmpy.solvers.ortools
    all_params = {
        'cp_model_probing_level': [0, 1, 2, 3],
        'linearization_level': [0, 1, 2],
        'symmetry_level': [0, 1, 2]
    }
    
    # Defaults are required by ParameterTuner if all_params is custom
    defaults = {
        'cp_model_probing_level': 2,
        'linearization_level': 1,
        'symmetry_level': 2
    }

    # initialize tuner; can use ParameterTuner (SMBO-based) or GridSearchTuner (Exhaustive)
    tuner = GridSearchTuner(
        solvername="ortools", 
        model=model, 
        all_params=all_params, 
        defaults=defaults
    )

    
    # 4. Run Tuning
    # time_limit: total budget for the whole process
    # max_tries: limit number of configs to test
    best_config = tuner.tune(time_limit=60, verbose=1)

    print("\n--- Tuning Results ---")
    print(f"Best Configuration found: {best_config}")
    print(f"Best Runtime: {round(tuner.best_runtime, 4)} seconds")
    print(f"Default Runtime (Initial): {round(tuner.base_runtime, 4)} seconds")


def nqueens(n=8):
    """ N-queens problem """
    queens = cp.intvar(1, n, shape=n)
    return cp.Model(
        cp.AllDifferent(queens),
        cp.AllDifferent([queens[i] + i for i in range(n)]),
        cp.AllDifferent([queens[i] - i for i in range(n)]),
    )


if __name__ == '__main__':
    main()