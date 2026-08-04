#!/usr/bin/env python
"""
Example of gridsearch over solver parameters

Generally applicable, and demonstrated on n-queens with ortools
"""
import cpmpy as cp
from cpmpy.tools.tune_solver import ParameterTuner
import numpy as np

def main():
    model = nqueens(n=125)

    # a selection of parameters, see docs of solver to see which parameters are available for tuning
    all_params = {
        'cp_model_probing_level': [0, 1, 2, 3],
        'linearization_level': [0, 1, 2],
        'symmetry_level': [0, 1, 2],
    }
    
    # defaults are required if all_params is custom
    defaults = {
        'cp_model_probing_level': 2,
        'linearization_level': 1,
        'symmetry_level': 2
    }

    # initialize tuner instance
    tuner = ParameterTuner(
        solvername="ortools", 
        model=model, 
        all_params=all_params, 
        defaults=defaults
    )

    
    # run tuner with 60s total budget
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
        cp.AllDifferent(queens + np.arange(n)),
        cp.AllDifferent(queens - np.arange(n)),
    )


if __name__ == '__main__':
    main()