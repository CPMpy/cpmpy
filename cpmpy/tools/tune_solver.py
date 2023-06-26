"""
    This file implements parameter tuning for constraint solvers based on SMBO and using adaptive capping.
    Based on the following paper:
    Ignace Bleukx, Senne Berden, Lize Coenen, Nicholas Decleyre, Tias Guns (2022). Model-Based Algorithm
    Configuration with Adaptive Capping and Prior Distributions. In: Schaus, P. (eds) Integration of Constraint
    Programming, Artificial Intelligence, and Operations Research. CPAIOR 2022. Lecture Notes in Computer Science,
    vol 13292. Springer, Cham. https://doi.org/10.1007/978-3-031-08011-1_6

    DOI: https://doi.org/10.1007/978-3-031-08011-1_6
    Link to paper: https://rdcu.be/cQyWR
    Link to original code of paper: https://github.com/ML-KULeuven/DeCaprio

    This code currently only implements the author's 'Hamming' surrogate function.
    The parameter tuner iteratively finds better hyperparameters close to the current best configuration during the search.
    Searching and time-out start at the default configuration for a solver (if available in the solver class)
"""
import time
from random import shuffle

import numpy as np

from ..solvers.utils import SolverLookup, param_combinations
from ..solvers.solver_interface import ExitStatus

class ParameterTuner:

    def __init__(self, solvername, model, all_params=None, defaults=None):
        """
            Parameter tuner based on DeCaprio method [ref_to_decaprio]
            :param: solvername: Name of solver to tune
            :param: model: CPMpy model to tune parameters on
            :param: all_params: optional, dictionary with parameter names and values to tune. If None, use predefined parameter set.
        """
        self.solvername = solvername
        self.model = model
        self.all_params = all_params
        self.best_params = defaults
        if self.all_params is None:
            self.all_params = SolverLookup.lookup(solvername).tunable_params()
            self.best_params = SolverLookup.lookup(solvername).default_params()

        self._param_order = list(self.all_params.keys())
        self._best_config = self._params_to_np([self.best_params])

    def tune(self, time_limit=None, max_tries=None, fix_params={}):
        """
            :param: time_limit: Time budget to run tuner in seconds. Solver will be interrupted when time budget is exceeded
            :param: max_tries: Maximum number of configurations to test
            :param: fix_params: Non-default parameters to run solvers with.
        """
        if time_limit is not None:
            start_time = time.time()

        # Init solver
        solver = SolverLookup.get(self.solvername, self.model)
        solver.solve(**self.best_params)

        self.base_runtime = solver.status().runtime
        self.best_runtime = self.base_runtime

        # Get all possible hyperparameter configurations
        combos = list(param_combinations(self.all_params))
        combos_np = self._params_to_np(combos)

        # Ensure random start
        np.random.shuffle(combos_np)

        i = 0
        if max_tries is None:
            max_tries = len(combos_np)
        while len(combos_np) and i < max_tries:
            # Make new solver
            solver = SolverLookup.get(self.solvername, self.model)
            # Apply scoring to all combos
            scores = self._get_score(combos_np)
            max_idx = np.where(scores == scores.min())[0][0]
            # Get index of optimal combo
            params_np = combos_np[max_idx]
            # Remove optimal combo from combos
            combos_np = np.delete(combos_np, max_idx, axis=0)
            # Convert numpy array back to dictionary
            params_dict = self._np_to_params(params_np)
            # set fixed params
            params_dict |= fix_params
            timeout = self.best_runtime
            # set timeout depending on time budget
            if time_limit is not None:
                timeout = min(timeout, time_limit - (time.time() - start_time))
            # run solver
            solver.solve(**params_dict, time_limit=timeout)
            if solver.status().exitstatus == ExitStatus.OPTIMAL and  solver.status().runtime < self.best_runtime:
                self.best_runtime = solver.status().runtime
                # update surrogate
                self._best_config = params_np

            if time_limit is not None and (time.time() - start_time) >= time_limit:
                break
            i += 1

        self.best_params = self._np_to_params(self._best_config)
        self.best_params |= fix_params
        return self.best_params

    def _get_score(self, combos):
        """
            Return the hamming distance for each remaining configuration to the current best config.
            Lower score means better configuration, so exploit the current best configuration by only allowing small changes.
        """
        return np.count_nonzero(combos != self._best_config, axis=1)

    def _params_to_np(self,combos):
        arr = [[params[key] for key in self._param_order] for params in combos]
        return np.array(arr)

    def _np_to_params(self,arr):
        return {key: val for key, val in zip(self._param_order, arr)}



class GridSearchTuner(ParameterTuner):

    def __init__(self, solvername, model, all_params=None, defaults=None):
        super().__init__(solvername, model, all_params, defaults)

    def tune(self, time_limit=None, max_tries=None, fix_params={}):
        """
            :param: time_limit: Time budget to run tuner in seconds. Solver will be interrupted when time budget is exceeded
            :param: max_tries: Maximum number of configurations to test
            :param: fix_params: Non-default parameters to run solvers with.
        """
        if time_limit is not None:
            start_time = time.time()

        # Init solver
        solver = SolverLookup.get(self.solvername, self.model)
        solver.solve(**self.best_params)


        self.base_runtime = solver.status().runtime
        self.best_runtime = self.base_runtime

        # Get all possible hyperparameter configurations
        combos = list(param_combinations(self.all_params))
        shuffle(combos) # test in random order

        if max_tries is not None:
            combos = combos[:max_tries]

        for params_dict in combos:
            # Make new solver
            solver = SolverLookup.get(self.solvername, self.model)
            # set fixed params
            params_dict |= fix_params
            timeout = self.best_runtime
            # set timeout depending on time budget
            if time_limit is not None:
                timeout = min(timeout, time_limit - (time.time() - start_time))
            # run solver
            solver.solve(**params_dict, time_limit=timeout)
            if solver.status().exitstatus == ExitStatus.OPTIMAL and solver.status().runtime < self.best_runtime:
                self.best_runtime = solver.status().runtime
                # update surrogate
                self.best_params = params_dict

            if time_limit is not None and (time.time() - start_time) >= time_limit:
                break

        return self.best_params



