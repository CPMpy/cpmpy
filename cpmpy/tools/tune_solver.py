"""
    This file implements parameter tuning for constraint solvers based on SMBO and using adaptive capping.
    Based on the following paper:
    Ignace Bleukx, Senne Berden, Lize Coenen, Nicholas Decleyre, Tias Guns (2022). Model-Based Algorithm
    Configuration with Adaptive Capping and Prior Distributions. In: Schaus, P. (eds) Integration of Constraint
    Programming, Artificial Intelligence, and Operations Research. CPAIOR 2022. Lecture Notes in Computer Science,
    vol 13292. Springer, Cham. https://doi.org/10.1007/978-3-031-08011-1_6

    - DOI: https://doi.org/10.1007/978-3-031-08011-1_6
    - Link to paper: https://rdcu.be/cQyWR
    - Link to original code of paper: https://github.com/ML-KULeuven/DeCaprio

    This code currently only implements the author's 'Hamming' surrogate function.
    The parameter tuner iteratively finds better hyperparameters close to the current best configuration during the search.
    Searching and time-out start at the default configuration for a solver (if available in the solver class)
"""
import math
import time
from random import shuffle
import numpy as np
from ..solvers.utils import SolverLookup, param_combinations
from ..solvers.solver_interface import ExitStatus, SolverInterface, SolverStatus


class ParameterTuner:
    """
        Parameter tuner based on DeCaprio method [ref_to_decaprio]
    """

    def __init__(self, solvername, model, all_params=None, defaults=None):
        """            
            :param solvername: Name of solver to tune
            :param model: CPMpy model to tune parameters on
            :param all_params: optional, dictionary with parameter names and values to tune. If None, use predefined parameter set.
            :param defaults: required ONLY IF all_params is set, dictionary with corresponding default parameter values.
        """
        self.solvername = solvername
        self.model = model
        if all_params is None:
            self.all_params = SolverLookup.lookup(solvername).tunable_params()
            self.best_params = SolverLookup.lookup(solvername).default_params()
        else:
            self.all_params = all_params
            assert defaults is not None, "ParameterTuner: if 'all_params' is set then 'defaults' must too"
            self.best_params = defaults

        self._param_order = list(self.all_params.keys())
        self._best_config = self._params_to_np([self.best_params])[0]

    def tune(self, time_limit=None, max_tries=None, fix_params={}, verbose=1):
        """
            :param time_limit: Time budget to run tuner in seconds. Solver will be interrupted when time budget is exceeded
            :param max_tries: Maximum number of configurations to test
            :param fix_params: Non-default parameters to run solvers with.
            :param verbose: how much information to print (0=none)
        """
        if time_limit is not None:
            start_time = time.time()

        # Init solver
        if verbose >= 1:
            print(f"Running {self.solvername} with default parameters")
        if not isinstance(self.model, list):
            solver = SolverLookup.get(self.solvername, self.model)
        else:
            solver = MultiSolver(self.solvername, self.model)
        solver.solve(**self.best_params, time_limit=time_limit)
        if not _has_finished(solver):
            raise TimeoutError("Time's up before solving init solver call")

        self.base_runtime = solver.status().runtime
        self.best_runtime = self.base_runtime
        self._best_config = {}
        if verbose >= 1:
            print(f" - took {self.best_runtime:.1f} seconds")



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
            if not isinstance(self.model, list):
                solver = SolverLookup.get(self.solvername, self.model)
            else:
                solver = MultiSolver(self.solvername, self.model)
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
            params_dict.update(fix_params)
            timeout = self.best_runtime
            # set timeout depending on time budget
            if time_limit is not None:
                if (time.time() - start_time) >= time_limit:
                    break
                timeout = min(timeout, time_limit - (time.time() - start_time))

            if verbose >= 1:
                print(f"Starting trial {i+1}/{max_tries}, cap: {timeout:.1f}s  -- remaining configs: {len(combos_np)}" + f" budget: {time_limit-(time.time()-start_time):.1f}s" if time_limit else "")
            # run solver
            solver.solve(**params_dict, time_limit=timeout)
            if _has_finished(solver):
                self.best_runtime = solver.status().runtime
                # update surrogate
                self._best_config = params_np
                if verbose >= 1:
                    print(f" - new best runtime {self.best_runtime}")
                if verbose >= 2:
                    print(f" - new best params {self._np_to_params(self._best_config)}")

            i += 1

        self.best_params = self._np_to_params(self._best_config)
        self.best_params.update(fix_params)
        if verbose >= 2:
            print("Best runtime: {self.best_runtime} for params {self.best_params}")

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
    """
        Grid search parameter tuner that exhaustively tests all parameter combinations.
        Inherits from ParameterTuner but uses a simple grid search strategy
    """

    def __init__(self, solvername, model, all_params=None, defaults=None):
        """
            Initialize a grid search parameter tuner.
            
            :param solvername: Name of solver to tune
            :param model: CPMpy model to tune parameters on
            :param all_params: optional, dictionary with parameter names and values to tune. If None, use predefined parameter set.
            :param defaults: required ONLY IF all_params is set, dictionary with corresponding default parameter values.
        """
        super().__init__(solvername, model, all_params, defaults)

    def tune(self, time_limit=None, max_tries=None, fix_params={}, verbose=1):
        """
            :param time_limit: Time budget to run tuner in seconds. Solver will be interrupted when time budget is exceeded
            :param max_tries: Maximum number of configurations to test
            :param fix_params: Non-default parameters to run solvers with.
            :param verbose: how much information to print (0=none)
        """
        if time_limit is not None:
            start_time = time.time()

        # Init solver
        if verbose >= 1:
            print(f"Running {self.solvername} with default parameters")
        if not isinstance(self.model, list):
            solver = SolverLookup.get(self.solvername, self.model)
        else:
            solver = MultiSolver(self.solvername, self.model)
        solver.solve(**self.best_params,time_limit=time_limit)
        if not _has_finished(solver):
            raise TimeoutError("Time's up before solving init solver call")


        self.base_runtime = solver.status().runtime
        self.best_runtime = self.base_runtime
        if verbose >= 1:
            print(f" - took {self.best_runtime:.1f} seconds")
        # Get all possible hyperparameter configurations
        combos = list(param_combinations(self.all_params))
        shuffle(combos) # test in random order

        if max_tries is not None:
            combos = combos[:max_tries]

        for i, params_dict in enumerate(combos):
            # Make new solver
            if not isinstance(self.model, list):
                solver = SolverLookup.get(self.solvername, self.model)
            else:
                solver = MultiSolver(self.solvername, self.model)
            # set fixed params
            params_dict.update(fix_params)
            timeout = self.best_runtime
            # set timeout depending on time budget
            if time_limit is not None:
                if (time.time() - start_time) >= time_limit:
                    break
                timeout = min(timeout, time_limit - (time.time() - start_time))
            
            if verbose >= 1:
                print(f"Starting trial {i+1}/{len(combos)}, cap: {timeout:.1f}s" + f" budget: {time_limit-(time.time()-start_time):.1f}s" if time_limit else "")
            # run solver
            solver.solve(**params_dict, time_limit=timeout)
            if _has_finished(solver):
                self.best_runtime = solver.status().runtime
                # update surrogate
                self.best_params = params_dict
                if verbose >= 1:
                    print(f" - new best runtime {self.best_runtime:.1f}")
                if verbose >= 2:
                    print(f" - new best params {self.best_params}")

            if time_limit is not None and (time.time() - start_time) >= time_limit:
                break

        if verbose >= 2:
            print(f"Best runtime: {self.best_runtime:.1f} for params {self.best_params}")

        return self.best_params

def _has_finished(solver):
    """
        Check whether a given solver has found the target solution.
        Parameters
        ----------
        solver : SolverInterface

        Returns
        -------
        bool
            True if the solver has has found the target solution. This means:
            - For a `MultiSolver`: its own `has_finished()` method determines completion.
            - For a problem with an objective: status is OPTIMAL.
            - For a problem without an objective: status is FEASIBLE.
            - For an unsat problem: status is UNSATISFIABLE.
            False otherwise.
        """
    if isinstance(solver,MultiSolver):
        return solver.has_finished()
    elif (((solver.has_objective() and solver.status().exitstatus == ExitStatus.OPTIMAL) or
          (not solver.has_objective() and solver.status().exitstatus == ExitStatus.FEASIBLE)) or
          (solver.status().exitstatus == ExitStatus.UNSATISFIABLE)):
        return True
    return False



class MultiSolver(SolverInterface):
    """
    Class that manages multiple solver instances.
    Attributes
    ----------
    name : str
        Name of the solver used for all instances.
    solvers : list of SolverInterface
        The solver instances corresponding to each model.
    cpm_status : SolverStatus
        Aggregated solver status. Tracks runtime and per-solver exit statuses.
    """

    def __init__(self,solvername,models):
        """
        Initialize a MultiSolver with the given list of solvers.
        Parameters
        ----------
        solvername : str
            Name of the solver backend (e.g., "ortools", "gurobi").
        models : list of Model
            The models to create solver instances for.
        """

        self.name = solvername
        self.solvers = []
        for mdl in models:
            self.solvers.append(SolverLookup.get(solvername,mdl))

    def solve(self, time_limit=None, **kwargs):
        """
        Solve the models sequentially using the solvers.

        Parameters
        ----------
        time_limit :
            Global time limit in seconds for all solvers combined.
        **kwargs : dict
            Additional arguments passed to each solve method.

        Returns
        -------
        bool
            True if all solvers returned a solution, False otherwise.
        """
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.exitstatus = [ExitStatus.NOT_RUN] * len(self.solvers)
        all_has_sol = True
        # initialize exitstatus list
        init_start = time.time()
        for i, s in enumerate(self.solvers):
            # call solver
            start = time.time()
            has_sol = s.solve(time_limit=time_limit, **kwargs)
            # update only the current solver's exitstatus
            self.cpm_status.exitstatus[i] = s.status().exitstatus
            if time_limit is not None:
                time_limit = time_limit - (time.time() - start)
                if time_limit <= 0:
                    break
            all_has_sol = all_has_sol and has_sol
        end = time.time()
        # update runtime
        self.cpm_status.runtime = end - init_start
        return all_has_sol

    def has_finished(self):
        """
        Check whether all solvers in the MultiSolver have finished.

        A solver is considered finished if:
        - It has an objective and reached OPTIMAL, or
        - It has no objective and reached FEASIBLE, or
        - It reached UNSATISFIABLE.

        Returns
        -------
        bool
            True if all solvers have finished, False otherwise.
        """
        all_have_finished = True
        for s in self.solvers:
            finished = ((s.has_objective() and s.status().exitstatus == ExitStatus.OPTIMAL) or
                        (not s.has_objective() and s.status().exitstatus == ExitStatus.FEASIBLE) or
                        (s.status().exitstatus == ExitStatus.UNSATISFIABLE))
            all_have_finished =  all_have_finished and finished
        return all_have_finished


