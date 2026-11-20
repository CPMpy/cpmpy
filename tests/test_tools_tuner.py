import time

import pytest

import cpmpy as cp
from cpmpy.tools import ParameterTuner, GridSearchTuner
from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.solvers.exact import CPM_exact



class TestTuner:

    def test_ortools_tunable_params(self):
        """
        Test that all tunable parameters can be set without errors
        """
        x = cp.intvar(lb=0, ub=5, shape=3)
        model = cp.Model([x[0] < x[1], x[1] < x[2]])
        
        tunable = CPM_ortools.tunable_params()
        
        for param_name, values in tunable.items():
            for val in values:
                solver = CPM_ortools(model)
                assert solver.solve(**{param_name: val})
                assert solver.status() is not None

    def test_ortools(self):
        x = cp.intvar(lb=0, ub=10, shape=10)
        model = cp.Model([
            cp.AllDifferent(x),
        ])

        tuner = ParameterTuner("ortools", model)
        assert tuner.tune(max_tries=100, fix_params={"num_search_workers":1}) is not None
        assert tuner.best_runtime <= tuner.base_runtime

        # run again with grid search tuner
        tuner = GridSearchTuner("ortools", model)
        assert tuner.tune(max_tries=100, fix_params={"num_search_workers": 1}) is not None
        assert tuner.best_runtime <= tuner.base_runtime

    def test_ortools_custom(self):

        x = cp.intvar(lb=0,ub=10, shape=10)
        model = cp.Model([
            cp.AllDifferent(x),
        ])

        tunables = {
            "search_branching":[0,1,2],
            "linearization_level":[0,1]}
        defaults = {
            "search_branching": 0,
            "linearization_level": 1
        }

        tuner = ParameterTuner("ortools", model, tunables, defaults)
        assert tuner.tune(max_tries=100, fix_params={"num_search_workers":1}) is not None
        assert tuner.best_runtime <= tuner.base_runtime

        # run again with grid search tuner
        tuner = GridSearchTuner("ortools", model, tunables, defaults)
        assert tuner.tune(max_tries=100, fix_params={"num_search_workers": 1}) is not None
        assert tuner.best_runtime <= tuner.base_runtime



    def test_no_improvement_returns_defaults(self):
        """If no configuration is accepted (here: no trials at all), the tuner should return the defaults"""
        x = cp.intvar(lb=0, ub=10, shape=10)
        model = cp.Model([
            cp.AllDifferent(x),
        ])

        tunables = {"search_branching": [0, 1, 2]}
        defaults = {"search_branching": 0}

        tuner = ParameterTuner("ortools", model, tunables, defaults)
        assert tuner.tune(max_tries=0, fix_params={"num_search_workers": 1}) == {"search_branching": 0, "num_search_workers": 1}

        tuner = GridSearchTuner("ortools", model, tunables, defaults)
        assert tuner.tune(max_tries=0, fix_params={"num_search_workers": 1}) == {"search_branching": 0, "num_search_workers": 1}
        assert defaults == {"search_branching": 0} # user-provided defaults are not modified

    @pytest.mark.parametrize("tuner_cls", [ParameterTuner, GridSearchTuner])
    def test_fix_params(self, tuner_cls, monkeypatch):
        """Fixed parameters are used in every run (including the baseline) and are not tuned"""
        import cpmpy.tools.tune_solver as tune_solver
        runs = []
        def recording_run_solver(solvername, model, params, time_limit):
            runs.append(dict(params))
            return run_solver(solvername, model, params, time_limit)
        run_solver = tune_solver._run_solver
        monkeypatch.setattr(tune_solver, "_run_solver", recording_run_solver)

        x = cp.intvar(lb=0, ub=10, shape=10)
        model = cp.Model([
            cp.AllDifferent(x),
        ])
        tunables = {"search_branching": [0, 1, 2], "linearization_level": [0, 1]}
        defaults = {"search_branching": 0, "linearization_level": 1}
        fix_params = {"search_branching": 1, "num_search_workers": 1}

        tuner = tuner_cls("ortools", model, tunables, defaults)
        best = tuner.tune(fix_params=fix_params)

        assert len(runs) == 1 + 2 # baseline + only the values of `linearization_level`
        for params in runs:
            assert params["search_branching"] == 1 and params["num_search_workers"] == 1
        assert best["search_branching"] == 1 and best["num_search_workers"] == 1

    def test_ortools_timelimit(self):
        x = cp.intvar(lb=0, ub=10, shape=10)
        model = cp.Model([
            cp.AllDifferent(x),
        ])

        tuner = ParameterTuner("ortools", model)

        start = time.time()
        assert tuner.tune(max_tries=100, fix_params={"num_search_workers":1}) is not None
        end = time.time()

        assert 10 <= 1.05 * end - start
        assert tuner.best_runtime <= tuner.base_runtime


        # run again with grid search tuner
        tuner = GridSearchTuner("ortools", model)

        start = time.time()
        assert tuner.tune(max_tries=100, fix_params={"num_search_workers": 1}) is not None
        end = time.time()

        assert 10 <= 1.05 * end - start
        assert tuner.best_runtime <= tuner.base_runtime

    def test_ortools_multi_model(self):
        """Test tuner with a list of models (MultiSolver path)."""
        models = [
            cp.Model([cp.AllDifferent(cp.intvar(lb=0, ub=5, shape=5))]),
            cp.Model([cp.AllDifferent(cp.intvar(lb=0, ub=6, shape=6))]),
        ]
        tuner = ParameterTuner("ortools", models)
        assert tuner.tune(max_tries=20, fix_params={"num_search_workers": 1}) is not None
        assert tuner.best_runtime <= tuner.base_runtime

        tuner = GridSearchTuner("ortools", models)
        assert tuner.tune(max_tries=20, fix_params={"num_search_workers": 1}) is not None
        assert tuner.best_runtime <= tuner.base_runtime

    @pytest.mark.requires_solver("exact")
    def test_exact_tunable_params(self):
        """
        Test that all tunable parameters can be set without errors
        Exact only accepts parameters at initialisation
        """
        x = cp.intvar(lb=0, ub=5, shape=3)
        model = cp.Model([x[0] < x[1], x[1] < x[2]])

        tunable = CPM_exact.tunable_params()

        for param_name, values in tunable.items():
            for val in values:
                solver = CPM_exact(model, **{param_name: val})
                assert solver.solve()
                assert solver.status() is not None

    @pytest.mark.requires_solver("exact")
    def test_exact(self):
        x = cp.intvar(lb=0, ub=10, shape=10)
        model = cp.Model([
            cp.AllDifferent(x),
        ])

        tuner = ParameterTuner("exact", model)
        assert tuner.tune(max_tries=100) is not None
        assert tuner.best_runtime <= tuner.base_runtime

        # run again with grid search tuner
        tuner = GridSearchTuner("exact", model)
        assert tuner.tune(max_tries=100) is not None
        assert tuner.best_runtime <= tuner.base_runtime

    @pytest.mark.requires_solver("exact")
    def test_exact_multi_model(self):
        """Test tuner with a list of models (MultiSolver path)."""
        models = [
            cp.Model([cp.AllDifferent(cp.intvar(lb=0, ub=5, shape=5))]),
            cp.Model([cp.AllDifferent(cp.intvar(lb=0, ub=6, shape=6))]),
        ]
        tuner = ParameterTuner("exact", models)
        assert tuner.tune(max_tries=20) is not None
        assert tuner.best_runtime <= tuner.base_runtime

        tuner = GridSearchTuner("exact", models)
        assert tuner.tune(max_tries=20) is not None
        assert tuner.best_runtime <= tuner.base_runtime
