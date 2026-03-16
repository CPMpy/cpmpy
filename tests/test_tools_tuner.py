import time

import cpmpy as cp
from cpmpy.tools import ParameterTuner, GridSearchTuner
from cpmpy.solvers.ortools import CPM_ortools



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
