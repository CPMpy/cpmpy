import time
from unittest import TestCase

from cpmpy import *
from cpmpy.tools import ParameterTuner, GridSearchTuner



class TestTuner:

    def test_ortools(self):
        x = intvar(lb=0, ub=10, shape=10)
        model = Model([
            AllDifferent(x),
        ])

        tuner = ParameterTuner("ortools", model)
        assert tuner.tune(max_tries=100, fix_params={"num_search_workers":1}) is not None
        assert tuner.best_runtime <= tuner.base_runtime

        # run again with grid search tuner
        tuner = GridSearchTuner("ortools", model)
        assert tuner.tune(max_tries=100, fix_params={"num_search_workers": 1}) is not None
        assert tuner.best_runtime <= tuner.base_runtime

    def test_ortools_custom(self):

        x = intvar(lb=0,ub=10, shape=10)
        model = Model([
            AllDifferent(x),
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
        x = intvar(lb=0, ub=10, shape=10)
        model = Model([
            AllDifferent(x),
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
