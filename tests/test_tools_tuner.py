import time
from utils import TestCase

import pytest

import cpmpy as cp
from cpmpy.tools import ParameterTuner, GridSearchTuner

@pytest.mark.usefixtures("solver")
@pytest.mark.requires_solver("ortools")
class TunerTests(TestCase):

    def test_ortools(self):
        x = cp.intvar(lb=0, ub=10, shape=10)
        model = cp.Model([
            cp.AllDifferent(x),
        ])

        tuner = ParameterTuner("ortools", model)
        self.assertIsNotNone(tuner.tune(max_tries=100, fix_params={"num_search_workers":1}))
        self.assertLessEqual(tuner.best_runtime, tuner.base_runtime)

        # run again with grid search tuner
        tuner = GridSearchTuner("ortools", model)
        self.assertIsNotNone(tuner.tune(max_tries=100, fix_params={"num_search_workers": 1}))
        self.assertLessEqual(tuner.best_runtime, tuner.base_runtime)

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
        self.assertIsNotNone(tuner.tune(max_tries=100, fix_params={"num_search_workers":1}))
        self.assertLessEqual(tuner.best_runtime, tuner.base_runtime)

        # run again with grid search tuner
        tuner = GridSearchTuner("ortools", model, tunables, defaults)
        self.assertIsNotNone(tuner.tune(max_tries=100, fix_params={"num_search_workers": 1}))
        self.assertLessEqual(tuner.best_runtime, tuner.base_runtime)



    def test_ortools_timelimit(self):
        x = cp.intvar(lb=0, ub=10, shape=10)
        model = cp.Model([
            cp.AllDifferent(x),
        ])

        tuner = ParameterTuner("ortools", model)

        start = time.time()
        self.assertIsNotNone(tuner.tune(max_tries=100, fix_params={"num_search_workers":1}))
        end = time.time()

        self.assertLessEqual(10, 1.05 * end - start)
        self.assertLessEqual(tuner.best_runtime, tuner.base_runtime)


        # run again with grid search tuner
        tuner = GridSearchTuner("ortools", model)

        start = time.time()
        self.assertIsNotNone(tuner.tune(max_tries=100, fix_params={"num_search_workers": 1}))
        end = time.time()

        self.assertLessEqual(10, 1.05 * end - start)
        self.assertLessEqual(tuner.best_runtime, tuner.base_runtime)
