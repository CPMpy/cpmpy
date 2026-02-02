import time
from unittest import TestCase

from cpmpy import *
from cpmpy.tools import ParameterTuner, GridSearchTuner
from cpmpy.solvers.ortools import CPM_ortools



class TunerTests(TestCase):

    def test_ortools_tunable_params(self):
        """
        Test that all tunable parameters can be set without errors
        """
        x = intvar(lb=0, ub=5, shape=3)
        model = Model([x[0] < x[1], x[1] < x[2]])
        
        tunable = CPM_ortools.tunable_params()
        
        for param_name, values in tunable.items():
            for val in values:
                solver = CPM_ortools(model)
                self.assertTrue(solver.solve(**{param_name: val}))
                self.assertIsNotNone(solver.status())

    def test_ortools(self):
        x = intvar(lb=0, ub=10, shape=10)
        model = Model([
            AllDifferent(x),
        ])

        tuner = ParameterTuner("ortools", model)
        self.assertIsNotNone(tuner.tune(max_tries=100, fix_params={"num_search_workers":1}))
        self.assertLessEqual(tuner.best_runtime, tuner.base_runtime)

        # run again with grid search tuner
        tuner = GridSearchTuner("ortools", model)
        self.assertIsNotNone(tuner.tune(max_tries=100, fix_params={"num_search_workers": 1}))
        self.assertLessEqual(tuner.best_runtime, tuner.base_runtime)

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
        self.assertIsNotNone(tuner.tune(max_tries=100, fix_params={"num_search_workers":1}))
        self.assertLessEqual(tuner.best_runtime, tuner.base_runtime)

        # run again with grid search tuner
        tuner = GridSearchTuner("ortools", model, tunables, defaults)
        self.assertIsNotNone(tuner.tune(max_tries=100, fix_params={"num_search_workers": 1}))
        self.assertLessEqual(tuner.best_runtime, tuner.base_runtime)



    def test_ortools_timelimit(self):
        x = intvar(lb=0, ub=10, shape=10)
        model = Model([
            AllDifferent(x),
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
