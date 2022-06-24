from unittest import TestCase

from cpmpy import *
from cpmpy.tools import ParameterTuner


class TunerTests(TestCase):

    def test_ortools(self):
        x = intvar(lb=0, ub=10, shape=10)
        model = Model([
            AllDifferent(x),
        ])

        tuner = ParameterTuner("ortools", model)

        self.assertIsNotNone(tuner.tune(max_tries=100))


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

        self.assertIsNotNone(tuner.tune(max_tries=100))
