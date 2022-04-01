import unittest
import tempfile
import os
from os.path import join

from numpy import logaddexp
import cpmpy as cp
from cpmpy.expressions.variables import NullShapeError, _IntVarImpl, _BoolVarImpl, NegBoolView, NDVarArray


class TestInequalityChaining(unittest.TestCase):
    def test_single_inequality(self):
        pass
    def test_chaining_lt(self):
        pass
    def test_chaining_lte(self):
        pass
    def test_chaining_gt(self):
        pass
    def test_chaining_gte(self):
        pass
    def test_chaining_mix_inequalities(self):
        pass
if __name__ == '__main__':
    unittest.main()