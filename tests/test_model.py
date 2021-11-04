import unittest
import cpmpy as cp
from cpmpy.expressions.variables import NullShapeError, _IntVarImpl, _BoolVarImpl, NegBoolView, NDVarArray


class TestModel(unittest.TestCase):
    def test_ndarray(self):
        iv = cp.intvar(1,9, shape=3)
        m = cp.Model( iv > 3 )
        m += (iv[0] == 5)
        self.assertTrue(m.solve())

    def test_empty(self):
        m = cp.Model()
        m += [] # should do nothing
        assert(len(m.constraints) == 0)
