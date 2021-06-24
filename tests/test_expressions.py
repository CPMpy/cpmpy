# TODO
"""
def test_value():
    returns a non-None .value():
    - NumVarImpl, BoolVarImpl
    - Operator (all of them)
    - Comparison (all of them)
    - Element (both 2 and 3 args)
"""
import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.expressions.core import *
from cpmpy.expressions.variables import *
from cpmpy.expressions.utils import *

class TestSum(unittest.TestCase):

    def setUp(self):
        self.iv = cp.IntVar(0, 10)

    def test_add_int(self):
        expr = self.iv + 4
        self.assertIsInstance(expr, Operator)
        self.assertTrue(expr.name is 'sum')

    def test_addsub_int(self):
        expr = self.iv + 3 - 1
        self.assertIsInstance(expr, Operator)
        self.assertTrue(expr.name is 'sum')
        self.assertEqual(len(expr.args), 3)

    def test_subadd_int(self):
        expr = self.iv -10 + 3
        self.assertIsInstance(expr, Operator)
        self.assertTrue(expr.name is 'sum')
        self.assertEqual(len(expr.args), 3)

    def test_add_iv(self):
        expr = self.iv + cp.IntVar(2,4)
        self.assertIsInstance(expr, Operator)
        self.assertTrue(expr.name is 'sum')

    def test_addsub_iv_int(self):
        expr = self.iv + cp.IntVar(2,4) - 1
        self.assertIsInstance(expr, Operator)
        self.assertTrue(expr.name is 'sum')
        self.assertEqual(len(expr.args), 3)

    def test_subadd_iv_int(self):
        expr = self.iv - cp.IntVar(2,4) + 1
        self.assertIsInstance(expr, Operator)
        self.assertTrue(expr.name is 'sum')
        self.assertEqual(len(expr.args), 3)
