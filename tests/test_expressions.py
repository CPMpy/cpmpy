import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.expressions.core import Operator

class TestSum(unittest.TestCase):

    def setUp(self):
        self.iv = cp.intvar(0, 10)

    def test_add_int(self):
        expr = self.iv + 4
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'sum')

    def test_addsub_int(self):
        expr = self.iv + 3 - 1
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'sum')
        self.assertEqual(len(expr.args), 3)

    def test_subadd_int(self):
        expr = self.iv -10 + 3
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'sum')
        self.assertEqual(len(expr.args), 3)

    def test_add_iv(self):
        expr = self.iv + cp.intvar(2,4)
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'sum')

    def test_addsub_iv_int(self):
        expr = self.iv + cp.intvar(2,4) - 1
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'sum')
        self.assertEqual(len(expr.args), 3)

    def test_subadd_iv_int(self):
        expr = self.iv - cp.intvar(2,4) + 1
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'sum')
        self.assertEqual(len(expr.args), 3)
