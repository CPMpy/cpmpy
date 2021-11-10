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

class TestWeightedSum(unittest.TestCase):
    def setUp(self) -> None:
        self.ivs = cp.intvar(lb=0, ub=5, shape=4)

    def test_weightedadd_int(self):
        expr = self.ivs[0] * 4 + 3
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'sum')
        expr2 = 3 + self.ivs[0] * 4
        self.assertIsInstance(expr2, Operator)
        self.assertEqual(expr2.name, 'sum')

    def test_weightedadd_iv(self):

        expr = self.ivs[0] * 4 + self.ivs[1]
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'wsum')
        self.assertEqual(str([self.ivs[0], self.ivs[1]]) + " .* " + str([4, 1]), str(expr))

        expr2 = self.ivs[1] + self.ivs[0] * 4
        self.assertIsInstance(expr2, Operator)
        self.assertEqual(expr2.name, 'wsum')
        self.assertEqual(str([self.ivs[1], self.ivs[0]]) + " .* " + str([1, 4]), str(expr2))

        expr3 = self.ivs[0] * 4 - self.ivs[1]
        self.assertIsInstance(expr3, Operator)
        self.assertEqual(expr3.name, 'wsum')
        self.assertEqual(str([self.ivs[0], self.ivs[1]]) + " .* " + str([4, -1]), str(expr3))

        expr4 = - self.ivs[1] + self.ivs[0] * 4
        self.assertIsInstance(expr4, Operator)
        self.assertEqual(expr4.name, 'wsum')
        self.assertEqual(str([self.ivs[1], self.ivs[0]]) + " .* " + str([-1, 4]), str(expr4))

    def test_weightedadd_weighted_iv(self):
        expr = self.ivs[0] * 4 + 5 * self.ivs[1] + 6 * self.ivs[2]
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'wsum')
        self.assertEqual(str([self.ivs[0] ,self.ivs[1],self.ivs[2]]) + " .* " + str([4, 5, 6]), str(expr))

        expr2 = 5 * self.ivs[0] + self.ivs[1] * 4
        self.assertIsInstance(expr2, Operator)
        self.assertEqual(expr2.name, 'wsum')
        self.assertEqual(str([self.ivs[0] ,self.ivs[1]]) + " .* " + str([5, 4]), str(expr2))

        expr3 = self.ivs[0] * 4 - self.ivs[1] * 3
        self.assertIsInstance(expr3, Operator)
        self.assertEqual(expr3.name, 'wsum')
        self.assertEqual(str([self.ivs[0] ,self.ivs[1]]) + " .* " + str([4, -3]), str(expr3))

        expr4 = - self.ivs[0] + self.ivs[1] * 4 - 6 * self.ivs[2]
        self.assertIsInstance(expr4, Operator)
        self.assertEqual(expr4.name, 'wsum')
        self.assertEqual(str([self.ivs[0] ,self.ivs[1],self.ivs[2]]) + " .* " + str([-1, 4, -6]), str(expr4))

    def test_weightedadd_int(self):
        expr = self.ivs[0] * 4 + 5 * self.ivs[1] + 6
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'sum')
    
    def test_weighted_nested_epxressions(self):
        expr = self.ivs[0] * 4 + 5 * (self.ivs[1] + 6 * self.ivs[2])
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'wsum')
        print(expr)
        

if __name__ == '__main__':
    unittest.main()