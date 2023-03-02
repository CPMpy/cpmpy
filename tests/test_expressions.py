import unittest
import cpmpy as cp
import numpy as np
from cpmpy.expressions import *
from cpmpy.expressions.variables import NDVarArray
from cpmpy.expressions.core import Operator, Expression


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

    def test_sum_unary(self):
        v = cp.intvar(1,9)
        model = cp.Model(v>=1, minimize=sum([v]))
        self.assertTrue(model.solve())
        self.assertEqual(v.value(), 1)

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

        expr2 = self.ivs[1] + self.ivs[0] * 4
        self.assertIsInstance(expr2, Operator)
        self.assertEqual(expr2.name, 'wsum')

        expr3 = self.ivs[0] * 4 - self.ivs[1]
        self.assertIsInstance(expr3, Operator)
        self.assertEqual(expr3.name, 'wsum')

        expr4 = - self.ivs[1] + self.ivs[0] * 4
        self.assertIsInstance(expr4, Operator)
        self.assertEqual(expr4.name, 'wsum')

    def test_weightedadd_weighted_iv(self):
        expr = self.ivs[0] * 4 + 5 * self.ivs[1] + 6 * self.ivs[2]
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'wsum')

        expr2 = 5 * self.ivs[0] + self.ivs[1] * 4
        self.assertIsInstance(expr2, Operator)
        self.assertEqual(expr2.name, 'wsum')

        expr3 = self.ivs[0] * 4 - self.ivs[1] * 3
        self.assertIsInstance(expr3, Operator)
        self.assertEqual(expr3.name, 'wsum')

        expr4 = - self.ivs[0] + self.ivs[1] * 4 - 6 * self.ivs[2]
        self.assertIsInstance(expr4, Operator)
        self.assertEqual(expr4.name, 'wsum')

    def test_weightedadd_int(self):
        expr = self.ivs[0] * 4 + 5 * self.ivs[1] + 6
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'sum')

    def test_weightedadd_sub(self):
        expr = self.ivs[0] * 4 - 5 * self.ivs[1]
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'wsum')

    def test_negated_weightedadd(self):
        expr = self.ivs[0] * 4 - 5 *  self.ivs[1]
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'wsum')
        expr = -expr # negate, every arg should be negated
        self.assertEqual(expr.name, 'wsum')
        self.assertListEqual([-4, 5], expr.args[0])

    def test_weighted_nested_epxressions(self):
        expr = self.ivs[0] * 4 + 5 * (self.ivs[1] + 6 * self.ivs[2])
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'wsum')

    def test_weighted_nested_mul(self):
        # issue #137
        x = boolvar()
        expr = 100 * (x < 5) * (5 - x) + 10 * (x - 5)
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, 'wsum')
        # (if this surprises you, note that the first one is (100*(x<5))*(5-x) in Python

    def test_sum_generator(self):
        expr1 = sum(self.ivs)
        expr2 = sum([x for x in self.ivs])
        expr3 = sum(x for x in self.ivs)
        assert(str(expr1) == str(expr2))
        assert(str(expr1) == str(expr3))

class TestMul(unittest.TestCase):

    def setUp(self) -> None:
        self.bvar = boolvar(name="bv")
        self.ivar = boolvar(name="iv")

    def test_mul_const(self):
        expr = self.ivar * 10
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, "mul")
        self.assertIn(self.ivar, expr.args)

        expr = self.ivar * True
        self.assertEqual(expr.name, self.ivar.name)
        # same for numpy true
        expr = self.ivar * np.True_
        self.assertEqual(expr.name, self.ivar.name)

        expr = self.ivar * False
        self.assertEqual(0, expr)
        # same for numpy false
        expr = self.ivar * np.False_
        self.assertEqual(0, expr)



    def test_mul_var(self):
        #ivar and bvar
        expr = self.ivar * self.bvar
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, "mul")
        self.assertIn(self.ivar, expr.args)
        self.assertIn(self.bvar, expr.args)

        #ivar and ivar
        expr = self.ivar * self.ivar
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, "mul")
        self.assertIn(self.ivar, expr.args)

        #bvar and bvar
        expr = self.bvar * self.bvar
        self.assertIsInstance(expr, Operator)
        self.assertEqual(expr.name, "mul")
        self.assertIn(self.bvar, expr.args)

    def test_nullarg_mul(self):
        x = intvar(0,5,shape=3, name="x")
        a = np.array([0,1,1], dtype=bool)

        prod = x * a

        self.assertIsInstance(prod, NDVarArray)
        for expr in prod.args:
            self.assertTrue(isinstance(expr, Expression) or expr == 0)


class TestBounds(unittest.TestCase):
    def test_bounds_mul_sub_sum(self):
        x = intvar(-8,8)
        y = intvar(-4,6)
        for name in ['mul','sub','sum']:
            op = Operator(name,[x,y])
            lb, ub = op.get_bounds()
            for lhs in range(*x.get_bounds()):
                for rhs in range(*y.get_bounds()):
                    val = Operator(name,[lhs,rhs]).value()
                    self.assertTrue(val >= lb)
                    self.assertTrue(val <= ub)

    def test_bounds_wsum(self):
        x = intvar(-8, 8,3)
        weights = [2,4,-3]
        op = Operator('wsum',[weights,x])
        lb, ub = op.get_bounds()
        for x1 in range(*x[0].get_bounds()):
            for x2 in range(*x[1].get_bounds()):
                for x3 in range(*x[2].get_bounds()):
                    val = Operator('wsum',[weights,[x1,x2,x3]]).value()
                    self.assertTrue(val >= lb)
                    self.assertTrue(val <= ub)


    def test_bounds_div(self):
        x = intvar(-8, 8)
        y = intvar(-7,-1)
        z = intvar(1,9)
        op1 = Operator('div',[x,y])
        lb1,ub1 = op1.get_bounds()
        op2 = Operator('div',[x,z])
        lb2,ub2 = op2.get_bounds()
        for lhs in range(*x.get_bounds()):
            for rhs in range(*y.get_bounds()):
                val = Operator('div',[lhs,rhs]).value()
                self.assertTrue(val >= lb1)
                self.assertTrue(val <= ub1)
            for rhs in range(*z.get_bounds()):
                val = Operator('div', [lhs, rhs]).value()
                self.assertTrue(val >= lb2)
                self.assertTrue(val <= ub2)

    def test_bounds_mod(self):
        x = intvar(-8, 8)
        y = intvar(-7, -1)
        z = intvar(1, 9)
        op1 = Operator('mod',[x,y])
        lb1, ub1 = op1.get_bounds()
        op2 = Operator('mod',[x,z])
        lb2, ub2 = op2.get_bounds()
        for lhs in range(*x.get_bounds()):
            for rhs in range(*y.get_bounds()):
                val = Operator('mod',[lhs,rhs]).value()
                self.assertTrue(val >= lb1)
                self.assertTrue(val <= ub1)
            for rhs in range(*z.get_bounds()):
                val = Operator('mod', [lhs, rhs]).value()
                self.assertTrue(val >= lb2)
                self.assertTrue(val <= ub2)

    def test_bounds_pow(self):
        x = intvar(-8, 8)
        z = intvar(1, 9)
        # only nonnegative exponents allowed
        op = Operator('pow',[x,z])
        lb, ub = op.get_bounds()
        for lhs in range(*x.get_bounds()):
            for rhs in range(*z.get_bounds()):
                val = Operator('pow',[lhs,rhs]).value()
                self.assertTrue(val >= lb)
                self.assertTrue(val <= ub)

    def test_bounds_unary(self):
        x = intvar(-8, 8)
        y = intvar(-7, -1)
        z = intvar(1, 9)
        for name in ['-', 'abs']:
            op = Operator(name,[x])
            lb, ub = op.get_bounds()
            op1 = Operator(name,[z])
            lb1, ub1 = op.get_bounds()
            op2 = Operator(name,[y])
            lb2, ub2 = op.get_bounds()
            for lhs in range(*x.get_bounds()):
                val = Operator(name, [lhs]).value()
                self.assertTrue(val >= lb)
                self.assertTrue(val <= ub)
            for lhs in range(*z.get_bounds()):
                val = Operator(name, [lhs]).value()
                self.assertTrue(val >= lb1)
                self.assertTrue(val <= ub1)
            for lhs in range(*y.get_bounds()):
                val = Operator(name, [lhs]).value()
                self.assertTrue(val >= lb2)
                self.assertTrue(val <= ub2)

if __name__ == '__main__':
    unittest.main()
