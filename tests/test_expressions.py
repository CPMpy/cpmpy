import unittest
import cpmpy as cp
import numpy as np

from cpmpy.exceptions import IncompleteFunctionError
from cpmpy.expressions import *
from cpmpy.expressions.variables import NDVarArray
from cpmpy.expressions.core import Operator, Expression
from cpmpy.expressions.utils import get_bounds, argval

class TestComparison(unittest.TestCase):
    def test_comps(self):
        # from the docs
        # XXX is this the right place? it should be tested with all solvers...
        import cpmpy as cp

        bv = cp.boolvar()
        iv = cp.intvar(0, 10)

        m = cp.Model(
            bv == True,         # allowed
            bv > 0,             # allowed but silly
            iv > 0,             # allowed
            iv != 1,            # allowed
            iv == True,         # allowed but means `iv == 1`, avoid
            (iv != 0) == True,  # allowed
            iv == bv,           # allowed but means `(iv == 1) == bv`, avoid
            # bv & iv,          # not allowed, choose one of:
            bv & (iv == 1),     # allowed
            bv & (iv != 0),     # allowed
        )
        for c in m.constraints:
            self.assertTrue(cp.Model(c).solve())

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

class TestArrayExpressions(unittest.TestCase):

    def test_sum(self):
        x = intvar(0,5,shape=10, name="x")
        y = intvar(0, 1000, name="y")
        model = cp.Model(y == x.sum())
        model.solve()
        self.assertTrue(y.value() == sum(x.value()))
        # with axis arg
        x = intvar(0,5,shape=(10,10), name="x")
        y = intvar(0, 1000, shape=10, name="y")
        model = cp.Model(y == x.sum(axis=0))
        model.solve()
        res = np.array([sum(x[i, ...].value()) for i in range(len(y))])
        self.assertTrue(all(y.value() == res))

    def test_prod(self):
        x = intvar(0,5,shape=10, name="x")
        y = intvar(0, 1000, name="y")
        model = cp.Model(y == x.prod())
        model.solve()
        res = 1
        for v in x:
            res *= v.value()
        self.assertTrue(y.value() == res)
        # with axis arg
        x = intvar(0,5,shape=(10,10), name="x")
        y = intvar(0, 1000, shape=10, name="y")
        model = cp.Model(y == x.prod(axis=0))
        model.solve()
        for i,vv in enumerate(x):
            res = 1
            for v in vv:
                res *= v.value()
            self.assertTrue(y[i].value() == res)

    def test_max(self):
        x = intvar(0,5,shape=10, name="x")
        y = intvar(0, 1000, name="y")
        model = cp.Model(y == x.max())
        model.solve()
        self.assertTrue(y.value() == max(x.value()))
        # with axis arg
        x = intvar(0,5,shape=(10,10), name="x")
        y = intvar(0, 1000, shape=10, name="y")
        model = cp.Model(y == x.max(axis=0))
        model.solve()
        res = np.array([max(x[i, ...].value()) for i in range(len(y))])
        self.assertTrue(all(y.value() == res))

    def test_min(self):
        x = intvar(0,5,shape=10, name="x")
        y = intvar(0, 1000, name="y")
        model = cp.Model(y == x.min())
        model.solve()
        self.assertTrue(y.value() == min(x.value()))
        # with axis arg
        x = intvar(0,5,shape=(10,10), name="x")
        y = intvar(0, 1000, shape=10, name="y")
        model = cp.Model(y == x.min(axis=0))
        model.solve()
        res = np.array([min(x[i, ...].value()) for i in range(len(y))])
        self.assertTrue(all(y.value() == res))

    def test_any(self):
        from cpmpy.expressions.python_builtins import any
        x = boolvar(shape=10, name="x")
        y = boolvar(name="y")
        model = cp.Model(y == x.any())
        model.solve()
        self.assertTrue(y.value() == any(x.value()))
        # with axis arg
        x = boolvar(shape=(10,10), name="x")
        y = boolvar(shape=10, name="y")
        model = cp.Model(y == x.any(axis=0))
        model.solve()
        res = np.array([any(x[i, ...].value()) for i in range(len(y))])
        self.assertTrue(all(y.value() == res))

    def test_all(self):
        from cpmpy.expressions.python_builtins import all
        x = boolvar(shape=10, name="x")
        y = boolvar(name="y")
        model = cp.Model(y == x.all())
        model.solve()
        self.assertTrue(y.value() == all(x.value()))
        # with axis arg
        x = boolvar(shape=(10,10), name="x")
        y = boolvar(shape=10, name="y")
        model = cp.Model(y == x.all(axis=0))
        model.solve()
        res = np.array([all(x[i, ...].value()) for i in range(len(y))])
        self.assertTrue(all(y.value() == res))

        
def inclusive_range(lb,ub):
        return range(lb,ub+1)

class TestBounds(unittest.TestCase):
    def test_bounds_mul_sub_sum(self):
        x = intvar(-8,8)
        y = intvar(-4,6)
        for name, test_lb, test_ub in [('mul',-48,48),('sub',-14,12),('sum',-12,14)]:
            op = Operator(name,[x,y])
            lb, ub = op.get_bounds()
            self.assertEqual(test_lb,lb)
            self.assertEqual(test_ub,ub)
            for lhs in inclusive_range(*x.get_bounds()):
                for rhs in inclusive_range(*y.get_bounds()):
                    val = Operator(name,[lhs,rhs]).value()
                    self.assertGreaterEqual(val,lb)
                    self.assertLessEqual(val,ub)

    def test_bounds_wsum(self):
        x = intvar(-8, 8,3)
        weights = [2,4,-3]
        op = Operator('wsum',[weights,x])
        lb, ub = op.get_bounds()
        self.assertEqual(lb,-72)
        self.assertEqual(ub,72)
        for x1 in inclusive_range(*x[0].get_bounds()):
            for x2 in inclusive_range(*x[1].get_bounds()):
                for x3 in inclusive_range(*x[2].get_bounds()):
                    val = Operator('wsum',[weights,[x1,x2,x3]]).value()
                    self.assertGreaterEqual(val,lb)
                    self.assertLessEqual(val,ub)

    def test_bounds_div(self):
        x = intvar(-8, 8)
        y = intvar(-7,-1)
        z = intvar(3,9)
        op1 = Operator('div',[x,y])
        lb1,ub1 = op1.get_bounds()
        self.assertEqual(lb1,-8)
        self.assertEqual(ub1,8)
        op2 = Operator('div',[x,z])
        lb2,ub2 = op2.get_bounds()
        self.assertEqual(lb2,-3)
        self.assertEqual(ub2,2)
        for lhs in inclusive_range(*x.get_bounds()):
            for rhs in inclusive_range(*y.get_bounds()):
                val = Operator('div',[lhs,rhs]).value()
                self.assertGreaterEqual(val,lb1)
                self.assertLessEqual(val,ub1)
            for rhs in inclusive_range(*z.get_bounds()):
                val = Operator('div', [lhs, rhs]).value()
                self.assertGreaterEqual(val,lb2)
                self.assertLessEqual(val,ub2)

    def test_bounds_mod(self):
        x = intvar(-8, 8)
        y = intvar(-5, -1)
        z = intvar(1, 4)
        op1 = Operator('mod',[x,y])
        lb1, ub1 = op1.get_bounds()
        self.assertEqual(lb1,-4)
        self.assertEqual(ub1,0)
        op2 = Operator('mod',[x,z])
        lb2, ub2 = op2.get_bounds()
        self.assertEqual(lb2,0)
        self.assertEqual(ub2,3)
        for lhs in inclusive_range(*x.get_bounds()):
            for rhs in inclusive_range(*y.get_bounds()):
                val = Operator('mod',[lhs,rhs]).value()
                self.assertGreaterEqual(val,lb1)
                self.assertLessEqual(val,ub1)
            for rhs in inclusive_range(*z.get_bounds()):
                val = Operator('mod', [lhs, rhs]).value()
                self.assertGreaterEqual(val,lb2)
                self.assertLessEqual(val,ub2)

    def test_bounds_pow(self):
        x = intvar(-8, 5)
        z = intvar(1, 9)
        # only nonnegative exponents allowed
        op = Operator('pow',[x,z])
        lb, ub = op.get_bounds()
        self.assertEqual(lb,-134217728)
        self.assertEqual(ub,16777216)
        for lhs in inclusive_range(*x.get_bounds()):
            for rhs in inclusive_range(*z.get_bounds()):
                val = Operator('pow',[lhs,rhs]).value()
                self.assertGreaterEqual(val,lb)
                self.assertLessEqual(val,ub)

    def test_bounds_unary(self):
        x = intvar(-8, 5)
        y = intvar(-7, -2)
        z = intvar(1, 9)
        for var,test_lb,test_ub in [(x,-5,8),(y,2,7),(z,-9,-1)]:
            name = '-'
            op = Operator(name,[var])
            lb, ub = op.get_bounds()
            self.assertEqual(test_lb,lb)
            self.assertEqual(test_ub,ub)
            for lhs in inclusive_range(*var.get_bounds()):
                val = Operator(name, [lhs]).value()
                self.assertGreaterEqual(val,lb)
                self.assertLessEqual(val,ub)


    def test_incomplete_func(self):
        # element constraint
        arr = cpm_array([1,2,3])
        i = intvar(0,5,name="i")
        p = boolvar()

        cons = (arr[i] == 1).implies(p)
        m = cp.Model([cons, i == 5])
        self.assertTrue(m.solve())
        self.assertTrue(cons.value())

        # div constraint
        a,b = intvar(1,2,shape=2)
        cons = (42 // (a - b)) >= 3
        m = cp.Model([p.implies(cons), a == b])
        if cp.SolverLookup.lookup("z3").supported():
            self.assertTrue(m.solve(solver="z3")) # ortools does not support divisor spanning 0 work here
            self.assertRaises(IncompleteFunctionError, cons.value)
            self.assertFalse(argval(cons))

        # mayhem
        cons = (arr[10 // (a - b)] == 1).implies(p)
        m = cp.Model([cons, a == b])
        if cp.SolverLookup.lookup("z3").supported():
            self.assertTrue(m.solve(solver="z3"))
            self.assertTrue(cons.value())


    def test_list(self):

        # cpm_array
        iv = cp.intvar(0,10,shape=3)
        lbs, ubs = iv.get_bounds()
        self.assertListEqual([0,0,0], lbs.tolist())
        self.assertListEqual([10,10,10], ubs.tolist())
        # list
        iv = [cp.intvar(0,10) for _ in range(3)]
        lbs, ubs = get_bounds(iv)
        self.assertListEqual([0, 0, 0], lbs)
        self.assertListEqual([10, 10, 10], ubs)
        # nested list
        exprs = [intvar(0,1), [intvar(2,3), intvar(4,5)], [intvar(5,6)]]
        lbs, ubs = get_bounds(exprs)
        self.assertListEqual([0,[2,4],[5]], lbs)
        self.assertListEqual([1,[3,5],[6]], ubs)



    def test_not_operator(self):
        p = boolvar()
        q = boolvar()
        x = intvar(0,9)
        self.assertTrue(cp.Model([~p]).solve())
        #self.assertRaises(cp.exceptions.TypeError, cp.Model([~x]).solve())
        self.assertTrue(cp.Model([~(x == 0)]).solve())
        self.assertTrue(cp.Model([~~p]).solve())
        self.assertTrue(cp.Model([~(p & p)]).solve())
        self.assertTrue(cp.Model([~~~~~(p & p)]).solve())
        self.assertTrue(cp.Model([~cpm_array([p,q,p])]).solve())
        self.assertTrue(cp.Model([~p.implies(q)]).solve())
        self.assertTrue(cp.Model([~p.implies(~q)]).solve())
        self.assertTrue(cp.Model([p.implies(~q)]).solve())
        self.assertTrue(cp.Model([p == ~q]).solve())
        self.assertTrue(cp.Model([~~p == ~q]).solve())
        self.assertTrue(cp.Model([Operator('not',[p]) == q]).solve())
        self.assertTrue(cp.Model([Operator('not',[p])]).solve())

    def test_description(self):

        a,b = cp.boolvar(name="a"), cp.boolvar(name="b")
        cons = a | b
        cons.set_description("either a or b should be true, but not both")

        self.assertEqual(repr(cons), "(a) or (b)")
        self.assertEqual(str(cons), "either a or b should be true, but not both")

        # ensure nothing goes wrong due to calling __str__ on a constraint with a custom description
        for solver,cls in cp.SolverLookup.base_solvers():
            if not cls.supported():
                continue
            print("Testing", solver)
            self.assertTrue(cp.Model(cons).solve(solver=solver))

        ## test extra attributes of set_description
        cons = a | b
        cons.set_description("either a or b should be true, but not both",
                             override_print=False)

        self.assertEqual(repr(cons), "(a) or (b)")
        self.assertEqual(str(cons), "(a) or (b)")

        cons = a | b
        cons.set_description("either a or b should be true, but not both",
                             full_print=True)

        self.assertEqual(repr(cons), "(a) or (b)")
        self.assertEqual(str(cons), "either a or b should be true, but not both -- (a) or (b)")





if __name__ == '__main__':
    unittest.main()
