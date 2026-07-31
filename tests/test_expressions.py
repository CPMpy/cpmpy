import pytest

import cpmpy as cp
import numpy as np

from cpmpy.expressions.variables import boolvar, intvar, cpm_array
from cpmpy.expressions.globalfunctions import Maximum, Abs
from cpmpy.expressions.variables import NDVarArray
from cpmpy.expressions.core import Comparison, Operator, Expression
from cpmpy.expressions.utils import eval_comparison, get_bounds

from cpmpy.exceptions import MinizincNameException, NotSupportedError, TypeError as CPMpyTypeError

from utils import inclusive_range


class TestComparison:
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
            assert cp.Model(c).solve()

class TestSum:

    def setup_method(self):
        self.iv = cp.intvar(0, 10)

    def test_add_int(self):
        expr = self.iv + 4
        assert isinstance(expr, Operator)
        assert expr.name == 'sum'

    def test_addsub_int(self):
        expr = self.iv + 3 - 1
        assert isinstance(expr, Operator)
        assert expr.name == 'sum'
        assert len(expr.args) == 3

    def test_subadd_int(self):
        expr = self.iv -10 + 3
        assert isinstance(expr, Operator)
        assert expr.name == 'sum'
        assert len(expr.args) == 3

    def test_add_iv(self):
        expr = self.iv + cp.intvar(2,4)
        assert isinstance(expr, Operator)
        assert expr.name == 'sum'

    def test_addsub_iv_int(self):
        expr = self.iv + cp.intvar(2,4) - 1
        assert isinstance(expr, Operator)
        assert expr.name == 'sum'
        assert len(expr.args) == 3

    def test_subadd_iv_int(self):
        expr = self.iv - cp.intvar(2,4) + 1
        assert isinstance(expr, Operator)
        assert expr.name == 'sum'
        assert len(expr.args) == 3

    def test_sum_unary(self):
        v = cp.intvar(1,9)
        model = cp.Model(v>=1, minimize=sum([v]))
        assert model.solve()
        assert v.value() == 1

class TestWeightedSum:
    def setup_method(self) -> None:
        self.ivs = cp.intvar(lb=0, ub=5, shape=4)

    def test_weightedadd_int(self):
        expr = self.ivs[0] * 4 + 3
        assert isinstance(expr, Operator)
        assert expr.name == 'sum'
        expr2 = 3 + self.ivs[0] * 4
        assert isinstance(expr2, Operator)
        assert expr2.name == 'sum'
        expr3 = self.ivs[0] * 4 + 5 * self.ivs[1] + 6
        assert isinstance(expr3, Operator)
        assert expr3.name == 'sum'


    def test_weightedadd_iv(self):

        expr = self.ivs[0] * 4 + self.ivs[1]
        assert isinstance(expr, Operator)
        assert expr.name == 'wsum'

        expr2 = self.ivs[1] + self.ivs[0] * 4
        assert isinstance(expr2, Operator)
        assert expr2.name == 'wsum'

        expr3 = self.ivs[0] * 4 - self.ivs[1]
        assert isinstance(expr3, Operator)
        assert expr3.name == 'wsum'

        expr4 = - self.ivs[1] + self.ivs[0] * 4
        assert isinstance(expr4, Operator)
        assert expr4.name == 'wsum'

    def test_weightedadd_weighted_iv(self):
        expr = self.ivs[0] * 4 + 5 * self.ivs[1] + 6 * self.ivs[2]
        assert isinstance(expr, Operator)
        assert expr.name == 'wsum'

        expr2 = 5 * self.ivs[0] + self.ivs[1] * 4
        assert isinstance(expr2, Operator)
        assert expr2.name == 'wsum'

        expr3 = self.ivs[0] * 4 - self.ivs[1] * 3
        assert isinstance(expr3, Operator)
        assert expr3.name == 'wsum'

        expr4 = - self.ivs[0] + self.ivs[1] * 4 - 6 * self.ivs[2]
        assert isinstance(expr4, Operator)
        assert expr4.name == 'wsum'

    def test_weightedadd_sub(self):
        expr = self.ivs[0] * 4 - 5 * self.ivs[1]
        assert isinstance(expr, Operator)
        assert expr.name == 'wsum'

    def test_negated_weightedadd(self):
        expr = self.ivs[0] * 4 - 5 *  self.ivs[1]
        assert isinstance(expr, Operator)
        assert expr.name == 'wsum'
        expr = -expr # negate, every arg should be negated
        assert expr.name == 'wsum'
        assert [-4, 5] == expr.args[0]

    def test_weighted_nested_epxressions(self):
        expr = self.ivs[0] * 4 + 5 * (self.ivs[1] + 6 * self.ivs[2])
        assert isinstance(expr, Operator)
        assert expr.name == 'wsum'

    def test_weighted_nested_mul(self):
        # issue #137
        x = boolvar()
        expr = 100 * (x < 5) * (5 - x) + 10 * (x - 5)
        assert isinstance(expr, Operator)
        assert expr.name == 'wsum'
        # (if this surprises you, note that the first one is (100*(x<5))*(5-x) in Python

    def test_sum_generator(self):
        expr1 = sum(self.ivs)
        expr2 = sum([x for x in self.ivs])
        expr3 = sum(x for x in self.ivs)
        assert(str(expr1) == str(expr2))
        assert(str(expr1) == str(expr3))

    def test_reject_float_coefficients(self):
        m = cp.Model()
        x, y, z = cp.boolvar(shape=3, name=tuple("xyz"))
        with pytest.raises(CPMpyTypeError, match="float constants"):
            m.add(0.7 * x + 0.8 * y >= 1)

    def test_floatsum_objective_only(self):
        x = cp.boolvar(name="x")
        fs = cp.FloatSum([0.5], [x])
        with pytest.raises(CPMpyTypeError, match="cannot be used as an expression"):
            _ = fs >= 1

class TestMul:

    def setup_method(self) -> None:
        self.bvar = boolvar(name="bv")
        self.ivar = boolvar(name="iv")

    def test_mul_const(self):
        expr = self.ivar * 10
        assert isinstance(expr, cp.Multiplication)
        assert expr.name == "mul"
        assert self.ivar in set(expr.args)

        expr = self.ivar * True
        assert expr.name == self.ivar.name
        # same for numpy true
        expr = self.ivar * np.True_
        assert expr.name == self.ivar.name

        # TODO do we want the following? see issue #342
        # expr = self.ivar * False # this test failes now
        # self.assertEqual(0, expr)
        # # same for numpy false
        # expr = self.ivar * np.False_
        # self.assertEqual(0, expr)



    def test_mul_var(self):
        #ivar and bvar
        expr = self.ivar * self.bvar
        assert isinstance(expr, cp.Multiplication)
        assert expr.name == "mul"
        assert self.ivar in set(expr.args)
        assert self.bvar in set(expr.args)

        #ivar and ivar
        expr = self.ivar * self.ivar
        assert isinstance(expr, cp.Multiplication)
        assert expr.name == "mul"
        assert self.ivar in set(expr.args)

        #bvar and bvar
        expr = self.bvar * self.bvar
        assert isinstance(expr, cp.Multiplication)
        assert expr.name == "mul"
        assert self.bvar in set(expr.args)

    def test_mul_is_lhs_num(self):
        """Multiplication normalises const to first arg and sets is_lhs_num."""
        x = cp.intvar(0, 5, name="x")
        # const * var -> constant first, is_lhs_num True
        expr = 3 * x
        assert expr.is_lhs_num is True
        assert expr.args[0] == 3 and expr.args[1] is x
        # var * const -> swapped to constant first, is_lhs_num True
        expr = x * 3
        assert expr.is_lhs_num is True
        assert expr.args[0] == 3 and expr.args[1] is x
        # var * var -> no constant, is_lhs_num False
        y = cp.intvar(0, 5, name="y")
        expr = x * y
        assert expr.is_lhs_num is False
        assert expr.args[0] is x and expr.args[1] is y

    def test_nullarg_mul(self):
        x = intvar(0,5,shape=3, name="x")
        a = np.array([0,1,1], dtype=bool)

        prod = x * a

        assert isinstance(prod, NDVarArray)
        for expr in prod:
            assert isinstance(expr, Expression) or expr == 0

class TestNDVarArrayBroadcast:

    def test_numpy_array_mul(self):
        x = intvar(0, 10, shape=(3, 4), name="x")
        w = np.array([1, 2, 3, 4])
        expr = x * w
        ref = np.multiply(x, w)
        assert expr.shape == ref.shape
        for idx in np.ndindex(expr.shape):
            assert str(expr[idx]) == str(ref[idx])

    def test_numpy_scalar_mul(self):
        x = intvar(0, 10, shape=(3, 4), name="x")
        expr = x * 2
        ref = np.multiply(x, 2)
        for idx in np.ndindex(expr.shape):
            assert str(expr[idx]) == str(ref[idx])

    def test_numpy_expr_mul(self):
        x = intvar(0, 10, shape=(3, 4), name="x")
        e = cp.boolvar() | cp.boolvar()
        expr = x * e
        ref = np.multiply(x, e)
        assert expr.shape == ref.shape
        for idx in np.ndindex(expr.shape):
            assert str(expr[idx]) == str(ref[idx])

    def test_incompatible_broadcast_raises(self):
        x = intvar(0, 10, shape=(3, 4), name="x")
        with pytest.raises(ValueError, match="broadcast"):
            x * np.array([1, 2])


class TestNDVarArrayVectorizedIndex:

    def test_index_with_ivar_array(self):
        arr = intvar(0, 10, shape=5, name="arr")
        idx = intvar(0, 4, shape=3, name="idx")
        expr = arr[idx]
        assert isinstance(expr, NDVarArray)
        assert expr.shape == (3,)
        for i in range(3):
            assert isinstance(expr[i], cp.Element)
            assert expr[i].args[1] is idx[i]

    def test_index_with_ivar_list(self):
        arr = intvar(0, 10, shape=5, name="arr")
        i0, i1 = intvar(0, 4, name="i0"), intvar(0, 4, name="i1")
        expr = arr[[i0, i1]]
        assert isinstance(expr, NDVarArray)
        assert expr.shape == (2,)
        assert expr[0].args[1] is i0 and expr[1].args[1] is i1

    def test_index_preserves_shape(self):
        arr = intvar(0, 10, shape=5, name="arr")
        idx = intvar(0, 4, shape=(2, 2), name="idx")
        expr = arr[idx]
        assert expr.shape == (2, 2)
        assert all(isinstance(e, cp.Element) for e in expr.flat)

    def test_index_mixed_const_and_ivar(self):
        arr = intvar(0, 10, shape=5, name="arr")
        i = intvar(0, 4, name="i")
        expr = arr[[0, i, 2]]
        assert expr.shape == (3,)
        assert expr[0] is arr[0]
        assert isinstance(expr[1], cp.Element)
        assert expr[2] is arr[2]

    def test_bool_mask_raises(self):
        arr = intvar(0, 10, shape=5, name="arr")
        b = cp.boolvar(shape=5, name="b")
        with pytest.raises(TypeError, match="mask"):
            arr[b]

    def test_vectorized_index_solves(self):
        arr = cpm_array([10, 20, 30, 40])
        idx = intvar(0, 3, shape=2, name="idx")
        m = cp.Model(arr[idx] == [20, 40], idx[0] != idx[1])
        assert m.solve()
        assert sorted(idx.value().tolist()) == [1, 3]


class TestArrayExpressions:

    def test_scalar_expr_with_ndarray(self):
        x = intvar(0, 5, shape=3, name=tuple("abc"))
        y = intvar(0, 5, name="y")
        xy, yx = x * y, y * x
        assert isinstance(xy, NDVarArray) and isinstance(yx, NDVarArray)
        for i in range(3):
            assert set(xy[i].args) == set(yx[i].args)
            assert set((x + y)[i].args) == set((y + x)[i].args)
        assert str(x == y) == str(y == x)

    def test_scalar_expr_left_of_ndarray_mul(self):
        # y * x must broadcast element-wise, not wrap the whole array in one Multiplication
        from cpmpy.expressions.globalfunctions import Multiplication

        x = intvar(0, 10, shape=3, name=tuple("abc"))
        y = intvar(0, 10, name="y")
        assert isinstance(x * y, NDVarArray)
        yx = y * x
        assert isinstance(yx, NDVarArray)
        assert not isinstance(yx, Multiplication)
        assert yx.shape == (3,)
        assert all(a is y or b is y for a, b in (e.args for e in yx))

    def test_scalar_expr_with_numpy_array(self):
        e = sum(cp.boolvar(3))
        y = np.array([1, 2, 3])
        ey, ye = e * y, y * e
        assert isinstance(ey, NDVarArray)
        assert ey.shape == ye.shape == (3,)
        for a, b in zip(ey, ye):
            assert str(a) == str(b)

    def test_sum(self):
        x = intvar(0,5,shape=10, name="x")
        y = intvar(0, 1000, name="y")
        model = cp.Model(y == x.sum())
        model.solve()
        assert y.value() == sum(x.value())
        # with axis arg
        x = intvar(0,5,shape=(10,10), name="x")
        y = intvar(0, 1000, shape=10, name="y")
        model = cp.Model(y == x.sum(axis=0))
        model.solve()
        res = x.value().sum(axis=0)
        assert all(y.value() == res)

    def test_prod(self):
        x = intvar(0,5,shape=10, name="x")
        y = intvar(0, 1000, name="y")
        model = cp.Model(y == x.prod())
        model.solve()
        res = 1
        for v in x:
            res *= v.value()
        assert y.value() == res
        # with axis arg
        x = intvar(0,5,shape=(10,4), name="x")
        y = intvar(0, 200, shape=10, name="y")
        model = cp.Model(y == x.prod(axis=1))  # y[i] = product(x[i,:])
        model.solve()
        for i,vv in enumerate(x):
            res = 1
            for v in vv:
                res *= v.value()
            assert y[i].value() == res

    def test_max(self):
        x = intvar(0,5,shape=10, name="x")
        y = intvar(0, 1000, name="y")
        model = cp.Model(y == x.max())
        model.solve()
        assert y.value() == max(x.value())
        # with axis arg
        x = intvar(0,5,shape=(10,10), name="x")
        y = intvar(0, 1000, shape=10, name="y")
        model = cp.Model(y == x.max(axis=0))
        model.solve()
        res = x.value().max(axis=0)
        assert all(y.value() == res)

    def test_min(self):
        x = intvar(0,5,shape=10, name="x")
        y = intvar(0, 1000, name="y")
        model = cp.Model(y == x.min())
        model.solve()
        assert y.value() == min(x.value())
        # with axis arg
        x = intvar(0,5,shape=(10,10), name="x")
        y = intvar(0, 1000, shape=10, name="y")
        model = cp.Model(y == x.min(axis=0))
        model.solve()
        res = x.value().min(axis=0)
        assert all(y.value() == res)

    def test_any(self):
        from cpmpy.expressions.python_builtins import any as cpm_any
        x = boolvar(shape=10, name="x")
        y = boolvar(name="y")
        model = cp.Model(y == x.any())
        model.solve()
        assert y.value() == cpm_any(x.value())
        # with axis arg
        x = boolvar(shape=(10,10), name="x")
        y = boolvar(shape=10, name="y")
        model = cp.Model(y == x.any(axis=0))
        model.solve()
        res = x.value().any(axis=0)
        assert all(y.value() == res)
        

    def test_all(self):
        from cpmpy.expressions.python_builtins import all as cpm_all
        x = boolvar(shape=10, name="x")
        y = boolvar(name="y")
        model = cp.Model(y == x.all())
        model.solve()
        assert y.value() == cpm_all(x.value())
        # with axis arg
        x = boolvar(shape=(10,10), name="x")
        y = boolvar(shape=10, name="y")
        model = cp.Model(y == x.all(axis=0))
        model.solve()
        res = x.value().all(axis=0)
        assert all(y.value() == res)

    def test_multidim(self):

        functions = ["all", "any", "max", "min", "sum", "prod"]
        bv = cp.boolvar(shape=(5,4,3,2)) # high dimensional tensor
        arr = np.zeros(shape=bv.shape) # numpy "ground truth"

        for axis in range(len(bv.shape)):
            np_res = arr.sum(axis=axis)
            for func in functions:
                cpm_res = getattr(bv, func)(axis=axis)
                assert isinstance(cpm_res, NDVarArray)
                assert cpm_res.shape == np_res.shape

class TestBounds:
    def test_bounds_mul_sub_sum(self):
        x = intvar(-8,8)
        y = intvar(-4,6)
        for name, test_lb, test_ub in [('mul',-48,48),('sub',-14,12),('sum',-12,14)]:
            if name == 'mul':
                op = cp.Multiplication(x, y)
            else:
                op = Operator(name,[x,y])
            lb, ub = op.get_bounds()
            assert test_lb ==lb
            assert test_ub ==ub
            for lhs in inclusive_range(*x.get_bounds()):
                for rhs in inclusive_range(*y.get_bounds()):
                    if name == 'mul':
                        val = lhs * rhs
                    else:
                        val = Operator(name,[lhs,rhs]).value()
                    assert val >=lb
                    assert val <=ub

    def test_bounds_wsum(self):
        x = intvar(-8, 8,3)
        weights = [2,4,-3]
        op = Operator('wsum',[weights,x])
        lb, ub = op.get_bounds()
        assert lb ==-72
        assert ub ==72
        for x1 in inclusive_range(*x[0].get_bounds()):
            for x2 in inclusive_range(*x[1].get_bounds()):
                for x3 in inclusive_range(*x[2].get_bounds()):
                    val = Operator('wsum',[weights,[x1,x2,x3]]).value()
                    assert val >=lb
                    assert val <=ub

    def test_bounds_unary(self):
        x = intvar(-8, 5)
        y = intvar(-7, -2)
        z = intvar(1, 9)
        for var,test_lb,test_ub in [(x,-5,8),(y,2,7),(z,-9,-1)]:
            name = '-'
            op = Operator(name,[var])
            lb, ub = op.get_bounds()
            assert test_lb ==lb
            assert test_ub ==ub
            for lhs in inclusive_range(*var.get_bounds()):
                val = Operator(name, [lhs]).value()
                assert val >=lb
                assert val <=ub


    def test_list(self):

        # cpm_array
        iv = cp.intvar(0,10,shape=3)
        lbs, ubs = iv.get_bounds()
        assert [0,0,0] == lbs.tolist()
        assert [10,10,10] == ubs.tolist()
        # list
        iv = [cp.intvar(0,10) for _ in range(3)]
        lbs, ubs = get_bounds(iv)
        assert [0, 0, 0] == lbs
        assert [10, 10, 10] == ubs
        # nested list
        exprs = [intvar(0,1), [intvar(2,3), intvar(4,5)], [intvar(5,6)]]
        lbs, ubs = get_bounds(exprs)
        assert [0,[2,4],[5]] == lbs
        assert [1,[3,5],[6]] == ubs


    def test_array(self):
        m = intvar(-3,3, shape = (3,2), name= [['a','b'],['c','d'],['e','f']])
        assert str(cpm_array(m)) == '[[a b]\n [c d]\n [e f]]'
        assert str(cpm_array(m.T)) == '[[a c e]\n [b d f]]'

    def test_not_operator(self):
        p = boolvar()
        q = boolvar()
        x = intvar(0,9)
        assert cp.Model([~p]).solve()
        #self.assertRaises(cp.exceptions.TypeError, cp.Model([~x]).solve())
        assert cp.Model([~(x == 0)]).solve()
        assert cp.Model([~~p]).solve()
        assert cp.Model([~(p & p)]).solve()
        assert cp.Model([~~~~~(p & p)]).solve()
        assert cp.Model([~cpm_array([p,q,p])]).solve()
        assert cp.Model([~p.implies(q)]).solve()
        assert cp.Model([~p.implies(~q)]).solve()
        assert cp.Model([p.implies(~q)]).solve()
        assert cp.Model([p == ~q]).solve()
        assert cp.Model([~~p == ~q]).solve()
        assert cp.Model([Operator('not',[p]) == q]).solve()
        assert cp.Model([Operator('not',[p])]).solve()

    def test_description(self):

        a,b = cp.boolvar(name="a"), cp.boolvar(name="b")
        cons = a | b
        cons.set_description("either a or b should be true, but not both")

        assert repr(cons) == "(a) or (b)"
        assert str(cons) == "either a or b should be true, but not both"

        # ensure nothing goes wrong due to calling __str__ on a constraint with a custom description
        for solver,cls in cp.SolverLookup.base_solvers():
            if not cls.supported():
                continue
            if solver == "rc2":
                continue
            print("Testing", solver)
            assert cp.Model(cons).solve(solver=solver)

        ## test extra attributes of set_description
        cons = a | b
        cons.set_description("either a or b should be true, but not both",
                             override_print=False)

        assert repr(cons) == "(a) or (b)"
        assert str(cons) == "(a) or (b)"

        cons = a | b
        cons.set_description("either a or b should be true, but not both",
                             full_print=True)

        assert repr(cons) == "(a) or (b)"
        assert str(cons) == "either a or b should be true, but not both -- (a) or (b)"


    def test_dtype(self):

        x = cp.intvar(1,10,shape=(3,3), name="x")
        assert cp.Model(cp.sum(x) >= 10).solve()
        assert x.value() is not None
        # test all types of expressions
        assert int == type(x[0,0].value())# just the var
        for v in x[0]:
            assert int == type(v.value())# array of var
        assert int == type(cp.sum(x[0]).value())
        assert int == type(cp.sum(x).value())
        assert int == type(cp.sum([1,2,3] * x[0]).value())
        
        # also numpy should be converted to Python native when callig value()
        assert int == type(cp.sum(np.array([1, 2, 3]) * x[0]).value())
        
        # test binary operators
        a,b = x[0,[0,1]]
        assert int == type((-a).value())
        assert int == type((a - b).value())
        assert int == type((a * b).value())
        assert int == type((a // b).value())
        # self.assertEqual(int, type((a ** b).value())) -> We don't allow variables as exponent anymore
        assert int == type((a % b).value())

        # test binary operators with numpy constants
        a,b = x[0,0], np.int64(42)
        assert int == type((a - b).value())
        assert int == type((a * b).value())
        assert int == type((a // b).value())
        assert int == type((a ** b).value())
        assert int == type((a % b).value())

        # test comparisons
        a,b = x[0,[0,1]]
        assert bool == type((a < b).value())
        assert bool == type((a <= b).value())
        assert bool == type((a > b).value())
        assert bool == type((a >= b).value())
        assert bool == type((a == b).value())
        assert bool == type((a != b).value())

        # alsl comparisons with numpy values
        a,b = x[0,0], np.int64(42)
        assert bool == type((a < b).value())
        assert bool == type((a <= b).value())
        assert bool == type((a > b).value())
        assert bool == type((a >= b).value())
        assert bool == type((a == b).value())
        assert bool == type((a != b).value())

class TestBuildIns:

    def setup_method(self):
        self.x = cp.intvar(0,10,shape=3)

    def test_sum(self):
        gt = Operator("sum", list(self.x))

        assert str(gt) == str(cp.sum(self.x))
        assert str(gt) == str(cp.sum(list(self.x)))
        assert str(gt) == str(cp.sum(v for v in self.x))
        with pytest.raises(TypeError):  # Python sum does not accept sum(1,2,3)
            cp.sum(self.x[0], self.x[1], self.x[2])

    def test_max(self):
        gt = Maximum(self.x)

        assert str(gt) == str(cp.max(self.x))
        assert str(gt) == str(cp.max(list(self.x)))
        assert str(gt) == str(cp.max(v for v in self.x))
        assert str(gt) == str(cp.max(self.x[0], self.x[1], self.x[2]))

    def test_abs(self):
        gt = Abs(self.x[0])
        assert str(gt) == str(cp.abs(self.x[0]))
        self.x[0]._value = 1
        assert gt.value() == 1
        self.x[0]._value = -1
        assert gt.value() == 1
        self.x[0]._value = 0
        assert gt.value() == 0
        self.x[0]._value = None
        assert gt.value() is None

from cpmpy.transformations.get_variables import get_variables
class TestNullifyingArguments:

    def setup_method(self):
        self.x = cp.intvar(0,10, name="x")
        self.b = cp.boolvar(name="b")

    def test_bool(self):

        funcs = ["__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__"]

        for func in funcs:
            expr = getattr(self.b, func)(True)
            assert get_variables(expr) == [self.b]

            expr = getattr(self.b, func)(False)
            assert get_variables(expr) == [self.b]

    def test_num(self):
        funcs = ["__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
                 "__floordiv__", "__rfloordiv__",
                 "__mod__", "__rmod__"]

        for func in funcs:

            expr = getattr(self.x, func)(1)
            assert get_variables(expr) == [self.x]

            expr = getattr(self.x, func)(0)
            assert get_variables(expr) == [self.x]

        with pytest.warns(SyntaxWarning, match="We only support floordivision"):
            expr = self.x / 1
        assert get_variables(expr) == [self.x]

        with pytest.warns(SyntaxWarning, match="We only support floordivision"):
            expr = 1 / self.x
        assert get_variables(expr) == [self.x]





class TestContainer:

    def setup_method(self):
        self.x = cp.intvar(0,10,name="x")
        self.y = cp.intvar(0,10,name="y")
        self.z = cp.intvar(0,10,name="z")

    def test_list(self):
        lst = [self.x, self.y, self.z]
        assert lst == [self.x, self.y, self.z]
        assert not lst == [self.x, self.z, self.y]
        assert lst != [self.x, self.z, self.y]
     
    def test_set(self):
        s = {self.x, self.y, self.z}
        assert s == {self.x, self.y, self.z}
        assert {self.x} <= s # test subset
        assert s & {self.x} == {self.x} # test intersection
        assert s | {self.x} == s # test union
        assert s - {self.x} == {self.y, self.z} # test difference
        assert s ^ {self.x} == {self.y, self.z} # test symmetric difference
        assert s ^ {self.x, self.y} == {self.z}
        assert s ^ {self.x, self.y, self.z} == set()

        # tricky cases, force hash collisions
        self.x.__hash__ = lambda self: 1
        self.y.__hash__ = lambda self: 1
        assert s == {self.x, self.y, self.z}
        assert {self.x} <= s # test subset
        assert s & {self.x} == {self.x} # test intersection
        assert s | {self.x} == s # test union
        assert s - {self.x} == {self.y, self.z} # test difference
        assert s ^ {self.x} == {self.y, self.z} # test symmetric difference
        assert s ^ {self.x, self.y} == {self.z}
        assert s ^ {self.x, self.y, self.z} == set()

    def test_dict(self):
        d = {self.x: "x", self.y: "y", self.z: "z"}
        assert d == {self.x: "x", self.y: "y", self.z: "z"}
        assert d[self.x] == "x"
        assert d[self.y] == "y"
        assert d[self.z] == "z"

        # tricky cases, force hash collisions
        self.x.__hash__ = lambda self: 1
        self.y.__hash__ = lambda self: 1
        assert d == {self.x: "x", self.y: "y", self.z: "z"}
        assert d[self.x] == "x"
        assert d[self.y] == "y"
        assert d[self.z] == "z"

        
class TestUtils:

    def test_cpm_array(self):
        x = cp.intvar(0,10, shape=(5, 3))
        assert isinstance(cpm_array(x), NDVarArray)
        assert cpm_array(x).shape == (5, 3)

        assert isinstance(cpm_array(x.T), NDVarArray)
        assert cpm_array(x.T).shape == (3, 5)

    def test_eval_comparison(self):
        x = intvar(0,10, name="x")

        for comp in ["==", "!=", "<", "<=", ">", ">="]:
            expr = eval_comparison(comp, x, 5)
            assert isinstance(expr, Comparison)
            assert str(expr.args[0]) == "x"
            assert expr.args[1] == 5# should always put the constant on the right

            expr = eval_comparison(comp, 5, x)
            assert isinstance(expr, Comparison)
            assert str(expr.args[0]) == "x"
            assert expr.args[1] == 5# should always put the constant on the right

            # now, also check with numpy
            expr = eval_comparison(comp, x, np.int64(5))
            assert isinstance(expr, Comparison)
            assert str(expr.args[0]) == "x"
            assert expr.args[1] == 5# should always put the constant on the right

            expr = eval_comparison(comp, np.int64(5), x)
            assert isinstance(expr, Comparison)
            assert str(expr.args[0]) == "x"
            assert expr.args[1] == 5# should always put the constant on the right


            # also with Boolean constants

            expr = eval_comparison(comp, x, True)
            assert isinstance(expr, Comparison)
            assert str(expr.args[0]) == "x"
            assert expr.args[1] == True# should always put the constant on the right

            expr = eval_comparison(comp, True, x)
            assert isinstance(expr, Comparison)
            assert str(expr.args[0]) == "x"
            assert expr.args[1] == True# should always put the constant on the right

            # now, also check with numpy
            expr = eval_comparison(comp, x, np.bool_(True))
            assert isinstance(expr, Comparison)
            assert str(expr.args[0]) == "x"
            assert expr.args[1] == True# should always put the constant on the right

            expr = eval_comparison(comp, np.bool_(True), x)
            assert isinstance(expr, Comparison)
            assert str(expr.args[0]) == "x"
            assert expr.args[1] == True# should always put the constant on the right
