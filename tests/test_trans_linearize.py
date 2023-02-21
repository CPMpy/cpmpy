import unittest

import cpmpy as cp
from cpmpy.expressions import boolvar, intvar
from cpmpy.expressions.core import Operator
from cpmpy.transformations.linearize import linearize_constraint
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl


class TestTransLinearize(unittest.TestCase):

    def setUp(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0
        self.ivars = cp.intvar(1, 10, shape=(5,))
        self.bvars = cp.boolvar((3,))

    def test_linearize(self):

        # Boolean
        a, b, c = [boolvar(name=var) for var in "abc"]

        # and
        cons = linearize_constraint(a & b)[0]
        self.assertEqual("(a) + (b) >= 2", str(cons))

        # or
        cons = linearize_constraint(a | b)[0]
        self.assertEqual("(a) + (b) >= 1", str(cons))

        # xor
        cons = linearize_constraint(a ^ b)[0]
        self.assertEqual("(a) + (b) == 1", str(cons))

        # implies
        cons = linearize_constraint(a.implies(b))[0]
        self.assertEqual("(a) -> (b >= 1)", str(cons))
    
    def test_bug_168(self):
        from cpmpy.solvers import CPM_gurobi
        if CPM_gurobi.supported():
            bv = boolvar(shape=2)
            iv = intvar(1, 9)
            e1 = (bv[0] * bv[1] == iv)
            s1 = cp.Model(e1).solve("gurobi")
            s1 = cp.Model(e1).solve("ortools")
            self.assertTrue(s1)
            self.assertEqual([bv[0].value(), bv[1].value(), iv.value()],[True, True, 1])

    def test_constraint(self):
        x,y,z = [cp.intvar(0,5, name=n) for n in "xyz"]
        a,b,c = [cp.boolvar(name=n) for n in "abc"]

        # test and
        self.assertEqual(str(linearize_constraint(a & b & c)), "[sum([a, b, c]) >= 3]")
        self.assertEqual(str(linearize_constraint(a & b & (~c))), "[sum([a, b, ~c]) >= 3]")
        # test or
        self.assertEqual(str(linearize_constraint(a | b | c)), "[sum([a, b, c]) >= 1]")
        self.assertEqual(str(linearize_constraint(a | b | (~c))), "[sum([a, b, ~c]) >= 1]")
        # test implies
        self.assertEqual(str(linearize_constraint(a.implies(b))), "[(~a) + (b) >= 1]")
        self.assertEqual(str(linearize_constraint(a.implies(~b))), "[(~a) + (~b) >= 1]")
        self.assertEqual(str(linearize_constraint(a.implies(x+y+z >= 0))), "[(a) -> (sum([x, y, z]) >= 0)]")
        self.assertEqual(str(linearize_constraint(a.implies(x+y+z > 0))), "[(a) -> (sum([x, y, z]) >= 1)]")
        # test sub
        self.assertEqual(str(linearize_constraint(Operator("sub",[x,y]) >= z)), "[sum([1, -1, -1] * [x, y, z]) >= 0]")
        # test mul
        self.assertEqual(str(linearize_constraint(3 * x > 2)), "[sum([3] * [x]) >= 3]")
        # test <
        self.assertEqual((str(linearize_constraint(x + y  < z))), "[sum([1, 1, -1] * [x, y, z]) <= -1]")
        # test >
        self.assertEqual((str(linearize_constraint(x + y  > z))), "[sum([1, 1, -1] * [x, y, z]) >= 1]")
        # test !=
        c1,c2 = linearize_constraint(x + y  != z)
        self.assertEqual(str(c1), "(BV3) -> (sum([1, 1, -1] * [x, y, z]) <= -1)")
        self.assertEqual(str(c2), "(~BV3) -> (sum([1, 1, -1] * [x, y, z]) >= 1)")
        c1, c2 = linearize_constraint(a.implies(x != y))
        self.assertEqual(str(c1), "(a) -> (sum([1, -1, -6] * [x, y, BV4]) <= -1)")
        self.assertEqual(str(c2), "(a) -> (sum([1, -1, 6] * [x, y, BV4]) >= -5)")



class TestConstRhs(unittest.TestCase):

    def  test_numvar(self):
        a, b = [cp.intvar(0, 10, name=n) for n in "ab"]

        cons = linearize_constraint(a <= b)[0]
        self.assertEqual("sum([1, -1] * [a, b]) <= 0", str(cons))

    def test_sum(self):
        a,b,c = [cp.intvar(0,10,name=n) for n in "abc"]
        rhs = intvar(0,10,name="r")

        cons = linearize_constraint(cp.sum([a,b,c]) <= rhs)[0]
        self.assertEqual("sum([1, 1, 1, -1] * [a, b, c, r]) <= 0", str(cons))

    def test_wsum(self):
        a, b, c = [cp.intvar(0, 10,name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")

        cons = 1*a + 2*b + 3*c <= rhs
        cons = linearize_constraint(cons)[0]
        self.assertEqual("sum([1, 2, 3, -1] * [a, b, c, r]) <= 0", str(cons))

    def test_impl(self):
        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")
        cond = cp.boolvar(name="bv")

        cons = cond.implies(1 * a + 2 * b + 3 * c <= rhs)
        cons = linearize_constraint(cons)[0]
        self.assertEqual("(bv) -> (sum([1, 2, 3, -1] * [a, b, c, r]) <= 0)", str(cons))

        cons = (~cond).implies(1 * a + 2 * b + 3 * c <= rhs)
        cons = linearize_constraint(cons)[0]
        self.assertEqual("(~bv) -> (sum([1, 2, 3, -1] * [a, b, c, r]) <= 0)", str(cons))

    def test_others(self):

        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")

        cons = cp.max([a,b,c]) <= rhs
        cons = linearize_constraint(cons)[0]
        self.assertEqual("(max(a,b,c)) <= (r)", str(cons))

        cons = cp.AllDifferent([a,b,c])
        cons = linearize_constraint(cons)[0]
        self.assertEqual("alldifferent(a,b,c)", str(cons))


class TestVarsLhs(unittest.TestCase):

    def test_sum(self):
        a,b,c = [cp.intvar(0,10,name=n) for n in "abc"]
        rhs = 5

        cons = linearize_constraint(cp.sum([a,b,c,10]) <= rhs)[0]
        self.assertEqual("sum([a, b, c]) <= -5", str(cons))

    def test_wsum(self):
        a, b, c = [cp.intvar(0, 10,name=n) for n in "abc"]
        rhs = 5

        cons = Operator("wsum",[[1,2,3,-1],[a,b,c,10]]) <= rhs
        cons = linearize_constraint(cons)[0]
        self.assertEqual("sum([1, 2, 3] * [a, b, c]) <= 15", str(cons))

    def test_impl(self):
        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = 5
        cond = cp.boolvar(name="bv")

        cons = cond.implies(Operator("wsum",[[1,2,3,-1],[a,b,c,10]]) <= rhs)
        cons = linearize_constraint(cons)[0]
        self.assertEqual("(bv) -> (sum([1, 2, 3] * [a, b, c]) <= 15)", str(cons))

        cons = (~cond).implies(Operator("wsum",[[1,2,3,-1],[a,b,c,10]]) <= rhs)
        cons = linearize_constraint(cons)[0]
        self.assertEqual("(~bv) -> (sum([1, 2, 3] * [a, b, c]) <= 15)", str(cons))

    def test_others(self):

        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")

        cons = cp.max([a,b,c,5]) <= rhs
        cons = linearize_constraint(cons)[0]
        self.assertEqual("(max(a,b,c,5)) <= (r)", str(cons))

        cons = cp.AllDifferent([a, b, c])
        cons = linearize_constraint(cons)[0]
        self.assertEqual("alldifferent(a,b,c)", str(cons))


