import unittest

import cpmpy as cp
from cpmpy.expressions import boolvar, intvar
from cpmpy.expressions.core import Operator
from cpmpy.transformations.linearize import linearize_constraint, only_const_rhs, only_var_lhs

class TestTransLineariez(unittest.TestCase):

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



class TestTransConstRhs(unittest.TestCase):

    def test_sum(self):
        a,b,c = [cp.intvar(0,10,name=n) for n in "abc"]
        rhs = intvar(0,10,name="r")

        cons = only_const_rhs(cp.sum([a,b,c]) <= rhs)[0]
        self.assertEqual("sum([1, 1, 1, -1] * [a, b, c, r]) <= 0", str(cons))

    def test_wsum(self):
        a, b, c = [cp.intvar(0, 10,name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")

        cons = 1*a + 2*b + 3*c <= rhs
        cons = only_const_rhs(cons)[0]
        self.assertEqual("sum([1, 2, 3, -1] * [a, b, c, r]) <= 0", str(cons))

    def test_impl(self):
        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")
        cond = cp.boolvar(name="bv")

        cons = cond.implies(1 * a + 2 * b + 3 * c <= rhs)
        cons = only_const_rhs(cons)[0]
        self.assertEqual("(bv) -> (sum([1, 2, 3, -1] * [a, b, c, r]) <= 0)", str(cons))

        cons = (~cond).implies(1 * a + 2 * b + 3 * c <= rhs)
        cons = only_const_rhs(cons)[0]
        self.assertEqual("(~bv) -> (sum([1, 2, 3, -1] * [a, b, c, r]) <= 0)", str(cons))

    def test_others(self):

        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")

        cons = cp.max([a,b,c]) <= rhs
        cons = only_const_rhs(cons)[0]
        self.assertEqual("(max(a,b,c)) <= (r)", str(cons))

        cons = cp.AllDifferent([a,b,c])
        cons = only_const_rhs(cons)[0]
        self.assertEqual("alldifferent(a,b,c)", str(cons))


class TestTransVarsLhs(unittest.TestCase):

    def test_sum(self):
        a,b,c = [cp.intvar(0,10,name=n) for n in "abc"]
        rhs = 5

        cons = only_var_lhs(cp.sum([a,b,c,10]) <= rhs)[0]
        self.assertEqual("sum([a, b, c]) <= -5", str(cons))

    def test_wsum(self):
        a, b, c = [cp.intvar(0, 10,name=n) for n in "abc"]
        rhs = 5

        cons = Operator("wsum",[[1,2,3,-1],[a,b,c,10]]) <= rhs
        cons = only_var_lhs(cons)[0]
        self.assertEqual("sum([1, 2, 3] * [a, b, c]) <= 15", str(cons))

    def test_impl(self):
        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = 5
        cond = cp.boolvar(name="bv")

        cons = cond.implies(Operator("wsum",[[1,2,3,-1],[a,b,c,10]]) <= rhs)
        cons = only_var_lhs(cons)[0]
        self.assertEqual("(bv) -> (sum([1, 2, 3] * [a, b, c]) <= 15)", str(cons))

        cons = (~cond).implies(Operator("wsum",[[1,2,3,-1],[a,b,c,10]]) <= rhs)
        cons = only_var_lhs(cons)[0]
        self.assertEqual("(~bv) -> (sum([1, 2, 3] * [a, b, c]) <= 15)", str(cons))

    def test_others(self):

        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")

        cons = cp.max([a,b,c,5]) <= rhs
        cons = only_var_lhs(cons)[0]
        self.assertEqual("(max(a,b,c,5)) <= (r)", str(cons))

        cons = cp.AllDifferent([a, b, c])
        cons = only_var_lhs(cons)[0]
        self.assertEqual("alldifferent(a,b,c)", str(cons))


