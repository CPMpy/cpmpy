import unittest

import cpmpy as cp
from cpmpy.expressions import boolvar, intvar
from cpmpy.transformations.linearize import linearize_constraint

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
