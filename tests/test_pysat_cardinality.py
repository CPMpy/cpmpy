import unittest
import pytest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.expressions.core import Operator
from cpmpy.solvers.pysat import CPM_pysat, only_positive_coefficients

@pytest.mark.skipif(not CPM_pysat.supported(),
                    reason="PySAT not installed")
class TestCardinality(unittest.TestCase):
    def setUp(self):
        self.bv_before = boolvar(shape=7)
        self.bvs = cpm_array(boolvar(shape=2).tolist() + [~boolvar()])

    def test_pysat_atmost(self):

        atmost = cp.Model(
            sum(self.bvs) < 2
        )
        ps = CPM_pysat(atmost)
        self.assertTrue(ps.solve())
        # all must be true

        self.assertLess(sum(self.bvs.value()), 2)

    def test_pysat_atmost_edge_case(self):

        atmost = cp.Model(
            sum(self.bvs) < 8,
            sum(self.bvs) < 1,
        )
        ps = CPM_pysat(atmost)
        self.assertTrue(ps.solve())
        # all must be true
        self.assertLess(sum(self.bvs.value()), 2)

    def test_pysat_atleast(self):

        atmost = cp.Model(
            sum(self.bvs) > 2
        )
        ps = CPM_pysat(atmost)
        self.assertTrue(ps.solve())
        # all must be true
        self.assertEqual(sum(self.bvs.value()), 3)

    def test_pysat_atleast_edge_case(self):

        atmost = cp.Model(
            sum(self.bvs) < 0
        )

        with self.assertRaises(ValueError):
            ps = CPM_pysat(atmost)


    def test_pysat_equals(self):
        equals = cp.Model(
            sum(self.bvs) == 2
        )
        ps = CPM_pysat(equals)
        self.assertTrue(ps.solve())
        self.assertEqual(sum(self.bvs.value()), 2)

        equals2 = cp.Model(
            sum(self.bvs) >= 2,
            sum(self.bvs) <= 2,
        )
        ps2 = CPM_pysat(equals2)
        self.assertTrue(ps2.solve())
        self.assertEqual(sum(self.bvs.value()), 2)

        equals3 = cp.Model(
            sum(self.bvs) > 1,
            sum(self.bvs) < 3,
        )
        ps3 = CPM_pysat(equals3)
        self.assertTrue(ps3.solve())
        self.assertEqual(sum(self.bvs.value()), 2)

    def test_pysat_atmost_equals(self):
        atmost_equals = cp.Model(
            sum(self.bvs) <= 2,
        )
        ps = CPM_pysat(atmost_equals)
        self.assertTrue(ps.solve())
        self.assertLessEqual(sum(self.bvs.value()), 2)

    def test_pysat_atleast_equals(self):
        atleast_equals = cp.Model(
            sum(self.bvs) >= 2,
        )
        ps = CPM_pysat(atleast_equals)
        self.assertTrue(ps.solve())

        self.assertGreaterEqual(sum(self.bvs.value()), 2)

    def test_pysat_different(self):
        
        differrent = cp.Model(
            sum(self.bvs) != 3,
            sum(self.bvs) != 1,
            sum(self.bvs) != 0,
        )
        ps = CPM_pysat(differrent)
        self.assertTrue(ps.solve())
        self.assertGreaterEqual(sum(self.bvs.value()), 2)

    def test_pysat_card_implied(self):
        b = cp.boolvar()
        x = cp.boolvar(shape=5)

        cons = [b.implies(sum(x) > 3),
                b.implies(sum(x) <= 1),
                b.implies(sum(x) != 4),
                b == (sum(x) >= 2),
                b == (sum(x) < 3),
                b == (sum(x) == 2),
                b == (sum(x) != 2),
                (sum(x) > 3).implies(b),
                (sum(x) <= 4).implies(b),
                (sum(x) == 3).implies(b),
                (sum(x) != 3).implies(b),
               ]
        for c in cons:
            cp.Model(c).solve("pysat")
            self.assertTrue(c.value())

    def test_pysat_support_negative_coefficients(self):
        bvs = cp.boolvar(3)
        # this is linearized to `a+b-c>0`, which previously wasn't rewritten to the cardinality constraint `a+b+~c>1`
        c = sum(bvs[1:]) > bvs[0]
        s = cp.SolverLookup.get("pysat")
        s += c
        self.assertTrue(s.solve())

    def test_only_positive_coefficients(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        only_pos = only_positive_coefficients([Operator("wsum",[[1,1,-1],[a,b,c]]) > 0])
        self.assertEqual(str([Operator("sum",[a, b, ~c]) > 1]), str(only_pos))


if __name__ == '__main__':
    unittest.main()
