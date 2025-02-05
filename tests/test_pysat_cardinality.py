import unittest
import pytest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.expressions.core import Operator
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.linearize import only_positive_coefficients

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

    def test_pysat_linear_other(self):
        expressions = [
            self.bvs[0] + self.bvs[1] + self.bvs[2] > 0,
            # now with var/expr on RHS
            self.bvs[0] + self.bvs[1] > self.bvs[2],
            self.bvs[0] > self.bvs[1] + self.bvs[2],
            self.bvs[0] > (self.bvs[1] | self.bvs[2]),
        ]

        ## check all types of linear constraints are handled
        for expression in expressions:
            cp.Model(expression).solve("pysat")

    def test_pysat_oob(self):

        def test_encode_pb_oob(self):
            self.assertTrue(len(self.bv) == 3)
            # test out of bounds (meaningless) thresholds
            expressions = [
                sum(self.bv) <= 5,  # true
                sum(self.bv) <= 3,  # true
                sum(self.bv) <= -2,  # false
                sum(self.bv) <= 0,  # undecided

                sum(self.bv) >= -2,  # true
                sum(self.bv) >= 0,  # true
                sum(self.bv) >= 5,  # false
                sum(self.bv) >= 3,  # undecided
            ]

            ## check all types of linear constraints are handled
            for expression in expressions:
                cp.Model(expression).solve("pysat")

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

    def test_pysat_aggregate_sum_sub_expressions(self):
        bvs = cp.boolvar(3)
        c = bvs[0] > sum(bvs[1:])
        s = cp.SolverLookup.get("pysat")
        s += c
        self.assertTrue(s.solve())

    def test_pysat_aggregate_sum_sub_expressions_implied(self):
        bvs = cp.boolvar(3)
        c = bvs[0] > sum(bvs[1:])
        s = cp.SolverLookup.get("pysat")
        s += c
        self.assertTrue(s.solve())

    def test_pysat_aggregate_sum_sub_expressions_implied(self):
        a, b, c, p = [cp.boolvar(name=n) for n in "abcp"]
        self.assertTrue(cp.SolverLookup.get("pysat", cp.Model(p.implies(a+b-c < 2))).solve())

    @pytest.mark.skip(reason="TODO: PySAT does not linearize models at the moment, because there is no integer encoding layer, so adding non-linear expressions will always fail.")
    def test_pysat_linearize_example(self):
        self.assertTrue(cp.SolverLookup.get("pysat", cp.Model((a+b).implies(a+b-c < 2))).solve())


if __name__ == '__main__':
    unittest.main()
