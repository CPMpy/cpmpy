import unittest
import pytest
import cpmpy as cp 
from cpmpy import *
from cpmpy.solvers.pysat import CPM_pysat

import importlib # can check for modules *without* importing them
pysat_available = CPM_pysat.supported()
pblib_available = importlib.util.find_spec("pypblib") is not None

@pytest.mark.skipif(not (pysat_available and not pblib_available), reason="`pysat` is not installed" if not pysat_available else "`pypblib` is installed")
def test_pypblib_error():
    # NOTE if you want to run this but pypblib is already installed, run `pip uninstall pypblib && pip install -e .[pysat]`
    unittest.TestCase().assertRaises(
            ImportError, # just solve a pb constraint with pypblib not installed
            lambda : CPM_pysat(cp.Model(2*cp.boolvar() + 3 * cp.boolvar() + 5 * cp.boolvar() <= 6)).solve()
        )

    # this one should still work without `pypblib`
    assert CPM_pysat(cp.Model(1*cp.boolvar() + 1 * cp.boolvar() + 1 * cp.boolvar() <= 2)).solve()

@pytest.mark.skipif(not (pysat_available and pblib_available), reason="`pysat` is not installed" if not pysat_available else "`pypblib` not installed")
class TestEncodePseudoBooleanConstraint(unittest.TestCase):
    def setUp(self):
        self.bv = boolvar(shape=3)

    def test_pysat_simple_atmost(self):

        atmost = cp.Model(
            ## <
            - 2 * self.bv[0] < 3,
            ## <=
            - 3 * self.bv[1] <= 3,
            ## >
            2 * self.bv[2] > 1,
            ## >=
            4 * self.bv[2] >= 3,
        )
        ps = CPM_pysat(atmost)
        ps.solve()


    def test_pysat_wsum_triv_sat(self):
        ls = cp.Model(
            2 * self.bv[0] + 3 * self.bv[1] <= 10,
        )
        ps = CPM_pysat(ls)
        solved = ps.solve()
        self.assertTrue(solved)

    def test_pysat_unsat(self):
        ls = cp.Model(
            2 * self.bv[0] + 3 * self.bv[1] <= 3,
            self.bv[0] == 1,
            self.bv[1] == 1
        )

        ps = CPM_pysat(ls)
        solved = ps.solve()
        self.assertFalse(solved)

    def test_encode_pb_expressions(self):
        expressions = [
            - self.bv[2] == -1,
            - 2 * self.bv[2] == -2,
            self.bv[0] - self.bv[2] > 0,
            -self.bv[0] + self.bv[2] > 0,
            2 * self.bv[0] + 3 * self.bv[2] > 0,
            2 * self.bv[0] - 3 * self.bv[2] + 2 * self.bv[1] > 0,
            self.bv[0] - 3 * self.bv[2] > 0,
            self.bv[0] - 3 * (self.bv[2] + 2 * self.bv[1])> 0,
            # now with var on RHS
            self.bv[0] - 3 * self.bv[1] > self.bv[2],
        ]

        ## check all types of linear constraints are handled
        for expression in expressions:
            Model(expression).solve("pysat")

    def test_encode_pb_oob(self):
        # test out of bounds (meaningless) thresholds
        expressions = [
            sum(self.bv*[2,2,2]) <= 10,  # true
            sum(self.bv*[2,2,2]) <= 6,   # true
            sum(self.bv*[2,2,2]) >= 10,  # false
            sum(self.bv*[2,2,2]) >= 6,   # undecided
            sum(self.bv*[2,-2,2]) <= 10,  # true
            sum(self.bv*[2,-2,2]) <= 4,   # true
            sum(self.bv*[2,-2,2]) >= 10,  # false
            sum(self.bv*[2,-2,2]) >= 4,   # undecided
        ]

        ## check all types of linear constraints are handled
        for expression in expressions:
            Model(expression).solve("pysat")

if __name__ == '__main__':
    unittest.main()

