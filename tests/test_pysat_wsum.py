import unittest
import cpmpy as cp 
from cpmpy import *
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.to_cnf import to_cnf

class TestCardinality(unittest.TestCase):
    def setUp(self):
        self.bvs = boolvar(shape=5)

    def test_unit_wsum(self):
        v = self.bvs[0] + self.bvs[1] + self.bvs[2] < 2
        print(to_cnf(v))
        s = CPM_pysat(Model(v))
        s.solve()
        print(self.bvs.value())

    def test_wsum(self):
        formula = 2 * self.bvs[0] + self.bvs[1] + self.bvs[2] < 2
        cnf_formula = to_cnf(formula)
        print(formula, cnf_formula)
        s = CPM_pysat(Model(cnf_formula))
        s.solve()
        print(self.bvs.value())

if __name__ == '__main__':
    unittest.main()
