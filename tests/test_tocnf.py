import unittest
import numpy as np
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.core import Operator
from cpmpy.expressions.globalconstraints import Xor

class TestToCnf(unittest.TestCase):
    def test_tocnf(self):
        a,b,c = boolvar(shape=3)

        cases = [a,
                 a|b,
                 a&b,
                 a!=b,
                 a==b,
                 a.implies(b),
                 a.implies(b|c),
                 a.implies(b&c),
                 a.implies(b!=c),
                 a.implies(b==c),
                 a.implies(b.implies(c)),
                 (b|c).implies(a),
                 (b&c).implies(a),
                 (b!=c).implies(a),
                 (b==c).implies(a),
                 (b.implies(c)).implies(a),
                 Xor([a,b]),
                ]

        # test for equivalent solutions with/without to_cnf
        for case in cases:
            vs = cpm_array(get_variables(case))
            s1 = self.allsols([case], vs)
            s1.sort(axis=0)
            s2 = self.allsols(to_cnf(case), vs)
            s2.sort(axis=0)
            for ss1,ss2 in zip(s1,s2):
                self.assertTrue(np.all(ss1 == ss2), (case, s1, s2))

        # test for errors in edge cases of to_cnf
        bvs = boolvar(shape=3)
        ivs = intvar(lb=2, ub=3, shape=3)
        edge_cases = [
            # do not consider object as a double implcation, but as a sum
            (a + b + c) == 1,
            a * b == 1,
            a * b != 1,
            (a + b + c) != 1,
            sum(bvs) > 2,
            sum(bvs) <= 2,
            sum(ivs) <= 3
        ]

        # check for error in edge cases
        for case in edge_cases:
            cnf = to_cnf(case)
            # Expressions should not be decomposed at the to_cnf level!
            self.assertEqual(len(cnf), 1)

    def allsols(self, cons, vs):
        sols = []

        m = CPM_ortools(Model(cons))
        while m.solve():
            sols.append(vs.value())
            m += ~all(vs == vs.value())

        return np.array(sols)


if __name__ == '__main__':
    unittest.main()

