import unittest
import numpy as np
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.core import Operator

class TestToCnf(unittest.TestCase):
    def test_tocnf(self):
        a,b,c = boolvar(shape=3)

        cases = [a,
                 a|b,
                 a&b,
                 Operator("xor", [a,b]),
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

    def allsols(self, cons, vs):
        sols = []

        m = CPM_ortools(Model(cons))
        while m.solve():
            sols.append(vs.value())
            m += ~all(vs == vs.value())

        return np.array(sols)

