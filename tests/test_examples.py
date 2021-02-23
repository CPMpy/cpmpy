import unittest
import cppy
import numpy as np

class TestExamples(unittest.TestCase):

    def test_send_more_money(self):
        # Construct the model.
        s,e,n,d,m,o,r,y = cppy.IntVar(0,9, 8)

        constraint = []
        constraint += [ cppy.alldifferent([s,e,n,d,m,o,r,y]) ]
        constraint += [    sum(   [s,e,n,d] * np.flip(10**np.arange(4)) )
                         + sum(   [m,o,r,e] * np.flip(10**np.arange(4)) )
                        == sum( [m,o,n,e,y] * np.flip(10**np.arange(5)) ) ]
        constraint += [ s > 0, m > 0 ]

        model = cppy.Model(constraint)
        for solver in cppy.get_supported_solvers():
            _ = model.solve(solver=solver)
            self.assertEqual([x.value() for x in [s,e,n,d]], [9, 5, 6, 7])
            self.assertEqual([x.value() for x in [m,o,r,e]], [1, 0, 8, 5])
            self.assertEqual([x.value() for x in [m,o,n,e,y]], [1, 0, 6, 5, 2])
