import unittest
from cpmpy import *
from cpmpy.solvers import CPM_pysat, CPM_RC2

def frietkot():
    # Construct the model.
    (x1, x2, x3) = boolvar(3)
    weights = [10, 20,]

    # Pure CNF
    c1 = ~x1 | ~x2 | x3
    c2 = ~x1 | x2 | x3
    c3 = ~x2 | ~x3
    c4 = x1
    c5 = ~x1 | x2
    c6 = ~x1 | ~x3

    allwishes = cpm_array([c1, c2, c3, c4, c5, c6])
    assum_vars = boolvar(len(allwishes))

    model = Model(assum_vars.implies(allwishes), maximize=)
    return model, assum_vars, [x1, x2, x3]


class TestPySATInterrupt(unittest.TestCase):
    def test_maxsat(self):
        """Check if the instance still returns the expected results
        after adding interrupt to pysat solver.
        """
        frietkot_model, assum_vars, variables = frietkot()
        s = CPM_RC2(frietkot_model)
        status = s.solve(assumptions=assum_vars)
        print(status)

if __name__ == '__main__':
    unittest.main()