import unittest

from cpmpy import *
from cpmpy.solvers import CPM_pysat, CPM_RC2
from cpmpy.transformations.get_variables import get_variables_model

def frietkot():
    # Construct the model.
    (x1, x2, x3) = boolvar(3)
    weights = [10, 10, 3, 5, 40, 20]

    # Pure CNF
    c1 = ~x1 | ~x2 | x3
    c2 = ~x1 | x2 | x3
    c3 = ~x2 | ~x3
    c4 = x1
    c5 = ~x1 | x2
    c6 = ~x1 | ~x3

    allwishes = cpm_array([c1, c2, c3, c4, c5, c6])
    assum_vars = boolvar(len(allwishes))

    model = Model(assum_vars.implies(allwishes), maximize=sum(w*x for w, x in zip(weights, assum_vars)))
    assert not Model(allwishes).solve(), "UNSAT Model"

    return model, assum_vars, [x1, x2, x3]

def simple():
    (x1, x2) = boolvar(2)
    weights = [10, 7, 3, 5]

    # Pure CNF
    c1 = ~x1 | ~x2
    c2 = ~x1 | x2 
    c3 = x1 | x2
    c4 = x1 | ~x2

    allwishes = cpm_array([c1, c2, c3, c4])
    assum_vars = boolvar(len(allwishes))

    model = Model(assum_vars.implies(allwishes), maximize=sum(w*x for w, x in zip(weights, assum_vars)))
    assert not Model(allwishes).solve(), "UNSAT Model"

    return model, assum_vars, [x1, x2]


class TestPySATRC2(unittest.TestCase):
    def test_frietkot(self):
        """Check if the instance still returns the expected results
        after adding interrupt to pysat solver.
        """
        frietkot_model, assum_vars, _ = frietkot()
        s = CPM_RC2(frietkot_model)
        self.assertTrue(s.solve(), "Problem solvable with current weights")
        print()
    
    def test_simple(self):
        """Check if the instance still returns the expected results
        after adding interrupt to pysat solver.
        """
        simple_model, assum_vars, _ = simple()
        s = CPM_RC2(simple_model)
        s = CPM_ortools
        solcount = s.solveAll(display=get_variables_model(simple_model))
        # self.assertTrue(s.solve(), "Problem solvable with current weights")
        # print()


if __name__ == '__main__':
    unittest.main()