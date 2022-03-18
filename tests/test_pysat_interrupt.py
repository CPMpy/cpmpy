from email.policy import default
import unittest
from cpmpy import *
from cpmpy.solvers import CPM_pysat
from cpmpy.transformations.get_variables import get_variables_model

def frietkot():
    # Construct the model.
    (mayo, ketchup, curry, andalouse, samurai) = boolvar(5)

    # Pure CNF
    Nora = mayo | ketchup
    Leander = ~samurai | mayo
    Benjamin = ~andalouse | ~curry | ~samurai
    Behrouz = ketchup | curry | andalouse
    Guy = ~ketchup | curry | andalouse
    Daan = ~ketchup | ~curry | andalouse
    Celine = ~samurai
    Anton = mayo | ~curry | ~andalouse
    Danny = ~mayo | ketchup | andalouse | samurai
    Luc = ~mayo | samurai

    allwishes = [Nora, Leander, Benjamin, Behrouz, Guy, Daan, Celine, Anton, Danny, Luc]

    model = Model(allwishes)
    return model, [mayo, ketchup, curry, andalouse, samurai]

def frietkot_n(n):
    allvars = []
    allwishes = []

    # Construct the model.
    for i in range(n):
        (mayo, ketchup, curry, andalouse, samurai) = boolvar(5)
        allvars += [mayo, ketchup, curry, andalouse, samurai]
        # Pure CNF
        Nora = mayo | ketchup
        Leander = ~samurai | mayo
        Benjamin = ~andalouse | ~curry | ~samurai
        Behrouz = ketchup | curry | andalouse
        Guy = ~ketchup | curry | andalouse
        Daan = ~ketchup | ~curry | andalouse
        Celine = ~samurai
        Anton = mayo | ~curry | ~andalouse
        Danny = ~mayo | ketchup | andalouse | samurai
        Luc = ~mayo | samurai

        allwishes += [Nora, Leander, Benjamin, Behrouz, Guy, Daan, Celine, Anton, Danny, Luc]

    model = Model(allwishes)
    return model, allvars

class TestPySATInterrupt(unittest.TestCase):
    def test_small_isntance_no_interrupt(self):
        """Check if the instance still returns the expected results
        after adding interrupt to pysat solver.
        """
        frietkot_model, variables = frietkot()
        s = CPM_pysat(frietkot_model)
        status = s.solve()
        var_state = [v.value() for v in variables]
        self.assertTrue(status)
        self.assertEqual(var_state, [False, True, False, True, False])

    def test_large_instance_interrup(self):
        from pysat.examples.genhard import PHP
        from collections import defaultdict
        import time
        lit_cpmvar = defaultdict(lambda: boolvar())
        cpm_clauses = []
        m  = Model()
        for clause in PHP(nof_holes=10).clauses:
            new_clause = []
            for lit in clause:
                lit_var = lit_cpmvar[abs(lit)]
                new_clause.append(~lit_var if lit < 0 else lit_var)
            m +=any(c for c in new_clause)
        s = CPM_pysat(m)
        assumption = [lit_cpmvar[1]]
        # offset for additional stuff done by cpmpy after solving
        time_limit, time_offset = 1, 0.5
        tstart_solving = time.time()
        s.solve(assumptions=assumption, time_limit=time_limit)
        tend_solving = time.time()
        self.assertLessEqual(tend_solving - tstart_solving, time_limit + time_offset)

if __name__ == '__main__':
    unittest.main()