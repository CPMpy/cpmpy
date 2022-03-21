import unittest
from cpmpy import *
from cpmpy.solvers import CPM_pysat

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
        m  = Model()

        # Implementing pysat example for interrupt in cpmpy
        # https://pysathq.github.io/docs/html/api/solvers.html#pysat.solvers.Solver.interrupt
        for clause in PHP(nof_holes=10).clauses:
            m +=any(~lit_cpmvar[abs(lit)] if lit < 0 else lit_cpmvar[abs(lit)] for lit in clause)

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