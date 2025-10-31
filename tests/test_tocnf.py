import unittest
import cpmpy as cp


from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.globalconstraints import Xor
from cpmpy.expressions.utils import argvals
from cpmpy.solvers.pindakaas import CPM_pindakaas

import pytest


@pytest.mark.skipif(not CPM_pindakaas.supported(), reason="Pindakaas (required for `to_cnf`) not installed")
class TestToCnf(unittest.TestCase):
    def test_tocnf(self):
        a, b, clause = cp.boolvar(shape=3)
        x = cp.intvar(1, 2)
        y, z = cp.intvar(0, 1, shape=2)

        bvs = cp.boolvar(shape=3)
        cases = [
            a,
            a | b,
            a & b,
            a != b,
            a == b,
            a.implies(b),
            a.implies(b | clause),
            a.implies(b & clause),
            a.implies(b != clause),
            a.implies(b == clause),
            a.implies(b.implies(clause)),
            (b | clause).implies(a),
            (b & clause).implies(a),
            (b != clause).implies(a),
            (b == clause).implies(a),
            (b.implies(clause)).implies(a),
            Xor([a, b]),
            cp.sum([2 * x + 3 * y]) <= 4,
            cp.sum([2 * x + 3 * y + 5 * z]) <= 6,
            cp.sum([2 * cp.intvar(1, 2) + 3 * cp.intvar(0, 1)]) <= 4,
            cp.sum([3 * cp.intvar(0, 1)]) <= 4,
            (a + b + clause) == 1,
            # a * b == 1,  # TODO in linearization!
            # a * b != 1,
            (a + b + clause) != 1,
            a + b + clause > 2,
            a + b + clause <= 2,
            cp.sum(cp.intvar(lb=2, ub=3, shape=3)) <= 3,
        ]

        # test for equivalent solutions with/without to_cnf
        for case in cases:
            vs = cp.cpm_array(get_variables(case))
            s1 = self.allsols([case], vs)
            ivarmap = dict()
            cnf = to_cnf(case, ivarmap=ivarmap)

            # TODO
            # assert (
            #     cnf is False
            #     or isinstance(cnf, _BoolVarImpl)
            #     or cnf.name == "and"
            #     and all(
            #         clause.name == "or"
            #         and all([is_bool(lit) or isinstance(lit, _BoolVarImpl) for lit in clause.args])
            #         for clause in cnf.args
            #     )
            # ), f"The following was not CNF: {cnf}"

            s2 = self.allsols(cnf, vs, ivarmap=ivarmap)
            assert s1 == s2, f"The equivalence check failed for translaton from {case} to {cnf}"

    def allsols(self, cons, vs, ivarmap=None):
        m = cp.Model(cons)
        sols = set()

        def display():
            if ivarmap:
                for x_enc in ivarmap.values():
                    x_enc._x._value = x_enc.decode()
            sols.add(tuple(argvals(vs)))

        m.solveAll(solver="ortools", display=display, solution_limit=100)
        assert len(sols) < 100, sols
        return sols


if __name__ == "__main__":
    unittest.main()
