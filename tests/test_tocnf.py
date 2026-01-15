import unittest
import cpmpy as cp


from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.utils import argvals
from cpmpy.solvers.pindakaas import CPM_pindakaas

import pytest

a, b, c = cp.boolvar(shape=3, name=["a", "b", "c"])
x = cp.intvar(1, 2, name="x")
y, z = cp.intvar(0, 1, shape=2, name=["y", "z"])

cases = [
    a,
    a | b,
    a & b,
    a != b,
    a == b,
    a.implies(b),
    a.implies(b | c),
    a.implies(b & c),
    a.implies(b != c),
    a.implies(b == c),
    a.implies(b.implies(c)),
    (b | c).implies(a),
    (b & c).implies(a),
    (b != c).implies(a),
    (b == c).implies(a),
    (b.implies(c)).implies(a),
    cp.Xor([a, b]),
    cp.sum([2 * x + 3 * y]) <= 4,
    cp.sum([2 * x + 3 * y + 5 * z]) <= 6,
    cp.sum([2 * x + 3 * cp.intvar(0, 1)]) <= 4,
    cp.sum([3 * y]) <= 4,  # sat
    cp.sum([3 * y]) >= 20,  # unsat
    (a + b + c) == 1,
    # a * b == 1,  # todo in linearization!
    # a * b != 1,
    (a + b + c) != 1,
    a + b + c > 2,
    a + b + c <= 2,
    cp.sum(cp.intvar(lb=2, ub=3, shape=3)) <= 3,
    (~a & ~b) | (a & b), # https://github.com/cpmpy/cpmpy/issues/823
    c | (a & b),  # above minimized
]


@pytest.mark.skipif(not CPM_pindakaas.supported(), reason="Pindakaas (required for `to_cnf`) not installed")
class TestToCnf:
    def idfn(val):
        if isinstance(val, tuple):
            # solver name, class tuple
            return val
        else:
            return f"{val}"


    @pytest.mark.parametrize(
        "case",
        cases,
        ids=idfn,
    )
    def test_tocnf(self, case):
        # test for equivalent solutions with/without to_cnf
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
        assert s1 == s2, f"The equivalence check failed for translation from:\n\n{case}\n\nto:\n\n{cnf}"

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
