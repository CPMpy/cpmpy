import cpmpy as cp
from cpmpy.transformations.to_gcnf import to_gcnf
from cpmpy.transformations.get_variables import get_variables_model
from cpmpy.expressions.utils import argvals
from cpmpy.solvers.pindakaas import CPM_pindakaas
from cpmpy.tools.explain.utils import make_assump_model

import pytest


SOLVER = "ortools"
ENCODING = "auto"


def get_gcnf_cases():
    p, q = cp.boolvar(shape=2, name=["p", "q"])
    soft = [cp.sum([2 * p + 3 * q]) <= 4, p & q]
    hard = [p]
    yield soft, hard

    b = cp.boolvar(name="b")
    soft = [
        b.implies(cp.sum([2 * p + 3 * q]) <= 10),
        b | (p == 0),
    ]
    hard = [q >= 1]
    yield soft, hard

    x, y = cp.intvar(0, 2, shape=2, name=["x", "y"])
    soft = [(x == 0) | (x == 1), (y == 2)]
    hard = [y == 1]
    yield soft, hard

    bs = cp.boolvar(4, name="b")
    soft = [cp.sum(bs) >= 1, cp.sum(bs) <= 1]
    hard = []
    yield soft, hard

    xs = cp.intvar(0, 2, shape=3, name="x")
    soft = [cp.sum(xs) >= 2, cp.sum(xs) <= 2, cp.max(xs) < 0]
    hard = []
    yield soft, hard

    xs = cp.intvar(0, 2, shape=2, name="x")
    soft = [cp.max(xs) > 0]
    hard = []
    yield soft, hard

    xs = cp.intvar(0, 2, shape=2, name="x")
    soft = [cp.max(xs) > -1]
    hard = []
    yield soft, hard


@pytest.mark.skipif(not CPM_pindakaas.supported(), reason="Pindakaas (required for `to_gcnf`) not installed")
class TestToGcnf:
    def idfn(val):
        return f"{val}"

    @pytest.mark.parametrize(
        "case",
        get_gcnf_cases(),
        ids=idfn,
    )
    def test_to_gcnf(self, case):
        soft, hard = case

        model = cp.Model(soft + hard)
        assump_model, _, _ = make_assump_model(soft, hard, name="a")

        ivarmap = dict()
        gcnf_model, soft_, hard_, assumptions = to_gcnf(
            soft,
            hard,
            name="a",
            ivarmap=ivarmap,
            disjoint=True,
            encoding=ENCODING,
        )

        # with all assumptions enabled, both models must have the same
        # solutions when projected onto the original model's variables
        vs = cp.cpm_array(get_variables_model(model))
        s1 = self.allsols(assump_model.constraints, vs, assumptions=assumptions)

        assert len(s1) <= 100, "Find a smaller case!"
        s2 = self.allsols(gcnf_model.constraints, vs, assumptions=assumptions, ivarmap=ivarmap)
        assert s1 == s2

    def allsols(self, cons, vs, ivarmap=None, assumptions=None):
        m = cp.Model(cons)
        projected_solutions = set()

        solution_limit = 10000
        projected_solution_limit = 100
        n_solutions = [0]  # number of non-projected solutions, pass by reference hack!

        def display():
            if ivarmap:
                for x_enc in ivarmap.values():
                    x_enc._x._value = x_enc.decode()
            sol = tuple(argvals(vs))
            n_solutions[0] += 1

            projected_solutions.add(sol)
            if n_solutions[0] >= solution_limit:
                assert False, "Increase sol limit!"

        m.solveAll(
            solver=SOLVER,
            display=display,
            solution_limit=solution_limit,
            assumptions=assumptions,
        )
        assert len(projected_solutions) < projected_solution_limit, (
            f"Strict less is intentional ; We didn't find ALL projected solutions with the limit of {projected_solution_limit}, after finding {solution_limit} non-projected solutions"
        )
        return projected_solutions
