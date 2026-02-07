import itertools


import cpmpy as cp
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl
from cpmpy.transformations.to_cnf import to_cnf, to_gcnf
from cpmpy.transformations.get_variables import get_variables, get_variables_model
from cpmpy.expressions.utils import argvals
from cpmpy.solvers.pindakaas import CPM_pindakaas
from cpmpy.tools.explain.marco import make_assump_model

import pytest


SOLVER = "ortools"
ENCODING = "auto"
a, b, c = cp.boolvar(shape=3, name=["a", "b", "c"])
x = cp.intvar(1, 2, name="x")
y, z = cp.intvar(0, 1, shape=2, name=["y", "z"])


def get_cnf_cases():
    return (
        cnf
        for cnf in (
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
            (~a & ~b) | (a & b),  # https://github.com/cpmpy/cpmpy/issues/823
            c | (a & b),  # above minimized
        )
    )


def get_gcnf_cases():
    p, q = cp.boolvar(shape=2, name=["p", "q"])
    soft = (cp.sum([2 * p + 3 * q]) <= 4, p & q)
    hard = (p,)
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


@pytest.mark.skipif(not CPM_pindakaas.supported(), reason="Pindakaas (required for `to_cnf`) not installed")
class TestCnf:
    def idfn(val):
        if isinstance(val, tuple):
            # solver name, class tuple
            return val
        else:
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
        disjoint = True
        print("hard = ", hard)
        print("soft = ", soft)
        gcnf_model, soft_, hard_, assumptions = to_gcnf(
            soft,
            hard,
            name="a",
            ivarmap=ivarmap,
            disjoint=disjoint,
            encoding=ENCODING,
        )
        print("m", gcnf_model)
        print("s", soft_)
        print("h", hard_)
        print("a", assumptions)

        def powerset(iterable):
            "Subsequences of the iterable from shortest to longest."
            # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
            s = list(iterable)
            return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

        for assumptions_ in itertools.islice(powerset(assumptions), None):
            vs = cp.cpm_array(get_variables_model(model))
            s1 = self.allsols(assump_model.constraints, vs, assumptions=assumptions)

            assert len(s1) <= 100, "Find a smaller case!"
            s2 = self.allsols(gcnf_model.constraints, vs, assumptions=assumptions, ivarmap=ivarmap)
            assert s1 == s2

    @pytest.mark.parametrize(
        "case",
        itertools.chain(get_cnf_cases(), (soft + hard for soft, hard in get_gcnf_cases())),
        ids=idfn,
    )
    def test_to_cnf(self, case):
        # test for equivalent solutions with/without to_cnf
        vs = cp.cpm_array(get_variables(case))
        s1 = self.allsols([case], vs)
        ivarmap = dict()
        cnf = to_cnf(case, ivarmap=ivarmap, encoding=ENCODING)

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

    def allsols(self, cons, vs, ivarmap=None, assumptions=None, feasibility=False):
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

        if feasibility:
            sat = m.solve(solver=SOLVER, assumptions=assumptions)
            return [True] if sat else [False]
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
