import inspect
import pytest

import cpmpy as cp
from cpmpy.tools import mss_opt, marco, OCUSException
from cpmpy.tools.explain import mus, mus_naive, quickxplain, quickxplain_naive, optimal_mus, optimal_mus_naive, mss, mcs, ocus, ocus_naive, mus_native


# annotation to run both MUS and naive varaint for the test
run_mus_and_naive = pytest.mark.parametrize("naive", [False, True], ids=["mus", "naive"])


class TestMus:
    def setup_method(self):
        self.mus_func = mus
        self.naive_func = mus_naive

    def _supported_solver(self, solver):
        s = cp.SolverLookup.get(solver)
        solve_arguments = inspect.signature(s.solve).parameters
        return solve_arguments.get("assumptions") is not None

    def _unsupported_reason(self, solver):
        return f"Solver {solver} does not support assumption-based MUS"

    def _test_mus(self, cons, hard, solver, verify_func, naive=False, **kwargs):
        if solver == "hexaly":
            pytest.skip("Hexaly is too slow on UNSAT problems.")
        if naive:
            mus_cons = self.naive_func(soft=cons, hard=hard, solver=solver, **kwargs)
        else:
            if not self._supported_solver(solver):
                pytest.skip(self._unsupported_reason(solver))
            mus_cons = self.mus_func(soft=cons, hard=hard, solver=solver, **kwargs)
        assert verify_func(mus_cons)


    # test cases

    @run_mus_and_naive
    def test_circular(self, solver, naive):
        x = cp.intvar(0, 3, shape=4, name="x")
        # circular "bigger then", UNSAT
        cons = [
            x[0] > x[1],
            x[1] > x[2],
            x[2] > x[0],

            x[3] > x[0],
            (x[3] > x[1]).implies((x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2])))
        ]

        self._test_mus(cons, hard=[], solver=solver, naive=naive,
                       verify_func=lambda ms: set(ms) == set(cons[:3]))

    @run_mus_and_naive
    def test_bug_191(self, solver, naive):
        """
        Original Bug request: https://github.com/CPMpy/cpmpy/issues/191
        When assum is a single boolvar and candidates is a list (of length 1), it fails.
        """
        bv = cp.boolvar(name="x")
        hard = [~bv]
        soft = [bv]

        self._test_mus(soft, hard=hard, solver=solver, naive=naive,
                       verify_func=lambda ms: set(ms) == set(soft))

    @run_mus_and_naive
    def test_bug_191_many_soft(self, solver, naive):
        """
        Checking whether bugfix 191  doesn't break anything in the MUS tool chain,
        when the number of soft constraints > 1.
        """
        x = cp.intvar(-9, 9, name="x")
        y = cp.intvar(-9, 9, name="y")
        hard = [x > 2]
        soft = [
            x + y < 6,
            y == 4
        ]

        self._test_mus(soft, hard=hard, solver=solver, naive=naive,
                       verify_func=lambda ms: set(ms) == set(soft))

    @run_mus_and_naive
    def test_wglobal(self, solver, naive):
        x = cp.intvar(-9, 9, name="x")
        y = cp.intvar(-9, 9, name="y")

        cons = [
            x < 0,
            x > 2,
            x < 1,
            y > 0,
            y == 4,
            (x + y > 0) | (y < 0),
            (y >= 0) | (x >= 0),
            (y < 0) | (x < 0),
            (y > 0) | (x < 0),
            cp.AllDifferent(x,y)
        ]

        # non-determinstic
        self._test_mus(cons, hard=[], solver=solver, naive=naive,
                       verify_func=lambda ms: len(ms) < len(cons) and not cp.Model(ms).solve())

    @run_mus_and_naive
    def test_decomposed_global(self, solver, naive):
        x = cp.intvar(1, 5, shape=3, name="x")
        soft = [x[0] == x[1], x[1] == x[2]]
        hard = [cp.AllDifferent(x)]

        self._test_mus(soft, hard=hard, solver=solver, naive=naive,
                       verify_func=lambda ms: len(set(ms)) == 1)

    @run_mus_and_naive
    def test_cse_shared_subexpr(self, solver, naive):
        """Example with CSE in the defining constraints.

        Reproducer from https://github.com/CPMpy/cpmpy/pull/986
        """
        x = cp.intvar(-10, 10, name="x")
        y = cp.intvar(-10, 10, name="y")
        soft = [
            cp.abs(x) + y <= 15,  # satisfiable, not needed for the conflict
            cp.abs(x) + y >= 11,  # the real conflict with hard
        ]
        hard = [x == 0]

        self._test_mus(soft, hard=hard, solver=solver, naive=naive,
                       verify_func=lambda ms: set(ms) == {soft[1]})


@pytest.mark.requires_solver("exact", "gurobi", "cplex")
class TestNativeMus(TestMus):
    def setup_method(self):
        self.mus_func = mus_native
        self.naive_func = mus_naive

    def _supported_solver(self, solver):
        # True only if the concrete solver class overrides mus_native
        return "mus_native" in cp.SolverLookup.lookup(solver).__dict__

    def _unsupported_reason(self, solver):
        return f"Solver {solver} does not support native MUS"


class TestQuickXplain(TestMus):
    def setup_method(self):
        self.mus_func = quickxplain
        self.naive_func = quickxplain_naive

    @run_mus_and_naive
    def test_prefered(self, solver, naive):
        a,b,c,d = [cp.boolvar(name=n) for n in "abcd"]

        mus1 = [b,d]
        mus2 = [a,b,c]

        hard = [~cp.all(mus1), ~cp.all(mus2)]
        self._test_mus([a,b,c,d], hard=hard, solver=solver, naive=naive,
                       verify_func=lambda ms: set(ms) == {a,b,c})
        self._test_mus([d,c,b,a], hard=hard, solver=solver, naive=naive,
                       verify_func=lambda ms: set(ms) == {b,d})


class TestOptimalMUS(TestMus):

    def setup_method(self):
        self.mus_func = optimal_mus
        self.naive_func = optimal_mus_naive

    @run_mus_and_naive
    def test_weighted(self, solver, naive):
        a, b, c, d = [cp.boolvar(name=n) for n in "abcd"]

        mus1 = [b, d]
        mus2 = [a, b, c]

        hard = [~cp.all(mus1), ~cp.all(mus2)]
        self._test_mus([a, b, c, d], hard=hard, solver=solver, naive=naive,
                       weights=[1, 1, 2, 4], verify_func=lambda ms: set(ms) == {a, b, c})
        self._test_mus([a, b, c, d], hard=hard, solver=solver, naive=naive,
                       weights=[2, 3, 4, 2], verify_func=lambda ms: set(ms) == {b, d})
        self._test_mus([a, b, c, d], hard=hard, solver=solver, naive=naive,
                       verify_func=lambda ms: set(ms) == {b, d})


class TestOCUS(TestOptimalMUS):

    def setup_method(self):
        self.mus_func = ocus
        self.naive_func = ocus_naive

    @run_mus_and_naive
    def test_constrained(self, solver, naive):
        a, b, c, d = [cp.boolvar(name=n) for n in "abcd"]

        mus1 = [b, d]
        mus2 = [a, b, c]

        hard = [~cp.all(mus1), ~cp.all(mus2)]
        self._test_mus([a, b, c, d], hard=hard, solver=solver, naive=naive,
                       meta_constraint=~b | d, verify_func=lambda ms: set(ms) == {b, d})
        self._test_mus([a, b, c, d], hard=hard, solver=solver, naive=naive,
                       meta_constraint=a & d, verify_func=lambda ms: set(ms) == {a, b, d})  # not subset-minimal

    @run_mus_and_naive
    def test_no_such_mus(self, solver, naive):
        a, b, c, d = [cp.boolvar(name=n) for n in "abcd"]

        mus1 = [b, d]
        mus2 = [a, b, c]
        hard = [~cp.all(mus1), ~cp.all(mus2)]

        mus_func = self.naive_func if naive else self.mus_func
        if not naive and not self._supported_solver(solver):
            pytest.skip(self._unsupported_reason(solver))
        pytest.raises(OCUSException, lambda: mus_func([a, b, c, d], hard, meta_constraint=~b, solver=solver))


class TestMARCOMUS:

    def test_php(self):
        x = cp.boolvar(shape=(5,3), name="x")
        model = cp.Model()
        model += cp.cpm_array(x.sum(axis=1)) >= 1
        model += cp.cpm_array(x.sum(axis=0)) <= 1

        subsets = list(marco(soft=model.constraints))
        musses = [ss for kind, ss in subsets if kind == "MUS"]
        mcses = [ss for kind, ss in subsets if kind == "MCS"]
        assert len(musses) == 5
        assert len(mcses) == 13

        # also works when only enumerating MUSes?
        musses = list(marco(soft=model.constraints, return_mcs=False))
        assert len(musses) == 5
        # or only MCSes?
        mcses = list(marco(soft=model.constraints, return_mus=False))
        assert len(mcses) == 13# any combination of 3 pigeon constraints + 3 mcses with the hole constraints



class TestMSS:

    def test_circular(self):
        x = cp.intvar(0, 3, shape=4, name="x")
        # circular "bigger then", UNSAT
        cons = [
            x[0] > x[1],
            x[1] > x[2],
            x[2] > x[0],

            x[3] > x[0],
            (x[3] > x[1]).implies((x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2])))
        ]

        assert len(mss(cons)) < len(cons)
        assert cons[4] in set(mss_opt(cons, weights=[1,1,1,1,5]))# weighted version

class TestMCS:

    def test_circular(self):
        x = cp.intvar(0, 3, shape=4, name="x")
        # circular "bigger then", UNSAT
        cons = [
            x[0] > x[1],
            x[1] > x[2],
            x[2] > x[0],

            x[3] > x[0],
            (x[3] > x[1]).implies((x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2])))
        ]
        assert len(mcs(cons)) == 1
