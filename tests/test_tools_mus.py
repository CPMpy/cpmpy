import inspect
import pytest

import cpmpy as cp
from cpmpy.tools import mss_opt, marco, OCUSException
from cpmpy.tools.explain import mus, mus_naive, quickxplain, quickxplain_naive, optimal_mus, optimal_mus_naive, mss, mcs, ocus, ocus_naive, mus_native


ALL = [
    "mus", "mus_naive", "mus_native",
    "quickxplain", "quickxplain_naive",
    "optimal_mus", "optimal_mus_naive",
    "ocus", "ocus_naive",
]

MUS_FUNCS = dict(
    mus = mus,
    mus_naive = mus_naive,
    mus_native = mus_native,
    quickxplain = quickxplain,
    quickxplain_naive = quickxplain_naive,
    optimal_mus = optimal_mus,
    optimal_mus_naive = optimal_mus_naive,
    ocus = ocus,
    ocus_naive = ocus_naive,
)



class TestMUS:

    def _supported_solver(self, solver, variant):
        if variant.endswith("_naive"):
            return True
        if variant == "mus_native":
            return "mus_native" in cp.SolverLookup.lookup(solver).__dict__
        # assumption-based algorithms
        s = cp.SolverLookup.get(solver)
        return inspect.signature(s.solve).parameters.get("assumptions") is not None

    def _unsupported_reason(self, solver, variant):
        if variant == "mus_native":
            return f"Solver {solver} does not support native MUS"
        return f"Solver {solver} does not support assumption-based MUS"

    def _test_mus(self, cons, hard, solver, verify_func, variant, **kwargs):
        if solver == "hexaly":
            pytest.skip("Hexaly is too slow on UNSAT problems.")
        if not self._supported_solver(solver, variant):
            pytest.skip(self._unsupported_reason(solver, variant))
        mus_cons = MUS_FUNCS[variant](soft=cons, hard=hard, solver=solver, **kwargs)
        assert verify_func(mus_cons)

    # shared test cases
    @pytest.mark.parametrize("variant", ALL)
    def test_circular(self, solver, variant):
        x = cp.intvar(0, 3, shape=4, name="x")
        # circular "bigger then", UNSAT
        cons = [
            x[0] > x[1],
            x[1] > x[2],
            x[2] > x[0],

            x[3] > x[0],
            (x[3] > x[1]).implies((x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2])))
        ]

        self._test_mus(cons, hard=[], solver=solver, variant=variant,
                       verify_func=lambda ms: set(ms) == set(cons[:3]))

    @pytest.mark.parametrize("variant", ALL)
    def test_bug_191(self, solver, variant):
        """
        Original Bug request: https://github.com/CPMpy/cpmpy/issues/191
        When assum is a single boolvar and candidates is a list (of length 1), it fails.
        """
        if solver == "cpo":
            pytest.skip("CPO does not support hard constraints")
        bv = cp.boolvar(name="x")
        hard = [~bv]
        soft = [bv]

        self._test_mus(soft, hard=hard, solver=solver, variant=variant,
                       verify_func=lambda ms: set(ms) == set(soft))

    @pytest.mark.parametrize("variant", ALL)
    def test_bug_191_many_soft(self, solver, variant):
        """
        Checking whether bugfix 191  doesn't break anything in the MUS tool chain,
        when the number of soft constraints > 1.
        """

        if solver == "cpo":
            pytest.skip("CPO does not support hard constraints")

        x = cp.intvar(-9, 9, name="x")
        y = cp.intvar(-9, 9, name="y")
        hard = [x > 2]
        soft = [
            x + y < 6,
            y == 4
        ]

        self._test_mus(soft, hard=hard, solver=solver, variant=variant,
                       verify_func=lambda ms: set(ms) == set(soft))

    @pytest.mark.parametrize("variant", ALL)
    def test_wglobal(self, solver, variant):
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
        self._test_mus(cons, hard=[], solver=solver, variant=variant,
                       verify_func=lambda ms: len(ms) < len(cons) and not cp.Model(ms).solve())

    @pytest.mark.parametrize("variant", ALL)
    def test_decomposed_global(self, solver, variant):

        x = cp.intvar(1, 5, shape=3, name="x")
        cons = cp.AllDifferent(x)
        cons.name = "DummyAllDiff" # nonexisting name, force decompos


        soft = [cons, x[0] == x[1], x[1] == x[2]]

        mus_cons = self.mus_func(soft=soft, hard=[], solver=solver)
        assert len(set(mus_cons)) == 2
        assert "DummyAllDiff(x[0],x[1],x[2])" in set(map(str, mus_cons))
        mus_naive_cons = self.naive_func(soft=soft, hard=[])
        assert len(set(mus_naive_cons)) == 2
        assert "DummyAllDiff(x[0],x[1],x[2])" in set(map(str, mus_cons))

    def single_soft_constraint(self, solver):
        x = cp.intvar(1, 2, shape=3, name="x")
        soft = [cp.AllDifferent(x)]
        hard = []
        self._test_mus(soft, hard=hard, solver=solver, variant=variant,
                       verify_func=lambda ms: len(set(ms)) == 1)

    @pytest.mark.parametrize("variant", ALL)
    def test_cse_shared_subexpr(self, solver, variant):
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

        self._test_mus(soft, hard=hard, solver=solver, variant=variant,
                       verify_func=lambda ms: set(ms) == {soft[1]})

    # quickxplain-specific
    @pytest.mark.parametrize("variant", ["quickxplain", "quickxplain_naive"])
    def test_prefered(self, solver, variant):
        a,b,c,d = [cp.boolvar(name=n) for n in "abcd"]

        mus1 = [b,d]
        mus2 = [a,b,c]

        hard = [~cp.all(mus1), ~cp.all(mus2)]
        self._test_mus([a,b,c,d], hard=hard, solver=solver, variant=variant,
                       verify_func=lambda ms: set(ms) == {a,b,c})
        self._test_mus([d,c,b,a], hard=hard, solver=solver, variant=variant,
                       verify_func=lambda ms: set(ms) == {b,d})

    # optimal MUS-specific
    @pytest.mark.parametrize("variant", ["optimal_mus", "optimal_mus_naive"])
    def test_weighted(self, solver, variant):
        a, b, c, d = [cp.boolvar(name=n) for n in "abcd"]

        mus1 = [b, d]
        mus2 = [a, b, c]

        hard = [~cp.all(mus1), ~cp.all(mus2)]
        self._test_mus([a, b, c, d], hard=hard, solver=solver, variant=variant,
                       weights=[1, 1, 2, 4], verify_func=lambda ms: set(ms) == {a, b, c})
        self._test_mus([a, b, c, d], hard=hard, solver=solver, variant=variant,
                       weights=[2, 3, 4, 2], verify_func=lambda ms: set(ms) == {b, d})
        self._test_mus([a, b, c, d], hard=hard, solver=solver, variant=variant,
                       verify_func=lambda ms: set(ms) == {b, d})

    # OCUS-specific
    @pytest.mark.parametrize("variant", ["ocus", "ocus_naive"])
    def test_constrained(self, solver, variant):
        a, b, c, d = [cp.boolvar(name=n) for n in "abcd"]

        mus1 = [b, d]
        mus2 = [a, b, c]

        hard = [~cp.all(mus1), ~cp.all(mus2)]
        self._test_mus([a, b, c, d], hard=hard, solver=solver, variant=variant,
                       meta_constraint=~b | d, verify_func=lambda ms: set(ms) == {b, d})
        self._test_mus([a, b, c, d], hard=hard, solver=solver, variant=variant,
                       meta_constraint=a & d, verify_func=lambda ms: set(ms) == {a, b, d})  # not subset-minimal

    @pytest.mark.parametrize("variant", ["ocus", "ocus_naive"])
    def test_no_such_mus(self, solver, variant):
        a, b, c, d = [cp.boolvar(name=n) for n in "abcd"]

        mus1 = [b, d]
        mus2 = [a, b, c]
        hard = [~cp.all(mus1), ~cp.all(mus2)]

        if solver == "hexaly":
            pytest.skip("Hexaly is too slow on UNSAT problems.")
        if not self._supported_solver(solver, variant):
            pytest.skip(self._unsupported_reason(solver, variant))
        pytest.raises(OCUSException, lambda: MUS_FUNCS[variant](
            [a, b, c, d], hard, meta_constraint=~b, solver=solver))


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
