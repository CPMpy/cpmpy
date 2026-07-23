"""Tests for int2bool_nested: preserve Boolean nesting while encoding int comparisons."""

import pytest

import cpmpy as cp
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.int2bool_nested import int2bool_nested
from cpmpy.transformations.linearize import decompose_linear
from cpmpy.transformations.negation import push_down_negation
from cpmpy.transformations.normalize import simplify_boolean, toplevel_list
from cpmpy.transformations.safening import no_partial_functions
from cpmpy.solvers.paramita import CPM_paramita


def _prep(cons):
    return simplify_boolean(
        decompose_linear(
            push_down_negation(no_partial_functions(toplevel_list(cons))),
            supported=frozenset(),
            supported_reified=frozenset(),
        )
    )


def _has_domain_side_constraints(out, ivarmap):
    """True if domain constraints for encoded ints appear in the output list."""
    assert ivarmap, "expected at least one int encoding"
    # Domain constraints are implications (order) or sum==1 (direct), listed before/with exprs
    return any(
        (isinstance(c, Operator) and c.name == "->")
        or (isinstance(c, Comparison) and c.name == "==" and isinstance(c.args[0], Operator) and c.args[0].name == "sum")
        for c in out
    )


class TestInt2BoolNested:

    def test_preserves_or_over_comparison(self):
        x, y = cp.intvar(0, 3, shape=2, name="xy")
        a = cp.boolvar(name="a")
        ivarmap = {}
        out = int2bool_nested(_prep((x + y > 2) | a), ivarmap)

        ors = [c for c in out if isinstance(c, Operator) and c.name == "or"]
        assert len(ors) == 1
        # Nested PB | a — not an aux BV reifying the comparison
        assert a in ors[0].args
        assert any(isinstance(arg, Comparison) for arg in ors[0].args)
        assert _has_domain_side_constraints(out, ivarmap)

    def test_preserves_implies_diseq(self):
        x, y = cp.intvar(0, 3, shape=2, name="xy")
        b = cp.boolvar(name="b")
        ivarmap = {}
        out = int2bool_nested(_prep(b.implies(x != y)), ivarmap)

        imps = [c for c in out if isinstance(c, Operator) and c.name == "->"]
        # At least b -> (pb != 0); domain may also contribute ->
        assert any(
            isinstance(c, Operator)
            and c.name == "->"
            and c.args[0] is b
            and isinstance(c.args[1], Comparison)
            and c.args[1].name == "!="
            for c in out
        )
        assert imps  # domain and/or the implication itself
        assert _has_domain_side_constraints(out, ivarmap)

    def test_preserves_bool_reification(self):
        x, y = cp.intvar(0, 3, shape=2, name="xy")
        a = cp.boolvar(name="a")
        ivarmap = {}
        out = int2bool_nested(_prep(a == (x + y >= 2)), ivarmap)

        reifs = [
            c
            for c in out
            if isinstance(c, Comparison) and c.name == "==" and (c.args[0] is a or c.args[1] is a)
        ]
        assert len(reifs) == 1
        other = reifs[0].args[1] if reifs[0].args[0] is a else reifs[0].args[0]
        assert isinstance(other, Comparison)
        assert other.name == ">="
        assert _has_domain_side_constraints(out, ivarmap)

    def test_pure_bool_unchanged_shape(self):
        p, q = cp.boolvar(name="p"), cp.boolvar(name="q")
        ivarmap = {}
        out = int2bool_nested(_prep(p | q), ivarmap)
        assert ivarmap == {}
        assert len(out) == 1
        assert isinstance(out[0], Operator) and out[0].name == "or"

    def test_alldifferent_sum_of_eq_to_bools(self):
        """decompose_linear yields sum(x[i]==v); atoms must become encoding BVs (no IntVars left)."""
        from cpmpy.expressions.variables import _NumVarImpl

        x = cp.intvar(1, 3, shape=3, name="x")
        ivarmap = {}
        out = int2bool_nested(_prep(cp.AllDifferent(x)), ivarmap)
        assert ivarmap
        assert not any(
            isinstance(v, _NumVarImpl) and not v.is_bool() for v in get_variables(out)
        )
        assert any(isinstance(c, Comparison) and c.name in ("<=", "==") for c in out)
        assert _has_domain_side_constraints(out, ivarmap)


@pytest.mark.skipif(not CPM_paramita.supported(), reason="Paramita not installed or no solver plugins")
class TestParamitaNested:

    def test_solve_nested_int_bool(self):
        x, y = cp.intvar(0, 3, shape=2, name="xy")
        a, b = cp.boolvar(name="a"), cp.boolvar(name="b")
        cons = [
            ((x + y > 2) | a) & b.implies(x != y),
            (x == 1) | (y >= 2),
            ~(a & (x + 2 * y <= 3)),
        ]
        # Structure through Paramita.transform
        s = CPM_paramita()
        out = s.transform(cons)
        assert any(isinstance(c, Operator) and c.name == "or" for c in out)

        m = cp.Model(cons)
        assert m.solve(solver="paramita")
        # Feasible assignment must satisfy original constraints
        assert all(c.value() for c in cons)

    def test_solve_mul_abs_nested(self):
        x, y = cp.intvar(0, 3, name="x"), cp.intvar(0, 3, name="y")
        a = cp.boolvar(name="a")
        cons = ((x * y <= 4) | a) & (cp.abs(x - y) <= 2)
        m = cp.Model(cons)
        assert m.solve(solver="paramita")
        assert cons.value()

    def test_solve_parity_vs_ortools(self):
        x, y = cp.intvar(0, 2, shape=2, name="xy")
        a = cp.boolvar(name="a")
        cons = ((x + y >= 2) | a) & (x != y)
        user_vars = list(get_variables(cons))

        orto_sols = []
        cp.Model(cons).solveAll(
            solver="ortools",
            display=lambda: orto_sols.append(tuple(v.value() for v in user_vars)),
        )

        para_sols = []
        s = CPM_paramita()
        s += cons
        s.solveAll(display=lambda: para_sols.append(tuple(v.value() for v in user_vars)))

        assert sorted(orto_sols) == sorted(para_sols)

    def test_solve_alldifferent(self):
        x = cp.intvar(1, 3, shape=3, name="x")
        m = cp.Model(cp.AllDifferent(x))
        assert m.solve(solver="paramita")
        assert len(set(x.value())) == 3

    def test_solve_nqueens4(self):
        from pathlib import Path
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "csplib"))
        from prob054_n_queens import n_queens

        m, _ = n_queens(4)
        assert m.solve(solver="paramita")
