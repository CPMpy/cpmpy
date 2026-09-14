"""
Tests for the :class:`~cpmpy.expressions.globalconstraints.LexLess` global
constraint, aimed at its linear decomposition (``LexLess.decompose_linear_positive``),
which is what linear solvers such as Gurobi, SCIP, HiGHS and Exact use.

The reference semantics are the ones implemented by ``LexLess.value()``:
``X`` is lexicographically strictly smaller than ``Y`` iff there is an index
``i`` with ``X[i] < Y[i]`` and ``X[j] <= Y[j]`` for every ``j < i``.

Every solver-dependent test uses the ``solver`` fixture, so the module runs
against whichever solver is selected, e.g.::

    python -m pytest -k lexless -x --solver=gurobi

Both the module name and every test name contain "lexless", so ``-k lexless``
selects all of them.
"""

import itertools

import pytest

import cpmpy as cp
from cpmpy.expressions.utils import argvals

# Solvers that cannot handle integer decision variables at all.
NO_INTVAR_SOLVERS = ("pysat", "pysdd", "pindakaas", "rc2")


def _skip_if_no_intvars(solver):
    """Skip the calling test when the solver has no integer variables."""
    if solver in NO_INTVAR_SOLVERS:
        pytest.skip(f"{solver} does not support integer variables")


def _oracle(xs, ys):
    """Ground truth: value() evaluated on plain constants, no solver involved."""
    return cp.LexLess(list(xs), list(ys)).value()


def _pairs(n, lb, ub):
    """Every pair of assignments of two length-n lists over [lb, ub]."""
    domain = list(range(lb, ub + 1))
    for xs in itertools.product(domain, repeat=n):
        for ys in itertools.product(domain, repeat=n):
            yield xs, ys


def _check_all_pairs(solver, n, lb, ub):
    """Assert solver and reference semantics agree on every pair of assignments."""
    for xs, ys in _pairs(n, lb, ub):
        x = cp.intvar(lb, ub, shape=n, name="x")
        y = cp.intvar(lb, ub, shape=n, name="y")
        cons = cp.LexLess(x, y)
        sat = bool(cp.Model([cons, x == list(xs), y == list(ys)]).solve(solver=solver))
        expected = _oracle(xs, ys)
        assert sat == expected, (
            f"{solver}: LexLess({list(xs)}, {list(ys)}) reported "
            f"{'SAT' if sat else 'UNSAT'} but value() says {expected}"
        )
        if sat:
            assert cons.value() is True


# ---------------------------------------------------------------------------- #
#                            Core semantics sweep                              #
# ---------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "n,lb,ub",
    [
        (2, 0, 2),   # 81 pairs
        (3, 0, 1),   # 64 pairs
        (4, 0, 1),   # 256 pairs
    ],
    ids=str,
)
@pytest.mark.usefixtures("solver")
def test_lexless_exhaustive_matches_value(solver, n, lb, ub):
    """Solver and reference semantics must agree on every pair of assignments."""
    _skip_if_no_intvars(solver)
    _check_all_pairs(solver, n, lb, ub)


@pytest.mark.usefixtures("solver")
def test_lexless_negative_domain(solver):
    """Negative values, where the sign of the difference matters."""
    _skip_if_no_intvars(solver)
    _check_all_pairs(solver, n=2, lb=-2, ub=0)


@pytest.mark.usefixtures("solver")
def test_lexless_domain_spanning_zero(solver):
    """Domains straddling zero."""
    _skip_if_no_intvars(solver)
    _check_all_pairs(solver, n=2, lb=-1, ub=1)


# ---------------------------------------------------------------------------- #
#                             Degenerate shapes                                #
# ---------------------------------------------------------------------------- #

@pytest.mark.usefixtures("solver")
def test_lexless_empty_lists(solver):
    """Two empty lists: nothing can be strictly smaller, so the constraint is false."""
    _skip_if_no_intvars(solver)
    cons = cp.LexLess([], [])
    assert cons.value() is False
    assert not cp.Model(cons).solve(solver=solver)


@pytest.mark.usefixtures("solver")
def test_lexless_single_element(solver):
    """Single-element lists reduce to a plain strict inequality."""
    _skip_if_no_intvars(solver)
    x = cp.intvar(0, 2, name="x0")
    y = cp.intvar(0, 2, name="y0")
    cons = cp.LexLess([x], [y])

    for xv, yv in itertools.product(range(3), repeat=2):
        sat = bool(cp.Model([cons, x == xv, y == yv]).solve(solver=solver))
        assert sat == (xv < yv), f"{solver}: LexLess([{xv}], [{yv}]) gave {sat}"


@pytest.mark.usefixtures("solver")
def test_lexless_equal_lists(solver):
    """A list is never lexicographically strictly less than an equal list."""
    _skip_if_no_intvars(solver)
    x = cp.intvar(0, 2, shape=3, name="x")
    y = cp.intvar(0, 2, shape=3, name="y")
    assert not cp.Model([cp.LexLess(x, y), x == y]).solve(solver=solver)


@pytest.mark.usefixtures("solver")
def test_lexless_same_list_twice(solver):
    """The same variable array on both sides is unsatisfiable."""
    _skip_if_no_intvars(solver)
    x = cp.intvar(0, 2, shape=3, name="x")
    assert not cp.Model(cp.LexLess(x, x)).solve(solver=solver)


@pytest.mark.usefixtures("solver")
def test_lexless_shared_variables(solver):
    """Overlapping variables between the two lists."""
    _skip_if_no_intvars(solver)
    x = cp.intvar(0, 2, shape=2, name="x")
    # [x0, x1] <lex [x1, x0]  holds exactly when x0 < x1
    cons = cp.LexLess([x[0], x[1]], [x[1], x[0]])
    for a, b in itertools.product(range(3), repeat=2):
        sat = bool(cp.Model([cons, x == [a, b]]).solve(solver=solver))
        assert sat == (a < b), f"{solver}: x=[{a},{b}] gave {sat}"


def test_lexless_length_mismatch_rejected():
    """Lists of different length are rejected at construction time."""
    x = cp.intvar(0, 2, shape=3, name="x")
    y = cp.intvar(0, 2, shape=2, name="y")
    with pytest.raises(ValueError):
        cp.LexLess(x, y)
    with pytest.raises(ValueError):
        cp.LexLess([0, 1], [0, 1, 2])


# ---------------------------------------------------------------------------- #
#                    Constants and expressions as arguments                    #
# ---------------------------------------------------------------------------- #

@pytest.mark.usefixtures("solver")
def test_lexless_all_constants(solver):
    """Fully constant argument lists."""
    _skip_if_no_intvars(solver)
    assert cp.Model(cp.LexLess([0, 1, 2], [0, 1, 3])).solve(solver=solver)
    assert cp.Model(cp.LexLess([0, 1, 9], [0, 2, 0])).solve(solver=solver)
    assert not cp.Model(cp.LexLess([0, 1, 2], [0, 1, 2])).solve(solver=solver)
    assert not cp.Model(cp.LexLess([1, 0, 0], [0, 9, 9])).solve(solver=solver)


@pytest.mark.usefixtures("solver")
def test_lexless_mixed_constants_and_vars(solver):
    """Constants mixed in with decision variables, on both sides."""
    _skip_if_no_intvars(solver)
    x = cp.intvar(0, 2, shape=2, name="x")

    # [0, x0] <lex [0, x1]  holds exactly when x0 < x1
    cons = cp.LexLess([0, x[0]], [0, x[1]])
    assert cp.Model([cons, x == [0, 1]]).solve(solver=solver)
    assert cons.value() is True
    assert not cp.Model([cons, x == [1, 1]]).solve(solver=solver)

    # a constant that already decides the comparison at position 0
    assert not cp.Model(cp.LexLess([2, x[0]], [1, x[1]])).solve(solver=solver)
    assert cp.Model(cp.LexLess([0, x[0]], [1, x[1]])).solve(solver=solver)


@pytest.mark.usefixtures("solver")
def test_lexless_expressions_as_args(solver):
    """Non-variable expressions as list members."""
    _skip_if_no_intvars(solver)
    x = cp.intvar(0, 2, shape=2, name="x")
    y = cp.intvar(0, 2, shape=2, name="y")

    cons = cp.LexLess([x[0] + 1, x[1]], [y[0], y[1]])
    for xs, ys in _pairs(2, 0, 2):
        expected = _oracle((xs[0] + 1, xs[1]), ys)
        sat = bool(cp.Model([cons, x == list(xs), y == list(ys)]).solve(solver=solver))
        assert sat == expected, (
            f"{solver}: LexLess([{xs[0]}+1, {xs[1]}], {list(ys)}) gave {sat}"
        )


# ---------------------------------------------------------------------------- #
#                            Non-positive contexts                             #
# ---------------------------------------------------------------------------- #

"""
In a positive context the decomposition only has to be implied by the
constraint; in a non-positive context it must also exclude every violating
assignment. Each context below is checked against its full truth table.
"""

_CONTEXTS = [
    ("negated",          lambda c, bv: ~c,               lambda p, b: not p),
    ("double_negated",   lambda c, bv: ~(~c),            lambda p, b: p),
    ("antecedent",       lambda c, bv: c.implies(bv),    lambda p, b: (not p) or b),
    ("consequent",       lambda c, bv: bv.implies(c),    lambda p, b: (not b) or p),
    ("reified_eq",       lambda c, bv: bv == c,          lambda p, b: p == b),
    ("reified_neq",      lambda c, bv: c != bv,          lambda p, b: p != b),
    ("xor",              lambda c, bv: cp.Xor([c, bv]),  lambda p, b: p != b),
    ("nand",             lambda c, bv: ~(c & bv),        lambda p, b: not (p and b)),
    ("or_with_negation", lambda c, bv: cp.any([~c, bv]), lambda p, b: (not p) or b),
]


def _check_boolean_context(solver, context, expected, n, lb, ub):
    """Check one Boolean context against its truth table, for every pair."""
    for xs, ys in _pairs(n, lb, ub):
        holds = _oracle(xs, ys)
        for bv_val in (False, True):
            x = cp.intvar(lb, ub, shape=n, name="x")
            y = cp.intvar(lb, ub, shape=n, name="y")
            bv = cp.boolvar(name="bv")
            cons = cp.LexLess(x, y)
            fixed = [x == list(xs), y == list(ys), bv if bv_val else ~bv]
            sat = bool(cp.Model([context(cons, bv)] + fixed).solve(solver=solver))
            assert sat == expected(holds, bv_val), (
                f"{solver}: x={list(xs)} y={list(ys)} (lex-less: {holds}) bv={bv_val} "
                f"reported {'SAT' if sat else 'UNSAT'}, expected "
                f"{'SAT' if expected(holds, bv_val) else 'UNSAT'}"
            )


@pytest.mark.parametrize(
    "context,expected",
    [(ctx, exp) for _, ctx, exp in _CONTEXTS],
    ids=[name for name, _, _ in _CONTEXTS],
)
@pytest.mark.usefixtures("solver")
def test_lexless_in_non_positive_context(solver, context, expected):
    """The constraint must be encoded exactly, not merely implied."""
    _skip_if_no_intvars(solver)
    _check_boolean_context(solver, context, expected, n=2, lb=0, ub=2)


@pytest.mark.parametrize(
    "context,expected",
    [(ctx, exp) for _, ctx, exp in _CONTEXTS],
    ids=[name for name, _, _ in _CONTEXTS],
)
@pytest.mark.usefixtures("solver")
def test_lexless_in_non_positive_context_longer(solver, context, expected):
    """Same truth tables on three-element lists."""
    _skip_if_no_intvars(solver)
    _check_boolean_context(solver, context, expected, n=3, lb=0, ub=1)


@pytest.mark.usefixtures("solver")
def test_lexless_negated_is_lex_greater_equal(solver):
    """~LexLess(X, Y) must be exactly LexGreaterEq(X, Y)."""
    _skip_if_no_intvars(solver)
    for xs, ys in _pairs(2, 0, 2):
        x = cp.intvar(0, 2, shape=2, name="x")
        y = cp.intvar(0, 2, shape=2, name="y")
        fixed = [x == list(xs), y == list(ys)]
        neg = bool(cp.Model([~cp.LexLess(x, y)] + fixed).solve(solver=solver))
        assert neg == (not _oracle(xs, ys)), (
            f"{solver}: ~LexLess({list(xs)}, {list(ys)}) gave {neg}"
        )


# ---------------------------------------------------------------------------- #
#                               Side constraints                               #
# ---------------------------------------------------------------------------- #

_SIDE_CONSTRAINTS = [
    ("alldifferent_x",
     lambda x, y: cp.AllDifferent(x),
     lambda xs, ys: len(set(xs)) == len(xs)),
    ("x_non_decreasing",
     lambda x, y: [x[i] <= x[i + 1] for i in range(len(x) - 1)],
     lambda xs, ys: all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))),
    ("sum_bound",
     lambda x, y: cp.sum(y) <= 3,
     lambda xs, ys: sum(ys) <= 3),
    ("linked_lists",
     lambda x, y: y[0] == x[0],
     lambda xs, ys: ys[0] == xs[0]),
    ("fixed_head",
     lambda x, y: x[0] == 1,
     lambda xs, ys: xs[0] == 1),
]


@pytest.mark.parametrize(
    "side,side_ref",
    [(s, r) for _, s, r in _SIDE_CONSTRAINTS],
    ids=[name for name, _, _ in _SIDE_CONSTRAINTS],
)
@pytest.mark.usefixtures("solver")
def test_lexless_with_side_constraints(solver, side, side_ref):
    """LexLess together with a side constraint on the same variables."""
    _skip_if_no_intvars(solver)
    n, lb, ub = 2, 0, 2
    for xs, ys in _pairs(n, lb, ub):
        x = cp.intvar(lb, ub, shape=n, name="x")
        y = cp.intvar(lb, ub, shape=n, name="y")
        cons = cp.LexLess(x, y)
        fixed = [x == list(xs), y == list(ys)]
        sat = bool(cp.Model([cons, side(x, y)] + fixed).solve(solver=solver))
        expected = _oracle(xs, ys) and side_ref(xs, ys)
        assert sat == expected, (
            f"{solver}: x={list(xs)} y={list(ys)} with side constraint gave {sat}"
        )


@pytest.mark.parametrize(
    "side,side_ref",
    [(s, r) for _, s, r in _SIDE_CONSTRAINTS],
    ids=[name for name, _, _ in _SIDE_CONSTRAINTS],
)
@pytest.mark.parametrize("sense", ["maximize", "minimize"])
@pytest.mark.usefixtures("solver")
def test_lexless_optimum_with_side_constraints(solver, sense, side, side_ref):
    """The optimum over LexLess + a side constraint must match brute force.

    Leaving the variables free is the point: a spurious solution admitted by
    the decomposition shows up as a better objective than any feasible one.
    """
    _skip_if_no_intvars(solver)
    n, lb, ub = 3, 0, 2

    feasible = [(xs, ys) for xs, ys in _pairs(n, lb, ub)
                if _oracle(xs, ys) and side_ref(xs, ys)]

    x = cp.intvar(lb, ub, shape=n, name="x")
    y = cp.intvar(lb, ub, shape=n, name="y")
    cons = cp.LexLess(x, y)
    model = cp.Model([cons, side(x, y)])
    getattr(model, sense)(cp.sum(x) + cp.sum(y))

    if not feasible:
        assert not model.solve(solver=solver)
        return

    assert model.solve(solver=solver)
    best = (max if sense == "maximize" else min)(sum(xs) + sum(ys) for xs, ys in feasible)
    assert int(model.objective_value()) == best, (
        f"{solver}: {sense} gave {model.objective_value()}, brute force says {best}"
    )
    assert cons.value() is True
    assert side_ref(tuple(argvals(x)), tuple(argvals(y)))


# ---------------------------------------------------------------------------- #
#                    Structure of the linear decomposition                     #
# ---------------------------------------------------------------------------- #

"""
The defining part of a decomposition is posted unconditionally, in every
context. For that to be sound it must make the auxiliary variables a function
of the original ones: satisfiable for *every* assignment (totality, or an
infeasibility leaks into the parent model) and satisfiable in exactly *one*
way (functionality, or a negated context can pick convenient auxiliaries).
These two tests check those properties directly, independently of any solver
under test.
"""

@pytest.mark.parametrize("n,lb,ub", [(2, 0, 2), (3, 0, 1)], ids=str)
def test_lexless_linear_defining_is_total(n, lb, ub):
    """`defining` must be satisfiable for every assignment of the originals."""
    not_total = []
    for xs, ys in _pairs(n, lb, ub):
        x = cp.intvar(lb, ub, shape=n, name="x")
        y = cp.intvar(lb, ub, shape=n, name="y")
        _, defining = cp.LexLess(x, y).decompose_linear_positive()
        nsol = cp.Model(list(defining) + [x == list(xs), y == list(ys)]).solveAll(
            solver="ortools")
        if nsol == 0:
            not_total.append((list(xs), list(ys)))
    assert not not_total, (
        f"`defining` is unsatisfiable for {len(not_total)} assignment(s), "
        f"e.g. x={not_total[0][0]} y={not_total[0][1]}; an infeasibility there "
        f"leaks into the parent model in every context"
    )


@pytest.mark.parametrize("n,lb,ub", [(2, 0, 2), (3, 0, 1)], ids=str)
def test_lexless_linear_defining_is_functional(n, lb, ub):
    """`defining` must pin the auxiliaries down to exactly one assignment."""
    ambiguous = []
    for xs, ys in _pairs(n, lb, ub):
        x = cp.intvar(lb, ub, shape=n, name="x")
        y = cp.intvar(lb, ub, shape=n, name="y")
        _, defining = cp.LexLess(x, y).decompose_linear_positive()
        nsol = cp.Model(list(defining) + [x == list(xs), y == list(ys)]).solveAll(
            solver="ortools")
        if nsol > 1:
            ambiguous.append((list(xs), list(ys), nsol))
    assert not ambiguous, (
        f"`defining` admits several auxiliary assignments for "
        f"{len(ambiguous)} case(s), e.g. x={ambiguous[0][0]} y={ambiguous[0][1]} "
        f"has {ambiguous[0][2]} solutions; a negated context can pick whichever suits it"
    )


@pytest.mark.parametrize("n,lb,ub", [(2, 0, 2), (3, 0, 1)], ids=str)
def test_lexless_linear_agrees_with_decompose(n, lb, ub):
    """The two decompositions must accept exactly the same assignments.

    Compared pairwise rather than by counting solutions, so that auxiliary
    variables (of which the two decompositions have different numbers) do not
    affect the comparison.
    """
    disagree = []
    for xs, ys in _pairs(n, lb, ub):
        results = {}
        for kind in ("decompose", "decompose_linear_positive"):
            x = cp.intvar(lb, ub, shape=n, name="x")
            y = cp.intvar(lb, ub, shape=n, name="y")
            con, deff = getattr(cp.LexLess(x, y), kind)()
            results[kind] = bool(cp.Model(
                list(con) + list(deff) + [x == list(xs), y == list(ys)]
            ).solve(solver="ortools"))
        if results["decompose"] != results["decompose_linear_positive"] or \
                results["decompose"] != _oracle(xs, ys):
            disagree.append((list(xs), list(ys), results, _oracle(xs, ys)))

    assert not disagree, (
        f"{len(disagree)} assignment(s) where the decompositions disagree with "
        f"each other or with value(), e.g. {disagree[0]}"
    )
