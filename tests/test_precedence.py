"""
Edge-case tests for the :class:`~cpmpy.expressions.globalconstraints.Precedence`
global constraint.

The tests in this module are written against the semantics implemented by
``Precedence.value()``, which is the reference definition of the constraint:
for every consecutive pair ``(s, t)`` in the precedence list, a variable may
only take value ``t`` if some earlier variable already took value ``s``. The
values in the precedence list must be pairwise distinct; duplicates are
rejected at construction time.

Every test here uses the ``solver`` fixture, so the whole module runs against
whichever solver is selected on the command line, e.g.::

    python -m pytest -k precedence -x --solver=gurobi

Both the module name and every test name contain "precedence", so ``-k
precedence`` selects all of them.
"""

import itertools

import numpy as np
import pytest

import cpmpy as cp
from cpmpy.exceptions import TypeError
from cpmpy.expressions.utils import argvals

# Solvers that cannot handle integer decision variables at all.
NO_INTVAR_SOLVERS = ("pysat", "pysdd", "pindakaas", "rc2")


def _skip_if_no_intvars(solver):
    """Skip the calling test when the solver has no integer variables."""
    if solver in NO_INTVAR_SOLVERS:
        pytest.skip(f"{solver} does not support integer variables")


def _oracle(vals, precedence):
    """Ground truth for an assignment.

    Builds the constraint over plain constants, so ``value()`` evaluates the
    reference semantics directly without involving any solver.
    """
    return cp.Precedence(list(vals), list(precedence)).value()


def _check_all_assignments(solver, n, lb, ub, precedence):
    """Assert solver and reference semantics agree on every assignment.

    Enumerates the full cube ``[lb, ub]^n``, posts the assignment together with
    the ``Precedence`` constraint, and checks that satisfiability matches
    ``Precedence.value()`` on those same constants.
    """
    for vals in itertools.product(range(lb, ub + 1), repeat=n):
        x = cp.intvar(lb, ub, shape=n, name="x")
        cons = cp.Precedence(x, precedence)
        sat = bool(cp.Model([cons, x == list(vals)]).solve(solver=solver))
        expected = _oracle(vals, precedence)
        assert sat == expected, (
            f"{solver}: Precedence({list(vals)}, {list(precedence)}) "
            f"reported {'SAT' if sat else 'UNSAT'} but value() says {expected}"
        )
        if sat:
            assert cons.value() is True


# ---------------------------------------------------------------------------- #
#                        Degenerate precedence lists                           #
# ---------------------------------------------------------------------------- #

@pytest.mark.usefixtures("solver")
def test_precedence_empty_list(solver):
    """An empty precedence list has no consecutive pairs, so it constrains nothing."""
    _skip_if_no_intvars(solver)
    _check_all_assignments(solver, n=3, lb=0, ub=2, precedence=[])


@pytest.mark.usefixtures("solver")
def test_precedence_single_value(solver):
    """A one-element precedence list also has no pairs and constrains nothing."""
    _skip_if_no_intvars(solver)
    _check_all_assignments(solver, n=3, lb=0, ub=2, precedence=[0])


@pytest.mark.usefixtures("solver")
def test_precedence_single_variable(solver):
    """With a single variable, only the 'first position' rule can bite."""
    _skip_if_no_intvars(solver)
    x = cp.intvar(0, 2, name="x0")

    cons = cp.Precedence([x], [0, 1])
    # x cannot be 1: nothing can precede it.
    assert not cp.Model([cons, x == 1]).solve(solver=solver)
    for val in (0, 2):
        assert cp.Model([cons, x == val]).solve(solver=solver)
        assert cons.value() is True


# ---------------------------------------------------------------------------- #
#                    Length mismatch between vars and values                   #
# ---------------------------------------------------------------------------- #

@pytest.mark.usefixtures("solver")
def test_precedence_more_values_than_vars(solver):
    """The precedence list may be longer than the variable array."""
    _skip_if_no_intvars(solver)
    _check_all_assignments(solver, n=2, lb=0, ub=2, precedence=[0, 1, 2])


@pytest.mark.usefixtures("solver")
def test_precedence_many_more_values_than_vars(solver):
    """Precedence list far longer than the variable array."""
    _skip_if_no_intvars(solver)
    _check_all_assignments(solver, n=2, lb=0, ub=3, precedence=[0, 1, 2, 3])


@pytest.mark.usefixtures("solver")
def test_precedence_more_vars_than_values(solver):
    """The variable array may be longer than the precedence list."""
    _skip_if_no_intvars(solver)
    _check_all_assignments(solver, n=4, lb=0, ub=2, precedence=[0, 1])


# ---------------------------------------------------------------------------- #
#                          Unusual precedence lists                            #
# ---------------------------------------------------------------------------- #

@pytest.mark.usefixtures("solver")
def test_precedence_reversed_order(solver):
    """Precedence values need not be sorted."""
    _skip_if_no_intvars(solver)
    _check_all_assignments(solver, n=3, lb=0, ub=2, precedence=[2, 1, 0])


@pytest.mark.usefixtures("solver")
def test_precedence_values_outside_domain(solver):
    """Values that no variable can take."""
    _skip_if_no_intvars(solver)
    # Neither 5 nor 6 is reachable, so the constraint is vacuously true.
    _check_all_assignments(solver, n=3, lb=0, ub=2, precedence=[5, 6])
    # 5 is unreachable, so the value 1 that depends on it becomes unreachable too.
    _check_all_assignments(solver, n=3, lb=0, ub=2, precedence=[5, 1])
    # 5 is unreachable but only as the tail of the pair, so nothing is forbidden.
    _check_all_assignments(solver, n=3, lb=0, ub=2, precedence=[1, 5])


@pytest.mark.usefixtures("solver")
def test_precedence_negative_values(solver):
    """Negative domains and negative precedence values."""
    _skip_if_no_intvars(solver)
    _check_all_assignments(solver, n=3, lb=-3, ub=-1, precedence=[-2, -1])


@pytest.mark.usefixtures("solver")
def test_precedence_domain_spanning_zero(solver):
    """Domains that straddle zero, with 0 itself in the precedence list."""
    _skip_if_no_intvars(solver)
    _check_all_assignments(solver, n=3, lb=-1, ub=1, precedence=[-1, 0, 1])


# ---------------------------------------------------------------------------- #
#                     Constants and expressions as arguments                   #
# ---------------------------------------------------------------------------- #

@pytest.mark.usefixtures("solver")
def test_precedence_all_constants(solver):
    """A fully constant argument list must still be handled by the solver."""
    _skip_if_no_intvars(solver)
    assert cp.Model(cp.Precedence([0, 1, 2], [0, 1, 2])).solve(solver=solver)
    assert not cp.Model(cp.Precedence([1, 0, 2], [0, 1, 2])).solve(solver=solver)


@pytest.mark.usefixtures("solver")
def test_precedence_mixed_constants_and_vars(solver):
    """Constants mixed in between decision variables."""
    _skip_if_no_intvars(solver)
    x = cp.intvar(0, 2, shape=2, name="x")

    # A leading constant 1 already violates [0, 1]: nothing precedes it.
    assert not cp.Model(cp.Precedence([1, x[0], x[1]], [0, 1])).solve(solver=solver)

    # A leading constant 0 licenses a later 1.
    cons = cp.Precedence([0, x[0], x[1]], [0, 1])
    assert cp.Model([cons, x[0] == 1]).solve(solver=solver)
    assert cons.value() is True


@pytest.mark.usefixtures("solver")
def test_precedence_expressions_as_vars(solver):
    """Non-variable expressions as members of the argument array."""
    _skip_if_no_intvars(solver)
    x = cp.intvar(0, 2, shape=3, name="x")

    cons = cp.Precedence([x[0] + 1, x[1], x[2]], [1, 2])
    # x[0] + 1 == 1 (so x[0] == 0) is the only way to license a later 2.
    assert cp.Model([cons, x[1] == 2]).solve(solver=solver)
    assert cons.value() is True
    assert x[0].value() == 0

    assert not cp.Model([cons, x[0] == 1, x[1] == 2]).solve(solver=solver)


# ---------------------------------------------------------------------------- #
#                          Negation and reification                            #
# ---------------------------------------------------------------------------- #

@pytest.mark.usefixtures("solver")
def test_precedence_negated(solver):
    """A negated Precedence must accept exactly the violating assignments."""
    _skip_if_no_intvars(solver)

    x = cp.intvar(0, 2, shape=3, name="x")
    cons = cp.Precedence(x, [0, 1])

    # [1, 0, 0] violates the precedence, so its negation accepts it.
    assert cp.Model([~cons, x == [1, 0, 0]]).solve(solver=solver)
    assert cons.value() is False

    # [0, 1, 0] satisfies the precedence, so its negation rejects it.
    assert not cp.Model([~cons, x == [0, 1, 0]]).solve(solver=solver)


@pytest.mark.usefixtures("solver")
def test_precedence_reified(solver):
    """Reification must agree with value() in both directions."""
    _skip_if_no_intvars(solver)

    x = cp.intvar(0, 2, shape=3, name="x")
    bv = cp.boolvar(name="bv")
    cons = cp.Precedence(x, [0, 1])

    assert cp.Model([bv == cons, x == [0, 1, 2]]).solve(solver=solver)
    assert bv.value() is True

    assert cp.Model([bv == cons, x == [1, 0, 2]]).solve(solver=solver)
    assert bv.value() is False

    # half-reification
    assert not cp.Model([bv.implies(cons), bv, x == [1, 0, 2]]).solve(solver=solver)


# ---------------------------------------------------------------------------- #
#                            Non-positive contexts                             #
# ---------------------------------------------------------------------------- #

"""
A global constraint in a positive context only has to be *implied* by its
encoding; in a non-positive context the encoding must also rule out the
assignments that violate it. An encoding whose auxiliary variables are only
constrained in one direction happily passes the positive tests and silently
fails here, so every context below is checked against the full truth table
rather than on a couple of hand-picked assignments.
"""

def _check_boolean_context(solver, context, expected, precedence, n=3, lb=0, ub=2):
    """Check one Boolean context against its truth table, for every assignment.

    Arguments:
        context: callable ``(cons, bv) -> Expression`` embedding the constraint
        expected: callable ``(prec_holds, bv_val) -> bool`` giving the truth table
    """
    for vals in itertools.product(range(lb, ub + 1), repeat=n):
        prec_holds = _oracle(vals, precedence)
        for bv_val in (False, True):
            x = cp.intvar(lb, ub, shape=n, name="x")
            bv = cp.boolvar(name="bv")
            cons = cp.Precedence(x, precedence)
            fixed = [x == list(vals), bv if bv_val else ~bv]
            sat = bool(cp.Model([context(cons, bv)] + fixed).solve(solver=solver))
            assert sat == expected(prec_holds, bv_val), (
                f"{solver}: with x={list(vals)} (precedence holds: {prec_holds}) "
                f"and bv={bv_val}, the context reported "
                f"{'SAT' if sat else 'UNSAT'} but should be "
                f"{'SAT' if expected(prec_holds, bv_val) else 'UNSAT'}"
            )


# (id, how to embed the constraint, the truth table it must obey).
# `p` is Precedence.value() on the assignment, `b` is the value of bv.
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


@pytest.mark.parametrize(
    "context,expected",
    [(ctx, exp) for _, ctx, exp in _CONTEXTS],
    ids=[name for name, _, _ in _CONTEXTS],
)
@pytest.mark.usefixtures("solver")
def test_precedence_in_non_positive_context(solver, context, expected):
    """The constraint must be encoded exactly, not just implied."""
    _skip_if_no_intvars(solver)
    _check_boolean_context(solver, context, expected, precedence=[0, 1])


@pytest.mark.parametrize(
    "context,expected",
    [(ctx, exp) for _, ctx, exp in _CONTEXTS],
    ids=[name for name, _, _ in _CONTEXTS],
)
@pytest.mark.usefixtures("solver")
def test_precedence_in_non_positive_context_longer_chain(solver, context, expected):
    """Same truth tables, but with a chain of three values."""
    _skip_if_no_intvars(solver)
    _check_boolean_context(solver, context, expected, precedence=[0, 1, 2])


@pytest.mark.parametrize(
    "n,ub,precedence",
    [
        (3, 2, []),            # vacuously true, so its negation is unsatisfiable
        (3, 2, [0]),           # idem
        (3, 2, [2, 0]),        # unsorted
        (2, 2, [0, 1, 2, 3]),  # more values than variables
        (4, 1, [0, 1]),        # more variables than values
    ],
    ids=str,
)
@pytest.mark.usefixtures("solver")
def test_precedence_negated_edge_shapes(solver, n, ub, precedence):
    """Negation combined with the degenerate argument shapes."""
    _skip_if_no_intvars(solver)
    _check_boolean_context(
        solver, lambda c, bv: ~c, lambda p, b: not p, precedence, n=n, lb=0, ub=ub
    )


@pytest.mark.usefixtures("solver")
def test_precedence_two_constraints_in_non_positive_context(solver):
    """Two Precedence constraints sharing variables, both under a negation."""
    _skip_if_no_intvars(solver)
    x = cp.intvar(0, 2, shape=3, name="x")

    for vals in itertools.product(range(0, 3), repeat=3):
        c1 = cp.Precedence(x, [0, 1])
        c2 = cp.Precedence(x, [1, 2])
        # ~(c1 & c2) == (~c1 | ~c2)
        expected = not (_oracle(vals, [0, 1]) and _oracle(vals, [1, 2]))
        sat = bool(cp.Model([~(c1 & c2), x == list(vals)]).solve(solver=solver))
        assert sat == expected, f"{solver}: ~(c1 & c2) wrong for x={list(vals)}"


# ---------------------------------------------------------------------------- #
#                               Side constraints                               #
# ---------------------------------------------------------------------------- #

"""
A decomposition introduces auxiliary variables. It stays correct under side
constraints on the original variables only if those auxiliaries are pinned
down by the originals: if some auxiliary is free to move, the encoding can
admit an assignment the constraint forbids (spurious solutions) or block one
it allows (over-constraining). Neither shows up when the constraint is tested
on its own, so each side constraint below is checked both assignment by
assignment and through an optimum, where a spurious solution changes the
objective value.
"""

# (id, how to build the side constraint, the same predicate in Python)
_SIDE_CONSTRAINTS = [
    ("alldifferent",
     lambda x: cp.AllDifferent(x),
     lambda v: len(set(v)) == len(v)),
    ("non_decreasing",
     lambda x: [x[i] <= x[i + 1] for i in range(len(x) - 1)],
     lambda v: all(v[i] <= v[i + 1] for i in range(len(v) - 1))),
    ("sum_bound",
     lambda x: cp.sum(x) <= 4,
     lambda v: sum(v) <= 4),
    ("fixed_middle",
     lambda x: x[1] == 2,
     lambda v: v[1] == 2),
    ("disequality",
     lambda x: x[0] != x[-1],
     lambda v: v[0] != v[-1]),
    ("count",
     lambda x: cp.Count(x, 0) >= 1,
     lambda v: list(v).count(0) >= 1),
]


@pytest.mark.parametrize(
    "side,side_ref",
    [(s, r) for _, s, r in _SIDE_CONSTRAINTS],
    ids=[name for name, _, _ in _SIDE_CONSTRAINTS],
)
@pytest.mark.usefixtures("solver")
def test_precedence_with_side_constraints(solver, side, side_ref):
    """Precedence together with a side constraint on the same variables."""
    _skip_if_no_intvars(solver)
    n, lb, ub, precedence = 3, 0, 2, [0, 1]

    for vals in itertools.product(range(lb, ub + 1), repeat=n):
        x = cp.intvar(lb, ub, shape=n, name="x")
        cons = cp.Precedence(x, precedence)
        sat = bool(cp.Model([cons, side(x), x == list(vals)]).solve(solver=solver))
        expected = _oracle(vals, precedence) and side_ref(vals)
        assert sat == expected, (
            f"{solver}: x={list(vals)} with a side constraint reported "
            f"{'SAT' if sat else 'UNSAT'}, expected {'SAT' if expected else 'UNSAT'}"
        )


@pytest.mark.parametrize(
    "side,side_ref",
    [(s, r) for _, s, r in _SIDE_CONSTRAINTS],
    ids=[name for name, _, _ in _SIDE_CONSTRAINTS],
)
@pytest.mark.parametrize("sense", ["maximize", "minimize"])
@pytest.mark.usefixtures("solver")
def test_precedence_optimum_with_side_constraints(solver, sense, side, side_ref):
    """The optimum over Precedence + a side constraint must match brute force.

    Leaving the variables free is what makes this different from the checks
    above: a spurious solution admitted by the encoding would show up as a
    better objective than any genuinely feasible assignment.
    """
    _skip_if_no_intvars(solver)
    n, lb, ub, precedence = 4, 0, 3, [0, 1, 2]

    feasible = [v for v in itertools.product(range(lb, ub + 1), repeat=n)
                if _oracle(v, precedence) and side_ref(v)]

    x = cp.intvar(lb, ub, shape=n, name="x")
    cons = cp.Precedence(x, precedence)
    model = cp.Model([cons, side(x)])
    getattr(model, sense)(cp.sum(x))

    if not feasible:
        assert not model.solve(solver=solver)
        return

    assert model.solve(solver=solver)
    best = (max if sense == "maximize" else min)(sum(v) for v in feasible)
    assert int(model.objective_value()) == best, (
        f"{solver}: {sense} gave {model.objective_value()}, brute force says {best}"
    )
    # the reported solution must itself be feasible, not just hit the right value
    assert cons.value() is True
    assert side_ref(tuple(argvals(x)))


# ---------------------------------------------------------------------------- #
#                         Exhaustive consistency sweep                         #
# ---------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "n,precedence",
    [
        (2, [0, 1]),
        (3, [0, 1]),
        (3, [0, 1, 2]),
        (3, [2, 0]),
        (3, [1, 2, 0]),
        (4, [0, 2]),
    ],
    ids=str,
)
@pytest.mark.usefixtures("solver")
def test_precedence_exhaustive_matches_value(solver, n, precedence):
    """Sweep small instances and require solver/reference agreement everywhere."""
    _skip_if_no_intvars(solver)
    _check_all_assignments(solver, n=n, lb=0, ub=2, precedence=precedence)


# ---------------------------------------------------------------------------- #
#                              Argument checking                               #
# ---------------------------------------------------------------------------- #

def test_precedence_argument_type_errors():
    """Bad argument shapes are rejected at construction time."""
    x = cp.intvar(0, 2, shape=3, name="x")

    with pytest.raises(TypeError):
        cp.Precedence(x[0], [0, 1])          # not a list of variables
    with pytest.raises(TypeError):
        cp.Precedence(x, 0)                  # not a list of values
    with pytest.raises(TypeError):
        cp.Precedence(x, [0, x[1]])          # values must be constants


@pytest.mark.parametrize(
    "precedence",
    [
        [0, 0],        # a value that would have to precede itself
        [1, 1, 2],     # repeat at the head of a longer chain
        [0, 1, 0],     # cyclic: 0 before 1 before 0
        [2, 0, 1, 0],  # repeat at a distance
    ],
    ids=str,
)
def test_precedence_duplicate_values_rejected(precedence):
    """Duplicate precedence values are rejected at construction time.

    A repeated value ``v`` would require an occurrence of ``v`` to be strictly
    preceded by another occurrence of ``v``, which no assignment can satisfy for
    the leftmost one. The constraint is defined over distinct values (Law & Lee),
    so such a list is an error rather than a way to forbid the value.
    """
    x = cp.intvar(0, 2, shape=3, name="x")

    with pytest.raises(ValueError):
        cp.Precedence(x, precedence)
    with pytest.raises(ValueError):
        cp.Precedence([0, 1, 2], precedence)  # also on constant arguments


def test_precedence_distinct_values_accepted():
    """Lists that merely look degenerate but hold distinct values stay legal."""
    x = cp.intvar(0, 2, shape=3, name="x")

    cp.Precedence(x, [])           # no pairs at all
    cp.Precedence(x, [0])          # single value, no pairs
    cp.Precedence(x, [2, 1, 0])    # unsorted
    cp.Precedence(x, [-1, 0, 1])   # negative values
    cp.Precedence(x, np.array([0, 1, 2]))  # numpy integers
