"""
Tests for Gurobi solver transformations and expression tree support.

These tests verify that CPMpy correctly transforms constraints to Gurobi's
expression tree format by comparing the generated LP file output.

https://docs.gurobi.com/projects/optimizer/en/current/features/nonlinear.html
"""

import pytest
import tempfile
import cpmpy as cp
from cpmpy.solvers.gurobi import CPM_gurobi
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl


def get_lp_string(solver):
    """Write the Gurobi model to LP format and return as string."""
    solver.grb_model.update()
    with tempfile.NamedTemporaryFile(suffix=".lp", delete=False, mode="w") as f:
        solver.grb_model.write(f.name)
    with open(f.name) as rf:
        return rf.read()


def extract_constraints(lp_string):
    """Extract constraints from 'Subject To' and 'General Constraints' sections as a list."""
    lines = lp_string.split("\n")
    in_section = False
    constraints = []
    for line in lines:
        if line.strip() in ("Subject To", "General Constraints"):
            in_section = True
            continue
        if line.strip() in ("Bounds", "Binaries", "Generals", "End"):
            in_section = False
        if in_section and line.strip():
            constraints.append(line.strip())
    return constraints


def reset_counters():
    _IntVarImpl.counter = 0
    _BoolVarImpl.counter = 0


def expression_tree_cases():
    for c in expression_tree_cases_():
        reset_counters()
        yield c


def expression_tree_cases_():
    """Generator yielding (name, constraint_func, expected_lp) tuples."""

    x, y, z = [cp.intvar(-2, 2, name=name) for name in "xyz"]
    p, q, r = [cp.boolvar(name=name) for name in "pqr"]

    yield (
        "BV",
        p,
        ["p"],
        ["R0: p >= 1"],
    )

    yield (
        "True",
        cp.BoolVal(True),
        ["boolval(True)"],
        ["R0: C0 = 1"],
    )

    yield (
        "False",
        cp.BoolVal(False),
        ["boolval(False)"],
        ["R0: C0 = 0"],
    )

    yield (
        "pow",
        x**2 + y == 9,
        ["(pow(x,2)) + (y) == 9"],
        ["qc0: y + [ x ^2 ] = 9"],
    )

    yield (
        "neq",
        x != 1,
        ["(z) * (sum([1, -1] * [x, y])) == 1"],
        [
            "R0: BV2 >= 1",
            "GC0: BV0 = 1 -> x >= 2",
            "GC1: BV1 = 1 -> x <= 0",
            "GC2: BV2 = OR ( BV0 , BV1 )",
        ],  # TODO not sure how it did this, but happy with it
    )

    """Positive implications"""
    yield (
        "positive_implication",
        p.implies(x + y <= 3),
        ["(p) -> ((x) + (y) <= 3)"],
        ["GC0: p = 1 -> x + y <= 3"],
    )

    """Negative implications"""
    yield (
        "negative_implication",
        (~p).implies(x + y <= 3),
        ["(~p) -> ((x) + (y) <= 3)"],
        ["GC0: p = 0 -> x + y <= 3"],
    )

    """While NL constraint pow can be a expression tree node, it cannot be reified"""
    yield (
        "imp_quad",
        p.implies(x * y == 3),
        ["((x) * (y)) == (IV0)", "(p) -> (sum(IV0) == 3)"],
        ["qc0: IV0 + [ - x * y ] = 0", "GC0: p = 1 -> IV0 = 3"],
    )

    yield (
        "pow_bool",
        p**2 + q == 2,
        ["(pow(p,2)) + (q) == 2"],
        ["qc0: q + [ p ^2 ] = 2"],
    )

    yield (
        "multiplication",
        z + x * y == 12,
        ["(z) + ((x) * (y)) == 12"],
        ["qc0: z + [ x * y ] = 12"],
    )

    # yield (
    #     "div",
    #     z + x / y == 12,
    #     ["(z) + (IV0) == 12", "(max(x,y)) == (IV0)"],
    #     ["R0: z + IV0 = 12", "GC0: IV0 = MAX ( x , y )"],
    # )

    yield (
        "maximum",
        z + cp.Maximum([x, y]) == 12,
        ["(z) + (IV0) == 12", "(max(x,y)) == (IV0)"],
        ["R0: IV0 + z = 12", "GC0: IV0 = MAX ( x , y )"],
    )

    yield (
        "nested",
        z + (x - 3) * ((-y) ** 2) - 3 == 12,
        ["(z) + (((x) + -3) * (pow(sum([-1] * [y]),2))) == 15"],
        [
            "\\ C3 = (z + (sqr(y) * (-3 + x))) + -3",
            "GC0: C3 = NL : ( PLUS , -1 , -1 ) ( PLUS , -1 , 0 ) ( VARIABLE , z , 1 )",
            "( MULTIPLY , -1 , 1 ) ( SQUARE , -1 , 3 ) ( VARIABLE , y , 4 )",
            "( PLUS , -1 , 3 ) ( CONSTANT , -3 , 6 ) ( VARIABLE , x , 6 )",
            "( CONSTANT , -3 , 0 )",
        ],
    )

    # yield (
    #     "nested_obj",
    #     cp.Model(minimize=z + (x - 3) * ((-y) ** 2) - 3),
    #     ["(z) + (((x) + -3) * (pow(sum([-1] * [y]),2))) == 15"],
    #     [
    #         "\\ C3 = z + (sqr(y) * (-3 + x))",
    #         "GC0: C3 = NL : ( PLUS , -1 , -1 ) ( VARIABLE , z , 0 )",
    #         # TODO not totally clean MULTIPLY node?
    #         "( MULTIPLY , -1 , 0 ) ( SQUARE , -1 , 2 ) ( VARIABLE , y , 3 )",
    #         "( PLUS , -1 , 2 ) ( CONSTANT , -3 , 5 ) ( VARIABLE , x , 5 )",
    #     ],
    # )

    # # TODO needlessly reifying
    # yield (
    #     "subtract",
    #     -(x * y) == 12,
    #     ["(z) + ((x) * (y)) == 12"],
    #     ["qc0: z + [ x * y ] = 12"],
    # )

    # TODO divide (semantic may be slightly different from gurobi?)

    yield (
        "abs",
        cp.Abs(x) + y == 3,
        ["(IV0) + (y) == 3", "(abs(x)) == (IV0)"],
        ["R0: IV0 + y = 3", "GC0: IV0 = ABS ( x )"],
    )

    """Mul is supported in expression tree, but not abs"""
    yield (
        "abs_in_mul",
        cp.Abs(x) * y + z == 3,
        ["((IV0) * (y)) + (z) == 3", "(abs(x)) == (IV0)"],
        ["qc0: z + [ IV0 * y ] = 3", "GC0: IV0 = ABS ( x )"],
    )

    yield (
        "mul_in_abs",
        cp.Abs(x * y) + z == 3,
        ["(IV1) + (z) == 3", "(abs(IV0)) == (IV1)", "((x) * (y)) == (IV0)"],
        ["R0: IV0 + z = 3", "qc0: C2 + [ - x * y ] = 0", "GC0: IV0 = ABS ( C2 )"],
    )

    # TODO keep as operator?
    yield (
        "minus_in",
        z * (x - y) == 1,
        ["(z) * (sum([1, -1] * [x, y])) == 1"],
        ["qc0: [ z * x - z * y ] = 1"],  # TODO not sure how it did this, but happy with it
    )

    yield (
        "minus_out",
        z * -(x + y) == 1,
        ["(z) * (sum([-1, -1] * [x, y])) == 1"],
        ["qc0: [ - z * x - z * y ] = 1"],
    )

    # yield (
    #     "tmp",
    #     (~p).implies(x >= 2),
    #     ["(z) * (sum([-1, -1] * [x, y])) == 1"],
    #     ["qc0: [ - z * x - z * y ] = 1"],
    # )

    yield (
        "reification",
        z * (x == 2) == 1,
        [
            "(z) * (x == 2) == 1",
        ],
        [
            "qc0: [ BV0 * z ] = 1",
            "GC0: BV0 = 1 -> x = 2",
            "GC1: BV1 = 1 -> x >= 3",
            "GC2: BV2 = 1 -> x <= 1",
            "GC4: BV0 = 0 -> BV3 >= 1",
            "GC3: BV3 = OR ( BV1 , BV2 )",
        ],
    )

    yield (
        "disjunction",
        p | q,
        ["(p) or (q)"],
        ["R0: BV0 >= 1", "GC0: BV0 = OR ( p , q )"],
# TODO perhaps ["GC0: C2 = OR ( p , q )"],
    )

    # yield (
    #     "conjunction",
    #     p & q,
    #     ["(p) and (q)"],
    #     ["R0: BV0 >= 1", "GC0: BV0 = OR ( p , q )"],
    # )

    yield (
        "conjunction_in_disjunction",
        (p | (q & r)),
        ["(p) or (BV0)", "(BV0) == ((q) and (r))"],
        ["R0: BV1 >= 1", "GC0: BV0 = AND ( q , r )", "GC1: BV1 = OR ( p , BV0 )"],
    )


@pytest.mark.requires_solver("gurobi")
@pytest.mark.parametrize(
    "name,constraint,expected_tf,expected_lp", list(expression_tree_cases()), ids=[c[0] for c in expression_tree_cases()]
)
def test_gurobi_expression_tree(name, constraint, expected_tf, expected_lp):
    """Test that Gurobi transformation generates expected LP output."""
    reset_counters()

    print("CONSTRAINT")
    print(constraint)
    transformed = [str(c) for c in CPM_gurobi().transform(constraint)]
    print("TF", ", ".join(transformed))
    reset_counters()

    if not isinstance(constraint, cp.Model):
        m = cp.Model(constraint)
    solver = CPM_gurobi(m)

    lp = get_lp_string(solver)
    print(lp)

    constraints = extract_constraints(lp)
    # assert transformed == expected_tf, f"From {constraint} to TF\n{transformed}"
    assert constraints == expected_lp, f"From {constraint} to LP\n{constraints}\n\nFull LP:\n{lp}"
    is_sat = solver.solve()
    assert not is_sat or constraint.value(), "Incorrect constraint value"
    # assert is_sat == m.solve(), "Unexpected solve result"
