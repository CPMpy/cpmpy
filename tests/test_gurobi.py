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
        ["1"],
        # ["boolval(True)"],
        [],
    )

    yield (
        "False",
        cp.BoolVal(False),
        ["0"],
        # ["boolval(False)"],
        ["R0: <= -1"],
    )

    """A quadratic constraint"""
    yield (
        "pow",
        x**2 + y == 9,
        ["(pow(x,2)) + (y) == 9"],
        ["qc0: y + [ x ^2 ] = 9"],
    )

    # yield (
    #     "implication",
    #     p.implies(q),
    #     ["(p) -> (q >= 1)"],
    #     ["GC0: p = 1 -> q >= 1"],
    # )

    yield (
        "neq",
        x != 1,
        # z * (x - y) == 1
        [
            "(BV0) -> (x >= 2)",
            "(~BV0) -> (x <= 1)",
            "True",
            "(BV1) -> (x <= 0)",
            "(~BV1) -> (x >= 1)",
            "True",
            "(BV0) or (BV1)",
            "1",
        ],
        [
            "GC0: BV0 = 1 -> x >= 2",
            "GC1: BV0 = 0 -> x <= 1",
            "GC2: BV1 = 1 -> x <= 0",
            "GC3: BV1 = 0 -> x >= 1",
            "GC4: C3 = OR ( BV0 , BV1 )",
        ],
    )

    yield (
        "implies_neq",
        p.implies(x != 1),
        [
            "(BV0) -> (x >= 2)",
            "(~BV0) -> (x <= 1)",
            "True",
            "(BV1) -> (x <= 0)",
            "(~BV1) -> (x >= 1)",
            "True",
            "(BV2) == ((BV0) or (BV1))",
            "(p) -> (BV2 >= 1)",
        ],
        [
            "GC0: BV0 = 1 -> x >= 2",
            "GC1: BV0 = 0 -> x <= 1",
            "GC2: BV1 = 1 -> x <= 0",
            "GC3: BV1 = 0 -> x >= 1",
            "GC5: p = 1 -> BV2 >= 1",
            "GC4: BV2 = OR ( BV0 , BV1 )",
        ],
    )

    yield (
        "implies_x_ge_1",
        p.implies(x >= 1),
        # ["(p) -> (x >= 1)"],
        ["(p) -> (x >= 1)"],
        ["GC0: p = 1 -> x >= 1"],
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

    """While NL constraint pow can be a expression tree node, it cannot be reified as indicator constraint require linears"""
    yield (
        "implies_quad",
        p.implies(x * y >= 4),
        ["(IV0) == ((x) * (y))", "(p) -> (IV0 >= 4)"],
        ["qc0: IV0 + [ - x * y ] = 0", "GC0: p = 1 -> IV0 >= 4"],
    )

    """An indicator LHS has to be a BV"""
    yield (
        "quad_implies",
        (x * y >= 2).implies(z <= 3),
        [
            "(IV0) == ((x) * (y))",
            "(BV0) -> (IV0 >= 2)",
            "(~BV0) -> (IV0 <= 1)",
            "True",
            "(BV0) -> (z <= 3)",
        ],
        [
            # TODO duplicated
            "qc0: IV0 + [ - x * y ] = 0",
            "GC0: BV0 = 1 -> IV0 >= 2",
            "GC1: BV0 = 0 -> IV0 <= 1",
            "GC2: BV0 = 1 -> z <= 3",
        ],
    )

    nl_con = z + (x - 3) * ((-y) ** 2) - 3 == 12

    # """An indicator LHS has to be a BV"""
    # yield (
    #     "nl_implies",
    #     (x + (y>=3) <= 2).implies(z <= 3),
    #     ["qc0: IV0 + [ - x * y ] = 0", "GC0: p = 1 -> IV0 = 3"],
    #     [
    #         "GC1: BV0 = 1 -> IV0 + z = 15",
    #         "GC2: BV1 = 1 -> IV0 + z >= 16",
    #         "GC3: BV2 = 1 -> IV0 + z <= 14",
    #         "GC5: BV0 = 0 -> BV3 >= 1",
    #         "GC6: BV0 = 1 -> z <= 3",
    #         "\\ IV0 = sqr(y) * (-3 + x)",
    #         "GC0: IV0 = NL : ( MULTIPLY , -1 , -1 ) ( SQUARE , -1 , 0 )",
    #         "( VARIABLE , y , 1 ) ( PLUS , -1 , 0 ) ( CONSTANT , -3 , 3 )",
    #         "( VARIABLE , x , 3 )",
    #         "GC4: BV3 = OR ( BV1 , BV2 )",
    #     ],
    # )

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
        ["(IV0) == (max(x,y))", "(z) + (IV0) == 12"],
        ["R0: IV0 + z = 12", "GC0: IV0 = MAX ( x , y )"],
    )

    yield (
        "nested",
        nl_con,
        # ["(z) + (((x) + -3) * (pow(sum([-1] * [y]),2))) == 15"],
        None,
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
    #     ["(z) + (((x) + -3) * pow(sum([-1] * [y]),2)) == 15"],
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
        [
            "(IV0) == (abs(x))",
            "(IV0) + (y) == 3",
        ],
        ["R0: IV0 + y = 3", "GC0: IV0 = ABS ( x )"],
    )

    """Mul is supported in expression tree, but not abs"""
    yield (
        "abs_in_mul",
        cp.Abs(x) * y + z == 3,
        ["(IV0) == (abs(x))", "((IV0) * (y)) + (z) == 3"],
        ["qc0: z + [ IV0 * y ] = 3", "GC0: IV0 = ABS ( x )"],
    )

    yield (
        "mul_in_abs",
        cp.Abs(x * y) + z == 3,
        ["(IV0) == ((x) * (y))", "(IV1) == (abs(IV0))", "(IV1) + (z) == 3"],
        None,
        # ["R0: IV0 + z = 3", "qc0: IV1 + [ - x * y ] = 0", "GC0: IV0 = ABS ( IV1 )"],
    )

    # TODO keep as operator?
    yield (
        "minus_in",
        z * (x - y) == 1,
        ["(z) * ((x) + (-(y))) == 1"],
        ["qc0: [ z * x - z * y ] = 1"],  # TODO not sure how it did this, but happy with it
    )

    yield (
        "minus_out",
        z * -(x + y) == 1,
        ["(z) * (-((x) + (y))) == 1"],
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
        # TODO apply direct encoding Should help a lot here
        [
            "(BV0) -> (x == 2)",
            "(BV1) -> (x >= 3)",
            "(~BV1) -> (x <= 2)",
            "True",
            "(BV2) -> (x <= 1)",
            "(~BV2) -> (x >= 2)",
            "True",
            "(BV3) == ((BV1) or (BV2))",
            "(~BV0) -> (BV3 >= 1)",
            "True",
            "(z) * (BV0) == 1",
        ],
        None,
    )

    yield (
        "disjunction",
        p | q,
        ["(p) or (q)", "1"],  # TODO avoid BV0
        ["GC0: C2 = OR ( p , q )"],
        # TODO perhaps ["GC0: C2 = OR ( p , q )"],
    )

    # yield (
    #     "maximum_root",
    #     1 == cp.Maximum([x, y]),
    #     None,
    #     ["GC0: C2 = MAX ( x , y )"],
    # )

    yield (
        "maximum_bv_root",
        1 == cp.Maximum([p, q]),
        ["(IV0) == (max(p,q))", "IV0 == 1"],
        ["R0: IV0 = 1", "GC0: IV0 = MAX ( p , q )"],
    )

    # (x) * (pow(y,2)) <= 4
    # (x) * (pow(y,2)) - 4 <= 0
    # y <= 0, y = (x) * (pow(y,2)) - 4
    yield (
        "unnormalized_quad",
        (x * y) <= 4,
        ["(x) * (y) <= 4"],
        ["qc0: [ x * y ] <= 4"],
        # TODO perhaps ["GC0: C2 = OR ( p , q )"],
    )

    yield (
        "normalized_nonlinear",
        (x * (y**2)) <= 4,
        ["(x) * (pow(y,2)) <= 4"],
        [
            "R0: C2 <= 4",
            "\\ C2 = sqr(y) * x",
            "GC0: C2 = NL : ( MULTIPLY , -1 , -1 ) ( SQUARE , -1 , 0 )",
            "( VARIABLE , y , 1 ) ( VARIABLE , x , 0 )",
        ],
        # TODO perhaps ["GC0: C2 = OR ( p , q )"],
    )

    yield (
        "normalize_nonlinear_on_rhs",
        (p | q) <= (x == 2) + (y**2),
        [
            "(BV0) == ((p) or (q))",
            "(BV1) -> (x == 2)",
            "(BV2) -> (x >= 3)",
            "(~BV2) -> (x <= 2)",
            "True",
            "(BV3) -> (x <= 1)",
            "(~BV3) -> (x >= 2)",
            "True",
            "(BV4) == ((BV2) or (BV3))",
            "(~BV1) -> (BV4 >= 1)",
            "True",
            "(BV0) <= ((BV1) + (pow(y,2)))",
        ],  # TODO avoid BV0
        None,
        # TODO perhaps ["GC0: C2 = OR ( p , q )"],
    )

    yield (
        "neg_disjunction",
        ~(p | q),
        ["(~p) == (BV0)", "(~q) == (BV1)", "(BV2) == ((BV0) and (BV1))", "BV2"],
        ["R0: - p - BV0 = -1", "R1: - q - BV1 = -1", "R2: BV2 >= 1", "GC0: BV2 = AND ( BV0 , BV1 )"],
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
        ["(BV0) == ((q) and (r))", "(p) or (BV0)", "1"],
        ["GC0: BV0 = AND ( q , r )", "GC1: C4 = OR ( p , BV0 )"],
    )

    # yield (
    #     "all_different",
    #     cp.AllDifferent(cp.intvar(1, 2, shape=2, name="X")),
    #     ["(BV0) == ((q) and (r))", "(BV1) == ((p) or (BV0))", "BV1"],
    #     ["R0: BV1 >= 1", "GC0: BV0 = AND ( q , r )", "GC1: BV1 = OR ( p , BV0 )"],
    # )


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

    if expected_tf is not None:
        assert transformed == expected_tf, f"From {constraint} to TF\n{transformed}"

    solver = CPM_gurobi(m)

    lp = get_lp_string(solver)
    constraints = extract_constraints(lp)
    print(lp)

    if expected_lp is not None:
        assert constraints == expected_lp, f"From {constraint} to LP\n{constraints}\n\nFull LP:\n{lp}"

    is_sat = solver.solve()
    if is_sat:
        assert constraint.value(), "Incorrect constraint value"

    assert is_sat is m.solve()
    # assert is_sat == m.solve(), "Unexpected solve result"
