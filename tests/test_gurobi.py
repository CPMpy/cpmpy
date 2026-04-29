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


cp.transformations.into_tree.NAMED = True


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
        "nBV",
        ~p,
        ["~p"],
        ["R0: - p >= 0"],
    )

    # TODO
    # yield (
    #     "True",
    #     cp.BoolVal(True),
    #     ["1"],
    #     # ["boolval(True)"],
    #     [],
    # )

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
        x**2 + y == 6,
        ["(pow(x,2)) + (y) == 6"],
        ["qc0: y + [ x ^2 ] = 6"],
    )

    """Accor"""
    yield (
        "accor",
        (x >= 2) | (y == z),
        ["True", "(BV0) -> (x >= 2)", "(~BV0) -> ((y) == (z))"],
        None,
    )

    """Accor"""
    yield (
        # 474. `((pt[12] <= 12) and (pf[12] >= 14)) -> ((x[12][13]) == (x[13][13]))`
        "accor2",
        ((x >= 1) & (y >= 1)).implies(x == z),
        [
            "(BV[(BV[x >= 1]) and (BV[y >= 1])]) -> ((x) == (z))",
            "(BV[x >= 1]) -> (x >= 1)",
            "(~BV[x >= 1]) -> (x <= 0)",
            "True",
            "(BV[y >= 1]) -> (y >= 1)",
            "(~BV[y >= 1]) -> (y <= 0)",
            "True",
            "(BV[(BV[x >= 1]) and (BV[y >= 1])]) == ((BV[x >= 1]) and (BV[y >= 1]))",
        ],
        None,
    )

    # yield (
    #     "implication",
    #     p.implies(q),
    #     ["(p) -> (q >= 1)"],
    #     ["GC0: p = 1 -> q >= 1"],
    # )

    if cp.transformations.into_tree.BIG_M_NEQ:
        yield (
            "neq",
            x != 1,
            ["True", "sum([1, -2] * [x, BV0]) <= 0", "sum([1, -4] * [x, BV0]) >= -2"],
            None,
        )

    yield (
        "reified_neq",
        (x != 1) | p,
        [
            "(BV[~BV[x == 1]]) + (p) >= 1",
            "sum(BV[x == -2], BV[x == -1], BV[x == 0], BV[x == 1], BV[x == 2]) == 1",
            "((sum([0, 1, 2, 3, 4] * [BV[x == -2], BV[x == -1], BV[x == 0], BV[x == 1], BV[x == 2]])) + -2) == (x)",
            "(~BV[x == 1]) == (BV[~BV[x == 1]])",
        ],
        None,
    )

    yield (
        "implies_neq",
        p.implies(x != 1),
        [
            "True",
            "(p) -> (sum([1, -2] * [x, BV0]) <= 0)",
            "(p) -> (sum([1, -4] * [x, BV0]) >= -2)",
        ],
        None,
    )

    yield (
        "implies_x_ge_1",
        p.implies(x >= 1),
        # ["(p) -> (x >= 1)"],
        ["(p) -> (x >= 1)"],
        ["GC0: p = 1 -> x >= 1"],
    )

    yield (
        "pos_implies",
        p.implies(x + y <= 3),
        ["(p) -> ((x) + (y) <= 3)"],
        ["GC0: p = 1 -> x + y <= 3"],
    )

    yield (
        "neg_implies",
        (~p).implies(x + y <= 3),
        ["(~p) -> ((x) + (y) <= 3)"],
        ["GC0: p = 0 -> x + y <= 3"],
    )

    """While NL constraint pow can be a expression tree node, it cannot be reified as indicator constraint require linears"""
    yield (
        "implies_quad",
        p.implies(x * y >= 4),
        ["(p) -> (IV[(x) * (y)] >= 4)", "(IV[(x) * (y)]) == ((x) * (y))"],
        ["qc0: IV[(x)_*_(y)] + [ - x * y ] = 0", "GC0: p = 1 -> IV[(x)_*_(y)] >= 4"],
    )

    """An indicator LHS has to be a BV"""
    yield (
        "quad_implies",
        (x * y >= 2).implies(z <= 1),
        [
            "(BV[(x) * (y) >= 2]) -> (z <= 1)",
            "(IV[(x) * (y)]) == ((x) * (y))",
            "(BV[(x) * (y) >= 2]) -> (IV[(x) * (y)] >= 2)",
            "(~BV[(x) * (y) >= 2]) -> (IV[(x) * (y)] <= 1)",
            "True",
        ],
        None,
    )

    """Indicator body linearization should not leak into subexpressions:
    x*y+z<=3 linearizes the sum (reifying x*y into IV0), but the defining
    constraint IV0==x*y should keep x*y as a tree node, not linearize further."""
    yield (
        "implies_nested_quad",
        p.implies(2 * (x * y) + 3 * z <= 5),
        ["(p) -> (sum([2, 3] * [IV[(x) * (y)], z]) <= 5)", "(IV[(x) * (y)]) == ((x) * (y))"],
        None,
    )

    # ##TODO improve?
    # yield (
    #     "quad_implies",
    #     p.implies(q.implies(x >= 2)),
    #     None,
    #     []
    # )

    yield (
        "pow_bool",
        p**2 + q == 2,
        ["(pow(p,2)) + (q) == 2"],
        ["qc0: q + [ p ^2 ] = 2"],
    )

    yield (
        "multiplication",
        z + x * y == 6,
        ["(z) + ((x) * (y)) == 6"],
        ["qc0: z + [ x * y ] = 6"],
    )

    # yield (
    #     "div",
    #     z + x / y == 12,
    #     ["(z) + (IV0) == 12", "(max(x,y)) == (IV0)"],
    #     ["R0: z + IV0 = 12", "GC0: IV0 = MAX ( x , y )"],
    # )

    yield (
        "maximum",
        z + cp.Maximum([x, y]) == 4,
        ["(z) + (IV[max(x,y)]) == 4", "(IV[max(x,y)]) == (max(x,y))"],
        ["R0: z + IV[max(x,y)] = 4", "GC0: IV[max(x,y)] = MAX ( x , y )"],
    )

    yield (
        "nested",
        z + (cp.max([x, y]) - 3) * ((-y) ** 2) - 3 <= -6,
        # ["sum(z, ((IV0) + -3) * (pow(-(y),2)), -3) <= -6", "(IV0) == (max(x,y))"],
        ["sum(z, ((IV[max(x,y)]) + -3) * (pow(-(y),2)), -3) <= -6", "(IV[max(x,y)]) == (max(x,y))"],
        [
            "R0: C3 <= -6",
            "\\ C3 = (z + (sqr(y) * (-3 + IV[max(x,y)]))) + -3",
            "GC0: C3 = NL : ( PLUS , -1 , -1 ) ( PLUS , -1 , 0 ) ( VARIABLE , z , 1 )",
            "( MULTIPLY , -1 , 1 ) ( SQUARE , -1 , 3 ) ( VARIABLE , y , 4 )",
            "( PLUS , -1 , 3 ) ( CONSTANT , -3 , 6 ) ( VARIABLE , IV[max(x,y)] , 6 )",
            "( CONSTANT , -3 , 0 )",
            "GC1: IV[max(x,y)] = MAX ( x , y )",
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
        ["(IV[abs(x)]) + (y) == 3", "(IV[abs(x)]) == (abs(x))"],
        ["R0: IV[abs(x)] + y = 3", "GC0: IV[abs(x)] = ABS ( x )"],
    )

    """Mul is supported in expression tree, but not abs"""
    yield (
        "abs_in_mul",
        cp.Abs(x) * y + z == 3,
        ["((IV[abs(x)]) * (y)) + (z) == 3", "(IV[abs(x)]) == (abs(x))"],
        ["qc0: z + [ IV[abs(x)] * y ] = 3", "GC0: IV[abs(x)] = ABS ( x )"],
    )

    yield (
        "mul_in_abs",
        cp.Abs(x * y) + z == 3,
        [
            "(IV[abs(IV[(x) * (y)])]) + (z) == 3",
            "(IV[(x) * (y)]) == ((x) * (y))",
            "(IV[abs(IV[(x) * (y)])]) == (abs(IV[(x) * (y)]))",
        ],
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
        [
            "(z) * (BV[x == 2]) == 1",
            "sum(BV[x == -2], BV[x == -1], BV[x == 0], BV[x == 1], BV[x == 2]) == 1",
            "((sum([0, 1, 2, 3, 4] * [BV[x == -2], BV[x == -1], BV[x == 0], BV[x == 1], BV[x == 2]])) + -2) == (x)",
        ],
        None,
    )

    yield (
        "disjunction",
        p | q,
        ["(p) + (q) >= 1"],
        ["R0: p + q >= 1"],
    )

    yield (
        "two_disjunctions",
        (p | q) & (q | r),
        None,
        ["R0: p + q >= 1", "R1: q + r >= 1"],
    )

    yield (
        "disjunction_of_negs",
        (~p) | (~q),
        # TODO 1-p + 1-q >= 1 === -p-q>=-1 === p+q<=1
        ["(BV[~p]) + (BV[~q]) >= 1", "(~p) == (BV[~p])", "(~q) == (BV[~q])"],
        None,
    )

    yield (
        "conjunction_implies_comparison",
        ((x >= 2) & (y <= 10)).implies(z >= 2),
        ["(BV[x >= 2]) -> (z >= 2)", "(BV[x >= 2]) -> (x >= 2)", "(~BV[x >= 2]) -> (x <= 1)", "True"],
        None,
    )

    yield (
        "reified_not_eq",
        (p == (~q)) == p,
        [
            "True",
            "(p) -> ((~q) == (p))",
            "(~p) -> ((sum([1, -2] * [~q, BV0])) <= ((p) + -1))",
            "(~p) -> ((sum([1, -2] * [~q, BV0])) >= (sum(p, -2, 1)))",
            "True",
        ],
        None,
    )

    # yield (
    #     "maximum_root",
    #     1 == cp.Maximum([x, y]),
    #     None,
    #     ["GC0: C2 = MAX ( x , y )"],
    # )

    # yield (
    #     "maximum_bv_root",
    #     1 == cp.Maximum([p, q]),
    #     ["IV0 == 1", "(IV0) == (max(p,q))"],
    #     ["R0: IV0 = 1", "GC0: IV0 = MAX ( p , q )"],
    # )

    """If we find a general constraint already in the proper form of `y = f(x)`, we should not reify"""
    yield (
        "general_constraint_in_normal_form",
        x == cp.Maximum([y, z]),
        ["(x) == (max(y,z))"],
        ["GC0: x = MAX ( y , z )"],
    )

    # (x) * (pow(y,2)) <= 4
    # (x) * (pow(y,2)) - 4 <= 0
    # y <= 0, y = (x) * (pow(y,2)) - 4
    yield (
        "unnormalized_quad",
        (x * y) <= 3,
        ["(x) * (y) <= 3"],
        ["qc0: [ x * y ] <= 3"],
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
        (p | q) <= (x == 2) + (y**2) - 4,
        [
            "(BV[(p) or (q)]) <= (sum(BV[x == 2], pow(y,2), -4))",
            "(BV[(p) or (q)]) == ((p) or (q))",
            "sum(BV[x == -2], BV[x == -1], BV[x == 0], BV[x == 1], BV[x == 2]) == 1",
            "((sum([0, 1, 2, 3, 4] * [BV[x == -2], BV[x == -1], BV[x == 0], BV[x == 1], BV[x == 2]])) + -2) == (x)",
        ],
        None,
    )

    yield (
        "disjunction_of_equalities",
        (x == 1) | (y == 2),
        [
            "(BV[x == 1]) + (BV[y == 2]) >= 1",
            "sum(BV[x == -2], BV[x == -1], BV[x == 0], BV[x == 1], BV[x == 2]) == 1",
            "((sum([0, 1, 2, 3, 4] * [BV[x == -2], BV[x == -1], BV[x == 0], BV[x == 1], BV[x == 2]])) + -2) == (x)",
            "sum(BV[y == -2], BV[y == -1], BV[y == 0], BV[y == 1], BV[y == 2]) == 1",
            "((sum([0, 1, 2, 3, 4] * [BV[y == -2], BV[y == -1], BV[y == 0], BV[y == 1], BV[y == 2]])) + -2) == (y)",
        ],
        None,
    )

    yield (
        "disjunction_of_disequalities",
        (x != 1) | (y == 2),
        [
            "(BV[~BV[x == 1]]) + (BV[y == 2]) >= 1",
            "sum(BV[x == -2], BV[x == -1], BV[x == 0], BV[x == 1], BV[x == 2]) == 1",
            "((sum([0, 1, 2, 3, 4] * [BV[x == -2], BV[x == -1], BV[x == 0], BV[x == 1], BV[x == 2]])) + -2) == (x)",
            "sum(BV[y == -2], BV[y == -1], BV[y == 0], BV[y == 1], BV[y == 2]) == 1",
            "((sum([0, 1, 2, 3, 4] * [BV[y == -2], BV[y == -1], BV[y == 0], BV[y == 1], BV[y == 2]])) + -2) == (y)",
            "(~BV[x == 1]) == (BV[~BV[x == 1]])",
        ],
        None,
    )

    yield (
        "neg_disjunction",
        ~(p | q),
        ["True", "~p", "~q"],
        None,
    )

    yield (
        "conjunction",
        p & q,
        ["p", "q"],
        ["R0: p >= 1", "R1: q >= 1"],
    )

    yield (
        "conjunction_in_disjunction",
        (p | (q & r)),
        ["(p) + (BV[(q) and (r)]) >= 1", "(BV[(q) and (r)]) == ((q) and (r))"],
        ["R0: p + BV[(q)_and_(r)] >= 1", "GC0: BV[(q)_and_(r)] = AND ( q , r )"],
    )

    STAR = "*"
    a, b, c = cp.intvar(0, 2, shape=3, name="st")
    yield (
        "short_table",
        cp.ShortTable([a, b, c], [[0, 1, STAR], [1, STAR, 0], [STAR, 0, 1]]),
        [],
        None,
    )

    yield (
        "table",
        cp.Table([a, b], [[0, 1], [1, 2], [2, 0]]),
        None,
        None,
    )

    yield (
        "neg_table",
        cp.NegativeTable([a, b], [[0, 1], [1, 2], [2, 0]]),
        [],
        None,
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
    CPM_gurobi.verbose = True

    print("CONSTRAINT")
    print(constraint)
    transformed = [str(c) for c in CPM_gurobi().transform(constraint)]
    print("TF", "\n".join(transformed))
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
