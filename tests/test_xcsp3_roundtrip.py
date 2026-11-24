"""Semantic CPMpy -> XCSP3 -> CPMpy checks using the real parser.

Run: python -m pytest tests/test_xcsp3_roundtrip.py -q
Requires pycsp3 and a CPMpy solver supporting solveAll (e.g. OR-Tools).
"""
import json
import subprocess
import sys
import pytest

cp = pytest.importorskip("cpmpy")
pytest.importorskip("pycsp3")
pytest.importorskip("ortools")

from cpmpy.tools.xcsp3.exporter import write_xcsp3
from cpmpy.tools.xcsp3.parser import _parse_xcsp3, _load_xcsp3


def _import_model(path):
    return _load_xcsp3(_parse_xcsp3(path))


def _imported_variables(imported, variables):
    from cpmpy.transformations.get_variables import get_variables
    expressions = list(imported.constraints)
    if imported.has_objective():
        expressions.append(imported.objective_)
    by_name = {}
    for var in get_variables(expressions):
        assert var.name not in by_name, f"Duplicate imported variable name: {var.name}"
        by_name[var.name] = var
    missing = [v.name for v in variables if v.name not in by_name]
    assert not missing, f"Original variables absent from import: {missing}"
    return [by_name[v.name] for v in variables]


def _solutions(model, variables, limit=10000):
    seen = set()
    def record():
        seen.add(tuple(int(v.value()) for v in variables))
    count = model.solveAll(solver="ortools", display=record, solution_limit=limit)
    assert count < limit, "Solution limit reached; reduce variable domains"
    return seen


# Run the imported model in a fresh interpreter.  CPMpy's XCSP3 loader
# warns when variables have already been created in the current interpreter;
# merely filtering that warning would hide possible name aliasing.
_IMPORT_WORKER = r"""
import json
import sys
from cpmpy.tools.xcsp3.parser import _parse_xcsp3, _load_xcsp3
from cpmpy.transformations.get_variables import get_variables

path, operation, names_json = sys.argv[1:4]
export_path = sys.argv[4] if len(sys.argv) > 4 else None
names = json.loads(names_json)
model = _load_xcsp3(_parse_xcsp3(path))
expressions = list(model.constraints)
if model.has_objective():
    expressions.append(model.objective_)
by_name = {}
for var in get_variables(expressions):
    if var.name in by_name:
        raise AssertionError(f"Duplicate imported variable name: {var.name}")
    by_name[var.name] = var
missing = [name for name in names if name not in by_name]
if missing:
    raise AssertionError(f"Original variables absent from import: {missing}")
if operation in ("solutions", "solutions_export"):
    selected = [by_name[name] for name in names]
    seen = set()
    def record():
        seen.add(tuple(int(v.value()) for v in selected))
    limit = 10000
    count = model.solveAll(solver="ortools", display=record, solution_limit=limit)
    assert count < limit, "Solution limit reached; reduce variable domains"
    result = {"solutions": sorted(seen)}
elif operation in ("optimum", "optimum_export"):
    sat = bool(model.solve(solver="ortools"))
    result = {"sat": sat, "objective": model.objective_value() if sat else None,
              "is_min": model.objective_is_min}
else:
    raise ValueError(f"Unknown operation: {operation}")
if operation.endswith("_export"):
    from cpmpy.tools.xcsp3.exporter import write_xcsp3
    write_xcsp3(model, export_path)
print("XCSP3_ISOLATED_RESULT=" + json.dumps(result))
"""


def _isolated_import_result(path, operation, variables, export_path=None):
    """Import and solve in a clean Python process, preserving the original oracle."""
    names = [v if isinstance(v, str) else v.name for v in variables]
    assert len(names) == len(set(names)), "Duplicate original variable names"
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_WORKER, str(path), operation,
         json.dumps(names)] + ([str(export_path)] if export_path is not None else []),
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, (
        f"Isolated XCSP3 import failed for {path}:\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    marker = "XCSP3_ISOLATED_RESULT="
    lines = [line[len(marker):] for line in completed.stdout.splitlines()
             if line.startswith(marker)]
    assert len(lines) == 1, (
        f"Expected one isolated result for {path}, got {len(lines)}:\n"
        f"{completed.stdout}\n{completed.stderr}"
    )
    return json.loads(lines[0])



def _inverse_roundtrip_solutions(source, exported, names, expected):
    """Check both XCSP3 imports in separate interpreters, preserving the oracle."""
    first = _isolated_import_result(source, "solutions_export", names, exported)
    assert {tuple(v) for v in first["solutions"]} == expected
    second = _isolated_import_result(exported, "solutions", names)
    assert {tuple(v) for v in second["solutions"]} == expected


def _inverse_roundtrip_optimum(source, exported, names, expected, direction):
    first = _isolated_import_result(source, "optimum_export", names, exported)
    second = _isolated_import_result(exported, "optimum", names)
    for result in (first, second):
        assert result["is_min"] == (direction == "minimize")
        assert result["sat"]
        assert result["objective"] == expected

def _check_csp(tmp_path, name, model, variables):
    path = tmp_path / f"{name}.xml"
    write_xcsp3(model, path)
    before = _solutions(model, variables)
    after = {tuple(row) for row in
             _isolated_import_result(path, "solutions", variables)["solutions"]}
    assert before == after, (
        f"{name}: semantic mismatch\n"
        f"Only original: {sorted(before - after)[:20]}\n"
        f"Only imported: {sorted(after - before)[:20]}"
    )


def _check_optimum(tmp_path, name, model, variables):
    path = tmp_path / f"{name}.xml"
    write_xcsp3(model, path)
    original_sat = bool(model.solve(solver="ortools"))
    original_objective = model.objective_value() if original_sat else None
    imported = _isolated_import_result(path, "optimum", variables)
    assert original_sat == imported["sat"], f"{name}: different satisfiability"
    if original_sat:
        assert original_objective == imported["objective"], (
            f"{name}: optimum changed from {original_objective} "
            f"to {imported['objective']}"
        )


def test_roundtrip_arithmetic(tmp_path):
    x, y = cp.intvar(0, 3, name="x"), cp.intvar(0, 3, name="y")
    _check_csp(tmp_path, "arithmetic", cp.Model(x + y == 3, x < y), [x, y])


def test_roundtrip_boolean(tmp_path):
    a, b, c = [cp.boolvar(name=n) for n in ("a", "b", "c")]
    _check_csp(tmp_path, "boolean", cp.Model(a.implies(b), cp.Xor([a, b, c])), [a, b, c])


def test_roundtrip_alldifferent(tmp_path):
    x = [cp.intvar(0, 2, name=f"x{i}") for i in range(3)]
    _check_csp(tmp_path, "alldifferent", cp.Model(cp.AllDifferent(x), x[0] < x[2]), x)


def test_roundtrip_table(tmp_path):
    x, y = cp.intvar(0, 2, name="x"), cp.intvar(0, 2, name="y")
    _check_csp(tmp_path, "table", cp.Model(cp.Table([x, y], [(0, 1), (1, 2)])), [x, y])


def test_roundtrip_domain_holes(tmp_path):
    x = cp.intvar(0, 5, name="x")
    _check_csp(tmp_path, "domain_holes", cp.Model(cp.InDomain(x, [0, 2, 5])), [x])


def test_roundtrip_minimize(tmp_path):
    x, y = cp.intvar(0, 5, name="x"), cp.intvar(0, 5, name="y")
    model = cp.Model(x + y >= 5, x >= 1)
    model.minimize(2*x + y)
    _check_optimum(tmp_path, "minimize", model, [x, y])


def test_roundtrip_maximize(tmp_path):
    x, y = cp.intvar(0, 4, name="x"), cp.intvar(0, 4, name="y")
    model = cp.Model(x + y <= 5)
    model.maximize(x + 2*y)
    _check_optimum(tmp_path, "maximize", model, [x, y])

# Extended semantic regression cases. These compare complete projected solution
# sets, not just XML structure or one satisfying assignment.

@pytest.mark.parametrize("case", [
    "unsat", "negative_domains", "count", "among", "element",
    "minimum_maximum", "cardinality", "increasing", "decreasing",
    "all_equal", "negative_table", "short_table", "lex_less",
])
def test_roundtrip_extended_constraints(tmp_path, case):
    x = [cp.intvar(-1, 2, name=f"{case}_x{i}") for i in range(3)]
    if case == "unsat":
        model = cp.Model(x[0] == 0, x[0] == 1)
    elif case == "negative_domains":
        model = cp.Model(x[0] + x[1] == -1, x[2] < 1)
    elif case == "count":
        model = cp.Model(cp.Count(x, 1) == 2)
    elif case == "among":
        model = cp.Model(cp.Among(x, [-1, 1]) == 2)
    elif case == "element":
        # Keep the index domain within the list to avoid undefined indexing.
        i = cp.intvar(0, 2, name=f"{case}_idx")
        model = cp.Model(cp.Element(x, i) == 1)
        _check_csp(tmp_path, case, model, [*x, i])
        return
    elif case == "minimum_maximum":
        model = cp.Model(cp.Minimum(x) == -1, cp.Maximum(x) == 2)
    elif case == "cardinality":
        model = cp.Model(cp.GlobalCardinalityCount(x, [-1, 1], [1, 1], closed=False))
    elif case == "increasing":
        model = cp.Model(cp.Increasing(x))
    elif case == "decreasing":
        model = cp.Model(cp.Decreasing(x))
    elif case == "all_equal":
        model = cp.Model(cp.AllEqual(x))
    elif case == "negative_table":
        model = cp.Model(cp.NegativeTable(x[:2], [(-1, -1), (0, 0)]))
    elif case == "short_table":
        model = cp.Model(cp.ShortTable(x[:2], [(-1, '*'), ('*', 2)]))
    elif case == "lex_less":
        model = cp.Model(cp.LexLess(x[:2], x[1:]))
    selected = x[:1] if case == "unsat" else x[:2] if case in ("negative_table", "short_table") else x
    _check_csp(tmp_path, case, model, selected)


@pytest.mark.parametrize("case", ["no_overlap", "cumulative", "circuit"])
def test_roundtrip_scheduling_and_graph(tmp_path, case):
    if case == "no_overlap":
        starts = [cp.intvar(0, 3, name=f"{case}_s{i}") for i in range(2)]
        model = cp.Model(cp.NoOverlap(starts, [2, 2]))
        variables = starts
    elif case == "cumulative":
        starts = [cp.intvar(0, 3, name=f"{case}_s{i}") for i in range(2)]
        model = cp.Model(cp.Cumulative(starts, [2, 2], demand=[2, 2], capacity=3))
        variables = starts
    else:
        successors = [cp.intvar(0, 2, name=f"{case}_s{i}") for i in range(3)]
        model = cp.Model(cp.Circuit(successors))
        variables = successors
    _check_csp(tmp_path, case, model, variables)


@pytest.mark.parametrize("case", ["unsat_minimize", "negative_minimize", "multiple_optima"])
def test_roundtrip_extended_optimization(tmp_path, case):
    x, y = cp.intvar(-2, 3, name=f"{case}_x"), cp.intvar(-2, 3, name=f"{case}_y")
    if case == "unsat_minimize":
        model = cp.Model(x == 0, x == 1)
        model.minimize(x + y)
    elif case == "negative_minimize":
        model = cp.Model(x + y >= -1)
        model.minimize(2*x + y)
    else:
        model = cp.Model(x + y == 1)
        model.minimize(x + y)
    _check_optimum(tmp_path, case, model, [x, y])

# Additional semantic checks: compare the complete projected solution sets.
# These are intentionally not xfailed: an unsupported or incorrect conversion
# must be reported as a failing test.
@pytest.mark.parametrize("case", [
    "arithmetic_subtraction", "arithmetic_multiplication", "arithmetic_abs",
    "arithmetic_modulo", "boolean_equivalence", "boolean_disjunction",
    "all_different_except", "all_equal_except", "strict_increasing",
    "strict_decreasing", "lex_lesseq", "lex_chain_less",
    "lex_chain_lesseq", "nvalue", "nvalue_except", "precedence",
    "inverse", "if_then_else", "element_constant_array",
])
def test_roundtrip_additional_semantics(tmp_path, case):
    # Small finite domains make exhaustive solution-set comparisons affordable.
    two_var_cases = {"arithmetic_subtraction", "arithmetic_multiplication",
                     "arithmetic_abs", "arithmetic_modulo",
                     "if_then_else", "element_constant_array"}
    size = 2 if case in two_var_cases else 3
    x = [cp.intvar(0, 2, name=f"{case}_x{i}") for i in range(size)]
    a, b = x[:2]
    if case == "arithmetic_subtraction":
        model = cp.Model(a - b == 1)
    elif case == "arithmetic_multiplication":
        model = cp.Model(a * b == 2)
    elif case == "arithmetic_abs":
        model = cp.Model(abs(a - b) == 1)
    elif case == "arithmetic_modulo":
        model = cp.Model((a + b) % 2 == 1)
    elif case == "boolean_equivalence":
        p, q = cp.boolvar(name=f"{case}_p"), cp.boolvar(name=f"{case}_q")
        _check_csp(tmp_path, case, cp.Model(p == q), [p, q])
        return
    elif case == "boolean_disjunction":
        p, q = cp.boolvar(name=f"{case}_p"), cp.boolvar(name=f"{case}_q")
        _check_csp(tmp_path, case, cp.Model(p | q), [p, q])
        return
    elif case == "all_different_except":
        model = cp.Model(cp.AllDifferentExceptN(x, [0]))
    elif case == "all_equal_except":
        model = cp.Model(cp.AllEqualExceptN(x, [0]))
    elif case == "strict_increasing":
        model = cp.Model(cp.IncreasingStrict(x))
    elif case == "strict_decreasing":
        model = cp.Model(cp.DecreasingStrict(x))
    elif case == "lex_lesseq":
        model = cp.Model(cp.LexLessEq(x[:2], x[1:]))
    elif case == "lex_chain_less":
        model = cp.Model(cp.LexChainLess([x[:2], x[1:]]))
    elif case == "lex_chain_lesseq":
        model = cp.Model(cp.LexChainLessEq([x[:2], x[1:]]))
    elif case == "nvalue":
        model = cp.Model(cp.NValue(x) == 2)
    elif case == "nvalue_except":
        model = cp.Model(cp.NValueExcept(x, 0) == 1)
    elif case == "precedence":
        model = cp.Model(cp.Precedence(x, [0, 1, 2]))
    elif case == "inverse":
        y = [cp.intvar(0, 2, name=f"{case}_y{i}") for i in range(3)]
        _check_csp(tmp_path, case, cp.Model(cp.Inverse(x, y)), [*x, *y])
        return
    elif case == "if_then_else":
        p = cp.boolvar(name=f"{case}_p")
        model = cp.Model(cp.IfThenElse(p, a == 1, b == 2))
        _check_csp(tmp_path, case, model, [*x, p])
        return
    elif case == "element_constant_array":
        model = cp.Model(cp.Element([2, 0, 1], a) == b)
    else:
        raise AssertionError(f"Unknown roundtrip case: {case}")
    _check_csp(tmp_path, case, model, x)

# Further semantic roundtrips: automata, combined constraints and objectives.
@pytest.mark.parametrize("case", ["regular", "mdd", "regular_with_arithmetic",
                                  "mdd_with_boolean", "combined_globals"])
def test_roundtrip_advanced_semantics(tmp_path, case):
    if case == "regular":
        x = [cp.intvar(0, 1, name=f"{case}_x{i}") for i in range(3)]
        transitions = [("A", 0, "A"), ("A", 1, "B"), ("B", 0, "B")]
        model = cp.Model(cp.Regular(x, transitions, "A", ["B"]))
        variables = x
    elif case == "mdd":
        x = [cp.intvar(0, 1, name=f"{case}_x{i}") for i in range(2)]
        model = cp.Model(cp.MDD(x, [("A", 0, "B"), ("A", 1, "B"),
                                      ("B", 0, "C"), ("B", 1, "C")]))
        variables = x
    elif case == "regular_with_arithmetic":
        x = [cp.intvar(0, 1, name=f"{case}_x{i}") for i in range(3)]
        transitions = [("A", 0, "A"), ("A", 1, "B"), ("B", 0, "B")]
        model = cp.Model(cp.Regular(x, transitions, "A", ["B"]),
                         x[0] + x[1] + x[2] == 1)
        variables = x
    elif case == "mdd_with_boolean":
        x = [cp.intvar(0, 1, name=f"{case}_x{i}") for i in range(2)]
        b = cp.boolvar(name=f"{case}_b")
        model = cp.Model(cp.MDD(x, [("A", 0, "B"), ("A", 1, "B"),
                                      ("B", 0, "C"), ("B", 1, "C")]),
                         b.implies(x[0] == x[1]))
        variables = [*x, b]
    else:
        x = [cp.intvar(0, 2, name=f"{case}_x{i}") for i in range(3)]
        model = cp.Model(cp.AllDifferent(x), cp.Increasing(x),
                         x[0] + x[1] + x[2] == 3)
        variables = x
    _check_csp(tmp_path, case, model, variables)


@pytest.mark.parametrize("case", ["nested_minimum", "nested_maximum",
                                  "weighted_objective", "absolute_objective"])
def test_roundtrip_advanced_optimization(tmp_path, case):
    x = [cp.intvar(0, 3, name=f"{case}_x{i}") for i in range(2)]
    a, b = x
    model = cp.Model(a + b >= 2, a <= b + 1)
    if case == "nested_minimum":
        model.minimize((a - b) * (a - b) + a)
    elif case == "nested_maximum":
        model.maximize((a + 1) * (b + 1))
    elif case == "weighted_objective":
        model.minimize(3 * a + 2 * b)
    else:
        model.minimize(abs(a - b) + a)
    _check_optimum(tmp_path, case, model, x)

# Native XCSP3 globals whose CPMpy equivalents are decompositions rather than
# first-class public CPMpy constraints. Compare against an independent CPMpy
# reference model, including every assignment (not just satisfiability).
def _check_native_global(tmp_path, name, variables, global_name, arguments, reference):
    import xml.etree.ElementTree as ET
    from types import SimpleNamespace
    from cpmpy.tools.xcsp3 import exporter as writer

    path = tmp_path / f"{name}.xml"
    root = ET.Element("instance", format="XCSP3", type="CSP")
    declarations = ET.SubElement(root, "variables")
    for var in variables:
        ET.SubElement(declarations, "var", id=var.name).text = f"{var.lb}..{var.ub}"
    constraints = ET.SubElement(root, "constraints")
    writer._constraint(constraints, SimpleNamespace(name=global_name, args=arguments),
                       {id(var): var.name for var in variables})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    before = _solutions(reference, variables)
    after = {tuple(row) for row in
             _isolated_import_result(path, "solutions", variables)["solutions"]}
    assert before == after, (
        f"{name}: native XCSP3 semantic mismatch\n"
        f"Only reference: {sorted(before - after)[:20]}\n"
        f"Only imported: {sorted(after - before)[:20]}"
    )


@pytest.mark.parametrize("case", ["limits", "loads", "condition_le", "condition_eq"])
def test_roundtrip_native_binpacking(tmp_path, case):
    if case == "condition_eq":
        # Three items, three bins: each bin must receive exactly one item.
        x = [cp.intvar(0, 2, name=f"bp_{case}_{i}") for i in range(3)]
        weights = [2, 2, 2]
        loads = [sum(w * (v == bin_id) for v, w in zip(x, weights))
                 for bin_id in range(3)]
        kind, spec = "binpacking", ("eq", 2)
        model = cp.Model([load == 2 for load in loads])
    else:
        x = [cp.intvar(0, 1, name=f"bp_{case}_{i}") for i in range(3)]
        weights = [2, 3, 1]
        load0 = sum(w * (v == 0) for v, w in zip(x, weights))
        load1 = sum(w * (v == 1) for v, w in zip(x, weights))
        if case == "limits":
            kind, spec = "binpacking_limits", [3, 4]
            model = cp.Model(load0 <= 3, load1 <= 4)
        elif case == "loads":
            kind, spec = "binpacking_loads", [3, 3]
            model = cp.Model(load0 == 3, load1 == 3)
        else:
            kind, spec = "binpacking", ("le", 4)
            model = cp.Model(load0 <= 4, load1 <= 4)
    _check_native_global(tmp_path, f"binpacking_{case}", x, kind,
                         (x, weights, spec), model)


@pytest.mark.parametrize("case", ["feasible", "infeasible", "exact_capacity"])
def test_roundtrip_native_knapsack(tmp_path, case):
    x = [cp.boolvar(name=f"ks_{case}_{i}") for i in range(3)]
    weights, profits = [2, 3, 4], [3, 5, 6]
    weight = sum(w * v for w, v in zip(weights, x))
    profit = sum(p * v for p, v in zip(profits, x))
    if case == "feasible":
        wc, pc = ("le", 5), ("ge", 5)
        model = cp.Model(weight <= 5, profit >= 5)
    elif case == "infeasible":
        wc, pc = ("le", 2), ("ge", 12)
        model = cp.Model(weight <= 2, profit >= 12)
    else:
        wc, pc = ("eq", 5), ("ge", 8)
        model = cp.Model(weight == 5, profit >= 8)
    _check_native_global(tmp_path, f"knapsack_{case}", x, "knapsack",
                         (x, weights, wc, profits, pc), model)


@pytest.mark.parametrize("case", ["two_dimensions", "three_dimensions"])
def test_roundtrip_native_no_overlap_nd(tmp_path, case):
    dimensions = 2 if case == "two_dimensions" else 3
    origins = [[cp.intvar(0, 2, name=f"nd_{case}_{i}_{d}")
                for d in range(dimensions)] for i in range(2)]
    lengths = [[1] * dimensions, [2] + [1] * (dimensions - 1)]
    reference = cp.Model(cp.any([
        (origins[0][d] + lengths[0][d] <= origins[1][d]) |
        (origins[1][d] + lengths[1][d] <= origins[0][d])
        for d in range(dimensions)
    ]))
    flat = [v for row in origins for v in row]
    _check_native_global(tmp_path, f"no_overlap_{case}", flat, "no_overlap_nd",
                         (origins, lengths), reference)


@pytest.mark.parametrize("case", ["regular_unsat", "mdd_unsat", "regular_boundary", "mdd_boundary"])
def test_roundtrip_automata_edge_cases(tmp_path, case):
    x = [cp.intvar(0, 1, name=f"{case}_{i}") for i in range(2)]
    if case.startswith("regular"):
        transitions = [("s", 0, "s"), ("s", 1, "t"), ("t", 0, "t")]
        model = cp.Model(cp.Regular(x, transitions, "s", ["t"]))
    else:
        transitions = [("s", 0, "m"), ("m", 1, "t")]
        model = cp.Model(cp.MDD(x, transitions))
    if case.endswith("unsat"):
        model += (x[0] == 1)
        if case.startswith("regular"):
            model += (x[1] == 1)
    else:
        model += (x[0] == 0)
        model += (x[1] == 1)
    _check_csp(tmp_path, case, model, x)


@pytest.mark.parametrize("case", ["negative_weighted", "nested_max", "negative_nested_min"])
def test_roundtrip_objective_edge_cases(tmp_path, case):
    x, y = (cp.intvar(-2, 2, name=f"obj_{case}_{i}") for i in range(2))
    model = cp.Model(x + y >= -1, x - y <= 2)
    if case == "negative_weighted":
        model.minimize(-3*x + 2*y)
    elif case == "nested_max":
        model.maximize((x + 2) * (y + 2) - x)
    else:
        model.minimize(abs(x - y) - 2*x)
    _check_optimum(tmp_path, case, model, [x, y])

# Additional importer/exporter parity: compare complete solution sets.
@pytest.mark.parametrize('case', ['single_assignment', 'multiple_assignments'])
def test_roundtrip_native_instantiation(tmp_path, case):
    x = [cp.intvar(0, 2, name=f'inst_{case}_{i}') for i in range(3)]
    values = [1, 0, 2] if case == 'single_assignment' else [2, 2, 0]
    reference = cp.Model([v == k for v, k in zip(x, values)])
    _check_native_global(tmp_path, f'instantiation_{case}', x,
                         'instantiation', (x, values), reference)


@pytest.mark.parametrize('case', ['two_rows', 'three_rows'])
def test_roundtrip_native_alldifferent_lists(tmp_path, case):
    nrows = 2 if case == 'two_rows' else 3
    rows = [[cp.intvar(0, 1, name=f'adl_{case}_{i}_{j}')
             for j in range(2)] for i in range(nrows)]
    reference = cp.Model([
        cp.any([rows[i][j] != rows[k][j] for j in range(2)])
        for i in range(nrows) for k in range(i + 1, nrows)
    ])
    variables = [v for row in rows for v in row]
    _check_native_global(tmp_path, f'alldifferent_lists_{case}',
                         variables, 'alldifferent_lists', rows, reference)


@pytest.mark.parametrize('case', ['square', 'rectangular'])
def test_roundtrip_native_alldifferent_matrix(tmp_path, case):
    nrows, ncols = (2, 2) if case == 'square' else (2, 3)
    matrix = [[cp.intvar(0, 2, name=f'adm_{case}_{i}_{j}')
               for j in range(ncols)] for i in range(nrows)]
    reference = cp.Model(
        [cp.AllDifferent(row) for row in matrix] +
        [cp.AllDifferent([matrix[i][j] for i in range(nrows)])
         for j in range(ncols)]
    )
    variables = [v for row in matrix for v in row]
    _check_native_global(tmp_path, f'alldifferent_matrix_{case}',
                         variables, 'alldifferent_matrix', (matrix,), reference)


# Further importer/exporter semantic parity: exhaustive solution-set checks.
@pytest.mark.parametrize("case", ["strict", "nonstrict"])
def test_roundtrip_native_lex_matrix(tmp_path, case):
    matrix = [[cp.intvar(0, 2, name=f"lm_{case}_{r}_{c}")
               for c in range(2)] for r in range(2)]
    rows = [list(row) for row in matrix]
    cols = [[matrix[r][c] for r in range(2)] for c in range(2)]
    lex = cp.LexLess if case == "strict" else cp.LexLessEq
    reference = cp.Model(
        [lex(rows[i], rows[i + 1]) for i in range(len(rows) - 1)] +
        [lex(cols[i], cols[i + 1]) for i in range(len(cols) - 1)]
    )
    variables = [v for row in matrix for v in row]
    _check_native_global(tmp_path, f"lex_matrix_{case}", variables,
                         "lex_matrix", (matrix, "lt" if case == "strict" else "le"),
                         reference)


@pytest.mark.parametrize("case", ["variable_occurs", "fixed_occurs", "closed"])
def test_roundtrip_cardinality_more_cases(tmp_path, case):
    x = [cp.intvar(0, 2, name=f"gcc_{case}_{i}") for i in range(3)]
    if case == "variable_occurs":
        occurrences = [cp.intvar(0, 3, name=f"gcc_{case}_count_{i}")
                       for i in range(2)]
        model = cp.Model(cp.GlobalCardinalityCount(
            x, [0, 1], occurrences, closed=False))
        variables = x + occurrences
    elif case == "fixed_occurs":
        model = cp.Model(cp.GlobalCardinalityCount(
            x, [0, 1], [1, 1], closed=False))
        variables = x
    else:
        model = cp.Model(cp.GlobalCardinalityCount(
            x, [0, 1], [1, 2], closed=True))
        variables = x
    _check_csp(tmp_path, f"cardinality_{case}", model, variables)


@pytest.mark.parametrize("case", ["variable_lengths", "fixed_lengths"])
def test_roundtrip_native_no_overlap_lengths(tmp_path, case):
    starts = [cp.intvar(0, 3, name=f"nol_{case}_s{i}") for i in range(2)]
    if case == "variable_lengths":
        lengths = [cp.intvar(1, 2, name=f"nol_{case}_d{i}") for i in range(2)]
    elif case == "fixed_lengths":
        lengths = [1, 2]
    reference = cp.Model(
        (starts[0] + lengths[0] <= starts[1]) |
        (starts[1] + lengths[1] <= starts[0])
    )
    variables = starts + [v for v in lengths if not isinstance(v, int)]
    _check_native_global(tmp_path, f"no_overlap_lengths_{case}", variables,
                         "no_overlap", (starts, lengths), reference)

# Channel parity: native XCSP3 channel constraints against independent
# finite-domain inverse-permutation reference models.
@pytest.mark.parametrize("size", [3, 4])
def test_roundtrip_native_channel_two_lists(tmp_path, size):
    left = [cp.intvar(0, size - 1, name=f"ch{size}_a{i}") for i in range(size)]
    right = [cp.intvar(0, size - 1, name=f"ch{size}_b{i}") for i in range(size)]
    reference = cp.Model([
        (left[i] == j) == (right[j] == i)
        for i in range(size) for j in range(size)
    ])
    _check_native_global(tmp_path, f"channel_two_lists_{size}", left + right,
                         "inverse", (left, right), reference)


@pytest.mark.parametrize("size", [3, 4])
def test_roundtrip_native_channel_one_list(tmp_path, size):
    x = [cp.intvar(0, size - 1, name=f"ch1_{size}_{i}") for i in range(size)]
    reference = cp.Model([
        (x[i] == j) == (x[j] == i)
        for i in range(size) for j in range(size)
    ])
    _check_native_global(tmp_path, f"channel_one_list_{size}", x,
                         "inverse_one", (x,), reference)


@pytest.mark.parametrize("case", ["constant", "variable", "inequality"])
def test_roundtrip_native_element_matrix(tmp_path, case):
    matrix = [[cp.intvar(0, 2, name=f"em_{case}_{i}_{j}")
               for j in range(2)] for i in range(2)]
    row = cp.intvar(0, 1, name=f"em_{case}_row")
    col = cp.intvar(0, 1, name=f"em_{case}_col")
    target = cp.intvar(0, 2, name=f"em_{case}_target") if case == "variable" else 1
    op = "!=" if case == "inequality" else "=="
    reference = cp.Model(cp.any([
        (row == i) & (col == j) &
        ((matrix[i][j] != target) if case == "inequality" else (matrix[i][j] == target))
        for i in range(2) for j in range(2)
    ]))
    variables = [v for line in matrix for v in line] + [row, col]
    if case == "variable":
        variables.append(target)
    _check_native_global(tmp_path, f"element_matrix_{case}", variables,
                         "element_matrix", (matrix, row, col, op, target), reference)


@pytest.mark.parametrize("operator", ["==", "!=", "<", "<=", ">", ">="])
def test_roundtrip_native_element_matrix_all_comparisons(tmp_path, operator):
    """Compare all assignments for every native matrix-element comparison."""
    matrix = [[cp.intvar(0, 2, name=f"emop_{operator.replace('=', 'e').replace('<', 'l').replace('>', 'g').replace('!', 'n')}_{r}_{c}")
               for c in range(2)] for r in range(2)]
    row = cp.intvar(0, 1, name="emop_row")
    col = cp.intvar(0, 1, name="emop_col")
    target = 1
    compare = {
        "==": lambda a: a == target,
        "!=": lambda a: a != target,
        "<": lambda a: a < target,
        "<=": lambda a: a <= target,
        ">": lambda a: a > target,
        ">=": lambda a: a >= target,
    }[operator]
    reference = cp.Model(cp.any([
        (row == r) & (col == c) & compare(matrix[r][c])
        for r in range(2) for c in range(2)
    ]))
    variables = [v for line in matrix for v in line] + [row, col]
    _check_native_global(tmp_path, "element_matrix_operator_" + operator.replace("=", "e").replace("<", "l").replace(">", "g").replace("!", "n"),
                         variables, "element_matrix", (matrix, row, col, operator, target), reference)


@pytest.mark.parametrize("shape", [(2, 3), (3, 2)])
def test_roundtrip_native_element_matrix_rectangular(tmp_path, shape):
    """Exercise non-square matrices and both index-domain dimensions."""
    rows, cols = shape
    matrix = [[cp.intvar(0, 2, name=f"emrect_{rows}_{cols}_{r}_{c}")
               for c in range(cols)] for r in range(rows)]
    row = cp.intvar(0, rows - 1, name=f"emrect_{rows}_{cols}_row")
    col = cp.intvar(0, cols - 1, name=f"emrect_{rows}_{cols}_col")
    reference = cp.Model(cp.any([
        (row == r) & (col == c) & (matrix[r][c] == 1)
        for r in range(rows) for c in range(cols)
    ]))
    variables = [v for line in matrix for v in line] + [row, col]
    _check_native_global(tmp_path, f"element_matrix_rectangular_{rows}_{cols}",
                         variables, "element_matrix", (matrix, row, col, "==", 1), reference)

# Precedence parity: every occurrence of a later value must have an earlier
# occurrence of its predecessor. Values not in the ordered list are unrestricted.
@pytest.mark.parametrize("size,domain,values", [
    (2, 2, (0, 1)),
    (3, 3, (0, 1, 2)),
    (4, 3, (0, 1, 2)),
    (3, 4, (1, 3)),
    (4, 4, (1, 3)),
])
def test_roundtrip_native_precedence(tmp_path, size, domain, values):
    x = [cp.intvar(0, domain, name=f"prec_{size}_{domain}_{'_'.join(map(str, values))}_{i}")
         for i in range(size)]
    conditions = []
    for previous, following in zip(values, values[1:]):
        conditions.append(x[0] != following)
        for j in range(1, size):
            conditions.append((x[j] != following) |
                              cp.any([x[k] == previous for k in range(j)]))
    reference = cp.Model(conditions)
    _check_native_global(tmp_path,
                         f"precedence_{size}_{domain}_{'_'.join(map(str, values))}",
                         x, "precedence", (x, list(values)), reference)


@pytest.mark.parametrize("shape", [(2, 3), (3, 2)])
@pytest.mark.parametrize("operator", ["==", "!=", "<", "<=", ">", ">="])
def test_roundtrip_native_element_matrix_rectangular_comparisons(tmp_path, shape, operator):
    """Exhaustively compare rectangular element semantics for all six operators."""
    rows, cols = shape
    tag = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[operator]
    matrix = [[cp.intvar(0, 1, name=f"emrc_{rows}_{cols}_{tag}_{r}_{c}")
               for c in range(cols)] for r in range(rows)]
    row = cp.intvar(0, rows - 1, name=f"emrc_{rows}_{cols}_{tag}_row")
    col = cp.intvar(0, cols - 1, name=f"emrc_{rows}_{cols}_{tag}_col")
    comparisons = {
        "==": lambda value: value == 1,
        "!=": lambda value: value != 1,
        "<": lambda value: value < 1,
        "<=": lambda value: value <= 1,
        ">": lambda value: value > 1,
        ">=": lambda value: value >= 1,
    }
    reference = cp.Model(cp.any([
        (row == r) & (col == c) & comparisons[operator](matrix[r][c])
        for r in range(rows) for c in range(cols)
    ]))
    variables = [v for line in matrix for v in line] + [row, col]
    _check_native_global(tmp_path, f"element_matrix_rect_{rows}_{cols}_{tag}",
                         variables, "element_matrix", (matrix, row, col, operator, 1), reference)


@pytest.mark.parametrize("shape", [(2, 3), (3, 2)])
def test_roundtrip_native_element_matrix_rectangular_variable_target(tmp_path, shape):
    """Check the rectangular fallback when the comparison target is a variable."""
    rows, cols = shape
    matrix = [[cp.intvar(0, 1, name=f"emrt_{rows}_{cols}_{r}_{c}")
               for c in range(cols)] for r in range(rows)]
    row = cp.intvar(0, rows - 1, name=f"emrt_{rows}_{cols}_row")
    col = cp.intvar(0, cols - 1, name=f"emrt_{rows}_{cols}_col")
    target = cp.intvar(0, 1, name=f"emrt_{rows}_{cols}_target")
    reference = cp.Model(cp.any([
        (row == r) & (col == c) & (matrix[r][c] == target)
        for r in range(rows) for c in range(cols)
    ]))
    variables = [v for line in matrix for v in line] + [row, col, target]
    _check_native_global(tmp_path, f"element_matrix_rect_target_{rows}_{cols}",
                         variables, "element_matrix", (matrix, row, col, "==", target), reference)


@pytest.mark.parametrize("operator", ["==", "!=", "<", "<=", ">", ">="])
def test_roundtrip_native_element_matrix_square_variable_comparisons(tmp_path, operator):
    """Check all six square-matrix comparisons against a variable target."""
    tag = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[operator]
    matrix = [[cp.intvar(0, 2, name=f"emsv_{tag}_{r}_{c}")
               for c in range(2)] for r in range(2)]
    row = cp.intvar(0, 1, name=f"emsv_{tag}_row")
    col = cp.intvar(0, 1, name=f"emsv_{tag}_col")
    target = cp.intvar(0, 2, name=f"emsv_{tag}_target")
    comparisons = {
        "==": lambda value: value == target,
        "!=": lambda value: value != target,
        "<": lambda value: value < target,
        "<=": lambda value: value <= target,
        ">": lambda value: value > target,
        ">=": lambda value: value >= target,
    }
    reference = cp.Model(cp.any([
        (row == r) & (col == c) & comparisons[operator](matrix[r][c])
        for r in range(2) for c in range(2)
    ]))
    variables = [v for line in matrix for v in line] + [row, col, target]
    _check_native_global(tmp_path, f"element_matrix_square_variable_{tag}",
                         variables, "element_matrix", (matrix, row, col, operator, target), reference)

# Independent automata/circuit parity: the reference is constructed from
# accepted words/permutations, not from the same global constraint.
from itertools import product, permutations


def _word_reference(variables, accepted):
    """Explicit finite-language semantics, independent of Regular/MDD."""
    words = [cp.all([var == value for var, value in zip(variables, word)])
             for word in accepted]
    return cp.Model(cp.any(words)) if words else cp.Model(variables[0] < 0)


@pytest.mark.parametrize("length", [2, 3, 4, 5])
@pytest.mark.parametrize("language", [
    "ends_one", "contains_one", "no_adjacent_ones", "even_parity",
    "starts_zero", "at_most_one_one",
])
def test_roundtrip_regular_independent_language(tmp_path, length, language):
    """Compare a deterministic automaton with an independent word oracle."""
    x = [cp.intvar(0, 1, name=f"rl_{language}_{length}_{i}")
         for i in range(length)]
    if language == "ends_one":
        transitions = [(s, v, "one" if v else "zero")
                       for s in ("zero", "one") for v in (0, 1)]
        start, final = "zero", ["one"]
        accepts = lambda w: w[-1] == 1
    elif language == "contains_one":
        transitions = [("none", 0, "none"), ("none", 1, "seen"),
                       ("seen", 0, "seen"), ("seen", 1, "seen")]
        start, final = "none", ["seen"]
        accepts = lambda w: 1 in w
    elif language == "no_adjacent_ones":
        transitions = [("zero", 0, "zero"), ("zero", 1, "one"),
                       ("one", 0, "zero")]
        start, final = "zero", ["zero", "one"]
        accepts = lambda w: all(a + b < 2 for a, b in zip(w, w[1:]))
    elif language == "even_parity":
        transitions = [("even", 0, "even"), ("even", 1, "odd"),
                       ("odd", 0, "odd"), ("odd", 1, "even")]
        start, final = "even", ["even"]
        accepts = lambda w: sum(w) % 2 == 0
    elif language == "starts_zero":
        transitions = [("start", 0, "ok"), ("ok", 0, "ok"),
                       ("ok", 1, "ok")]
        start, final = "start", ["ok"]
        accepts = lambda w: w[0] == 0
    else:
        transitions = [("none", 0, "none"), ("none", 1, "one"),
                       ("one", 0, "one")]
        start, final = "none", ["none", "one"]
        accepts = lambda w: sum(w) <= 1
    accepted = [w for w in product((0, 1), repeat=length) if accepts(w)]
    reference = _word_reference(x, accepted)
    _check_native_global(tmp_path, f"regular_{language}_{length}", x,
                         "regular", (x, transitions, start, final), reference)


@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("language", [
    "all_zero", "exactly_one_one", "first_equals_last",
    "nondecreasing", "alternating", "sum_even",
])
def test_roundtrip_mdd_independent_language(tmp_path, length, language):
    """Layered MDD from a prefix trie, compared with explicit accepted words."""
    x = [cp.intvar(0, 1, name=f"md_{language}_{length}_{i}")
         for i in range(length)]
    predicates = {
        "all_zero": lambda w: all(v == 0 for v in w),
        "exactly_one_one": lambda w: sum(w) == 1,
        "first_equals_last": lambda w: w[0] == w[-1],
        "nondecreasing": lambda w: list(w) == sorted(w),
        "alternating": lambda w: all(a != b for a, b in zip(w, w[1:])),
        "sum_even": lambda w: sum(w) % 2 == 0,
    }
    accepted = [w for w in product((0, 1), repeat=length)
                if predicates[language](w)]
    # A unique terminal state is shared by all accepted words.
    edges = set()
    for word in accepted:
        for i, value in enumerate(word):
            prefix = word[:i]
            src = "root" if i == 0 else "p" + "".join(map(str, prefix))
            dst = "terminal" if i == length - 1 else "p" + "".join(map(str, word[:i + 1]))
            edges.add((src, value, dst))
    transitions = sorted(edges, key=lambda edge: (edge[0] != "root", edge))
    reference = _word_reference(x, accepted)
    # MDD's transition mapping is constructed by the CPMpy global itself.
    _check_csp(tmp_path, f"mdd_{language}_{length}",
               cp.Model(cp.MDD(x, transitions, start="root")), x)
    # Also independently check its accepted language, not just a self-roundtrip.
    assert _solutions(cp.Model(cp.MDD(x, transitions, start="root")), x) == _solutions(reference, x)


@pytest.mark.parametrize("size", [3, 4, 5])
def test_roundtrip_circuit_independent_permutations(tmp_path, size):
    """Hamiltonian successor cycles checked against enumerated permutations."""
    x = [cp.intvar(0, size - 1, name=f"cycle_{size}_{i}") for i in range(size)]
    valid = []
    for perm in permutations(range(size)):
        visited = {0}
        node = perm[0]
        while node not in visited:
            visited.add(node)
            node = perm[node]
        if node == 0 and len(visited) == size:
            valid.append(perm)
    reference = _word_reference(x, valid)
    _check_native_global(tmp_path, f"circuit_independent_{size}", x,
                         "circuit", tuple(x), reference)


# Additional independent ternary-language checks.  The trie is generated from
# an explicit truth predicate, while the oracle is an ordinary CPMpy disjunction.
def _ternary_language_words(length, language):
    def accepts(w):
        if language == "at_least_two_distinct":
            return len(set(w)) >= 2
        if language == "nondecreasing":
            return all(a <= b for a, b in zip(w, w[1:]))
        if language == "nonincreasing":
            return all(a >= b for a, b in zip(w, w[1:]))
        if language == "exactly_one_two":
            return w.count(2) == 1
        if language == "no_adjacent_equal":
            return all(a != b for a, b in zip(w, w[1:]))
        if language == "sum_divisible_three":
            return sum(w) % 3 == 0
        if language == "first_less_last":
            return w[0] < w[-1]
        if language == "contains_zero_and_two":
            return 0 in w and 2 in w
        if language == "at_most_one_zero":
            return w.count(0) <= 1
        if language == "palindrome":
            return w == w[::-1]
        raise ValueError(language)
    return [w for w in product((0, 1, 2), repeat=length) if accepts(w)]


def _ternary_trie(accepted, length):
    edges = set()
    for word in accepted:
        for index, value in enumerate(word):
            prefix = word[:index]
            source = "root" if index == 0 else "p" + "".join(map(str, prefix))
            target = "terminal" if index == length - 1 else "p" + "".join(map(str, word[:index + 1]))
            edges.add((source, value, target))
    assert edges and any(source == "root" for source, _, _ in edges)
    return sorted(edges, key=lambda edge: (edge[0] != "root", edge))


@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("language", [
    "at_least_two_distinct", "nondecreasing", "nonincreasing", "exactly_one_two",
    "no_adjacent_equal", "sum_divisible_three", "first_less_last",
    "contains_zero_and_two", "at_most_one_zero", "palindrome",
])
def test_roundtrip_regular_ternary_independent(tmp_path, length, language):
    """Ternary deterministic trie automata checked against explicit words."""
    variables = [cp.intvar(0, 2, name=f"rt_{language}_{length}_{i}")
                 for i in range(length)]
    accepted = _ternary_language_words(length, language)
    transitions = _ternary_trie(accepted, length)
    reference = _word_reference(variables, accepted)
    _check_native_global(tmp_path, f"regular_ternary_{language}_{length}",
                         variables, "regular",
                         (variables, transitions, "root", ["terminal"]), reference)


@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("language", [
    "at_least_two_distinct", "nondecreasing", "nonincreasing", "exactly_one_two",
    "no_adjacent_equal", "sum_divisible_three", "first_less_last",
    "contains_zero_and_two", "at_most_one_zero", "palindrome",
])
def test_roundtrip_mdd_ternary_independent(tmp_path, length, language):
    """Ternary layered MDD, with an independent accepted-word oracle."""
    variables = [cp.intvar(0, 2, name=f"mt_{language}_{length}_{i}")
                 for i in range(length)]
    accepted = _ternary_language_words(length, language)
    transitions = _ternary_trie(accepted, length)
    original = cp.Model(cp.MDD(variables, transitions, start="root"))
    reference = _word_reference(variables, accepted)
    assert _solutions(original, variables) == _solutions(reference, variables)
    _check_csp(tmp_path, f"mdd_ternary_{language}_{length}", original, variables)

# Cross-family compositions with a Python truth-table oracle.  These test
# interactions between globals and arithmetic rather than isolated globals.
@pytest.mark.parametrize("domain", [2, 3, 4])
@pytest.mark.parametrize("scenario", [
    "distinct_sum", "distinct_order", "equal_sum", "increasing_bound",
    "decreasing_bound", "count_sum", "among_sum", "minmax_sum",
    "negative_table_sum", "table_sum", "lex_sum", "element_sum",
])
def test_roundtrip_composed_globals_independent(tmp_path, domain, scenario):
    x = [cp.intvar(0, domain - 1, name=f"mix_{scenario}_{domain}_{i}")
         for i in range(3)]
    if scenario == "element_sum":
        x[0] = cp.intvar(0, 2, name=f"mix_{scenario}_{domain}_0")
    a, b, c = x
    bound = domain
    if scenario == "distinct_sum":
        constraints = [cp.AllDifferent(x), a + b + c == bound]
        accepts = lambda w: len(set(w)) == 3 and sum(w) == bound
    elif scenario == "distinct_order":
        constraints = [cp.AllDifferent(x), a < c]
        accepts = lambda w: len(set(w)) == 3 and w[0] < w[2]
    elif scenario == "equal_sum":
        constraints = [cp.AllEqual(x[:2]), a + b + c == bound]
        accepts = lambda w: w[0] == w[1] and sum(w) == bound
    elif scenario == "increasing_bound":
        constraints = [cp.Increasing(x), a + c >= bound - 1]
        accepts = lambda w: w[0] <= w[1] <= w[2] and w[0] + w[2] >= bound - 1
    elif scenario == "decreasing_bound":
        constraints = [cp.Decreasing(x), a + c <= bound]
        accepts = lambda w: w[0] >= w[1] >= w[2] and w[0] + w[2] <= bound
    elif scenario == "count_sum":
        constraints = [cp.Count(x, 1) == 1, a + b + c >= 2]
        accepts = lambda w: w.count(1) == 1 and sum(w) >= 2
    elif scenario == "among_sum":
        constraints = [cp.Among(x, [0, 1]) == 2, a + c >= 1]
        accepts = lambda w: sum(v in (0, 1) for v in w) == 2 and w[0] + w[2] >= 1
    elif scenario == "minmax_sum":
        constraints = [cp.Minimum(x) == 0, cp.Maximum(x) == domain - 1, b <= c]
        accepts = lambda w: min(w) == 0 and max(w) == domain - 1 and w[1] <= w[2]
    elif scenario == "negative_table_sum":
        constraints = [cp.NegativeTable(x[:2], [(0, 0), (1, 1)]), a + c == bound - 1]
        accepts = lambda w: (w[0], w[1]) not in ((0, 0), (1, 1)) and w[0] + w[2] == bound - 1
    elif scenario == "table_sum":
        allowed = [(i, (i + 1) % domain) for i in range(domain)]
        constraints = [cp.Table(x[:2], allowed), a + c <= bound]
        accepts = lambda w: (w[0], w[1]) in allowed and w[0] + w[2] <= bound
    elif scenario == "lex_sum":
        constraints = [cp.LexLessEq(x[:2], x[1:]), a + c == bound - 1]
        accepts = lambda w: (w[0], w[1]) <= (w[1], w[2]) and w[0] + w[2] == bound - 1
    else:
        constraints = [cp.Element([0, domain - 1, 1], a) == b, b + c == domain - 1]
        accepts = lambda w: (0, domain - 1, 1)[w[0]] == w[1] and w[1] + w[2] == domain - 1
    expected = {w for w in product(*(range(v.lb, v.ub + 1) for v in x)) if accepts(w)}
    model = cp.Model(*constraints)
    assert _solutions(model, x) == expected, f"Incorrect original oracle for {scenario}/{domain}"
    _check_csp(tmp_path, f"mix_{scenario}_{domain}", model, x)


# Optimization compositions: compare optimum with an independent enumeration,
# not only with the round-tripped CPMpy model.
@pytest.mark.parametrize("domain", [3, 4, 5])
@pytest.mark.parametrize("objective", [
    "linear_min", "linear_max", "abs_min", "abs_max",
    "product_min", "product_max", "quadratic_min", "quadratic_max",
])
def test_roundtrip_optimization_independent(tmp_path, domain, objective):
    a = cp.intvar(-1, domain - 2, name=f"opt_{objective}_{domain}_a")
    b = cp.intvar(-1, domain - 2, name=f"opt_{objective}_{domain}_b")
    variables = [a, b]
    model = cp.Model(a + b >= 0, a <= b + 1)
    funcs = {
        "linear_min": lambda x, y: 3 * x - 2 * y,
        "linear_max": lambda x, y: 3 * x - 2 * y,
        "abs_min": lambda x, y: abs(x - y) + x,
        "abs_max": lambda x, y: abs(x - y) + x,
        "product_min": lambda x, y: (x + 2) * (y + 2),
        "product_max": lambda x, y: (x + 2) * (y + 2),
        "quadratic_min": lambda x, y: (x - y) * (x - y) + y,
        "quadratic_max": lambda x, y: (x - y) * (x - y) + y,
    }
    expressions = {
        "linear": 3 * a - 2 * b,
        "abs": abs(a - b) + a,
        "product": (a + 2) * (b + 2),
        "quadratic": (a - b) * (a - b) + b,
    }
    expression = expressions[objective.rsplit("_", 1)[0]]
    feasible = [(x, y) for x, y in product(range(-1, domain - 1), repeat=2)
                if x + y >= 0 and x <= y + 1]
    assert feasible
    values = [funcs[objective](x, y) for x, y in feasible]
    expected = min(values) if objective.endswith("min") else max(values)
    if objective.endswith("min"):
        model.minimize(expression)
    else:
        model.maximize(expression)
    assert model.solve(solver="ortools")
    assert model.objective_value() == expected
    _check_optimum(tmp_path, f"opt_{objective}_{domain}", model, variables)

# Deterministic property-based regression: the oracle is plain Python and does
# not use the exporter, parser, CPMpy expression evaluation, or a solver.
import random


@pytest.mark.parametrize("seed", range(96))
def test_roundtrip_generated_linear_systems(tmp_path, seed):
    rng = random.Random(91731 + seed)
    lower = rng.choice((-2, -1, 0))
    upper = lower + rng.choice((2, 3, 4))
    variables = [cp.intvar(lower, upper, name=f"gen_{seed}_{i}") for i in range(3)]
    specifications = []
    constraints = []
    for _ in range(3):
        coefficients = tuple(rng.choice((-3, -2, -1, 0, 1, 2, 3)) for _ in range(3))
        if not any(coefficients):
            coefficients = (1, 0, -1)
        rhs = rng.randint(-5, 5)
        operator = rng.choice(("==", "!=", "<=", ">=", "<", ">"))
        lhs = sum(coef * var for coef, var in zip(coefficients, variables))
        constraints.append({
            "==": lambda: lhs == rhs,
            "!=": lambda: lhs != rhs,
            "<=": lambda: lhs <= rhs,
            ">=": lambda: lhs >= rhs,
            "<": lambda: lhs < rhs,
            ">": lambda: lhs > rhs,
        }[operator]())
        specifications.append((coefficients, operator, rhs))

    def accepts(values):
        for coefficients, operator, rhs in specifications:
            lhs = sum(coef * value for coef, value in zip(coefficients, values))
            if not {
                "==": lhs == rhs, "!=": lhs != rhs,
                "<=": lhs <= rhs, ">=": lhs >= rhs,
                "<": lhs < rhs, ">": lhs > rhs,
            }[operator]:
                return False
        return True

    expected = {values for values in product(range(lower, upper + 1), repeat=3)
                if accepts(values)}
    original = cp.Model(*constraints)
    assert _solutions(original, variables) == expected, f"Original mismatch, seed={seed}"
    _check_csp(tmp_path, f"generated_linear_{seed}", original, variables)


@pytest.mark.parametrize("size", [4, 5])
@pytest.mark.parametrize("variant", ["plain", "first_zero", "first_one", "ordered"])
def test_roundtrip_nqueens_independent(tmp_path, size, variant):
    queens = [cp.intvar(0, size - 1, name=f"nq_{size}_{variant}_{i}")
              for i in range(size)]
    constraints = [cp.AllDifferent(queens)]
    for i in range(size):
        for j in range(i + 1, size):
            constraints.extend((queens[i] - queens[j] != j - i,
                                queens[j] - queens[i] != j - i))
    if variant == "first_zero":
        constraints.append(queens[0] == 0)
    elif variant == "first_one":
        constraints.append(queens[0] == 1)
    elif variant == "ordered":
        constraints.append(queens[0] < queens[-1])

    def accepts(values):
        if len(set(values)) != size:
            return False
        if any(abs(values[i] - values[j]) == j - i
               for i in range(size) for j in range(i + 1, size)):
            return False
        return (variant == "plain" or
                (variant == "first_zero" and values[0] == 0) or
                (variant == "first_one" and values[0] == 1) or
                (variant == "ordered" and values[0] < values[-1]))

    expected = {values for values in product(range(size), repeat=size) if accepts(values)}
    original = cp.Model(*constraints)
    assert _solutions(original, queens) == expected
    _check_csp(tmp_path, f"nqueens_{size}_{variant}", original, queens)


@pytest.mark.parametrize("horizon", [2, 3, 4, 5])
@pytest.mark.parametrize("durations", [(1, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("kind", ["no_overlap", "cumulative"])
def test_roundtrip_scheduling_independent(tmp_path, horizon, durations, kind):
    starts = [cp.intvar(0, horizon, name=f"sched_{kind}_{horizon}_{durations[0]}_{durations[1]}_{i}")
              for i in range(2)]
    if kind == "no_overlap":
        original = cp.Model(cp.NoOverlap(starts, list(durations)))
    else:
        original = cp.Model(cp.Cumulative(starts, list(durations),
                                          demand=[2, 2], capacity=3))
    expected = {(a, b) for a, b in product(range(horizon + 1), repeat=2)
                if a + durations[0] <= b or b + durations[1] <= a}
    assert _solutions(original, starts) == expected
    _check_csp(tmp_path, f"sched_{kind}_{horizon}_{durations}", original, starts)


@pytest.mark.parametrize("seed", range(24))
def test_roundtrip_generated_optimization_independent(tmp_path, seed):
    rng = random.Random(712011 + seed)
    coefficients = tuple(rng.choice((-3, -2, -1, 1, 2, 3)) for _ in range(3))
    bound = rng.randint(-2, 4)
    direction = rng.choice(("min", "max"))
    variables = [cp.intvar(-1, 3, name=f"genopt_{seed}_{i}") for i in range(3)]
    a, b, c = variables
    original = cp.Model(a + b + c >= bound, a <= b + 1)
    objective = sum(coef * var for coef, var in zip(coefficients, variables))
    if direction == "min":
        original.minimize(objective)
    else:
        original.maximize(objective)
    feasible = [values for values in product(range(-1, 4), repeat=3)
                if sum(values) >= bound and values[0] <= values[1] + 1]
    expected = [sum(coef * value for coef, value in zip(coefficients, values))
                for values in feasible]
    assert bool(original.solve(solver="ortools")) == bool(expected)
    if expected:
        assert original.objective_value() == (min(expected) if direction == "min" else max(expected))
    _check_optimum(tmp_path, f"generated_opt_{seed}", original, variables)

# Reverse-direction evidence: start with independently authored native XCSP3,
# import it, export the imported CPMpy model, and import the exported XML again.
# Every projected solution is checked against a plain Python oracle.
@pytest.mark.parametrize("domain", [2, 3, 4])
@pytest.mark.parametrize("case", [
    "intension_sum", "intension_implication", "intension_distance",
    "alldifferent", "allequal", "ordered", "extension_supports",
    "extension_conflicts", "sum_condition", "count_condition",
])
def test_native_xcsp3_inverse_exporter_oracle(tmp_path, domain, case):
    import xml.etree.ElementTree as ET
    from itertools import product

    root = ET.Element("instance", {"format": "XCSP3", "type": "CSP"})
    declarations = ET.SubElement(root, "variables")
    for name in ("rev_a", "rev_b", "rev_c"):
        ET.SubElement(declarations, "var", {"id": name}).text = f"0..{domain}"
    constraints = ET.SubElement(root, "constraints")
    if case == "intension_sum":
        ET.SubElement(constraints, "intension").text = "eq(add(rev_a,rev_b),rev_c)"
        accepts = lambda a, b, c: a + b == c
    elif case == "intension_implication":
        ET.SubElement(constraints, "intension").text = "imp(eq(rev_a,0),lt(rev_b,rev_c))"
        accepts = lambda a, b, c: a != 0 or b < c
    elif case == "intension_distance":
        ET.SubElement(constraints, "intension").text = "eq(dist(rev_a,rev_b),rev_c)"
        accepts = lambda a, b, c: abs(a-b) == c
    elif case in {"alldifferent", "allequal"}:
        elt = ET.SubElement(constraints, "allDifferent" if case == "alldifferent" else "allEqual")
        ET.SubElement(elt, "list").text = "rev_a rev_b rev_c"
        accepts = ((lambda a, b, c: len({a, b, c}) == 3) if case == "alldifferent"
                   else (lambda a, b, c: a == b == c))
    elif case == "ordered":
        elt = ET.SubElement(constraints, "ordered")
        ET.SubElement(elt, "list").text = "rev_a rev_b rev_c"
        ET.SubElement(elt, "operator").text = "le"
        accepts = lambda a, b, c: a <= b <= c
    elif case in {"extension_supports", "extension_conflicts"}:
        elt = ET.SubElement(constraints, "extension")
        ET.SubElement(elt, "list").text = "rev_a rev_b rev_c"
        forbidden = {(0, 0, 0), (1, 1, 1)}
        ET.SubElement(elt, "supports" if case == "extension_supports" else "conflicts").text = "(0,0,0) (1,1,1)"
        accepts = ((lambda a, b, c: (a,b,c) in forbidden) if case == "extension_supports"
                   else (lambda a, b, c: (a,b,c) not in forbidden))
    elif case == "sum_condition":
        elt = ET.SubElement(constraints, "sum")
        ET.SubElement(elt, "list").text = "rev_a rev_b rev_c"
        ET.SubElement(elt, "condition").text = f"(le,{domain})"
        accepts = lambda a, b, c: a+b+c <= domain
    else:
        elt = ET.SubElement(constraints, "count")
        ET.SubElement(elt, "list").text = "rev_a rev_b rev_c"
        ET.SubElement(elt, "values").text = "1"
        ET.SubElement(elt, "condition").text = "(eq,2)"
        accepts = lambda a, b, c: (a==1)+(b==1)+(c==1) == 2

    source = tmp_path / f"native_inverse_{case}_{domain}.xml"
    ET.ElementTree(root).write(source, encoding="utf-8", xml_declaration=True)
    expected = {t for t in product(range(domain+1), repeat=3) if accepts(*t)}
    names = ("rev_a", "rev_b", "rev_c")
    exported = tmp_path / f"reexport_{case}_{domain}.xml"
    _inverse_roundtrip_solutions(source, exported, names, expected)

# Native XCSP3 variants from the importer's partially implemented callbacks.
# Independent oracle, followed by XCSP3 -> CPMpy -> XCSP3 -> CPMpy.
@pytest.mark.parametrize('domain', [2, 3, 4])
@pytest.mark.parametrize('operator', ['le', 'lt', 'ge', 'gt'])
def test_native_ordered_operator_inverse_oracle(tmp_path, domain, operator):
    import xml.etree.ElementTree as ET
    from itertools import product
    from cpmpy.transformations.get_variables import get_variables

    names = ('ord_a', 'ord_b', 'ord_c')
    root = ET.Element('instance', {'format': 'XCSP3', 'type': 'CSP'})
    declarations = ET.SubElement(root, 'variables')
    for name in names:
        ET.SubElement(declarations, 'var', {'id': name}).text = f'0..{domain}'
    constraints = ET.SubElement(root, 'constraints')
    ordered = ET.SubElement(constraints, 'ordered')
    ET.SubElement(ordered, 'list').text = ' '.join(names)
    ET.SubElement(ordered, 'operator').text = operator
    compare = {'le': lambda x, y: x <= y, 'lt': lambda x, y: x < y,
               'ge': lambda x, y: x >= y, 'gt': lambda x, y: x > y}[operator]
    expected = {values for values in product(range(domain + 1), repeat=3)
                if compare(values[0], values[1]) and compare(values[1], values[2])}
    source = tmp_path / f'ordered_{operator}_{domain}.xml'
    ET.ElementTree(root).write(source, encoding='utf-8', xml_declaration=True)
    exported = tmp_path / f'ordered_reexport_{operator}_{domain}.xml'
    _inverse_roundtrip_solutions(source, exported, names, expected)


@pytest.mark.parametrize('domain', [2, 3])
@pytest.mark.parametrize('operator', ['le', 'lt', 'ge', 'gt'])
def test_native_lex_operator_inverse_oracle(tmp_path, domain, operator):
    import xml.etree.ElementTree as ET
    from itertools import product
    from cpmpy.transformations.get_variables import get_variables

    names = ('lex_a', 'lex_b', 'lex_c', 'lex_d')
    root = ET.Element('instance', {'format': 'XCSP3', 'type': 'CSP'})
    declarations = ET.SubElement(root, 'variables')
    for name in names:
        ET.SubElement(declarations, 'var', {'id': name}).text = f'0..{domain}'
    constraints = ET.SubElement(root, 'constraints')
    lex = ET.SubElement(constraints, 'lex')
    for pair in (names[:2], names[2:]):
        ET.SubElement(lex, 'list').text = ' '.join(pair)
    ET.SubElement(lex, 'operator').text = operator
    compare = {'le': lambda x, y: x <= y, 'lt': lambda x, y: x < y,
               'ge': lambda x, y: x >= y, 'gt': lambda x, y: x > y}[operator]
    expected = {values for values in product(range(domain + 1), repeat=4)
                if compare(values[:2], values[2:])}
    source = tmp_path / f'lex_{operator}_{domain}.xml'
    ET.ElementTree(root).write(source, encoding='utf-8', xml_declaration=True)
    exported = tmp_path / f'lex_reexport_{operator}_{domain}.xml'
    _inverse_roundtrip_solutions(source, exported, names, expected)


# Importer-partial-feature audit: independent native XCSP3 -> CPMpy -> XCSP3
# -> CPMpy, with an exhaustive Python oracle on small finite domains.
def _inverse_native_oracle(tmp_path, case_id, domain, constraint_builder, predicate,
                           *, names=("audit_a", "audit_b", "audit_c")):
    import xml.etree.ElementTree as ET
    from itertools import product
    from cpmpy.transformations.get_variables import get_variables

    root = ET.Element("instance", {"format": "XCSP3", "type": "CSP"})
    variables = ET.SubElement(root, "variables")
    for name in names:
        ET.SubElement(variables, "var", {"id": name}).text = f"0..{domain}"
    constraints = ET.SubElement(root, "constraints")
    constraint_builder(constraints, names)
    source = tmp_path / f"partial_{case_id}.xml"
    ET.ElementTree(root).write(source, encoding="utf-8", xml_declaration=True)
    expected = {values for values in product(range(domain + 1), repeat=len(names))
                if predicate(*values)}
    exported = tmp_path / f"partial_reexport_{case_id}.xml"
    _inverse_roundtrip_solutions(source, exported, names, expected)


@pytest.mark.parametrize("domain", [2, 3, 4])
@pytest.mark.parametrize("operator", ["le", "lt", "ge", "gt"])
@pytest.mark.parametrize("length", [0, 1])
def test_inverse_partial_ordered_lengths(tmp_path, domain, operator, length):
    import xml.etree.ElementTree as ET

    def build(parent, names):
        ordered = ET.SubElement(parent, "ordered")
        ET.SubElement(ordered, "list").text = " ".join(names)
        ET.SubElement(ordered, "lengths").text = f"{length} {length}"
        ET.SubElement(ordered, "operator").text = operator

    comparisons = {"le": lambda x, y: x <= y, "lt": lambda x, y: x < y,
                   "ge": lambda x, y: x >= y, "gt": lambda x, y: x > y}
    pred = lambda a, b, c: (comparisons[operator](a + length, b)
                             and comparisons[operator](b + length, c))
    _inverse_native_oracle(tmp_path, f"ordered_lengths_{operator}_{length}_{domain}",
                           domain, build, pred)


@pytest.mark.parametrize("domain", [2, 3, 4])
@pytest.mark.parametrize("except_values", [(0,), (0, 1), (1, 2)])
def test_inverse_partial_alldifferent_except(tmp_path, domain, except_values):
    import xml.etree.ElementTree as ET

    def build(parent, names):
        node = ET.SubElement(parent, "allDifferent")
        ET.SubElement(node, "list").text = " ".join(names)
        ET.SubElement(node, "except").text = " ".join(map(str, except_values))

    def pred(*values):
        filtered = [value for value in values if value not in except_values]
        return len(filtered) == len(set(filtered))

    _inverse_native_oracle(tmp_path, f"alldiff_except_{domain}_{'_'.join(map(str, except_values))}",
                           domain, build, pred)


@pytest.mark.parametrize("domain", [2, 3, 4])
@pytest.mark.parametrize("explicit", [True, False])
def test_inverse_partial_precedence_covered_false(tmp_path, domain, explicit):
    import xml.etree.ElementTree as ET

    values = (0, 1, 2) if explicit else tuple(range(domain + 1))

    def build(parent, names):
        node = ET.SubElement(parent, "precedence")
        ET.SubElement(node, "list").text = " ".join(names)
        if explicit:
            ET.SubElement(node, "values").text = " ".join(map(str, values))

    def pred(*sequence):
        for prev, following in zip(values[:-1], values[1:]):
            for i, value in enumerate(sequence):
                if value == following and prev not in sequence[:i]:
                    return False
        return True

    _inverse_native_oracle(tmp_path, f"precedence_{domain}_{explicit}",
                           domain, build, pred)


@pytest.mark.parametrize("domain", [2, 3, 4])
@pytest.mark.parametrize("expression,predicate", [
    ("and(eq(audit_a,1),eq(audit_b,1),eq(audit_c,1))", lambda a,b,c: a==b==c==1),
    ("or(eq(audit_a,1),eq(audit_b,1),eq(audit_c,1))", lambda a,b,c: 1 in (a,b,c)),
    ("xor(eq(audit_a,1),eq(audit_b,1),eq(audit_c,1))", lambda a,b,c: sum(v==1 for v in (a,b,c))%2==1),
    ("iff(eq(audit_a,1),eq(audit_b,1),eq(audit_c,1))", lambda a,b,c: (a==1)==(b==1)==(c==1)),
    ("imp(eq(audit_a,0),eq(audit_b,1))", lambda a,b,c: a!=0 or b==1),
    ("ne(audit_a,audit_b,audit_c)", lambda a,b,c: not (a==b==c)),
    ("eq(audit_a,audit_b,audit_c)", lambda a,b,c: a==b==c),
    ("in(audit_a,set(0,2))", lambda a,b,c: a in (0,2)),
    ("notin(audit_a,set(0))", lambda a,b,c: a != 0),
])
def test_inverse_partial_logic_intension(tmp_path, domain, expression, predicate):
    import xml.etree.ElementTree as ET

    def build(parent, names):
        ET.SubElement(parent, "intension").text = expression
        # Ensure the oracle's projected variables all occur in the model.
        ET.SubElement(parent, "intension").text = "ge(audit_c,0)"
        ET.SubElement(parent, "intension").text = "ge(audit_b,0)"

    case_id = str(abs(sum((i+1)*ord(ch) for i,ch in enumerate(expression))))
    _inverse_native_oracle(tmp_path, f"logic_{domain}_{case_id}", domain, build, predicate)


@pytest.mark.parametrize("domain", [2, 3, 4])
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("kind", ["sum", "maximum", "minimum", "nValues"])
def test_inverse_partial_special_objectives(tmp_path, domain, direction, kind):
    import xml.etree.ElementTree as ET
    from itertools import product
    from cpmpy.transformations.get_variables import get_variables

    names = ("objective_a", "objective_b", "objective_c")
    root = ET.Element("instance", {"format": "XCSP3", "type": "COP"})
    variables = ET.SubElement(root, "variables")
    for name in names:
        ET.SubElement(variables, "var", {"id": name}).text = f"0..{domain}"
    constraints = ET.SubElement(root, "constraints")
    ET.SubElement(constraints, "intension").text = "le(objective_a,objective_b)"
    ET.SubElement(constraints, "intension").text = "le(objective_b,objective_c)"
    objectives = ET.SubElement(root, "objectives")
    objective = ET.SubElement(objectives, direction, {"type": kind})
    ET.SubElement(objective, "list").text = " ".join(names)
    # This pycsp3 parser version unconditionally reads the second child
    # for typed objectives, even when coefficients are optional in XCSP3.
    # Unit coefficients preserve maximum/minimum/nValues semantics.
    ET.SubElement(objective, "coeffs").text = (
        "2 1 3" if kind == "sum" else "1 1 1"
    )

    def objective_value(a, b, c):
        return {"sum": lambda: 2*a+b+3*c,
                "maximum": lambda: max(a,b,c),
                "minimum": lambda: min(a,b,c),
                "nValues": lambda: len({a,b,c})}[kind]()

    feasible = [v for v in product(range(domain+1), repeat=3)
                if v[0] <= v[1] <= v[2]]
    expected = (min if direction == "minimize" else max)(map(lambda v: objective_value(*v), feasible))
    source = tmp_path / f"special_{direction}_{kind}_{domain}.xml"
    ET.ElementTree(root).write(source, encoding="utf-8", xml_declaration=True)
    exported = tmp_path / f"special_reexport_{direction}_{kind}_{domain}.xml"
    _inverse_roundtrip_optimum(source, exported, names, expected, direction)


# Regression: the importer may fail on a multi-value notin(set(...)) AST.
# The exporter must use ordinary Boolean comparisons, not emit that AST.
@pytest.mark.parametrize("domain", [2, 3, 4])
@pytest.mark.parametrize("excluded", [(0, 2), (1, 3), (0, 1, 4)])
def test_export_finite_set_membership_without_set_ast(tmp_path, domain, excluded):
    import xml.etree.ElementTree as ET
    from cpmpy.tools.xcsp3.exporter import _expr
    from cpmpy.transformations.get_variables import get_variables

    class Membership:
        def __init__(self, name, args):
            self.name, self.args = name, args

    x = cp.intvar(0, domain, name="member_a")
    membership = Membership("notin", (x, excluded))
    text = _expr(membership, {id(x): "member_a"})
    assert "set(" not in text
    root = ET.Element("instance", {"format": "XCSP3", "type": "CSP"})
    variables = ET.SubElement(root, "variables")
    ET.SubElement(variables, "var", {"id": "member_a"}).text = f"0..{domain}"
    constraints = ET.SubElement(root, "constraints")
    ET.SubElement(constraints, "intension").text = text
    path = tmp_path / f"membership_{domain}_{'_'.join(map(str,excluded))}.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    expected = {(i,) for i in range(domain + 1) if i not in excluded}
    actual = _isolated_import_result(path, "solutions", [x])["solutions"]
    assert {tuple(v) for v in actual} == expected


@pytest.mark.parametrize("domain", [2, 3, 4])
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("kind", ["maximum", "minimum", "nValues"])
@pytest.mark.parametrize("coeffs", [(2, 1, 3), (1, 2, 1)])
def test_inverse_weighted_special_objectives(tmp_path, domain, direction, kind, coeffs):
    """Independent weighted optimum oracle for each importable typed objective."""
    import xml.etree.ElementTree as ET
    from itertools import product
    from cpmpy.transformations.get_variables import get_variables

    names = ("weighted_a", "weighted_b", "weighted_c")
    root = ET.Element("instance", {"format": "XCSP3", "type": "COP"})
    variables = ET.SubElement(root, "variables")
    for name in names:
        ET.SubElement(variables, "var", {"id": name}).text = f"0..{domain}"
    constraints = ET.SubElement(root, "constraints")
    ET.SubElement(constraints, "intension").text = "le(weighted_a,weighted_b)"
    ET.SubElement(constraints, "intension").text = "le(weighted_b,weighted_c)"
    objectives = ET.SubElement(root, "objectives")
    objective = ET.SubElement(objectives, direction, {"type": kind})
    ET.SubElement(objective, "list").text = " ".join(names)
    ET.SubElement(objective, "coeffs").text = " ".join(map(str, coeffs))

    def score(values):
        weighted = [weight * value for weight, value in zip(coeffs, values)]
        return {"maximum": lambda: max(weighted),
                "minimum": lambda: min(weighted),
                "nValues": lambda: len(set(weighted))}[kind]()

    feasible = [v for v in product(range(domain + 1), repeat=3)
                if v[0] <= v[1] <= v[2]]
    expected = (min if direction == "minimize" else max)(map(score, feasible))
    source = tmp_path / f"weighted_{direction}_{kind}_{domain}_{coeffs}.xml"
    ET.ElementTree(root).write(source, encoding="utf-8", xml_declaration=True)
    exported = tmp_path / f"reexport_{source.name}"
    _inverse_roundtrip_optimum(source, exported, names, expected, direction)


@pytest.mark.parametrize('shape', [(2, 2), (2, 3), (3, 2)])
@pytest.mark.parametrize('operator', ['==', '!=', '<', '<=', '>', '>='])
@pytest.mark.parametrize('element_on_right', [False, True])
def test_roundtrip_cpmpy_nd_element_comparison(tmp_path, shape, operator, element_on_right):
    """Exercise the public writer's nd_element dispatch and compare every solution."""
    import numpy as np
    rows, cols = shape
    matrix = cp.cpm_array(np.arange(rows * cols).reshape(shape) % 3)
    row = cp.intvar(0, rows - 1, name='nd_row')
    col = cp.intvar(0, cols - 1, name='nd_col')
    target = cp.intvar(0, 2, name='nd_target')
    selected = matrix[row, col]
    left, right = (target, selected) if element_on_right else (selected, target)
    expr = {
        '==': lambda: left == right,
        '!=': lambda: left != right,
        '<': lambda: left < right,
        '<=': lambda: left <= right,
        '>': lambda: left > right,
        '>=': lambda: left >= right,
    }[operator]()
    # The reference is an explicit disjunction, independent of ElementMatrix.
    compare = {
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b,
        '<': lambda a, b: a < b,
        '<=': lambda a, b: a <= b,
        '>': lambda a, b: a > b,
        '>=': lambda a, b: a >= b,
    }[operator]
    reference = cp.Model(cp.any([
        (row == r) & (col == c) & (
            compare(target, int(matrix[r, c])) if element_on_right
            else compare(int(matrix[r, c]), target))
        for r in range(rows) for c in range(cols)
    ]))
    vars_ = [row, col, target]
    original_solutions = _solutions(cp.Model(expr), vars_)
    expected_solutions = _solutions(reference, vars_)
    assert original_solutions == expected_solutions
    tag = {'==': 'eq', '!=': 'ne', '<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge'}[operator]
    _check_csp(tmp_path,
               f'nd_matrix_{rows}_{cols}_{tag}_{"reverse" if element_on_right else "forward"}',
               cp.Model(expr), vars_)
