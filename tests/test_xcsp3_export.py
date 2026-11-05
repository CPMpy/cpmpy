"""Initial structural and semantic checks for the XCSP3 writer."""
import xml.etree.ElementTree as ET
import pytest
cp = pytest.importorskip("cpmpy")
from cpmpy.tools.xcsp3.exporter import write_xcsp3


def test_variables_and_intension(tmp_path):
    x, y = cp.intvar(0, 10, shape=2, name="v")
    path = tmp_path / "basic.xml"
    write_xcsp3(cp.Model(x + y == 10, x < y), path)
    root = ET.parse(path).getroot()
    ids = [v.attrib["id"] for v in root.findall("./variables/var")]
    assert len(ids) == len(set(ids)) == 2
    assert all("[" not in name for name in ids)
    expressions = [e.text for e in root.findall("./constraints/intension")]
    assert len(expressions) == 2
    assert all(any(name in expr for name in ids) for expr in expressions)


def test_globals_and_objective(tmp_path):
    x = cp.intvar(0, 5, shape=3, name="x")
    model = cp.Model(cp.AllDifferent(x), cp.sum(x) == 6)
    model.minimize(x[0])
    path = tmp_path / "optimization.xml"
    write_xcsp3(model, path)
    root = ET.parse(path).getroot()
    assert root.attrib["type"] == "COP"
    assert root.find("./constraints/allDifferent/list") is not None
    assert root.find("./objectives/minimize") is not None


def test_table_extension(tmp_path):
    x = cp.intvar(0, 5, shape=3, name="x")
    path = tmp_path / "table.xml"
    write_xcsp3(cp.Model(cp.Table(x, [(0, 1, 2), (2, 1, 0)])), path)
    root = ET.parse(path).getroot()
    ext = root.find("./constraints/extension")
    assert ext is not None
    assert len(ext.find("list").text.split()) == 3
    assert ext.find("supports").text == "(0,1,2) (2,1,0)"


def test_unsupported_does_not_write(tmp_path):
    x = cp.intvar(0, 5, name="x")
    path = tmp_path / "unsupported.xml"
    with pytest.raises(NotImplementedError):
        write_xcsp3(cp.Model(cp.AllDifferent([x, x + 1]) & cp.AllDifferent([x, x + 2])), path)
    assert not path.exists()


def test_unsupported_preserves_existing_file(tmp_path):
    x = cp.intvar(0, 5, name="x")
    path = tmp_path / "existing.xml"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(NotImplementedError):
        write_xcsp3(cp.Model(cp.AllDifferent([x, x + 1]) & cp.AllDifferent([x, x + 2])), path)
    assert path.read_text(encoding="utf-8") == "original"


def test_independent_variables(tmp_path):
    a, b = cp.intvar(-3, 7, shape=2, name="z")
    path = tmp_path / "variables.xml"
    write_xcsp3(cp.Model(a != b), path)
    root = ET.parse(path).getroot()
    ids = [e.attrib["id"] for e in root.findall("./variables/var")]
    assert len(set(ids)) == 2
    assert all("[" not in name for name in ids)


def test_short_table_wildcards(tmp_path):
    x = cp.intvar(0, 3, shape=3, name="x")
    path = tmp_path / "short.xml"
    write_xcsp3(cp.Model(cp.ShortTable(x, [(1, "*", 2), ("*", 0, 3)])), path)
    root = ET.parse(path).getroot()
    assert root.find("./constraints/extension/supports").text == "(1,*,2) (*,0,3)"


def test_negative_table(tmp_path):
    x = cp.intvar(0, 3, shape=2, name="x")
    path = tmp_path / "negative.xml"
    write_xcsp3(cp.Model(cp.NegativeTable(x, [(1, 2), (2, 1)])), path)
    root = ET.parse(path).getroot()
    assert root.find("./constraints/extension/conflicts").text == "(1,2) (2,1)"


def test_global_cardinality(tmp_path):
    x = cp.intvar(0, 2, shape=3, name="x")
    n = cp.intvar(0, 3, name="n")
    path = tmp_path / "gcc.xml"
    write_xcsp3(cp.Model(cp.GlobalCardinalityCount(x, [0, 1], [n, 2], closed=True)), path)
    root = ET.parse(path).getroot()
    gcc = root.find("./constraints/cardinality")
    assert gcc is not None
    assert gcc.attrib["closed"] == "true"
    assert gcc.find("values").text == "0 1"
    assert len(gcc.find("occurs").text.split()) == 2


def test_increasing(tmp_path):
    x = cp.intvar(0, 5, shape=3, name="x")
    path = tmp_path / "ordered.xml"
    write_xcsp3(cp.Model(cp.Increasing(x)), path)
    root = ET.parse(path).getroot()
    assert root.find("./constraints/ordered/operator").text == "le"


def test_count_function(tmp_path):
    x = cp.intvar(0, 3, shape=3, name="x")
    n = cp.intvar(0, 3, name="n")
    path = tmp_path / "count.xml"
    write_xcsp3(cp.Model(cp.Count(x, 2) == n), path)
    expr = ET.parse(path).getroot().find("./constraints/intension").text
    assert expr.startswith("eq(")
    assert expr.count("eq(") == 4


def test_among_function(tmp_path):
    x = cp.intvar(0, 3, shape=3, name="x")
    path = tmp_path / "among.xml"
    write_xcsp3(cp.Model(cp.Among(x, [1, 2]) == 2), path)
    expr = ET.parse(path).getroot().find("./constraints/intension").text
    assert "or(" in expr
    assert "add(" in expr


def test_element_function(tmp_path):
    x = cp.intvar(0, 5, shape=3, name="x")
    idx = cp.intvar(0, 2, name="idx")
    path = tmp_path / "element.xml"
    write_xcsp3(cp.Model(cp.Element(x, idx) == 3), path)
    element = ET.parse(path).getroot().find("./constraints/element")
    assert element is not None
    assert element.find("list") is not None
    assert element.find("index").text == "idx"
    assert element.find("condition").text == "(eq,3)"


def test_element_out_of_range_rejected(tmp_path):
    x = cp.intvar(0, 5, shape=3, name="x")
    idx = cp.intvar(-1, 3, name="idx")
    path = tmp_path / "invalid_element.xml"
    with pytest.raises(NotImplementedError):
        write_xcsp3(cp.Model(cp.Element(x, idx) == 3), path)
    assert not path.exists()


def test_minimum_maximum_functions(tmp_path):
    x = cp.intvar(0, 5, shape=3, name="x")
    path = tmp_path / "minmax.xml"
    write_xcsp3(cp.Model(cp.Minimum(x) == 1, cp.Maximum(x) == 4), path)
    exprs = [e.text for e in ET.parse(path).getroot().findall("./constraints/intension")]
    assert any("min(" in expr for expr in exprs)
    assert any("max(" in expr for expr in exprs)

# Extended exporter coverage. These are structural tests, not a claim of
# XCSP3 parser conformance or semantic equivalence.

@pytest.mark.parametrize("operator,expected", [
    ("eq", "eq("), ("ne", "ne("), ("lt", "lt("),
    ("le", "le("), ("gt", "gt("), ("ge", "ge("),
])
def test_comparison_operator_matrix(tmp_path, operator, expected):
    x, y = cp.intvar(-3, 3, shape=2, name="comparison")
    expressions = {
        "eq": x == y, "ne": x != y, "lt": x < y,
        "le": x <= y, "gt": x > y, "ge": x >= y,
    }
    path = tmp_path / f"comparison_{operator}.xml"
    write_xcsp3(cp.Model(expressions[operator]), path)
    root = ET.parse(path).getroot()
    assert root.find("./constraints/intension").text.startswith(expected)


@pytest.mark.parametrize("operation,token", [
    ("sum", "add("), ("subtract", "sub("), ("multiply", "mul("),
    ("absolute", "abs("), ("minimum", "min("), ("maximum", "max("),
])
def test_arithmetic_expression_matrix(tmp_path, operation, token):
    x, y, z = cp.intvar(-3, 3, shape=3, name="arithmetic")
    expressions = {
        "sum": x + y + z, "subtract": x - y,
        "multiply": x * y, "absolute": abs(x),
        "minimum": cp.Minimum([x, y, z]),
        "maximum": cp.Maximum([x, y, z]),
    }
    path = tmp_path / f"arithmetic_{operation}.xml"
    write_xcsp3(cp.Model(expressions[operation] == z), path)
    serialized = ET.parse(path).getroot().find("./constraints/intension").text
    assert token in serialized or (operation == "subtract" and "add(" in serialized and "neg(" in serialized)


@pytest.mark.parametrize("operation,token", [
    ("and", "and("), ("or", "or("), ("implies", "imp("),
])
def test_boolean_expression_matrix(tmp_path, operation, token):
    a, b = cp.boolvar(shape=2, name="bools")
    expressions = {
        "and": a & b, "or": a | b,
        "implies": a.implies(b),
    }
    path = tmp_path / f"boolean_{operation}.xml"
    write_xcsp3(cp.Model(expressions[operation]), path)
    serialized = ET.parse(path).getroot().find("./constraints/intension").text
    assert token in serialized or (operation == "subtract" and "add(" in serialized and "neg(" in serialized)


@pytest.mark.parametrize("bounds", [(-10, -1), (-2, 2), (0, 1), (1, 100)])
def test_integer_domain_matrix(tmp_path, bounds):
    x = cp.intvar(*bounds, name="domain_var")
    path = tmp_path / f"domain_{bounds[0]}_{bounds[1]}.xml"
    write_xcsp3(cp.Model(x == bounds[0]), path)
    root = ET.parse(path).getroot()
    assert root.find("./variables/var").text == f"{bounds[0]}..{bounds[1]}"


@pytest.mark.parametrize("kind,tag", [
    ("allDifferent", "allDifferent"), ("allEqual", "allEqual"),
])
def test_all_global_matrix(tmp_path, kind, tag):
    x = cp.intvar(0, 3, shape=3, name="globals")
    constraint = cp.AllDifferent(x) if kind == "allDifferent" else cp.AllEqual(x)
    path = tmp_path / f"{kind}.xml"
    write_xcsp3(cp.Model(constraint), path)
    assert ET.parse(path).getroot().find(f"./constraints/{tag}/list") is not None


@pytest.mark.parametrize("kind,tag", [
    ("increasing", "le"), ("decreasing", "ge"),
    ("strict_increasing", "lt"), ("strict_decreasing", "gt"),
])
def test_ordered_matrix(tmp_path, kind, tag):
    x = cp.intvar(0, 8, shape=3, name="ordered_vars")
    constructors = {
        "increasing": cp.Increasing,
        "decreasing": cp.Decreasing,
        "strict_increasing": cp.IncreasingStrict,
        "strict_decreasing": cp.DecreasingStrict,
    }
    path = tmp_path / f"{kind}.xml"
    write_xcsp3(cp.Model(constructors[kind](x)), path)
    assert ET.parse(path).getroot().find("./constraints/ordered/operator").text == tag


@pytest.mark.parametrize("kind,xml_tag", [
    ("positive", "supports"), ("negative", "conflicts"),
    ("short", "supports"),
])
def test_extension_matrix(tmp_path, kind, xml_tag):
    x = cp.intvar(0, 2, shape=2, name="ext")
    constructors = {
        "positive": lambda: cp.Table(x, [(0, 1), (1, 2)]),
        "negative": lambda: cp.NegativeTable(x, [(0, 1)]),
        "short": lambda: cp.ShortTable(x, [(0, "*"), ("*", 1)]),
    }
    path = tmp_path / f"ext_{kind}.xml"
    write_xcsp3(cp.Model(constructors[kind]()), path)
    assert ET.parse(path).getroot().find(f"./constraints/extension/{xml_tag}") is not None


@pytest.mark.parametrize("minimize", [True, False])
def test_objective_direction_matrix(tmp_path, minimize):
    x = cp.intvar(-3, 5, name="objective_var")
    model = cp.Model(x >= 0)
    if minimize:
        model.minimize(x)
    else:
        model.maximize(x)
    path = tmp_path / f"objective_{minimize}.xml"
    write_xcsp3(model, path)
    root = ET.parse(path).getroot()
    assert root.attrib["type"] == "COP"
    tag = "minimize" if minimize else "maximize"
    assert root.find(f"./objectives/{tag}") is not None


def test_constraint_only_model_is_csp(tmp_path):
    x = cp.intvar(0, 3, name="csp_var")
    path = tmp_path / "csp.xml"
    write_xcsp3(cp.Model(x == 2), path)
    root = ET.parse(path).getroot()
    assert root.attrib["type"] == "CSP"
    assert root.find("./objectives") is None


def test_multiple_constraints_preserve_count(tmp_path):
    x, y = cp.intvar(0, 5, shape=2, name="many")
    path = tmp_path / "many.xml"
    write_xcsp3(cp.Model(x < y, x != 2, y >= 1), path)
    assert len(ET.parse(path).getroot().findall("./constraints/intension")) == 3


def test_count_with_variable_target(tmp_path):
    x = cp.intvar(0, 3, shape=3, name="items")
    target = cp.intvar(0, 3, name="target")
    path = tmp_path / "variable_count.xml"
    write_xcsp3(cp.Model(cp.Count(x, target) == 2), path)
    serialized = ET.parse(path).getroot().find("./constraints/intension").text
    assert serialized.count("eq(") >= 3


def test_among_duplicate_values_do_not_double_count(tmp_path):
    x = cp.intvar(0, 3, shape=2, name="items")
    path = tmp_path / "among_duplicates.xml"
    write_xcsp3(cp.Model(cp.Among(x, [1, 1, 2]) == 1), path)
    serialized = ET.parse(path).getroot().find("./constraints/intension").text
    assert "or(eq(v_0,1),eq(v_0,2))" in serialized
    assert "or(eq(v_1,1),eq(v_1,2))" in serialized


def test_element_singleton_array(tmp_path):
    x = cp.intvar(0, 4, name="only")
    index = cp.intvar(0, 0, name="index")
    path = tmp_path / "element_singleton.xml"
    write_xcsp3(cp.Model(cp.Element([x], index) == 2), path)
    element = ET.parse(path).getroot().find("./constraints/element")
    assert element is not None
    assert element.find("list").text == "only"
    assert element.find("index").text == "index"
    assert element.find("condition").text == "(eq,2)"


def test_xml_is_well_formed_and_has_required_sections(tmp_path):
    x = cp.intvar(0, 2, name="section_var")
    path = tmp_path / "sections.xml"
    write_xcsp3(cp.Model(x == 1), path)
    assert path.read_text(encoding="utf-8").startswith('<?xml version="1.0"')
    root = ET.parse(path).getroot()
    assert root.tag == "instance" and root.attrib["format"] == "XCSP3"
    assert root.find("variables") is not None
    assert root.find("constraints") is not None


@pytest.mark.parametrize("feature,tag", [
    ("regular", "regular"), ("mdd", "mdd"),
    ("inverse", "channel"), ("precedence", "precedence"),
    ("lex_less", "lex"), ("lex_lesseq", "lex"),
    ("lex_chain_less", "lex"), ("lex_chain_lesseq", "lex"),
    ("no_overlap", "noOverlap"), ("cumulative", "cumulative"),
])
def test_extended_global_serialization(tmp_path, feature, tag):
    x = cp.intvar(0, 3, shape=3, name="extended")
    y = cp.intvar(0, 2, shape=3, name="reverse")
    factories = {
        "regular": lambda: cp.Regular(x, [("A", 0, "A"), ("A", 1, "B"), ("B", 0, "B")], "A", ["B"]),
        "mdd": lambda: cp.MDD(x[:2], [("A", 0, "B"), ("B", 1, "C")]),
        "inverse": lambda: cp.Inverse(y, y),
        "precedence": lambda: cp.Precedence(x, [0, 1, 2]),
        "lex_less": lambda: cp.LexLess(x[:2], x[1:]),
        "lex_lesseq": lambda: cp.LexLessEq(x[:2], x[1:]),
        "lex_chain_less": lambda: cp.LexChainLess([x[:2], x[1:]]),
        "lex_chain_lesseq": lambda: cp.LexChainLessEq([x[:2], x[1:]]),
        "no_overlap": lambda: cp.NoOverlap(x, [1, 2, 1]),
        "cumulative": lambda: cp.Cumulative(x, [1, 2, 1], demand=[1, 2, 1], capacity=2),
    }
    path = tmp_path / f"{feature}.xml"
    write_xcsp3(cp.Model(factories[feature]()), path)
    assert ET.parse(path).getroot().find(f"./constraints/{tag}") is not None


@pytest.mark.parametrize("feature", ["nvalue", "nvalue_except"])
def test_distinct_value_functions(tmp_path, feature):
    x = cp.intvar(0, 3, shape=3, name="distinct")
    expr = cp.NValue(x) if feature == "nvalue" else cp.NValueExcept(x, 0)
    path = tmp_path / f"{feature}.xml"
    write_xcsp3(cp.Model(expr == 2), path)
    serialized = ET.parse(path).getroot().find("./constraints/intension").text
    assert "ne(" in serialized
    assert "add(" in serialized

@pytest.mark.parametrize('size', [2, 3, 4, 5, 8])
def test_circuit_export(tmp_path, size):
    succ = cp.intvar(0, size - 1, shape=size, name='succ')
    path = tmp_path / f'circuit_{size}.xml'
    write_xcsp3(cp.Model(cp.Circuit(succ)), path)
    root = ET.parse(path).getroot()
    assert root.find('./constraints/circuit/list') is not None
    assert len(root.find('./constraints/circuit/list').text.split()) == size


def test_circuit_and_objective(tmp_path):
    succ = cp.intvar(0, 3, shape=4, name='succ')
    model = cp.Model(cp.Circuit(succ))
    model.minimize(succ[0])
    path = tmp_path / 'circuit_opt.xml'
    write_xcsp3(model, path)
    root = ET.parse(path).getroot()
    assert root.find('./constraints/circuit') is not None
    assert root.find('./objectives/minimize') is not None


def test_circuit_preserves_other_constraints(tmp_path):
    succ = cp.intvar(0, 2, shape=3, name='succ')
    path = tmp_path / 'circuit_other.xml'
    write_xcsp3(cp.Model(cp.Circuit(succ), succ[0] != 0), path)
    root = ET.parse(path).getroot()
    constraints = root.findall('./constraints/*')
    assert len(root.findall('./constraints/circuit')) == 1
    intensions = [node.text for node in constraints if node.tag == 'intension']
    assert len(intensions) == 4
    # Three anti-self-loop constraints plus the user's explicit constraint.
    assert sum(expr == 'ne(v_0,0)' for expr in intensions) == 2
    assert 'ne(v_1,1)' in intensions
    assert 'ne(v_2,2)' in intensions


@pytest.mark.parametrize("excluded", [0, -1, 2])
def test_all_different_except_n(tmp_path, excluded):
    x = cp.intvar(-1, 3, shape=4, name="ad")
    path = tmp_path / "except.xml"
    write_xcsp3(cp.Model(cp.AllDifferentExceptN(x, excluded)), path)
    root = ET.parse(path).getroot()
    assert root.find("./constraints/allDifferent/except").text == str(excluded)


def test_all_equal_except_n(tmp_path):
    x = cp.intvar(0, 4, shape=3, name="ae")
    path = tmp_path / "ae.xml"
    write_xcsp3(cp.Model(cp.AllEqualExceptN(x, 0)), path)
    root = ET.parse(path).getroot()
    assert len(root.findall("./constraints/intension")) >= 1


@pytest.mark.parametrize("domain", [[1], [1, 3], [-2, 0, 2]])
def test_in_domain(tmp_path, domain):
    x = cp.intvar(-3, 3, name="x")
    path = tmp_path / "domain.xml"
    write_xcsp3(cp.Model(cp.InDomain(x, domain)), path)
    text = ET.parse(path).getroot().find("./constraints/intension").text
    for val in domain:
        assert f"eq(x,{val})" in text


def test_if_then_else(tmp_path):
    a, b, c = cp.boolvar(shape=3, name="bool")
    path = tmp_path / "ite.xml"
    write_xcsp3(cp.Model(cp.IfThenElse(a, b, c)), path)
    assert ET.parse(path).getroot().find("./constraints/intension").text.startswith("if(")


def test_xor_global(tmp_path):
    x = cp.boolvar(shape=3, name="x")
    path = tmp_path / "xor.xml"
    write_xcsp3(cp.Model(cp.Xor(x)), path)
    assert "xor(" in ET.parse(path).getroot().find("./constraints/intension").text

# Targeted white-box tests for branches that are difficult to reach through
# ordinary CPMpy constructors. These supplement, rather than replace, model tests.
import types
import numpy as np
import cpmpy.tools.xcsp3.exporter as writer


def _node(name, *args, **extra):
    return types.SimpleNamespace(name=name, args=args, **extra)


def _fake_var(name, lb=0, ub=3):
    class Var:
        def __init__(self):
            self.name, self.lb, self.ub = name, lb, ub
    return Var()


def test_identifier_collision_and_invalid_names():
    variables = [_fake_var('same'), _fake_var('same'),
                 _fake_var('v_1'), _fake_var('bad[0]')]
    mapped = writer._identifier_map(variables)
    assert len(set(mapped.values())) == len(variables)
    assert all('[' not in name for name in mapped.values())


@pytest.mark.parametrize('value,expected', [(True, '1'), (False, '0'),
                                            (np.bool_(True), '1'), (np.int64(-3), '-3')])
def test_internal_scalar_expression(value, expected):
    assert writer._expr(value, {}) == expected


def test_internal_undeclared_variable():
    with pytest.raises(ValueError, match='Undeclared variable'):
        writer._expr(_fake_var('missing'), {})


def test_internal_unsupported_expression():
    with pytest.raises(NotImplementedError, match='Unsupported XCSP3 expression'):
        writer._expr(object(), {})
    with pytest.raises(NotImplementedError, match='Unsupported XCSP3 operator'):
        writer._expr(_node('unknown', 1), {})


@pytest.mark.parametrize('node,expected', [
    (_node('indomain', 1, []), '0'),
    (_node('indomain', 2, [2]), 'eq(2,2)'),
    (_node('xor'), '0'),
    (_node('xor', True), '1'),
    (_node('sqr', 3), 'mul(3,3)'),
    (_node('wsum', [2, -1], [3, 4]), 'add(mul(2,3),mul(-1,4))'),
    (_node('wsum', [], []), '0'),
    (_node('wsum', [2], [3]), 'mul(2,3)'),
    (_node('-', 5, 2), 'sub(5,2)'),
    (_node('count', [], 1), '0'),
    (_node('among', [], [1]), '0'),
    (_node('among', [2], []), '0'),
    (_node('nvalue'), '0'),
    (_node('nvalue_except', [], 0), '0'),
])
def test_internal_expression_edges(node, expected):
    assert writer._expr(node, {}) == expected


def test_internal_weighted_sum_mismatch():
    with pytest.raises(ValueError, match='Malformed weighted sum'):
        writer._expr(_node('wsum', [1, 2], [3]), {})


def test_internal_element_empty_and_invalid_index():
    idx = _fake_var('idx', 0, 1)
    with pytest.raises(ValueError, match='cannot be empty'):
        writer._expr(_node('element', [], idx), {})
    with pytest.raises(NotImplementedError, match='in-range'):
        writer._expr(_node('element', [1], idx), {})


def test_internal_nd_element_and_invalid_index():
    idx = _fake_var('idx', 0, 1)
    names = {id(idx): 'idx'}
    assert writer._expr(_node('nd_element', np.array([4, 5]), idx), names) == 'if(eq(idx,0),4,5)'
    with pytest.raises(NotImplementedError, match='in-range'):
        writer._expr(_node('nd_element', np.array([4]), idx), names)


def test_internal_nd_element_multidimensional():
    i, j = _fake_var('i', 0, 1), _fake_var('j', 0, 1)
    result = writer._expr(_node('nd_element', np.array([[1, 2], [3, 4]]), i, j),
                          {id(i): 'i', id(j): 'j'})
    assert 'and(eq(i,0),eq(j,0))' in result
    assert result.endswith(',4)))')


def test_internal_table_invalid_inputs():
    root = ET.Element('constraints')
    with pytest.raises(NotImplementedError, match='nonempty'):
        writer._table_constraint(root, _node('table', [], [(1,)]), {})
    var = _fake_var('x')
    names = {id(var): 'x'}
    with pytest.raises(ValueError, match='tuple length'):
        writer._table_constraint(root, _node('table', [var], [(1, 2)]), names)
    with pytest.raises(NotImplementedError, match='table entry'):
        writer._table_constraint(root, _node('table', [var], [('invalid',)]), names)


def test_internal_cardinality_invalid_inputs():
    root = ET.Element('constraints')
    var = _fake_var('x')
    names = {id(var): 'x'}
    with pytest.raises(NotImplementedError, match='variable arguments'):
        writer._gcc_constraint(root, _node('gcc', [], [0], [1]), names)
    with pytest.raises(ValueError, match='length mismatch'):
        writer._gcc_constraint(root, _node('gcc', [var], [0], [1, 2]), names)
    with pytest.raises(NotImplementedError, match='integers'):
        writer._gcc_constraint(root, _node('gcc', [var], ['a'], [1]), names)


def test_internal_lex_invalid_inputs():
    root = ET.Element('constraints')
    with pytest.raises(ValueError, match='at least two'):
        writer._lex_constraint(root, _node('lex_less', [1]), {})
    with pytest.raises(ValueError, match='equal length'):
        writer._lex_constraint(root, _node('lex_less', [1], [1, 2]), {})


def test_internal_circuit_invalid_inputs():
    with pytest.raises(NotImplementedError, match='at least two'):
        writer._circuit_constraint(ET.Element('constraints'), _node('circuit', 1), {})


def test_internal_empty_global():
    with pytest.raises(ValueError, match='Empty global'):
        writer._constraint(ET.Element('constraints'), _node('alldifferent'), {})


def test_internal_decomposition_rejections():
    root = ET.Element('constraints')
    class Missing:
        name = 'missing'
    with pytest.raises(NotImplementedError, match='Cannot export'):
        writer._decomposed_constraint(root, Missing(), {})
    class Auxiliary:
        name = 'auxiliary'
        def decompose(self):
            return [], [object()]
    with pytest.raises(NotImplementedError, match='auxiliary variables'):
        writer._decomposed_constraint(root, Auxiliary(), {})


def test_internal_decomposition_fallback():
    class Decomposable:
        name = 'custom'
        args = ()
        def decompose(self):
            return [_node('==', 1, 1)], []
    root = ET.Element('constraints')
    writer._constraint(root, Decomposable(), {})
    assert root.find('intension').text == 'eq(1,1)'


def test_internal_sum_helper_and_except_helper():
    root = ET.Element('constraints')
    writer._sum_condition(root, [1, 2], 'eq', 3, {}, coefficients=[2, 1])
    assert root.find('./sum/coeffs').text == '2 1'
    writer._all_different_except(root, [1, 2], [], {})
    assert root.find('./allDifferent/except') is None


def test_internal_scheduling_with_end_variables():
    root = ET.Element('constraints')
    writer._scheduling_constraint(root, _node('no_overlap', [1], [2], [3]), {})
    assert root.find('intension').text == 'eq(add(1,2),3)'
    writer._scheduling_constraint(root, _node('cumulative', [1], [2], [3], [1], 5), {})
    assert len(root.findall('intension')) == 2


def test_internal_except_zero_variant():
    root = ET.Element('constraints')
    writer._constraint(root, _node('alldifferent_except_0', [1, 2]), {})
    assert root.find('./allDifferent/except').text == '0'


def test_identifier_collision_requires_suffix():
    # v_1 is occupied by a prior variable when the second name is invalid.
    vars_ = [_fake_var('v_1'), _fake_var('bad[0]')]
    mapped = writer._identifier_map(vars_)
    assert mapped[id(vars_[0])] == 'v_1'
    assert mapped[id(vars_[1])] == 'v_1_'


def test_internal_gcc_open_and_closed():
    var = _fake_var('x')
    names = {id(var): 'x'}
    for closed in (False, True):
        root = ET.Element('constraints')
        writer._gcc_constraint(root, _node('gcc', [var], [0], [1], closed=closed), names)
        card = root.find('cardinality')
        assert card.get('closed') == ('true' if closed else None)


def test_internal_all_different_except_empty():
    root = ET.Element('constraints')
    var = _fake_var('x')
    writer._all_different_except(root, [var], [], {id(var): 'x'})
    assert root.find('allDifferent/except') is None


def test_internal_sum_condition_without_coefficients():
    root = ET.Element('constraints')
    var = _fake_var('x')
    writer._sum_condition(root, [var], 'eq', 1, {id(var): 'x'})
    assert root.find('sum/coeffs') is None
    assert root.find('sum/condition').text == '(eq,1)'

# v12: structure-level tests for callback families absent from v11.
@pytest.mark.parametrize('kind,expected', [
    ('binpacking_limits', 'limits'),
    ('binpacking_loads', 'loads'),
    ('binpacking', 'condition'),
])
def test_v12_binpacking_variants(kind, expected):
    x, y = _fake_var('x'), _fake_var('y')
    names = {id(x): 'x', id(y): 'y'}
    spec = ('le', 5) if kind == 'binpacking' else [5, 6]
    root = ET.Element('constraints')
    writer._constraint(root, _node(kind, [x, y], [2, 3], spec), names)
    elt = root.find('binPacking')
    assert elt is not None
    assert elt.find('sizes').text == '2 3'
    assert elt.find(expected) is not None


def test_v12_knapsack_structure():
    x, y = _fake_var('x', 0, 1), _fake_var('y', 0, 1)
    root = ET.Element('constraints')
    writer._constraint(root, _node('knapsack', [x, y], [2, 3], ('le', 4),
                                   [8, 9], ('ge', 8)), {id(x): 'x', id(y): 'y'})
    elt = root.find('knapsack')
    assert elt.find('weights').text == '2 3'
    assert elt.find('profits').text == '8 9'
    assert [c.text for c in elt.findall('condition')] == ['(le,4)', '(ge,8)']
    assert [child.tag for child in elt] == ['list', 'weights', 'condition', 'profits', 'condition']


@pytest.mark.parametrize('kind', ['inverse_one', 'channel_value', 'inverse'])
def test_v12_channel_variants(kind):
    x, y = _fake_var('x'), _fake_var('y')
    args = ([x, y],) if kind == 'inverse_one' else ([x, y], x if kind == 'channel_value' else [y, x])
    root = ET.Element('constraints')
    writer._constraint(root, _node(kind, *args), {id(x): 'x', id(y): 'y'})
    elt = root.find('channel')
    assert elt is not None
    assert len(elt.findall('list')) == (2 if kind == 'inverse' else 1)
    assert (elt.find('value') is not None) == (kind == 'channel_value')


def test_v12_all_different_lists():
    a, b, c, d = [_fake_var(s) for s in 'abcd']
    root = ET.Element('constraints')
    writer._constraint(root, _node('alldifferent_lists', [a, b], [c, d]),
                       {id(v): v.name for v in [a, b, c, d]})
    assert len(root.findall('./allDifferent/list')) == 2


def test_v12_matrix_variants():
    a, b, c, d = [_fake_var(s) for s in 'abcd']
    names = {id(v): v.name for v in [a, b, c, d]}
    for kind, args, tag in [('alldifferent_matrix', ([[a, b], [c, d]],), 'allDifferent'),
                            ('lex_matrix', ([[a, b], [c, d]], 'lt'), 'lex')]:
        root = ET.Element('constraints')
        writer._constraint(root, _node(kind, *args), names)
        assert root.find(f'{tag}/matrix') is not None


def test_v12_reject_malformed_binpacking():
    x = _fake_var('x')
    with pytest.raises(ValueError):
        writer._constraint(ET.Element('constraints'), _node('binpacking_limits', [x], [1, 2], [3]),
                           {id(x): 'x'})

# Additional coverage for the v12 callback families and the 2-D scheduling form.
@pytest.mark.parametrize('kind,args,error', [
    ('binpacking', ([1], [2], ('le', 3)), NotImplementedError),
    ('binpacking', ([], [], ('le', 3)), ValueError),
    ('binpacking', ([1], [2], ('eq', 3), 4), ValueError),
    ('binpacking_limits', ([], [], []), ValueError),
    ('knapsack', ([1], [2], ('le', 3), [4], ('ge', 0)), NotImplementedError),
    ('knapsack', ([], [], ('le', 3), [], ('ge', 0)), ValueError),
    ('knapsack', ([1], [2]), ValueError),
    ('channel', ([1],), ValueError),
    ('inverse', ([1],), ValueError),
    ('alldifferent_lists', ([],), ValueError),
    ('alldifferent_matrix', ([],), ValueError),
    ('lex_matrix', ([[1], [2]], 'gt'), NotImplementedError),
    ('no_overlap_2d', ([1], [2]), ValueError),
    ('no_overlap_2d', ([], [], [], [], [], []), ValueError),
])
def test_extended_callback_invalid_forms(kind, args, error):
    with pytest.raises(error):
        writer._constraint(ET.Element('constraints'), _node(kind, *args), {})


@pytest.mark.parametrize('spec', [
    ('wat', 3), ('eq',), ('le', 1, 2), 5,
])
def test_invalid_binpacking_condition(spec):
    x = _fake_var('x')
    with pytest.raises(NotImplementedError):
        writer._constraint(ET.Element('constraints'),
                           _node('binpacking', [x], [2], spec), {id(x): 'x'})


def test_binpacking_rejects_negative_and_noninteger_sizes():
    x = _fake_var('x')
    for sizes in ([-1], [1.5]):
        with pytest.raises(NotImplementedError):
            writer._constraint(ET.Element('constraints'),
                               _node('binpacking', [x], sizes, ('le', 5)), {id(x): 'x'})


def test_binpacking_rejects_empty_limits():
    x = _fake_var('x')
    with pytest.raises(ValueError):
        writer._constraint(ET.Element('constraints'),
                           _node('binpacking_limits', [x], [2], []), {id(x): 'x'})


def test_knapsack_rejects_noninteger_weight_and_profit():
    x = _fake_var('x')
    for weights, profits in [([2.5], [3]), ([2], [3.5])]:
        with pytest.raises(NotImplementedError):
            writer._constraint(ET.Element('constraints'),
                               _node('knapsack', [x], weights, ('le', 5), profits, ('ge', 1)),
                               {id(x): 'x'})


def test_all_different_lists_except_tuples():
    x, y = _fake_var('x'), _fake_var('y')
    root = ET.Element('constraints')
    writer._constraint(root, _node('alldifferent_lists_except_n', [[x], [y]], [[0]]),
                       {id(x): 'x', id(y): 'y'})
    assert root.find('./allDifferent/except').text == '(0)'


def test_channel_inverse_one_flat_arguments():
    x, y = _fake_var('x'), _fake_var('y')
    root = ET.Element('constraints')
    writer._constraint(root, _node('inverse_one', x, y), {id(x): 'x', id(y): 'y'})
    assert root.find('./channel/list').text == 'x y'


def test_no_overlap_2d_serialization():
    vars_ = [_fake_var(f'v{i}') for i in range(8)]
    sx, dx, ex, sy, dy, ey = ([vars_[0], vars_[1]], [2, 3],
                              [vars_[2], vars_[3]], [vars_[4], vars_[5]],
                              [4, 5], [vars_[6], vars_[7]])
    root = ET.Element('constraints')
    writer._constraint(root, _node('no_overlap_2d', sx, dx, ex, sy, dy, ey),
                       {id(v): v.name for v in vars_})
    assert root.find('./noOverlap/origins').text == '(v0,v4) (v1,v5)'
    assert root.find('./noOverlap/lengths').text == '(2,4) (3,5)'
    assert len(root.findall('intension')) == 4


@pytest.mark.parametrize('bad_indices', [
    (), (0, 1, 2),
])
def test_nd_element_rejects_wrong_number_of_indices(bad_indices):
    with pytest.raises(NotImplementedError):
        writer._expr(_node('nd_element', np.array([[1, 2], [3, 4]]), *bad_indices), {})


@pytest.mark.parametrize('op', ['sub', 'div', 'mod', 'pow', '->'])
def test_nonassociative_operators_reject_extra_arguments(op):
    with pytest.raises(NotImplementedError):
        writer._expr(_node(op, 1, 2, 3), {})


@pytest.mark.parametrize('operator,expected', [('<', 'lt'), ('<=', 'le'), ('lt', 'lt'), ('le', 'le')])
def test_lex_matrix_operator_normalization(operator, expected):
    x, y = _fake_var('x'), _fake_var('y')
    root = ET.Element('constraints')
    writer._constraint(root, _node('lex_matrix', [[x], [y]], operator),
                       {id(x): 'x', id(y): 'y'})
    assert root.find('./lex/operator').text == expected


# Importer callback parity: ctr_instantiation

def test_instantiation_callback_structure():
    x, y = _fake_var('x'), _fake_var('y')
    root = ET.Element('constraints')
    writer._constraint(root, _node('instantiation', [x, y], [-2, 3]),
                       {id(x): 'x', id(y): 'y'})
    assert root.find('./instantiation/list').text == 'x y'
    assert root.find('./instantiation/values').text == '-2 3'


@pytest.mark.parametrize('case,error', [
    ('empty', ValueError),
    ('nonvariable', NotImplementedError),
    ('arity', ValueError),
    ('mismatch', ValueError),
    ('noninteger', NotImplementedError),
])
def test_instantiation_callback_rejects_invalid_forms(case, error):
    x = _fake_var('x')
    args = {
        'empty': ([], []),
        'nonvariable': ([1], [1]),
        'arity': ([x],),
        'mismatch': ([x], [1, 2]),
        'noninteger': ([x], ['not-an-int']),
    }[case]
    root = ET.Element('constraints')
    with pytest.raises(error):
        writer._constraint(root, _node('instantiation', *args), {id(x): 'x'})

# Importer parity: non-contiguous XCSP3 variable declarations.
@pytest.mark.parametrize('domain,expected', [
    ([1, 3, 7], '1 3 7'),
    ([-2, -1, 0], '-2..0'),
    ([4], '4..4'),
])
def test_domain_declaration_from_indomain(tmp_path, domain, expected):
    x = cp.intvar(-5, 10, name='holes')
    path = tmp_path / 'holes.xml'
    write_xcsp3(cp.Model(cp.InDomain(x, domain)), path)
    root = ET.parse(path).getroot()
    assert root.find('./variables/var').text == expected
    # Keep an explicit constraint, so exporting is correct even if another
    # importer ignores non-contiguous variable declarations.
    assert root.find('./constraints/intension') is not None


def test_domain_declaration_intersects_multiple_positive_memberships(tmp_path):
    x = cp.intvar(0, 10, name='holes')
    path = tmp_path / 'intersection.xml'
    write_xcsp3(cp.Model(cp.InDomain(x, [1, 3, 5]), cp.InDomain(x, [3, 5, 7])), path)
    assert ET.parse(path).getroot().find('./variables/var').text == '3 5'


def test_domain_declaration_disjoint_memberships_remain_contradictory(tmp_path):
    x = cp.intvar(0, 10, name='holes')
    path = tmp_path / 'disjoint.xml'
    write_xcsp3(cp.Model(cp.InDomain(x, [1]), cp.InDomain(x, [2])), path)
    root = ET.parse(path).getroot()
    assert root.find('./variables/var').text == '0..10'
    assert len(root.findall('./constraints/intension')) == 2


# Additional importer intension callback coverage

@pytest.mark.parametrize("op,expected", [
    ("dist", "dist(2,5)"),
    ("iff", "iff(1,0)"),
])
def test_importer_intension_binary_callbacks(op, expected):
    assert writer._expr(_node(op, 2, 5) if op == "dist" else _node(op, True, False), {}) == expected


@pytest.mark.parametrize("op,expected", [
    ("in", "or(eq(2,1),eq(2,2),eq(2,3))"),
    ("notin", "and(ne(2,1),ne(2,2),ne(2,3))"),
])
def test_importer_membership_callbacks(op, expected):
    assert writer._expr(_node(op, 2, [3, 1, 2, 2]), {}) == expected


@pytest.mark.parametrize("op,expected", [("in", "0"), ("notin", "1")])
def test_importer_membership_empty_set(op, expected):
    assert writer._expr(_node(op, 2, []), {}) == expected


@pytest.mark.parametrize("op,args", [
    ("dist", (1,)),
    ("in", (1,)),
    ("notin", (1, 2, 3)),
    ("in", (1, "not-a-set")),
    ("notin", (1, [1, "bad"])),
])
def test_importer_intension_invalid_callback_forms(op, args):
    with pytest.raises(NotImplementedError):
        writer._expr(_node(op, *args), {})


def test_weighted_load_helper_full_paths():
    assert writer._weighted_load([0, 1], [3, 5], 0, {}) == (
        "add(mul(3,eq(0,0)),mul(5,eq(1,0)))"
    )
    with pytest.raises(ValueError):
        writer._weighted_load([0, 1], [3], 0, {})
    with pytest.raises(NotImplementedError):
        writer._weighted_load([0], [2.5], 0, {})
    assert writer._add([]) == "0"
    assert writer._add(["a"]) == "a"
    assert writer._add(["a", "b"]) == "add(a,b)"


def test_indomain_invalid_variable_argument():
    constraint = _node("indomain", 3, [1, 2, 3])
    result = writer._explicit_variable_domains([constraint])
    assert result == {}


def test_indomain_non_integer_values():
    x = _fake_var("x", 0, 10)
    constraint = _node("indomain", x, [1, 2.5, 3])
    with pytest.raises(NotImplementedError, match="integer domain values"):
        writer._explicit_variable_domains([constraint])

# N-ary intension semantics matching CallbacksCPMPy.funcmap
@pytest.mark.parametrize("values,expected", [
    ((), "1"),
    ((True,), "1"),
    ((True, False), "iff(1,0)"),
    ((True, False, True), "and(eq(1,0),eq(0,1))"),
])
def test_nary_iff_importer_semantics(values, expected):
    assert writer._expr(_node("iff", *values), {}) == expected


@pytest.mark.parametrize("operator,expected", [
    ("eq", "and(eq(1,2),eq(1,3))"),
    ("ne", "not(and(eq(1,2),eq(1,3)))"),
    ("==", "and(eq(1,2),eq(1,3))"),
    ("!=", "not(and(eq(1,2),eq(1,3)))"),
])
def test_nary_equality_importer_semantics(operator, expected):
    assert writer._expr(_node(operator, 1, 2, 3), {}) == expected

# Batch parity additions: specialized objectives, ordered lengths, auxiliaries.

def test_specialized_sum_objective(tmp_path):
    x = cp.intvar(0, 4, shape=3, name="obj")
    model = cp.Model(x[0] <= x[1])
    model.minimize(cp.sum(x))
    path = tmp_path / "sum_objective.xml"
    write_xcsp3(model, path)
    obj = ET.parse(path).getroot().find("./objectives/minimize")
    assert obj.get("type") == "sum"
    assert len(obj.find("list").text.split()) == 3


@pytest.mark.parametrize("kind,expected", [("min", "minimum"), ("max", "maximum")])
def test_specialized_extremum_objective(tmp_path, kind, expected):
    x = cp.intvar(0, 4, shape=3, name="obj")
    model = cp.Model(x[0] <= x[1])
    model.minimize(cp.Minimum(x) if kind == "min" else cp.Maximum(x))
    path = tmp_path / "extremum.xml"
    write_xcsp3(model, path)
    assert ET.parse(path).getroot().find("./objectives/minimize").get("type") == expected


def test_ordered_with_lengths_structure():
    x, y = _fake_var("x"), _fake_var("y")
    root = ET.Element("constraints")
    writer._ordered_constraint(root, _node("increasing", x, y, lengths=[2]),
                               {id(x): "x", id(y): "y"})
    assert root.find("./ordered/lengths").text == "2"
    assert root.find("./ordered/operator").text == "le"


def test_ordered_rejects_wrong_lengths():
    with pytest.raises(ValueError, match="lengths"):
        writer._ordered_constraint(ET.Element("constraints"),
                                   _node("increasing", 1, 2, lengths=[1, 2]), {})


def test_registry_registers_new_variables_without_collision():
    x = _fake_var("x")
    registry = writer._VariableRegistry([x])
    y = _fake_var("x")
    registry.register([x, y])
    assert len(registry.variables) == 2
    assert registry[id(x)] != registry[id(y)]


def test_decomposition_with_auxiliary_variables(tmp_path):
    x = cp.intvar(0, 5, name="x")
    aux = cp.intvar(0, 5, name="aux")

    class CustomConstraint:
        name = "custom_auxiliary"
        args = ()

        def decompose(self):
            return [aux == x], [aux >= 1]

    # Check the lower-level serializer with the same registry used by the writer.
    registry = writer._VariableRegistry([x])
    root = ET.Element("constraints")
    writer._decomposed_constraint(root, CustomConstraint(), registry)
    assert id(aux) in registry
    assert len(root.findall("intension")) == 2


# Importer parity extensions: multidimensional noOverlap and cumulative operators.
@pytest.mark.parametrize("dimensions", [1, 2, 3, 4])
def test_no_overlap_nd_dimensions(dimensions):
    origins = [[i + d for d in range(dimensions)] for i in (0, 5)]
    lengths = [[1] * dimensions, [2] * dimensions]
    root = ET.Element("constraints")
    writer._constraint(root, _node("no_overlap_nd", origins, lengths), {})
    xml = root.find("noOverlap")
    assert xml is not None
    assert len(xml.find("origins").text.split()) == 2
    assert xml.find("lengths").text.count("(") == 2


def test_no_overlap_nd_explicit_ends():
    root = ET.Element("constraints")
    writer._constraint(root, _node("no_overlap_nd", [[0, 1]], [[2, 3]], [[2, 4]]), {})
    assert len(root.findall("intension")) == 2


@pytest.mark.parametrize("bad", [([], []), ([[0, 1]], [[2]]), ([[0]], [[1], [2]])])
def test_no_overlap_nd_invalid_shapes(bad):
    with pytest.raises(ValueError):
        writer._constraint(ET.Element("constraints"), _node("no_overlap_nd", *bad), {})


def test_no_overlap_nd_invalid_ends():
    with pytest.raises(ValueError, match="ends"):
        writer._constraint(ET.Element("constraints"),
                           _node("no_overlap_nd", [[0, 1]], [[1, 1]], [[1]]), {})


@pytest.mark.parametrize("operator,expected", [("<", "lt"), ("<=", "le"),
                                                ("==", "eq"), ("!=", "ne"),
                                                (">=", "ge"), (">", "gt")])
def test_cumulative_condition_variants(operator, expected):
    root = ET.Element("constraints")
    writer._constraint(root, _node("cumulative", [0], [1], [2], 3,
                                   condition=(operator, 3)), {})
    assert root.find("cumulative/condition").text == f"({expected},3)"


@pytest.mark.parametrize("condition", [("bad", 2), "bad"])
def test_cumulative_condition_invalid(condition):
    with pytest.raises(NotImplementedError):
        writer._constraint(ET.Element("constraints"),
                           _node("cumulative", [0], [1], [2], 3,
                                 condition=condition), {})

# Coverage of auxiliary identifier collisions and specialized weighted objectives.
def test_registry_generated_auxiliary_name_collision():
    original = _fake_var('original')
    reserved = _fake_var('aux_2')
    auxiliary = _fake_var('invalid[0]')
    registry = writer._VariableRegistry([original, reserved])
    registry.register([auxiliary])
    assert registry[id(auxiliary)] == 'aux_2_'
    assert len(set(registry.values())) == 3
    assert registry.variables[-1] is auxiliary


def test_no_overlap_nd_rejects_invalid_arity():
    with pytest.raises(ValueError, match='requires origins and lengths'):
        writer._no_overlap_nd(ET.Element('constraints'), _node('no_overlap_nd', [[0, 1]]), {})


def test_weighted_objective_specialized_xml():
    objective = _node('wsum', [2, -3], [4, 5])
    root = ET.Element('minimize')
    writer._objective_xml(root, objective, {})
    assert root.get('type') == 'sum'
    assert root.find('list').text == '4 5'
    assert root.find('coeffs').text == '2 -3'


def test_weighted_objective_rejects_mismatched_coefficients():
    root = ET.Element('minimize')
    with pytest.raises(ValueError, match='Malformed weighted objective'):
        writer._objective_xml(root, _node('wsum', [2, 3], [4]), {})

# Regression coverage for nested Element expressions and reversed comparisons.
def test_nested_element_expression_fallback():
    idx = _fake_var('idx', 0, 2)
    names = {id(idx): 'idx'}
    expression = writer._expr(_node('element', [4, 5, 6], idx), names)
    assert expression == 'if(eq(idx,0),4,if(eq(idx,1),5,6))'


def test_element_comparison_rejects_wrong_arity():
    parent = ET.Element('constraints')
    assert writer._element_comparison(parent, _node('==', 1), {}) is False
    assert list(parent) == []


@pytest.mark.parametrize('operator,expected', [
    ('<', 'gt'), ('<=', 'ge'), ('>', 'lt'), ('>=', 'le'),
    ('==', 'eq'), ('!=', 'ne'),
])
def test_element_comparison_reversed_operand(operator, expected):
    idx = _fake_var('idx', 0, 1)
    x = _fake_var('x')
    y = _fake_var('y')
    names = {id(idx): 'idx', id(x): 'x', id(y): 'y'}
    element = _node('element', [x, y], idx)
    parent = ET.Element('constraints')
    assert writer._element_comparison(parent, _node(operator, 2, element), names)
    result = parent.find('element')
    assert result is not None
    assert result.find('list').text == 'x y'
    assert result.find('index').text == 'idx'
    assert result.find('condition').text == f'({expected},2)'


@pytest.mark.parametrize("op,expected", [
    ("in", "eq(2,2)"),
    ("notin", "ne(2,2)"),
])
def test_importer_membership_singleton_is_primitive(op, expected):
    assert writer._expr(_node(op, 2, [2]), {}) == expected


def test_importer_membership_large_set_no_set_ast():
    x = _fake_var("member")
    names = {id(x): "member"}
    expression = writer._expr(_node("notin", x, [3, 0, 2, 3, 1]), names)
    assert expression == "and(ne(member,0),ne(member,1),ne(member,2),ne(member,3))"
    assert "set(" not in expression


def test_allequal_single_element_serialization():
    from cpmpy import intvar
    from cpmpy.expressions.globalconstraints import AllEqual
    from cpmpy.tools.xcsp3.exporter import _expr

    x = intvar(0, 3, name="single_x")
    assert _expr(AllEqual([x]), {id(x): "single_x"}) == "1"

# ElementMatrix validation paths: each failure must leave the XML tree unchanged.
@pytest.mark.parametrize('matrix', [[], [[]], [[1, 2], [3]]])
def test_element_matrix_rejects_invalid_shape(matrix):
    row, col = _fake_var('row', 0, 0), _fake_var('col', 0, 0)
    parent = ET.Element('constraints')
    with pytest.raises(ValueError, match='nonempty rectangular matrix'):
        writer._element_matrix_constraint(parent, _node('element_matrix', matrix, row, col, '==', 1),
                                          {id(row): 'row', id(col): 'col'})
    assert len(parent) == 0


@pytest.mark.parametrize('row_bounds,col_bounds', [
    ((-1, 1), (0, 1)), ((0, 2), (0, 1)),
    ((0, 1), (-1, 1)), ((0, 1), (0, 2)),
])
def test_element_matrix_rejects_out_of_range_indices(row_bounds, col_bounds):
    row = _fake_var('row', *row_bounds)
    col = _fake_var('col', *col_bounds)
    parent = ET.Element('constraints')
    with pytest.raises(NotImplementedError, match='in-range variable indices'):
        writer._element_matrix_constraint(parent,
            _node('element_matrix', [[0, 1], [2, 3]], row, col, '==', 1),
            {id(row): 'row', id(col): 'col'})
    assert len(parent) == 0


def test_element_matrix_rejects_unsupported_operator():
    row, col = _fake_var('row', 0, 1), _fake_var('col', 0, 1)
    parent = ET.Element('constraints')
    with pytest.raises(NotImplementedError, match='Unsupported ElementMatrix operator'):
        writer._element_matrix_constraint(parent,
            _node('element_matrix', [[0, 1], [2, 3]], row, col, 'invalid', 1),
            {id(row): 'row', id(col): 'col'})
    assert len(parent) == 0


@pytest.mark.parametrize('shape', [(2, 2), (2, 3), (3, 2)])
@pytest.mark.parametrize('operator', ['==', '!=', '<', '<=', '>', '>='])
@pytest.mark.parametrize('element_on_right', [False, True])
def test_internal_nd_element_comparison_dispatch(shape, operator, element_on_right):
    rows, cols = shape
    row = _fake_var('row', 0, rows - 1)
    col = _fake_var('col', 0, cols - 1)
    matrix = np.arange(rows * cols).reshape(shape)
    element = _node('nd_element', matrix, row, col)
    comparison = (_node(operator, 1, element) if element_on_right
                  else _node(operator, element, 1))
    parent = ET.Element('constraints')
    assert writer._element_comparison(parent, comparison,
                                      {id(row): 'row', id(col): 'col'})
    if rows == cols:
        assert parent.find('./element/matrix') is not None
        assert parent.find('./element/index').text == 'row col'
    else:
        assert parent.find('./intension') is not None
        assert 'eq(row,0)' in parent.find('./intension').text
        assert 'eq(col,0)' in parent.find('./intension').text


@pytest.mark.parametrize('invalid', ['wrong_dimension', 'wrong_indices', 'out_of_range'])
def test_internal_nd_element_comparison_fallback(invalid):
    row, col = _fake_var('row', 0, 1), _fake_var('col', 0, 1)
    if invalid == 'wrong_dimension':
        element = _node('nd_element', np.array([1, 2]), row, col)
    elif invalid == 'wrong_indices':
        element = _node('nd_element', np.array([[1, 2], [3, 4]]), row)
    else:
        col = _fake_var('col', 0, 2)
        element = _node('nd_element', np.array([[1, 2], [3, 4]]), row, col)
    parent = ET.Element('constraints')
    assert writer._element_comparison(parent, _node('==', element, 1),
                                      {id(row): 'row', id(col): 'col'}) is False
    assert len(parent) == 0
