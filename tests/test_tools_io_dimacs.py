"""
DIMACS CNF specific tests for ``cpmpy.tools.io.dimacs``.

Generic load/write/round-trip coverage lives in ``test_tools_io.py``. These tests
focus on CNF parser edge cases, p-line validation, writer output shape, etc.
"""

from pathlib import Path

import pytest
import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.transformations.get_variables import get_variables_model

from cpmpy.tools.io.annotate_bool import SugarAnnotator, VeriPBAnnotator
from cpmpy.tools.io.dimacs import load_dimacs, write_dimacs


DATA_DIR = Path(__file__).parent / "data" / "io"
TSEITIN_CNF_PATH = DATA_DIR / "tseitin_n18.cnf"
TSEITIN_CNF = TSEITIN_CNF_PATH.read_text()


def _vars(model):
    return sorted(get_variables_model(model), key=str)


def _writer_model():
    a, b, c = [cp.boolvar(name=n) for n in "abc"]
    return cp.Model(cp.any([a, b, c]), b.implies(~c), a <= 0)


def _assert_unsat(model):
    assert not model.solve()
    assert model.status().exitstatus == ExitStatus.UNSATISFIABLE


def _annotation_names(text):
    return {line.split(maxsplit=2)[2] for line in text.splitlines() if line.startswith("c ")}


# --------------------------------------------------------------------------- #
#                              CNF parsing (load)                             #
# --------------------------------------------------------------------------- #

class TestLoadCNF:

    def test_basic(self):
        model = load_dimacs("p cnf 3 3\n-2 -3 0\n3 2 1 0\n-1 0\n")
        bvs = _vars(model)

        assert str(model) == str(cp.Model(
            cp.any([~bvs[1], ~bvs[2]]),
            cp.any([bvs[2], bvs[1], bvs[0]]),
            ~bvs[0],
        ))

    def test_with_comments(self):
        model = load_dimacs(
            "c starting comment\nc\n\n"
            "p cnf 3 3\n"
            "-2 -3 0\n"
            "c mid comment\n"
            "3 2 1 0\n"
            "-1 0\n"
        )

        assert len(model.constraints) == 3
        bvs = _vars(model)
        solutions = set()
        model.solveAll(display=lambda: solutions.add(tuple(v.value() for v in bvs)))
        assert solutions == {(False, False, True), (False, True, False)}

    @pytest.mark.parametrize(
        "text,n_vars,n_constraints",
        [
            pytest.param("-1 2 0\n-2 3 0\n", 3, 2, id="headerless"),
            pytest.param("p cnf 3 2\n1 2 0 -3 0\n", 3, 2, id="multiple-clauses-one-line"),
            pytest.param("p cnf 3 1\n1 2\n-3 0\n", 3, 1, id="clause-spans-lines"),
        ],
    )
    def test_clause_shape(self, text, n_vars, n_constraints):
        model = load_dimacs(text)
        assert len(get_variables_model(model)) == n_vars
        assert len(model.constraints) == n_constraints

    def test_negative_literal(self):
        model = load_dimacs("p cnf 2 2\n-1 0\n2 0\n")
        bvs = _vars(model)

        assert model.solve()
        assert bvs[0].value() is False
        assert bvs[1].value() is True

    def test_empty_formula(self):
        model = load_dimacs("p cnf 0 0")

        assert model.solve()
        assert model.status().exitstatus == ExitStatus.FEASIBLE

    def test_too_few_variables(self):
        model = load_dimacs("p cnf 2 1\n1 0")

        assert len(model.constraints) == 1
        assert len(get_variables_model(model)) == 1

    def test_empty_clause(self):
        model = load_dimacs("p cnf 2 2\n1 0\n0")
        _assert_unsat(model)

    def test_empty_clauses(self):
        model = load_dimacs("p cnf 0 2\n0\n0")
        assert not model.solve()
        assert model.status().exitstatus == ExitStatus.UNSATISFIABLE

    def test_explicit_type_cnf(self):
        model = load_dimacs("1 2 0\n2 3 0\n", type="cnf")

        assert not model.has_objective()
        assert len(model.constraints) == 2

    @pytest.mark.xfail(
        reason="headerless CNF with only positive first literals is detected as WCNF",
        strict=True,
    )
    def test_headerless_all_positive(self):
        model = load_dimacs("1 2 0\n2 3 0\n")
        assert not model.has_objective()


class TestLoadCNFErrors:

    def test_too_many_variables(self):
        with pytest.raises(AssertionError):
            load_dimacs("p cnf 2 1\n1 2 3 0")

    def test_too_many_clauses(self):
        with pytest.raises(AssertionError):
            load_dimacs("p cnf 2 2\n1 2 0\n1 0\n2 0")

    def test_too_few_clauses(self):
        with pytest.raises(AssertionError):
            load_dimacs("p cnf 2 2\n1 0")

    def test_non_int_literal(self):
        with pytest.raises(ValueError):
            load_dimacs("p cnf 2 1\n1 b 2 0")

    def test_non_terminated_final_clause(self):
        with pytest.raises(AssertionError):
            load_dimacs("p cnf 2 2\n1 2 0\n-1 -2 0\n2")

    def test_clause_count_mismatch_with_late_p_line(self):
        with pytest.raises(AssertionError):
            load_dimacs("-1 2 0\np cnf 2 2")

    def test_unsupported_format_in_p_line(self):
        with pytest.raises(ValueError):
            load_dimacs("p foo 2 2\n1 2 0")

    @pytest.mark.xfail(
        reason="clause-before-p-line input is detected as WCNF instead of failing CNF validation",
        strict=True,
    )
    def test_missing_p_line(self):
        with pytest.raises(AssertionError):
            load_dimacs("1 -2 0\np cnf 2 2")

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            load_dimacs("p cnf 1 1\n1 0", type="xml")


# --------------------------------------------------------------------------- #
#                                Writing (CNF)                                #
# --------------------------------------------------------------------------- #

@pytest.mark.requires_dependency("pindakaas")
class TestWriteCNF:

    def test_string_default(self):
        text = write_dimacs(_writer_model())
        lines = text.splitlines()

        assert not lines[0].startswith("c ")
        assert not any(line.startswith("p ") for line in lines)

    def test_file_default_header(self, tmp_path):
        path = tmp_path / "model.cnf"
        write_dimacs(_writer_model(), path=path)
        lines = path.read_text().splitlines()

        assert lines[0] == "c " + "-" * 100
        assert "File written by CPMpy" in lines[1]
        assert "Format: 'cnf'" in lines[2]

    def test_p_header(self):
        text = write_dimacs(_writer_model(), p_header=True, header="")

        assert text.splitlines()[0] == "p cnf 3 3"

    def test_clause_content(self):
        text = write_dimacs(_writer_model(), header="")

        assert set(text.splitlines()) == {"1 2 3 0", "-2 -3 0", "-1 0"}

    def test_empty_header_omits_comment(self):
        text = write_dimacs(_writer_model(), header="")

        assert not text.startswith("c ")

    def test_custom_header(self):
        text = write_dimacs(_writer_model(), header="line one\nline two")
        lines = text.splitlines()

        assert lines[0] == "c line one"
        assert lines[1] == "c line two"

    def test_negbool_literal(self):
        a, b = cp.boolvar(name="a"), cp.boolvar(name="b")
        text = write_dimacs(cp.Model(cp.any([a, ~b])), p_header=True, header="")
        clause = text.splitlines()[1]

        assert clause.count("-") == 1

    @pytest.mark.parametrize(
        "encoding,annotate,expected",
        [
            pytest.param("direct", SugarAnnotator(), {"px=1", "px=2", "px=3", "b"}, id="sugar-direct"),
            pytest.param("order", SugarAnnotator(), {"px,2", "px,3", "b"}, id="sugar-order"),
            pytest.param("binary", SugarAnnotator(), {"px#0", "px#1", "b"}, id="sugar-binary"),
            pytest.param("direct", VeriPBAnnotator(), {"x_eq_1", "x_eq_2", "x_eq_3", "b"}, id="veripb-direct"),
            pytest.param("order", VeriPBAnnotator(), {"x_ge_2", "x_ge_3", "b"}, id="veripb-order"),
            pytest.param("binary", VeriPBAnnotator(), {"x_bit0", "x_bit1", "b"}, id="veripb-binary"),
        ],
    )
    def test_annotate_writer_comments_for_integer_encodings(self, encoding, annotate, expected):
        x = cp.intvar(1, 3, name="x")
        b = cp.boolvar(name="b")
        model = cp.Model(x >= 2, b)

        text = write_dimacs(model, encoding=encoding, annotate_bool=annotate, header="")

        assert expected <= _annotation_names(text)


# --------------------------------------------------------------------------- #
#                              Real instance                                  #
# --------------------------------------------------------------------------- #

class TestRealInstance:

    def test_tseitin_unsat(self):
        model = load_dimacs(TSEITIN_CNF)

        assert len(get_variables_model(model)) == 27
        assert len(model.constraints) == 72
        _assert_unsat(model)

    def test_tseitin_string_and_file(self):
        from_string = load_dimacs(TSEITIN_CNF)
        from_file = load_dimacs(TSEITIN_CNF_PATH)

        assert len(from_string.constraints) == len(from_file.constraints) == 72
