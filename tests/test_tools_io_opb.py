import re
import pytest

import cpmpy as cp
from cpmpy.tools.io.opb import load_opb, write_opb
from cpmpy.tools.io.annotate_bool import SugarAnnotator, VeriPBAnnotator
from cpmpy.transformations.get_variables import get_variables_model
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView

def _assert_names_in_opb(text, expected_names):
    names = {token.removeprefix("~") for token in re.findall(r"[+-]\d+\s+(~?\S+)", text)}
    assert expected_names <= names

class TestLoadOPB:

    def setup_method(self):
        pass

    def test_wrong_header(self):
        with pytest.raises(ValueError):
            load_opb("min: +2 x1 +1 x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n")
        with pytest.raises(ValueError):
            load_opb("* #variable= 2 #constraint= 1\nmin: +2 x1 +1 x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n")
        with pytest.raises(ValueError):
            load_opb("* #variable= 4 #constraint= 1\nmin: +2 x1 +1 x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n")
        with pytest.raises(ValueError):
            load_opb("* #variable= 3 #constraint= 2\nmin: +2 x1 +1 x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n")
        with pytest.raises(ValueError):
            load_opb("* #variable= 3 #constraint= 0\nmin: +2 x1 +1 x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n")
        with pytest.raises(ValueError):
            load_opb("* #variable= 3 #constraint= 1\nmin: +2 x1 +1 x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n+1 x1 +2 x2 -1 x3 >= 2;\n")
        with pytest.raises(ValueError):
            load_opb("* #variable= 3 #constraint= 3\nmin: +2 x1 +1 x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n+1 x1 +2 x2 -1 x3 >= 2;\n")

    def test_parse_objective(self):
        m = load_opb("* #variable= 3 #constraint= 1\nmin: +2 x1 +1 x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n")
        assert m.has_objective()
        assert str(m.objective_) == "sum([2, 1] * [x1, x2])"

        m = load_opb("* #variable= 3 #constraint= 1\nmin: -2 x1 -1 ~x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n")
        assert m.has_objective()
        assert str(m.objective_) == "sum([-2, -1] * [x1, ~x2])"

    def test_parse_constraint(self):
        m = load_opb("* #variable= 3 #constraint= 1\nmin: +2 x1 +1 x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n")
        assert len(m.constraints) == 1
        assert str(m.constraints[0]) == "sum([1, 2, -1] * [x1, x2, x3]) >= 2"

        m = load_opb("* #variable= 3 #constraint= 1\nmin: +2 x1 +1 x2;\n+1 x1 +2 ~x2 -1 ~x3 >= 2;\n")
        assert len(m.constraints) == 1
        assert str(m.constraints[0]) == "sum([1, 2, -1] * [x1, ~x2, ~x3]) >= 2"

        m = load_opb("* #variable= 3 #constraint= 2\nmin: +2 x1 +1 x2;\n+1 x1 +2 ~x2 -1 ~x3 >= 2;\n+1 x1 +2 x2 -1 x3 >= 2;\n")
        assert len(m.constraints) == 2
        assert str(m.constraints[0]) == "sum([1, 2, -1] * [x1, ~x2, ~x3]) >= 2"
        assert str(m.constraints[1]) == "sum([1, 2, -1] * [x1, x2, x3]) >= 2"

        # Unsigned first term
        m = load_opb("* #variable= 2 #constraint= 1\n1 x1 +2 x2 >= 1;\n")
        assert len(m.constraints) == 1
        assert str(m.constraints[0]) == "sum([1, 2] * [x1, x2]) >= 1"

        m = load_opb("* #variable= 3 #constraint= 1\nmin: +1 x1 +1 x2 +1 x3;\n +1 x1 +2 x2 +1 x3 >= 2;\n")
        assert m.constraints[0].name == ">="

        m = load_opb("* #variable= 2 #constraint= 1\n+1 x1 +1 x2 = 1;\n")
        assert m.constraints[0].name == "=="


    def test_variables(self):
        m = load_opb("* #variable= 3 #constraint= 1\nmin: +2 x1 +1 x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n")
        assert len(get_variables_model(m)) == 3
        for v in get_variables_model(m):
            assert v.name in ["x1", "x2", "x3"]
            assert v.get_bounds() == (0, 1)

        m = load_opb("* #variable= 3 #constraint= 1\nmin: +2 x1 +1 x2;\n+1 ~x1 +2 ~x2 -1 ~x3 >= 2;\n")
        constraint = m.constraints[0]
        assert str(constraint) == "sum([1, 2, -1] * [~x1, ~x2, ~x3]) >= 2"
        variables = constraint.args[0].args[1]
        for v in variables:
            assert v.name in ["~x1", "~x2", "~x3"]
            assert v.get_bounds() == (0, 1)
            assert isinstance(v, _BoolVarImpl)
            assert isinstance(v, NegBoolView)


class TestWriteOPB:

    def _int_model(self):
        # Note: bool var name is >= 2 chars so it is a valid unquoted OPB identifier.
        x = cp.intvar(1, 3, name="x")
        b = cp.boolvar(name="bb")
        return cp.Model(x >= 2, b)

    # naming="veripb": descriptive names used directly (VeriPB names are valid identifiers)
    @pytest.mark.parametrize("encoding,expected_names", [
        ("direct", {"x_eq_1", "x_eq_2", "x_eq_3", "bb"}),
        ("order", {"x_ge_2", "x_ge_3", "bb"}),
        ("binary", {"x_bit0", "x_bit1", "bb"}),
    ])
    def test_naming_veripb_uses_descriptive_names(self, encoding, expected_names):
        text = write_opb(self._int_model(), encoding=encoding,
                         annotate_bool=VeriPBAnnotator(), naming="veripb", header="")
        _assert_names_in_opb(text, expected_names)

    def test_naming_veripb_has_no_comment_block(self):
        # In veripb naming the annotation is the variable name itself, so no
        # * name -> annotation comment mapping is written.
        text = write_opb(self._int_model(), encoding="order",
                         annotate_bool=VeriPBAnnotator(), naming="veripb", header="")
        comment_lines = [line for line in text.splitlines()
                         if line.startswith("* ") and "#variable" not in line]
        assert comment_lines == []

    # naming="extended": descriptive names used as quoted variable names
    @pytest.mark.parametrize("encoding,expected_names", [
        ("direct", {'"px=1"', '"px=2"', '"px=3"', '"bb"'}),
        ("order", {'"px,2"', '"px,3"', '"bb"'}),
        ("binary", {'"px#0"', '"px#1"', '"bb"'}),
    ])
    def test_naming_extended_quotes_descriptive_names(self, encoding, expected_names):
        text = write_opb(self._int_model(), encoding=encoding,
                         annotate_bool=SugarAnnotator(), naming="extended", header="")
        _assert_names_in_opb(text, expected_names)

    def test_naming_restricted_uses_ids_and_comments(self):
        text = write_opb(self._int_model(), encoding="order",
                         annotate_bool=VeriPBAnnotator(), naming="restricted", header="")
        # body uses competition-style x<i> identifiers
        _assert_names_in_opb(text, {"x1", "x2", "x3"})
        # descriptive names are recoverable from the * comment block
        comments = {line for line in text.splitlines() if line.startswith("* x")}
        assert any(c.endswith(" x_ge_2") for c in comments)
        assert any(c.endswith(" x_ge_3") for c in comments)
        assert any(c.endswith(" bb") for c in comments)

    def test_naming_veripb_rejects_single_char_name(self):
        # Single-character names are invalid unquoted OPB identifiers.
        model = cp.Model(cp.boolvar(name="b") | cp.boolvar(name="cc"))
        with pytest.raises(ValueError):
            write_opb(model, annotate_bool=VeriPBAnnotator(), naming="veripb", header="")

    def test_naming_restricted_roundtrips_through_load(self):
        p, q, r = [cp.boolvar(name=n) for n in "pqr"]
        model = cp.Model(cp.sum([p, q, r]) >= 2)

        text = write_opb(model, naming="restricted", header="")

        reloaded = load_opb(text)  # * x<i> comment lines are skipped by the loader
        assert len(get_variables_model(reloaded)) == 3
        assert len(reloaded.constraints) == 1

    def test_naming_veripb_rejects_invalid_identifier(self):
        # Sugar order names contain a comma, which is not a valid unquoted OPB name.
        with pytest.raises(ValueError):
            write_opb(self._int_model(), encoding="order",
                      annotate_bool=SugarAnnotator(), naming="veripb", header="")

    def test_unknown_naming_raises(self):
        with pytest.raises(ValueError):
            write_opb(self._int_model(), naming="bogus", header="")


