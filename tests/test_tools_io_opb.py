import re
import pytest

import cpmpy as cp
from cpmpy.tools.io.opb import load_opb, write_opb
from cpmpy.tools.io.annotate import annotate_cpmpy, annotate_sugar, annotate_veripb
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

    @pytest.mark.parametrize("encoding,annotate,expected_names", [
        ("direct", annotate_cpmpy, {"x=1", "x=2", "x=3", "b"}),
        ("order", annotate_cpmpy, {"x>=2", "x>=3", "b"}),
        ("binary", annotate_cpmpy, {"x[bit=0]", "x[bit=1]", "b"}),
        ("direct", annotate_sugar, {"px=1", "px=2", "px=3", "b"}),
        ("order", annotate_sugar, {"px,2", "px,3", "b"}),
        ("binary", annotate_sugar, {"px#0", "px#1", "b"}),
        ("direct", annotate_veripb, {"x_eq_1", "x_eq_2", "x_eq_3", "b"}),
        ("order", annotate_veripb, {"x_ge_2", "x_ge_3", "b"}),
        ("binary", annotate_veripb, {"x_bit0", "x_bit1", "b"}),
    ])
    def test_annotate_writer_names_for_integer_encodings(self, encoding, annotate, expected_names):
        x = cp.intvar(1, 3, name="x")
        b = cp.boolvar(name="b")
        model = cp.Model(x >= 2, b)

        text = write_opb(model, encoding=encoding, annotate=annotate, header="")

        _assert_names_in_opb(text, expected_names)


