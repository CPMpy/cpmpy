from multiprocessing.process import parent_process
from cpmpy.tools.io.opb import load_opb, write_opb
import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables_model, get_variables
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
import pytest

OPB_BASIC = "* #variable= 3 #constraint= 1\nmin: +2 x1 +1 x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n"

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




