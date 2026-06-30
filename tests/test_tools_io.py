from cpmpy.tools.io.writer import write
from cpmpy.tools.io.loader import load, load_formats
from cpmpy.tools.io.opb import load_opb, write_opb
from cpmpy.tools.io.scip import load_scip, write_scip
from cpmpy.tools.io.dimacs import load_dimacs, write_dimacs
import tempfile
import pytest
import cpmpy as cp
from cpmpy.tools.io import write_formats
from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl
from cpmpy.transformations.get_variables import get_variables_model
from cpmpy.tools.io.loader import _get_loader

WRITE_FORMATS = write_formats()
LOAD_FORMATS = load_formats()

OPB_BASIC = "* #variable= 3 #constraint= 1\nmin: +2 x1 +1 x2;\n+1 x1 +2 x2 -1 x3 >= 2;\n"



FORMAT_MODEL_CASES = [
    pytest.param("opb", OPB_BASIC, 3, 0, id="opb"),
]

# @pytest.mark.parametrize("format", LOAD_FORMATS)
# def test_get_loader(format):
#     loader = _get_loader(format)
#     assert loader is not None
#     assert loader == FORMAT_LOADER_WRITER[format][0]

# @pytest.mark.parametrize("format", WRITE_FORMATS)
# def test_write_to_string(format):
#     model = cp.Model([cp.boolvar() >= 0])
#     text = write(model, format=format)
#     assert isinstance(text, str)
#     assert text.strip() != ""

# @pytest.mark.parametrize("format", WRITE_FORMATS)
# def test_write_to_file(format):
#     model = cp.Model([cp.boolvar() >= 0])
#     with tempfile.NamedTemporaryFile(suffix=f".{format}") as temp_file:
#         text_returned = write(model, file_path=temp_file.name, format=format)
#         with open(temp_file.name, "r") as f:
#             text_written = f.read()
#         assert isinstance(text_written, str)
#         assert text_written.strip() != ""
#         assert text_returned == text_written

# @pytest.mark.parametrize("format,model,n_bool_vars, n_int_vars", FORMAT_MODEL_CASES)
# def test_format_autodetect_on_load(format, model, n_bool_vars, n_int_vars):
#     reference_model = load(model, format=format)
#     model = load(model)
#     assert str(model.constraints) == str(reference_model.constraints)
#     assert model.has_objective() == reference_model.has_objective()

# @pytest.mark.parametrize("format,model,n_bool_vars, n_int_vars", FORMAT_MODEL_CASES)
# def test_var_name_counter_updated(format, model, n_bool_vars, n_int_vars):
   
#     model = load(model, format=format)
#     assert len(get_variables_model(model)) == n_bool_vars + n_int_vars
#     with tempfile.NamedTemporaryFile(suffix=f".{format}") as temp_file:
#         write(model, file_path=temp_file.name, format=format)
#         loaded = load(temp_file.name, format=format)
#         assert len(get_variables_model(loaded)) == n_bool_vars + n_int_vars

#     b = cp.boolvar(shape=3)
#     model = cp.Model([b[0] + 2 * b[1] + -1 * b[2] >= 2])
#     model.minimize(2 * b[0] + b[1])
#     with tempfile.NamedTemporaryFile(suffix=f".{format}") as temp_file:
#         write(model, file_path=temp_file.name, format=format)
#         loaded = load(temp_file.name, format=format)
    
#     assert len(get_variables_model(loaded)) == 3
#     assert str(loaded.objective_) == str(model.objective_)
#     assert str(loaded.constraints) == str(model.constraints)
#     assert _BoolVarImpl.counter == n_bool_vars
#     assert _IntVarImpl.counter == n_int_vars



# write global

def generate_basic_csp() -> tuple[cp.Model, list[cp.BoolVar]]:
    model = cp.Model()
    variables = cp.boolvar(shape=3)
    model += cp.sum(variables) >= 2
    return model, list(variables)

def generate_basic_cop() -> tuple[cp.Model, list[cp.BoolVar]]:
    model, variables = generate_basic_csp()
    model.minimize(cp.sum(variables))
    return model, list(variables)

from test_solvers import _get_golomb_model

def generate_golomb_model() -> tuple[cp.Model, list[cp.IntVar]]:
    model, variables = _get_golomb_model(3)
    return model, list(variables)

class TestWriter:

    @pytest.mark.parametrize("generator", [generate_basic_csp, generate_basic_cop, generate_golomb_model])
    @pytest.mark.parametrize("format", ["opb"])
    def test_write_to_string(self, generator, format):
        # Test that the format writer returns a string
        model, variables = generator()
        text = write(model, format=format)
        assert isinstance(text, str)
        assert text.strip() != ""


    @pytest.mark.parametrize("generator", [generate_basic_csp, generate_basic_cop, generate_golomb_model])
    @pytest.mark.parametrize("format", ["opb"])
    def test_write_to_file(self, generator, format):
        # Test that the format writer writes to a file
        model, variables = generator()
        with tempfile.NamedTemporaryFile(suffix=f".{format}") as temp_file:
            text = write(model, path=temp_file.name, format=format, header="") # empty header to make written file identical to string
            with open(temp_file.name, "r") as f:
                text_written = f.read()
            assert isinstance(text_written, str)
            assert text_written.strip() != ""
            assert text == text_written

    @pytest.mark.parametrize("format", ["opb"])
    def test_header(self, format):
        # Test that the format writer accepts and writes the provided header
        model, variables = generate_basic_csp()
        header = "This is a header\n----------------"
        text = write(model, format=format, header=header)
        assert "This is a header" in text
        assert "----------------" in text

# class TestWriteAndLoad:

#     def test_write_and_load(self):
#         model, variables = generate_basic_csp()
#         text = write(model, format="opb")
#         print(text)
#         model_loaded = load(text, format="opb")
#         assert model_loaded is not None
#         assert str(model_loaded.constraints) == str(model.constraints)
#         assert str(model_loaded.objective_) == str(model.objective_)
#         assert str(get_variables_model(model_loaded)) == str(get_variables_model(model))
#         assert model_loaded.solve() == model.solve()



class TestLoader:

    def test_load(problem: str, format: str):
        model = load(problem, format=format)
        assert model is not None

    @pytest.mark.parametrize("format,model,n_bool_vars, n_int_vars", FORMAT_MODEL_CASES)
    def test_load_from_string(self, format, model, n_bool_vars, n_int_vars):
        model = load(model, format=format)
        assert model is not None
        assert len(get_variables_model(model)) == n_bool_vars + n_int_vars
        assert str(model.constraints) == str(model.constraints)
        assert str(model.objective_) == str(model.objective_)
        assert model.solve() == model.solve()

    def test_load_from_file(self, format):
        pass

    def test_load_from_textio(self, format):
        pass

