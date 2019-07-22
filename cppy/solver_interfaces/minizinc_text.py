# translate expression tree to MiniZinc textual model

from ..expressions import *
from ..variables import *
from ..model_tools.get_variables import get_variables
from . import *

class MiniZincText(SolverInterface):
    # does not do solving itself (you can subclass it)
    # does provide conversion to text model
    def __init__(self):
        self.name = "minizinc_text"

    def solve(self, model):
        print("Model:",self.convert(model))

    def convert(self, model):
        modelvars = get_variables(model)
        txt_vars = "\n"
        for var in modelvars:
            if isinstance(var, BoolVarImpl):
                txt_vars += "var bool: BV{};\n".format(var.id)
            if isinstance(var, IntVarImpl):
                txt_vars += "var {}..{}: IV{};\n".format(var.lb, var.ub, var.id)

        txt_cons = self.convert_constraints(model.constraints)
        txt_obj  = self.convert_objective(model.objective)
                
        return txt_vars+"\n"+txt_cons+txt_obj

    def convert_constraints(self, cons):
        if cons is None:
            return ""
        # stick to default outputs for now...
        out = ""
        for con in cons:
            out += "constraint {};\n".format(con)
        return out+"\n"

    def convert_objective(self, obj):
        if obj is None:
            return "solve satisfy;"
        if not isinstance(obj, Objective):
            raise Exception("Only single objective supported by minizinc_text")

        return "{} {}".format(obj.name.lower(), obj.args)

