from ..model import *
from ..expressions import *
from ..variables import *

def cnf_to_pysat(constraints, output=None):
    py_cnf = []

    for ci in constraints:
        formula = []
        # single lit
        if isinstance(ci, Comparison):
            formula.append(- (ci.args[0].name + 1))
        elif isinstance(ci, BoolVarImpl):
            formula.append(ci.args[0].name + 1)
        elif isinstance(ci, Operator):
            for lit in ci.args:
                if isinstance(lit, Comparison):
                    formula.append(-(lit.args[0].name + 1))
                elif isinstance(lit, BoolVarImpl):
                    formula.append(lit.name + 1)
                else:
                    raise f"lit: {lit} in {ci} not handled"
        else:
            raise f"ci: {ci} not handled"
        py_cnf.append(formula)

    cnf_sets = [set(clause) for clause in py_cnf]
    return cnf_sets