"""
Linearization of constraints to use in MIP-like solvers

"""
from ..expressions.core import Comparison
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.utils import is_any_list
from ..expressions.variables import _BoolVarImpl
from ..transformations.flatten_model import flatten_constraint

def linearize_constraint(cpm_expr):

    if is_any_list(cpm_expr):
        lin_cons = [linearize_constraint(expr) for expr in cpm_expr]
        return [c for l in lin_cons for c in l]

    if cpm_expr.name == "and":
        if all(arg.is_bool() for arg in cpm_expr.args):
            return [sum(cpm_expr.args) == len(cpm_expr.args)]
        else:
            raise Exception("Numeric constants or numeric variables not allowed as base constraint")

    if cpm_expr.name == "or":
        if all(arg.is_bool() for arg in cpm_expr.args):
            return [sum(cpm_expr.args) >= 1]
        raise Exception("Numeric constants or numeric variables not allowed as base constraint")

    if cpm_expr.name == "xor":
        if all(arg.is_bool() for arg in cpm_expr.args):
            return [sum(cpm_expr.args) == 1]
        raise Exception("Numeric constants or numeric variables not allowed as base constraint")

    #Binary operators
    if cpm_expr.name == "<":
        lhs, rhs = cpm_expr.args
        return [lhs + 1 <= rhs]

    if cpm_expr.name == ">":
        lhs, rhs = cpm_expr.args
        return [rhs + 1 <= lhs]

    if cpm_expr.name == "!=":
        lhs, rhs = cpm_expr.args
        return linearize_constraint(flatten_constraint((lhs >= rhs + 1) | (lhs <= rhs - 1)))

    if cpm_expr.name == "->":
        lhs, rhs = cpm_expr.args
        if lhs.is_bool() and rhs.is_bool():
            return [lhs <= rhs]
        raise Exception(f"Numeric constants or numeric variables not allowed as base constraint, cannot linearize {cpm_expr}")

    return [cpm_expr]


def no_global_constraints(cpm_expr):
    # TODO: maybe not an explicit transformation? Can also be done in the solver itself.
    # OR maybe we put it somewhere else?
    if is_any_list(cpm_expr):
        ng_cons = [no_global_constraints(expr) for expr in cpm_expr]
        return [c for l in ng_cons for c in l]

    if isinstance(cpm_expr, GlobalConstraint):
        return cpm_expr.decompose()

    return [cpm_expr]