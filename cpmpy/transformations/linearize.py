"""
Linearization of constraints to use in MIP-like solvers

"""
from cpmpy.expressions.variables import _BoolVarImpl


def linearize(cpm_expr):

    if cpm_expr.name == "and":
        if all(isinstance(arg, _BoolVarImpl) for arg in cpm_expr.args):
            return sum(cpm_expr.args) >= len(cpm_expr.args)
        else:
            raise NotImplementedError("Linearization of non-boolean and not supported")
    elif cpm_expr.name == "or":
        return sum(cpm_expr.args) >= 1

    elif cpm_expr.name == "<":
        return cpm_expr.args[0] + 1 <= cpm_expr.args[1]

    elif cpm_expr.name == ">":
        return cpm_expr.args[1] + 1 <= cpm_expr.args[0]

    elif cpm_expr.name == "!=":
        lhs, rhs = cpm_expr.args
        return (lhs >= rhs + 1) | (lhs <= rhs - 1)

    raise NotImplementedError(f"Automatic linearization of {cpm_expr} not supported yet, report on github.")
