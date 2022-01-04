"""
Linearization of constraints to use in MIP-like solvers

"""
import numpy as np

from ..expressions.core import Comparison
from ..expressions.globalconstraints import GlobalConstraint, AllDifferent
from ..expressions.utils import is_any_list
from ..expressions.variables import _BoolVarImpl, boolvar
from ..transformations.flatten_model import flatten_constraint, get_or_make_var


def linearize_constraint(cpm_expr):

    if is_any_list(cpm_expr):
        lin_cons = [linearize_constraint(expr) for expr in cpm_expr]
        return [c for l in lin_cons for c in l]

    if cpm_expr.name == "and":
        if all(arg.is_bool() for arg in cpm_expr.args):
            return [sum(cpm_expr.args) == len(cpm_expr.args)]
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
        rhs_minus_1, cons = get_or_make_var(rhs - 1)
        return cons + [lhs <= rhs_minus_1]

    if cpm_expr.name == ">":
        lhs, rhs = cpm_expr.args
        rhs_plus_1, cons = get_or_make_var(rhs + 1)
        return cons + [lhs >= rhs_plus_1]

    if cpm_expr.name == "!=":
        lhs, rhs = cpm_expr.args
        if isinstance(lhs, _BoolVarImpl) and isinstance(rhs, _BoolVarImpl):
            return [lhs + rhs == 1]
        M = 1e20 # Arbitrary VERY large number
        z = boolvar()

        rhs_plus_1, cons_plus = get_or_make_var(rhs + 1)
        rhs_minus_1, cons_minus = get_or_make_var(rhs - 1)

        c1 = M * z + lhs >= rhs_plus_1
        c2 = -M(1-z) + lhs <= rhs_minus_1
        return [M * z + lhs >= y]

        # TODO fix this, very ugly and inefficient
        return flatten_constraint(linearize_constraint(lhs + 1 <= rhs or lhs >= 1 + rhs))

    if cpm_expr.name == "->":
        lhs, rhs = cpm_expr.args
        if lhs.is_bool() and rhs.is_bool():
            return [lhs <= rhs]
        raise Exception(f"Numeric constants or numeric variables not allowed as base constraint, cannot linearize {cpm_expr}")

    if cpm_expr.name == "alldifferent":
        """
            More efficient implementations possible
            http://yetanothermathprogrammingconsultant.blogspot.com/2016/05/all-different-and-mixed-integer.html
            This method avoids bounds computation
            Introduces n^2 new boolean variables
        """

        # Boolean variables
        if all(isinstance(arg, _BoolVarImpl) for arg in cpm_expr.args):
            lb, ub = 0, 1
        elif all(not isinstance(arg, _BoolVarImpl) for arg in cpm_expr.args):
            # All intvars, check lower and upper bounds
            lb, ub = cpm_expr.args[0].lb, cpm_expr.args[0].ub
            if not all(arg.lb == lb and arg.ub == ub for arg in cpm_expr.args):
                return linearize_constraint(cpm_expr.decompose())
        else:
            # Mix of ints and bools, check lower and upper bounds of ints
            if not all(arg.is_bool() or (arg.lb == 0 and arg.ub == 1) for arg in cpm_expr.args):
                return linearize_constraint(cpm_expr.decompose())
            lb, ub = 0, 1

        sigma = boolvar(shape=(len(cpm_expr.args), 1 + ub - lb))

        constraints = [sum(row) == 1 for row in sigma] # Exactly one value
        constraints += [sum(col) <= 1 for col in sigma.T] # All diff values

        for arg, row in zip(cpm_expr.args, sigma):
            constraints += [sum(np.arange(lb, ub+1) * row) == arg]

        return constraints

    return [cpm_expr]