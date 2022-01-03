"""
Linearization of constraints to use in MIP-like solvers

"""
import numpy as np

from ..expressions.core import Comparison
from ..expressions.globalconstraints import GlobalConstraint, AllDifferent
from ..expressions.utils import is_any_list
from ..expressions.variables import _BoolVarImpl, boolvar
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
        # TODO implement this --> Ignace
        raise NotImplementedError("!= is not implemented yet: TODO")

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
            constraints += [arg == sum(np.arange(lb, ub+1) * row)]

        return constraints

    return [cpm_expr]