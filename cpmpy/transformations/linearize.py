"""
Linearization of constraints to use in MIP-like solvers

"""
import numpy as np

from ..expressions.core import Comparison, Operator
from ..expressions.globalconstraints import GlobalConstraint, AllDifferent
from ..expressions.utils import is_any_list
from ..expressions.variables import _BoolVarImpl, boolvar
from ..transformations.flatten_model import flatten_constraint, get_or_make_var

M = int(10e10)  # Arbitrary VERY large number

def linearize_constraint(cpm_expr):

    if is_any_list(cpm_expr):
        lin_cons = [linearize_constraint(expr) for expr in cpm_expr]
        return [c for l in lin_cons for c in l]

    if isinstance(cpm_expr, _BoolVarImpl):
        return [cpm_expr >= 1]

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

    if cpm_expr.name == "->":
        lhs, rhs = cpm_expr.args
        if not lhs.is_bool() or not rhs.is_bool():
            raise Exception(
                f"Numeric constants or numeric variables not allowed as base constraint, cannot linearize {cpm_expr}")
        if isinstance(rhs, _BoolVarImpl):
            return [lhs <= rhs]
        if isinstance(lhs, _BoolVarImpl):
            return [rhs >= lhs]
        raise Exception(f"{cpm_expr} should be of the form Var -> BoolExpr or BoolExpr -> Var")

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
        z = boolvar()

        Mz, cons_Mz = get_or_make_var(M * z)
        Mmz, cons_Mmz = get_or_make_var(M * (z-1))

        c1 = Mz + lhs - 1 >= rhs
        c2 = Mmz + lhs + 1 <= rhs

        return cons_Mz + cons_Mmz + [c1, c2]

    if cpm_expr.name in [">=", "<=", "=="] and cpm_expr.args[0].name == "mul":
        if all(isinstance(arg, _BoolVarImpl) for arg in cpm_expr.args[0].args):
            return [Comparison(cpm_expr.name, sum(cpm_expr.args[0].args), cpm_expr.args[1])]

        lhs, rhs = cpm_expr.args
        var, constraints = get_or_make_var(lhs.args[0] * lhs.args[1])
        for arg in lhs.args[2:]:
            var, cons = get_or_make_var(var * arg)
            constraints += cons

        return constraints + [Comparison(cpm_expr.name, var, rhs)]


    if cpm_expr.name == "==" and isinstance(cpm_expr.args[0], Comparison):
        lhs, rhs = cpm_expr.args
        llhs, lrhs = lhs.args

        if lhs.name == ">=":
            Mz, cons_Mz = get_or_make_var(M * rhs)
            Mmz, cons_Mmz = get_or_make_var(M * (1 - rhs))
            return [llhs - lrhs <= Mz, lrhs - llhs - 1 <= Mmz]

        elif lhs.name == "<=":
            Mz, cons_Mz = get_or_make_var(M * rhs)
            Mmz, cons_Mmz = get_or_make_var(M * (1 - rhs))
            return [llhs - lrhs >= Mz, lrhs - llhs - 1 >= Mmz]

        elif lhs.name == "==":
            # Model as <= & >=
            z1, z2 = boolvar(shape=2)
            cons = linearize_constraint((llhs >= lrhs) == z1)
            cons += linearize_constraint((llhs <= lrhs) == z2)

            cons += [rhs <= z1, rhs <= z2, z1 + z1 -1 <= rhs]
            return cons
        raise Exception(f"Not a supported expression to linearize {cpm_expr}")



    if cpm_expr.name == "alldifferent":
        """
            More efficient implementations possible
            http://yetanothermathprogrammingconsultant.blogspot.com/2016/05/all-different-and-mixed-integer.html
            This method avoids bounds computation
            Introduces n^2 new boolean variables
        """
        # TODO check performance of implementation
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