"""
Linearization of constraints to use in MIP-like solvers

"""
import numpy as np

from ..expressions.core import Comparison, Operator
from ..expressions.globalconstraints import GlobalConstraint, AllDifferent
from ..expressions.utils import is_any_list
from ..expressions.variables import _BoolVarImpl, boolvar, NegBoolView, intvar
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
        # Special case: BV != BV
        if isinstance(lhs, _BoolVarImpl) and isinstance(rhs, _BoolVarImpl):
            return [lhs + rhs == 1]

        # Normal case: big M implementation
        z = boolvar()

        Mz, cons_Mz = get_or_make_var(M * z)
        Mmz, cons_Mmz = get_or_make_var(M - Mz)

        lhs, cons_lhs = get_or_make_var(lhs)

        c1 = Mz + lhs - 1 >= rhs
        c2 = M - Mz + lhs + 1 <= rhs

        return cons_Mz + cons_lhs + [c1, c2]

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


def no_negation(cpm_expr):
    """
        Replaces all occurences of ~BV with 1 - BV in the expression.
        cpm_expr is expected to be linearized.
    """

    if is_any_list(cpm_expr):
        nn_cons = [no_negation(expr) for expr in cpm_expr]
        return [c for l in nn_cons for c in l]


    def simplify(cpm_var):
        # TODO: come up with a better name for this function
        if isinstance(cpm_var, NegBoolView):
            return 1 - cpm_var._bv
        return cpm_var


    if isinstance(cpm_expr, NegBoolView):
        # Base case
        return [1 - cpm_expr._bv]


    if isinstance(cpm_expr, Comparison):

        lhs, rhs = cpm_expr.args
        nn_rhs, cons_r = get_or_make_var(simplify(rhs))

        if isinstance(lhs, Operator):
            # sum, wsum, and, or, min, max, abs, mul, div, pow
            cons_l = []
            if lhs.name == "wsum":
                vars, weights = lhs.args
                nn_args = []
                for nn_expr in map(simplify,vars):
                    nn_var, cons =  get_or_make_var(nn_expr)
                    cons_l += cons
                    nn_args.append(nn_var)
                nn_lhs = Operator("wsum", [vars,weights])

            else:
                nn_args = []
                for nn_expr in map(simplify, lhs.args):
                    nn_var, cons = get_or_make_var(nn_expr)
                    cons_l += cons
                    nn_args.append(nn_var)

                nn_lhs = Operator(lhs.name, nn_args)

        else:
            nn_lhs , cons_l = get_or_make_var(simplify(lhs))

        return cons_l + cons_r + [Comparison(cpm_expr.name, nn_lhs, nn_rhs)]



    return [cpm_expr] # Constant or non negative _BoolVarImpl





if __name__ == "__main__":

    a, b, c = [boolvar(name=n) for n in "abc"]
    i,j,k = [intvar(lb=0, ub=5, name=n) for n in "ijk"]

    expr = a & ~b >= i
    print(expr, no_negation(expr))

    expr = a.implies(~b) >= i
    print(expr, no_negation(expr))

    expr = a <= ~i
    print(expr, no_negation(expr))

    expr = a & ~b >= ~c
    print(expr, no_negation(expr))