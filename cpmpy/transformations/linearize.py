"""
Transformations regarding linearization of constraints.

Linearized constraints have one of the following forms:

NumExpr >=< Constant
NumExpr >=< Var

(NumExpr >=< Constant) == Var
(NumExpr >=< Var) == Var

Var -> Boolexpr

Numexpr:

        - Operator (non-Boolean) with all args Var/constant (examples: +,*,/,mod,wsum)
                                                           (CPMpy class 'Operator', not is_bool())
        - Global constraint (non-Boolean) (examples: Max,Min,Element)
                                                           (CPMpy class 'GlobalConstraint', not is_bool()))


This file implements:
    - linearize_constraint
    - no_negation


"""
import numpy as np

from ..expressions.core import Comparison, Operator, Expression
from ..expressions.globalconstraints import GlobalConstraint, AllDifferent
from ..expressions.utils import is_any_list, is_num
from ..expressions.variables import _BoolVarImpl, boolvar, NegBoolView, intvar, _NumVarImpl
from ..transformations.flatten_model import flatten_constraint, get_or_make_var, negated_normal

M = int(10e10)  # Arbitrary VERY large number


def linearize_constraint(cpm_expr):
    """
    Transforms all constraints to a linear form.
    This function assumes all constraints are in 'flat normal form'. Hence only apply after 'flatten()'.
    """

    if is_any_list(cpm_expr):
        lin_cons = [linearize_constraint(expr) for expr in cpm_expr]
        return [c for l in lin_cons for c in l]

    if isinstance(cpm_expr, _BoolVarImpl):
        if isinstance(cpm_expr, NegBoolView):
            return [cpm_expr._bv <= 0]
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
        assert len(cpm_expr.args) == 2, "Only supports xor with 2 vars"
        if all(arg.is_bool() for arg in cpm_expr.args):
            return [sum(cpm_expr.args) == 1]
        raise Exception("Numeric constants or numeric variables not allowed as base constraint")

    if cpm_expr.name == "->":
        cond, expr = cpm_expr.args
        if not cond.is_bool() or not expr.is_bool():
            raise Exception(
                f"Numeric constants or numeric variables not allowed as base constraint, cannot linearize {cpm_expr}")
        if isinstance(expr, _BoolVarImpl) and not isinstance(cond, _BoolVarImpl):
            # BE -> BV => ~BV -> BE
            return linearize_constraint((~expr).implies(negated_normal(cond)))

        lin_expr = linearize_constraint(expr)
        assert len(lin_expr) == 1, f"Not a supported operator to linearize: {expr} in constraint {cpm_expr}"
        lin_expr = lin_expr[0]

        # Bring all variables to left side of expr
        lhs, rhs = lin_expr.args
        new_var, cons = get_or_make_var(-rhs)
        return cons + [cond.implies(Comparison(lin_expr.name, lhs+new_var, 0))]


    # Binary operators
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
        lhs, cons_lhs = get_or_make_var(lhs)

        c1 = M + lhs - 1 >= rhs
        c2 = - Mz + lhs + 1 + M <= rhs

        return cons_Mz + cons_lhs + flatten_constraint([c1, c2])

    if cpm_expr.name in [">=", "<=", "=="] and isinstance(cpm_expr, Operator) and cpm_expr.args[0].name == "mul":
        if all(isinstance(arg, _BoolVarImpl) for arg in cpm_expr.args[0].args):
            return [Comparison(cpm_expr.name, sum(cpm_expr.args[0].args), cpm_expr.args[1])]

        lhs, rhs = cpm_expr.args
        var, constraints = get_or_make_var(lhs.args[0] * lhs.args[1])
        for arg in lhs.args[2:]:
            var, cons = get_or_make_var(var * arg)
            constraints += cons

        return constraints + [Comparison(cpm_expr.name, var, rhs)]

    gen_constr = ["and","or","min","max","abs","pow"] #General constraints supported by gurobi, TODO: make composable for other solvers
    if cpm_expr.name == '==':
        lhs, rhs = cpm_expr.args
        if lhs.is_bool() and not isinstance(lhs, _BoolVarImpl) and isinstance(rhs, _BoolVarImpl):
            # BE == BV :: ~BV -> ~BE, BV -> BE
            if lhs.name in gen_constr:
                return [cpm_expr]
            else:
                c1 = (~rhs).implies(negated_normal(lhs))
                c2 = rhs.implies(lhs)
                return linearize_constraint([c1,c2])

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

        constraints = [sum(row) == 1 for row in sigma]  # Exactly one value
        constraints += [sum(col) <= 1 for col in sigma.T]  # All diff values

        for arg, row in zip(cpm_expr.args, sigma):
            constraints += [sum(np.arange(lb, ub + 1) * row) == arg]

        return constraints

    return [cpm_expr]


def no_negation(cpm_expr):
    """
        Replaces all occurences of ~BV with 1 - BV in the expression.
        cpm_expr is expected to be linearized. Hence, only apply after applying linearize_constraint(cpm_expr)
    """

    if is_any_list(cpm_expr):
        nn_cons = [no_negation(expr) for expr in cpm_expr]
        return [c for l in nn_cons for c in l]

    def simplify(cpm_var):
        # TODO: come up with a better name for this function
        if isinstance(cpm_var, NegBoolView):
            return 1 - cpm_var._bv
        return cpm_var

    if isinstance(cpm_expr, Comparison):
        # >!=<
        lhs, rhs = cpm_expr.args
        cons_l, cons_r = [], []

        if isinstance(lhs, _BoolVarImpl):
            # ~BV1 >=< ~BV2 => BV2 >=< BV1
            if isinstance(lhs, NegBoolView) and isinstance(rhs, NegBoolView):
                return [Comparison(cpm_expr.name, rhs._bv, lhs._bv)]
            # ~BV1 >=< BV2 => BV1 + BV2 >=< 1
            if isinstance(lhs, NegBoolView):
                return [Comparison(cpm_expr.name, lhs._bv + rhs, 1)]
            if isinstance(rhs, NegBoolView):
                return [Comparison(cpm_expr.name, lhs + rhs._bv, 1)]
            return [cpm_expr]


        if isinstance(lhs, Operator):
            # sum, wsum, and, or, min, max, abs, mul, div, pow
            if lhs.name == "sum" or lhs.name == "wsum":
                if lhs.name == "sum":
                    # Bring negative variables to other side
                    new_lhs = sum(arg for arg in lhs.args if not isinstance(arg, NegBoolView))
                    new_rhs = sum(arg._bv for arg in lhs.args if isinstance(arg, NegBoolView))
                else: #wsum
                    new_lhs = sum((weight * var) for weight, var in zip(*lhs.args) if not isinstance(var, NegBoolView))
                    new_rhs = sum((weight * var._bv) for weight, var in zip(*lhs.args) if isinstance(var, NegBoolView))

                if isinstance(rhs, NegBoolView):
                    new_lhs += rhs._bv
                else:
                    new_rhs += rhs

                if (not isinstance(new_rhs, Operator) or len(new_rhs.args) > 1) \
                        and (not isinstance(new_lhs, Operator) or len(new_lhs.args) <= 1):
                    # Left hand side only contained negative values
                    return [Comparison(cpm_expr.name, new_rhs, new_lhs)]
                new_var, cons = get_or_make_var(new_rhs)
                return cons + [Comparison(cpm_expr.name, new_lhs, new_var)]

            else:
                nn_args = []
                for nn_expr in map(simplify, lhs.args):
                    nn_var, cons = get_or_make_var(nn_expr)
                    cons_l += cons
                    nn_args.append(nn_var)

                nn_lhs = Operator(lhs.name, nn_args)
                nn_rhs, cons_r = get_or_make_var(simplify(rhs))

                return cons_l + cons_r + [Comparison(cpm_expr.name, nn_lhs, nn_rhs)]

    if isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
        cond, expr = cpm_expr.args
        assert isinstance(cond, _BoolVarImpl), f"Left hand side of implication {cpm_expr} should be boolvar"
        assert isinstance(expr, Comparison), f"Right hand side of implication {cpm_expr} should be comparison"
        lhs, rhs = expr.args

        if isinstance(lhs, _NumVarImpl):
            if is_num(rhs):
                return [cpm_expr]
            cons, neg_rhs = get_or_make_var(-rhs)
            return cons + [cond.implies(Comparison(expr.name, lhs + neg_rhs, 0))]

        if isinstance(lhs, Operator) and (lhs.name == "sum" or lhs.name == "wsum"):
            if lhs.name == "sum":
                new_lhs = sum(arg for arg in lhs.args if not isinstance(arg, NegBoolView))
                neg_vars = [arg._bv for arg in lhs.args if isinstance(arg, NegBoolView)]
                new_var, cons = get_or_make_var(-sum(neg_vars))
                new_lhs += new_var
                new_lhs += len(neg_vars)

            if lhs.name == "wsum":
                new_lhs = sum(w * arg for w, arg in zip(*lhs.args) if not isinstance(arg, NegBoolView))
                neg_vars = [(w,arg._bv) for w,arg in zip(lhs.args) if isinstance(arg, NegBoolView)]
                new_var, cons = get_or_make_var(-sum(w * v for w,v in neg_vars))
                new_lhs += new_var
                new_lhs += sum(w for w,_ in neg_vars)

            return cons + [cond.implies(Comparison(expr.name, new_lhs, rhs))]
        else:
            raise NotImplementedError(f"Operator {lhs} is not supported on left right hand side of implication in {cpm_expr}")


    raise Exception(f"{cpm_expr} is not linear or is not supported. Please report on github")














































