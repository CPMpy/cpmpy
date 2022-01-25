"""
Transformations regarding linearization of constraints.

Linearized constraints have one of the following forms:


Linear comparison:
--------------------------
- LinExpr == Constant/Var
- LinExpr >= Constant/Var
- LinExpr <= Constant/Var
# TODO: do we want to put all vars on lhs (so rhs = 0)?

    LinExpr can be any of  NumVarImpl, div, mul, sum or wsum


Indicator constraints
--------------------------
- Boolvar -> LinExpr


General Constraints
---------------------------
GenExpr == var

    GenExpr can be any of: min, max, pow, abs

"""
import numpy as np

from .reification import only_bv_implies
from ..expressions.core import Comparison, Operator, Expression, _wsum_should, _wsum_make
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
            return [sum(cpm_expr.args) >= len(cpm_expr.args)]
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
        cond, sub_expr = cpm_expr.args
        var_cons = []
        if not cond.is_bool() or not sub_expr.is_bool():
            raise Exception(
                f"Numeric constants or numeric variables not allowed as base constraint, cannot linearize {cpm_expr}")
        lin_comps = linearize_constraint(sub_expr)
        lin_exprs = []
        for l_expr in lin_comps:
            lhs, rhs = l_expr.args
            if lhs.name == "mul" and _wsum_should(lhs):
                lhs = Operator("wsum", _wsum_make(lhs))

            if isinstance(rhs, _NumVarImpl):
                # Vars on rhs of linexpr not supported in indicator constraints
                if isinstance(lhs, _NumVarImpl):
                    new_lhs = lhs + (-1) * rhs
                elif lhs.name == "sum":
                    new_lhs = Operator("wsum", [[1]*len(lhs.args) + [-1], lhs.args + [rhs]])
                elif lhs.name == "wsum":
                    new_lhs = lhs + (-1 * rhs)
                else:
                    raise Exception(f"Unsupported expression {lhs} on right hand side of implication {l_expr}, resulting from linearization of {cpm_expr}")
                lin_exprs += [Comparison(l_expr.name, new_lhs, 0)]
            else:
                lin_exprs += [l_expr]

        return var_cons + [cond.implies(l_expr) for l_expr in lin_exprs]

    # Binary operators
    if cpm_expr.name == "<":
        # TODO: this can be optimized, see https://github.com/CPMpy/cpmpy/issues/97
        lhs, rhs = cpm_expr.args
        if lhs.name == "wsum":
            return [Operator("wsum", [lhs.args[0] + [1], lhs.args[1] + [1]]) <= rhs]
        else:
            cons = lhs + 1 <= rhs
        cons = flatten_constraint(cons)
        cons = linearize_constraint(cons)
        return cons

    if cpm_expr.name == ">":
        # TODO: this can be optimized, see https://github.com/CPMpy/cpmpy/issues/97
        lhs, rhs = cpm_expr.args
        if lhs.name == "wsum":
            return [Operator("wsum", [lhs.args[0] + [-1], lhs.args[1] + [1]]) >= rhs]
        else:
            cons = lhs - 1 >= rhs
        cons = flatten_constraint(cons)
        cons = linearize_constraint(cons)
        return cons

    if cpm_expr.name == "!=":
        lhs, rhs = cpm_expr.args
        # Special case: BV != BV
        if isinstance(lhs, _BoolVarImpl) and isinstance(rhs, _BoolVarImpl):
            return [lhs + rhs == 1]
        # Normal case: big M implementation
        z = boolvar()
        # TODO: dynamically calculate M!!
        Mz, cons_Mz = get_or_make_var(M * z)
        lhs, cons_lhs = get_or_make_var(lhs)

        c1 = M + lhs + -1 >= rhs
        c2 = - Mz + lhs + 1 <= rhs

        return cons_Mz + cons_lhs + flatten_constraint([c1, c2])

    if cpm_expr.name in [">=", "<=", "=="] and \
            isinstance(cpm_expr.args[0], Operator) and cpm_expr.args[0].name == "mul":
        if all(isinstance(arg, _BoolVarImpl) for arg in cpm_expr.args[0].args):
            return [Comparison(cpm_expr.name, sum(cpm_expr.args[0].args), cpm_expr.args[1])]

        lhs, rhs = cpm_expr.args
        if len(lhs.args) == 2:
            if _wsum_should(lhs):
                # Convert to wsum
                return [Comparison(cpm_expr.name, Operator("wsum", _wsum_make(lhs)), rhs)]
            return [cpm_expr]

        lhs, rhs = cpm_expr.args
        var, constraints = get_or_make_var(lhs.args[0] * lhs.args[1])
        for arg in lhs.args[2:]:
            var, cons = get_or_make_var(var * arg)
            constraints += cons

        return constraints + [Comparison(cpm_expr.name, var, rhs)]

    gen_constr = ["min","max","abs","pow"] #General constraints supported by gurobi, TODO: make composable for other solvers
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

        # Linear decomposition of alldifferent using bipartite matching
        sigma = boolvar(shape=(len(cpm_expr.args), 1 + ub - lb))

        constraints = [sum(row) == 1 for row in sigma]  # Exactly one value
        constraints += [sum(col) <= 1 for col in sigma.T]  # All diff values

        for arg, row in zip(cpm_expr.args, sigma):
            constraints += [sum(np.arange(lb, ub + 1) * row) == arg]

        return constraints

    if cpm_expr.name in [">=", "<=", "=="] and cpm_expr.args[0].name == "element":
        """
            Decomposition of element constraint according to:
            ```
            Refalo, P. (2000, September). Linear formulation of constraint programming models and hybrid solvers. 
            In International Conference on Principles and Practice of Constraint Programming (pp. 369-383). Springer, Berlin, Heidelberg.
            ```            
        """
        arr, idx = cpm_expr.args[0].args
        # Assuming 1-d array
        assert len(arr.shape) == 1, f"Only support 1-d element constraints, not {cpm_expr} which has shape {cpm_expr.shape}"

        n = len(arr)
        sigma = boolvar(shape=n)

        constraints  = [sum(sigma) == 1]
        constraints += [sum(np.arange(n) * sigma) == idx]
        constraints += [Comparison(cpm_expr.name, np.dot(arr,sigma), cpm_expr.args[1])]

        return linearize_constraint(flatten_constraint(constraints))





    if isinstance(cpm_expr, GlobalConstraint):
        return linearize_constraint(only_bv_implies(flatten_constraint(cpm_expr.decompose())))

    return [cpm_expr]


def is_lin(cpm_expr):
    """
        Returns whether cmp_expr is a linear constraint.
    """

    if isinstance(cpm_expr, Comparison):
        lhs, rhs = cpm_expr.args
        return isinstance(lhs, _NumVarImpl) or lhs.name == "sum" or lhs.name == "wsum"

    if isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
        cond, subsexpr = cpm_expr.args
        return isinstance(cond, _BoolVarImpl) and is_lin(subsexpr) and is_num(subsexpr.args[1])


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

        if isinstance(lhs, _NumVarImpl):
            return [cpm_expr]

        if isinstance(lhs, Operator) and lhs.name == "sum" or lhs.name == "wsum":
            # sum, wsum
            if lhs.name == "sum":
                if not any(isinstance(arg, NegBoolView) for arg in lhs.args):
                    # No NegBoolViews in sum, return
                    return [cpm_expr]
                # Convert to wsum
                lhs = Operator("wsum", [[1] * len(lhs.args), lhs.args])

            if lhs.name == "wsum":
                # TODO: can be optimized with sum/wsum improvements, see: https://github.com/CPMpy/cpmpy/issues/97
                weights, vars = [],[]
                for w, arg in zip(*lhs.args):
                    if isinstance(arg, NegBoolView):
                        weights += [w,-w]
                        vars += [1, arg._bv]
                    else:
                        weights += [w]
                        vars += [arg]
                new_lhs = Operator("wsum", [weights, vars])

            if isinstance(rhs, NegBoolView):
                new_lhs += 1 * rhs._bv
                new_rhs = 1
            else:
                new_rhs = rhs

            return [Comparison(cpm_expr.name, new_lhs, new_rhs)]

        if isinstance(lhs, Operator):
            # min, max
            simplied_args = [simplify(arg) for arg in lhs.args]
            nn_args, cons_l = zip(*[get_or_make_var(arg) for arg in simplied_args])

            nn_lhs = Operator(lhs.name, nn_args)
            nn_rhs, cons_r = get_or_make_var(simplify(rhs))

            var_cons = [c for con in cons_l for c in con] + [c for con in cons_r for c in con]
            return var_cons + [Comparison(cpm_expr.name, nn_lhs, nn_rhs)]

        if isinstance(lhs, GlobalConstraint):
            # min, max
            simplied_args = [simplify(arg) for arg in lhs.args]
            nn_args, cons_l = zip(*[get_or_make_var(arg) for arg in simplied_args])

            nn_lhs = GlobalConstraint(lhs.name, nn_args)
            nn_rhs, cons_r = get_or_make_var(simplify(rhs))

            var_cons = [c for con in cons_l for c in con] + [c for con in cons_r for c in con]
            return var_cons + [Comparison(cpm_expr.name, nn_lhs, nn_rhs)]


    if isinstance(cpm_expr, Operator) and cpm_expr.name == "->":

        cond, subexpr = cpm_expr.args
        assert isinstance(cond, _BoolVarImpl), f"Left hand side of implication {cpm_expr} should be boolvar"
        assert isinstance(subexpr, Comparison), f"Right hand side of implication {cpm_expr} should be comparison"
        lhs, rhs = subexpr.args

        if isinstance(lhs, _NumVarImpl):
            if is_num(rhs):
                return [cpm_expr]
            new_lhs = lhs + -1 * rhs
            return [cond.implies(Comparison(subexpr.name, new_lhs, 0))]

        if isinstance(lhs, Operator) and (lhs.name == "sum" or lhs.name == "wsum"):

            nn_subsexpr = no_negation(subexpr)
            return linearize_constraint([cond.implies(nn_expr) for nn_expr in nn_subsexpr])
        else:
            raise NotImplementedError(f"Operator {lhs} is not supported on left right hand side of implication in {cpm_expr}")

    raise Exception(f"{cpm_expr} is not linear or is not supported. Please report on github")














































