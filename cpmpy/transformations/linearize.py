"""
Transformations regarding linearization of constraints.

Linearized constraints have one of the following forms:


Linear comparison:
--------------------------
- LinExpr == Constant/Var
- LinExpr >= Constant/Var
- LinExpr <= Constant/Var
# TODO: do we want to put all vars on lhs (so rhs = 0)?

    LinExpr can be any of:
        - NumVarImpl
        - div               (is_num(Operator.args[1])
        - mul               (len(Operator.args) == 2)
        - sum
        - wsum



Indicator constraints
--------------------------
- Boolvar -> LinExpr == Constant
- Boolvar -> LinExpr >= Constant
- Boolvar -> LinExpr <= Constant


General Constraints
---------------------------
GenExpr == var

    GenExpr can be any of: min, max, pow, abs

"""
import numpy as np

from .reification import only_bv_implies
from .flatten_model import flatten_constraint, get_or_make_var, negated_normal

from ..expressions.core import Comparison, Operator, _wsum_should, _wsum_make
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.utils import is_any_list, is_num
from ..expressions.variables import _BoolVarImpl, boolvar, NegBoolView, _NumVarImpl

def linearize_constraint(cpm_expr):
    """
    Transforms all constraints to a linear form.
    This function assumes all constraints are in 'flat normal form'.
    Only apply after 'cpmpy.transformations.flatten_model.flatten_constraint()'.
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
                # multiplication with var and constant, convert to wsum
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
        # TODO: this can be optimized, see https://github.com/CPMpy/cpmpy/pull/93/files#r824163315
        lhs, rhs = cpm_expr.args
        # Special case: BV != BV
        if isinstance(lhs, _BoolVarImpl) and isinstance(rhs, _BoolVarImpl):
            return [lhs + rhs == 1]
        # Normal case: big M implementation
        z = boolvar()

        # Calculate bounds of M = |lhs - rhs| + 1
        bound1, _ = get_or_make_var(1 + lhs - rhs)
        bound2, _ = get_or_make_var(1 + rhs - lhs)
        M = max(bound1.ub, bound2.ub)

        if lhs.name == "sum":
            lhs = Operator("wsum", [[1]*len(lhs.args), lhs.args])
            cons_lhs = []
        if _wsum_should(lhs):
            lhs, cons_lhs = Operator("wsum",_wsum_make(lhs)), []
        if lhs.name != "wsum":
            lhs, cons_lhs = get_or_make_var(lhs)
        # Rewrite of constraints:
        #   c1 = lhs - rhs <= Mz - 1
        #   c2 = lhs - rhs >= -M(1-z) + 1
        c1 = Operator("wsum", [[-M, 1],       [z, 1]]) + lhs <= rhs      # TODO: this can be optimized, see https://github.com/CPMpy/cpmpy/issues/97
        c2 = Operator("wsum", [[-M, (M - 1)], [z, 1]]) + lhs >= rhs      # TODO: this can be optimized, see https://github.com/CPMpy/cpmpy/issues/97

        return linearize_constraint(flatten_constraint(cons_lhs)) + [c1, c2]

    if cpm_expr.name in [">=", "<=", "=="] and \
            isinstance(cpm_expr.args[0], Operator) and cpm_expr.args[0].name == "mul":
        if all(isinstance(arg, _BoolVarImpl) for arg in cpm_expr.args[0].args):
            return [Comparison(cpm_expr.name, sum(cpm_expr.args[0].args), cpm_expr.args[1])]
        lhs, rhs = cpm_expr.args
        if len(lhs.args) == 2:
            # multiplication of var and constant
            if _wsum_should(lhs):
                return [Comparison(cpm_expr.name, Operator("wsum", _wsum_make(lhs)), rhs)]
            return [cpm_expr]

        lhs, rhs = cpm_expr.args
        var, constraints = get_or_make_var(lhs.args[0] * lhs.args[1])
        # decompose long multiplication in several multiplications with two args
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
        lb, ub = min(arg.lb for arg in cpm_expr.args), max(arg.ub for arg in cpm_expr.args)
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

        constraints = [sum(sigma) == 1]
        constraints += [sum(np.arange(n) * sigma) == idx]
        constraints += [Comparison(cpm_expr.name, np.dot(arr,sigma), cpm_expr.args[1])]

        return linearize_constraint(flatten_constraint(constraints))

    return [cpm_expr]


def only_positive_bv(cpm_expr):
    """
        Replaces constraints containing NegBoolView with equivalent expression using only BoolVar.
        cpm_expr is expected to be linearized. Only apply after applying linearize_constraint(cpm_expr)

        Resulting expression is linear.
    """

    if is_any_list(cpm_expr):
        nn_cons = [only_positive_bv(expr) for expr in cpm_expr]
        return [c for l in nn_cons for c in l]

    def simplify(cpm_var):
        # TODO: come up with a better name for this function
        if isinstance(cpm_var, NegBoolView):
            return 1 - cpm_var._bv
        return cpm_var

    if isinstance(cpm_expr, Comparison):
        # >!=<
        lhs, rhs = cpm_expr.args
        if isinstance(lhs, _BoolVarImpl):
            # ~BV1 >=< ~BV2 => BV2 >=< BV1
            if isinstance(lhs, NegBoolView) and isinstance(rhs, NegBoolView):
                return [Comparison(cpm_expr.name, rhs._bv, lhs._bv)]
            # ~BV1 >=< IV => BV1 + IV >=< 1
            if isinstance(lhs, NegBoolView):
                return [Comparison(cpm_expr.name, lhs._bv + rhs, 1)]
            # BV1 >=< ~BV2 => BV1 + BV2 >=< 1
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

            nn_lhs = type(lhs)(nn_args)
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

            nn_subsexpr = only_positive_bv(subexpr)
            return linearize_constraint([cond.implies(nn_expr) for nn_expr in nn_subsexpr])
        else:
            raise NotImplementedError(f"Operator {lhs} is not supported on left right hand side of implication in {cpm_expr}")

    if isinstance(cpm_expr, GlobalConstraint):
        return [cpm_expr]

    raise Exception(f"{cpm_expr} is not linear or is not supported. Please report on github")














































