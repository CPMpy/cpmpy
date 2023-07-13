"""
Transformations regarding linearization of constraints.

Linearized constraints have one of the following forms:


Linear comparison:
------------------
- LinExpr == Constant
- LinExpr >= Constant
- LinExpr <= Constant

    LinExpr can be any of:
        - NumVar
        - sum
        - wsum

Indicator constraints:
----------------------
- BoolVar -> LinExpr == Constant
- BoolVar -> LinExpr >= Constant
- BoolVar -> LinExpr <= Constant

- BoolVar -> GenExpr                    (GenExpr.name in supported, GenExpr.is_bool())
- BoolVar -> GenExpr >= Var/Constant    (GenExpr.name in supported, GenExpr.is_num())
- BoolVar -> GenExpr <= Var/Constant    (GenExpr.name in supported, GenExpr.is_num())
- BoolVar -> GenExpr == Var/Constant    (GenExpr.name in supported, GenExpr.is_num())

Where BoolVar is a boolean variable or its negation.

General comparisons or expressions
-----------------------------------
- GenExpr                               (GenExpr.name in supported, GenExpr.is_bool())
- GenExpr == Var/Constant               (GenExpr.name in supported, GenExpr.is_num())
- GenExpr <= Var/Constant               (GenExpr.name in supported, GenExpr.is_num())
- GenExpr >= Var/Constant               (GenExpr.name in supported, GenExpr.is_num())


"""
import copy
import numpy as np

from .flatten_model import flatten_constraint, get_or_make_var
from ..exceptions import TransformationNotImplementedError

from ..expressions.core import Comparison, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.utils import is_any_list, is_num, eval_comparison, is_bool

from ..expressions.variables import _BoolVarImpl, boolvar, NegBoolView, _NumVarImpl

def linearize_constraint(cpm_expr, supported={"sum","wsum"}, reified=False):
    """
    Transforms all constraints to a linear form.
    This function assumes all constraints are in 'flat normal form' with only boolean variables on the lhs of an implication.
    Only apply after 'cpmpy.transformations.flatten_model.flatten_constraint()' 'and only_bv_implies()'.

    `AllDifferent` has a special linearization and is decomposed as such if not in `supported`.
    Any other unsupported global constraint should be decomposed using `cpmpy.transformations.decompose_global.decompose_global()`

    """

    if is_any_list(cpm_expr):
        lin_cons = [linearize_constraint(expr, supported=supported, reified=reified) for expr in cpm_expr]
        return [c for l in lin_cons for c in l]

    # boolvar
    if isinstance(cpm_expr, _BoolVarImpl):
        return [sum([cpm_expr]) >= 1]

    # Boolean operators
    if isinstance(cpm_expr, Operator) and cpm_expr.is_bool():
        # conjunction
        if cpm_expr.name == "and":
            return [sum(cpm_expr.args) >= len(cpm_expr.args)]

        # disjunction
        elif cpm_expr.name == "or":
            return [sum(cpm_expr.args) >= 1]

        # xor
        elif cpm_expr.name == "xor" and len(cpm_expr.args) == 2:
            return [sum(cpm_expr.args) == 1]

        # reification
        elif cpm_expr.name == "->":
            # determine direction of implication
            cond, sub_expr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"Linearization of {cpm_expr} is not supported, lhs of implication must be boolvar. Apply `only_bv_implies` before calling `linearize_constraint`"

            if isinstance(cond, _BoolVarImpl) and isinstance(sub_expr, _BoolVarImpl):
                # shortcut for BV -> BV, convert to disjunction and apply linearize on it
                return linearize_constraint(cond <= sub_expr)

            # BV -> LinExpr
            if isinstance(cond, _BoolVarImpl):
                lin_sub = linearize_constraint(sub_expr, supported=supported, reified=True)
                return [cond.implies(lin) for lin in lin_sub]

    # comparisons
    if isinstance(cpm_expr, Comparison):
        lhs, rhs = cpm_expr.args

        if lhs.name == "sub":
            # convert to wsum
            lhs = sum([1 * lhs.args[0] + -1 * lhs.args[1]])

        # linearize unsupported operators
        elif isinstance(lhs, Operator) and lhs.name not in supported: # TODO: add mul, (abs?), (mod?), (pow?)

            if lhs.name == "mul" and is_num(lhs.args[0]):
                lhs = Operator("wsum",[[lhs.args[0]], [lhs.args[1]]])
            else:
                raise TransformationNotImplementedError(f"lhs of constraint {cpm_expr} cannot be linearized, should be any of {supported | set(['sub'])} but is {lhs}. Please report on github")

        elif isinstance(lhs, GlobalConstraint) and lhs.name not in supported:
            raise ValueError("Linearization of `lhs` not supported, run `cpmpy.transformations.decompose_global.decompose_global() first")

        if is_num(lhs) or isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name in {"sum","wsum"}):
            # bring all vars to lhs
            if isinstance(rhs, _NumVarImpl):
                if isinstance(lhs, Operator) and lhs.name == "sum":
                    lhs, rhs = sum([1 * a for a in lhs.args]+[-1 * rhs]), 0
                elif isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name == "wsum"):
                    lhs, rhs = lhs + -1*rhs, 0
                else:
                    raise ValueError(f"unexpected expression on lhs of expression, should be sum,wsum or intvar but got {lhs}")

            assert not is_num(lhs), "lhs cannot be an integer at this point!"
            # bring all const to rhs
            if lhs.name == "sum":
                new_args = []
                for i, arg in enumerate(lhs.args):
                    if is_num(arg):
                        rhs -= arg
                    else:
                        new_args.append(arg)
                lhs = Operator("sum", new_args)

            elif lhs.name == "wsum":
                new_weights, new_args = [],[]
                for i, (w, arg) in enumerate(zip(*lhs.args)):
                    if is_num(arg):
                        rhs -= w * arg
                    else:
                        new_weights.append(w)
                        new_args.append(arg)
                lhs = Operator("wsum",[new_weights, new_args])

        if isinstance(lhs, Operator) and lhs.name == "mul" and len(lhs.args) == 2 and is_num(lhs.args[0]):
            # convert to wsum
            lhs = Operator("wsum",[[lhs.args[0]],[lhs.args[1]]])

        # now fix the comparisons themselves
        if cpm_expr.name == "<":
            new_rhs, cons = get_or_make_var(rhs - 1) # if rhs is constant, will return new constant
            return [lhs <= new_rhs] + linearize_constraint(cons)
        if cpm_expr.name == ">":
            new_rhs, cons = get_or_make_var(rhs + 1) # if rhs is constant, will return new constant
            return [lhs >= new_rhs] + linearize_constraint(cons)
        if cpm_expr.name == "!=":
            # Special case: BV != BV
            if isinstance(lhs, _BoolVarImpl) and isinstance(rhs, _BoolVarImpl):
                return [lhs + rhs == 1]

            if reified or (isinstance(lhs, (Operator, GlobalConstraint)) and lhs.name not in {"sum","wsum"}):
                # lhs is sum/wsum and rhs is constant OR
                # lhs is GenExpr and rhs is constant or var
                #  ... what requires less new variables?
                # Big M implementation
                # M is chosen so that
                # lhs - rhs + 1 <= M*z
                # rhs - lhs + 1 <= M*~z
                # holds
                z = boolvar()
                # Calculate bounds of M = |lhs - rhs| + 1
                _, M1 = (lhs - rhs + 1).get_bounds()
                _, M2 = (rhs - lhs + 1).get_bounds()
                cons = [lhs + -M1*z <= rhs-1, lhs  + -M2*z >= rhs-M2+1]
                return linearize_constraint(flatten_constraint(cons), supported=supported, reified=reified)

            else:
                # introduce new indicator constraints
                z = boolvar()
                constraints = [z.implies(lhs < rhs), (~z).implies(lhs > rhs)]
                return linearize_constraint(constraints, supported=supported, reified=reified)


        return [Comparison(cpm_expr.name, lhs, rhs)]

    elif cpm_expr.name == "alldifferent" and cpm_expr.name not in supported:
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

        constraints = [sum(row) == 1 for row in sigma]  # Each var has exactly one value
        constraints += [sum(col) <= 1 for col in sigma.T]  # Each value is assigned to at most 1 variable

        for arg, row in zip(cpm_expr.args, sigma):
            constraints += [sum(np.arange(lb, ub + 1) * row) + -1*arg == 0]

        return constraints

    elif isinstance(cpm_expr, GlobalConstraint) and cpm_expr.name not in supported:
        raise ValueError(f"Linearization of global constraint {cpm_expr} not supported, run `cpmpy.transformations.decompose_global.decompose_global() first")

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

    if isinstance(cpm_expr, Comparison):
        lhs, rhs = cpm_expr.args
        new_cons = []

        if isinstance(lhs, _NumVarImpl):
            if isinstance(lhs,NegBoolView):
                lhs, rhs = Operator("wsum",[[-1], [lhs._bv]]), 1 - rhs

        if lhs.name == "sum" and any(isinstance(a, NegBoolView) for a in lhs.args):
            lhs = Operator("wsum",[[1]*len(lhs.args), lhs.args])

        if lhs.name == "wsum":
            weights, args = lhs.args
            idxes = {i for i, a in enumerate(args) if isinstance(a, NegBoolView)}
            nw, na = zip(*[(-w,a._bv) if i in idxes else (w,a) for i, (w,a) in enumerate(zip(weights, args))])
            lhs = Operator("wsum", [nw, na]) # force making wsum, even for arity = 1
            rhs -= sum(weights[i] for i in idxes)

        if isinstance(lhs, Operator) and lhs.name not in {"sum","wsum"}:
        # other operators in comparison such as "min", "max"
            lhs = copy.copy(lhs)
            for i,arg in enumerate(list(lhs.args)):
                if isinstance(arg, NegBoolView):
                    new_arg, cons = get_or_make_var(1 - arg)
                    lhs.args[i] = new_arg
                    new_cons += cons

        return [Comparison(cpm_expr.name, lhs, rhs)] + linearize_constraint(new_cons)

    # reification
    if cpm_expr.name == "->":
        cond, subexpr = cpm_expr.args
        assert isinstance(cond, _BoolVarImpl), f"{cpm_expr} is not a supported linear expression. Apply `linearize_constraint` before calling `only_positive_bv`"
        if isinstance(cond, _BoolVarImpl): # BV -> Expr
            subexpr = only_positive_bv(subexpr)
            return[cond.implies(expr) for expr in subexpr]

    if isinstance(cpm_expr, (GlobalConstraint, BoolVal, DirectConstraint)):
        return [cpm_expr]

    raise Exception(f"{cpm_expr} is not linear or is not supported. Please report on github")