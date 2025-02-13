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
import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables

from cpmpy.transformations.reification import only_implies, only_bv_reifies


from .decompose_global import decompose_in_tree

from .flatten_model import flatten_constraint, get_or_make_var
from .normalize import toplevel_list
from .. import Abs
from ..exceptions import TransformationNotImplementedError

from ..expressions.core import Comparison, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.utils import is_num, eval_comparison, get_bounds, is_true_cst, is_false_cst

from ..expressions.variables import _BoolVarImpl, boolvar, NegBoolView, _NumVarImpl, intvar


def linearize_constraint(lst_of_expr, supported={"sum","wsum"}, reified=False):
    """
    Transforms all constraints to a linear form.
    This function assumes all constraints are in 'flat normal form' with only boolean variables on the lhs of an implication.
    Only apply after 'cpmpy.transformations.flatten_model.flatten_constraint()' 'and only_implies()'.

    `AllDifferent` has a special linearization and is decomposed as such if not in `supported`.
    Any other unsupported global constraint should be decomposed using `cpmpy.transformations.decompose_global.decompose_global()`

    """

    newlist = []
    for cpm_expr in lst_of_expr:

        # boolvar
        if isinstance(cpm_expr, _BoolVarImpl):
            newlist.append(sum([cpm_expr]) >= 1)

        # Boolean operators
        elif isinstance(cpm_expr, Operator) and cpm_expr.is_bool():
            # conjunction
            if cpm_expr.name == "and":
                newlist.append(sum(cpm_expr.args) >= len(cpm_expr.args))

            # disjunction
            elif cpm_expr.name == "or":
                newlist.append(sum(cpm_expr.args) >= 1)

            # xor
            elif cpm_expr.name == "xor" and len(cpm_expr.args) == 2:
                newlist.append(sum(cpm_expr.args) == 1)

            # reification
            elif cpm_expr.name == "->":
                # determine direction of implication
                cond, sub_expr = cpm_expr.args
                assert isinstance(cond, _BoolVarImpl), f"Linearization of {cpm_expr} is not supported, lhs of " \
                                                       f"implication must be boolvar. Apply `only_implies` before " \
                                                       f"calling `linearize_constraint`"

                if isinstance(cond, _BoolVarImpl) and isinstance(sub_expr, _BoolVarImpl):
                    # shortcut for BV -> BV, convert to disjunction and apply linearize on it
                    newlist.append(1 * cond + -1 * sub_expr <= 0)

                # BV -> LinExpr
                elif isinstance(cond, _BoolVarImpl):
                    lin_sub = linearize_constraint([sub_expr], supported=supported, reified=True)
                    newlist += [cond.implies(lin) for lin in lin_sub]
                    # ensure no new solutions are created
                    new_vars = set(get_variables(lin_sub)) - set(get_variables(sub_expr))
                    newlist += linearize_constraint([(~cond).implies(nv == nv.lb) for nv in new_vars], supported=supported, reified=reified)


        # comparisons
        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args

            if lhs.name == "sub":
                # convert to wsum
                lhs = sum([1 * lhs.args[0] + -1 * lhs.args[1]])
                cpm_expr = eval_comparison(cpm_expr.name, lhs, rhs)

            # linearize unsupported operators
            elif isinstance(lhs, Operator) and lhs.name not in supported: # TODO: add mul, (abs?), (mod?), (pow?)

                if lhs.name == "mul" and is_num(lhs.args[0]):
                    lhs = Operator("wsum",[[lhs.args[0]], [lhs.args[1]]])
                    cpm_expr = eval_comparison(cpm_expr.name, lhs, rhs)

                elif lhs.name == "pow" and "pow" not in supported:
                    if "mul" not in supported:
                        raise NotImplementedError("Cannot linearize power without multiplication")
                    if not is_num(lhs.args[1]):
                        raise NotImplementedError("Cannot linearize power with ")
                    # only `POW(b,n) == IV` supported, with n being an integer, post as b*b*...*b (n times) == IV
                    x, n = lhs.args
                    new_lhs = 1
                    for exp in range(n):
                        new_lhs, new_cons = get_or_make_var(x * new_lhs)
                        newlist.extend(new_cons)
                    cpm_expr = eval_comparison(cpm_expr.name, new_lhs, rhs)


                elif lhs.name == "mod" and "mod" not in supported:
                    if "mul" not in supported:
                        raise NotImplementedError("Cannot linearize modulo without multiplication")

                    if cpm_expr.name != "==":
                        new_rhs, newcons = get_or_make_var(lhs)
                        newlist.append(eval_comparison(cpm_expr.name, new_rhs, rhs))
                        newlist += linearize_constraint(newcons, supported=supported, reified=reified)
                        continue
                    else:
                        # mod != remainder after division because defined on integer div (rounding towards 0)
                        #   e.g., 7 % -5 = 2 and -7 % 5 = -2
                        # implement x % y == z as k * y + z == x with |z| < |y| and sign(x) = sign(z)
                        # https://marcelkliemannel.com/articles/2021/dont-confuse-integer-division-with-floor-division/
                        x, y = lhs.args
                        lby, uby = get_bounds(y)
                        if lby <= 0 <= uby:
                            raise ValueError("Attempting linerarization of unsafe modulo, safen expression first (cpmpy/transformations/safen.py)")

                        # k * y + z == x
                        k = intvar(*get_bounds((x - rhs) // y))
                        mult_res, side_cons = get_or_make_var(k * y)
                        cpm_expr = (mult_res + rhs) == x
                        # |z| < |y|
                        abs_of_z = cp.intvar(*get_bounds(abs(rhs)))
                        if "abs" in supported:
                            side_cons.append(abs(rhs) == abs_of_z)
                        else: # need to linearize abs...
                            z_is_pos = cp.boolvar()
                            side_cons += [z_is_pos.implies(rhs >= 0), (~z_is_pos).implies(rhs < 0)]
                            side_cons += [z_is_pos.implies(abs_of_z == rhs), (~z_is_pos).implies(abs_of_z == -rhs)]
                        # TODO: do the following in constructor of abs instead?
                        # we know y is strictly positive or negative due to safening.
                        if lby >= 0:
                            side_cons.append(abs_of_z < y)
                        if uby <= 0:
                            side_cons.append(abs_of_z < -y)
                        # sign(x) = sign(z)
                        lbx,ubx = get_bounds(x)
                        if lbx >= 0:
                            side_cons.append(rhs >= 0)
                        elif ubx <= 0:
                            side_cons.append(rhs <= 0)
                        else: # x can be pos or neg
                            x_is_pos, x_is_neg = cp.boolvar(), cp.boolvar()
                            side_cons += [(~x_is_pos).implies(x <= 0), (~x_is_neg).implies(x >= 0),
                                           x_is_pos.implies(rhs >= 0), (~x_is_pos).implies(rhs < 0),
                                           x_is_neg.implies(rhs <= 0), (~x_is_neg).implies(rhs > 0)]

                        side_cons = toplevel_list(side_cons) # get rid of bools that may result from the above
                        newlist += linearize_constraint(side_cons, supported, reified=reified)

                elif lhs.name == 'div' and 'div' not in supported:
                    if "mul" not in supported:
                        raise NotImplementedError("Cannot linearize division without multiplication")
                    a, b = lhs.args
                    # if division is total, b is either strictly negative or strictly positive!
                    lb, ub = get_bounds(b)
                    if lb <= 0 <= ub:
                        raise TypeError(
                            f"Can't divide by a domain containing 0, safen the expression first")
                    # div(a,b) = rhs  -->  a = b * rhs + r with r the remainder
                    # remainder can be both positive and negative (round towards 0, so negative r if a and b are both negative)
                    # abs(r) < abs(b), otherwise it wouldn't be a remainder.
                    # we need abs here because one or both of these can be negative.
                    r = intvar(*get_bounds(a % b)) # r is the remainder, reuse our bound calculations
                    cpm_expr = [eval_comparison(cpm_expr.name, a, b * rhs + r)]
                    # b * rhs <= a, otherwise we can both overshoot and undershoot the division, with r having positive and negative options.
                    cond = [Abs(r) < Abs(b), b * rhs <= a]
                    decomp = toplevel_list(decompose_in_tree(cond))  # decompose abs
                    cpm_exprs = toplevel_list(decomp + cpm_expr)
                    exprs = linearize_constraint(flatten_constraint(cpm_exprs), supported=supported)
                    newlist.extend(exprs)
                    continue

                else:
                    raise TransformationNotImplementedError(f"lhs of constraint {cpm_expr} cannot be linearized, should"
                                                            f" be any of {supported | {'sub'} } but is {lhs}. "
                                                            f"Please report on github")

            elif isinstance(lhs, GlobalConstraint) and lhs.name not in supported:
                raise ValueError(f"Linearization of `lhs` ({lhs}) not supported, run "
                                 "`cpmpy.transformations.decompose_global.decompose_global() first")

            [cpm_expr] = canonical_comparison([cpm_expr])  # just transforms the constraint, not introducing new ones
            lhs, rhs = cpm_expr.args

            # now fix the comparisons themselves
            if cpm_expr.name == "<":
                new_rhs, cons = get_or_make_var(rhs - 1) # if rhs is constant, will return new constant
                newlist.append(lhs <= new_rhs)
                newlist += linearize_constraint(cons)
            elif cpm_expr.name == ">":
                new_rhs, cons = get_or_make_var(rhs + 1) # if rhs is constant, will return new constant
                newlist.append(lhs >= new_rhs)
                newlist += linearize_constraint(cons)
            elif cpm_expr.name == "!=":
                # Special case: BV != BV
                if isinstance(lhs, _BoolVarImpl) and isinstance(rhs, _BoolVarImpl):
                    newlist.append(lhs + rhs == 1)

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
                    newlist += linearize_constraint(flatten_constraint(cons), supported=supported, reified=reified)

                else:
                    # introduce new indicator constraints
                    z = boolvar()
                    constraints = [z.implies(lhs < rhs), (~z).implies(lhs > rhs)]
                    newlist += linearize_constraint(constraints, supported=supported, reified=reified)
            else:
                # supported comparison
                newlist.append(eval_comparison(cpm_expr.name, lhs, rhs))

        elif cpm_expr.name == "alldifferent" and cpm_expr.name in supported:
            newlist.append(cpm_expr)
        elif cpm_expr.name == "alldifferent" and cpm_expr.name not in supported:
            """
                More efficient implementations possible
                http://yetanothermathprogrammingconsultant.blogspot.com/2016/05/all-different-and-mixed-integer.html
                Introduces n^2 new boolean variables
                Decomposes through bi-partite matching
            """
            # TODO check performance of implementation
            if reified is True:
                raise ValueError("Linear decomposition of AllDifferent does not work reified. "
                                 "Ensure 'alldifferent' is not in the 'supported_nested' set of 'decompose_in_tree'")

            lbs, ubs = get_bounds(cpm_expr.args)
            lb, ub = min(lbs), max(ubs)
            n_vals = (ub-lb) + 1

            x = boolvar(shape=(len(cpm_expr.args), n_vals))

            newlist += [sum(row) == 1 for row in x]   # each var has exactly one value
            newlist += [sum(col) <= 1 for col in x.T] # each value can be taken at most once

            # link Boolean matrix and integer variable
            for arg, row in zip(cpm_expr.args, x):
                if is_num(arg): # constant, fix directly
                    newlist.append(row[arg-lb] == 1)
                else: # ensure result is canonical
                    newlist.append(sum(np.arange(lb, ub + 1) * row) + -1 * arg == 0)

        elif isinstance(cpm_expr, (DirectConstraint, BoolVal)):
            newlist.append(cpm_expr)

        elif isinstance(cpm_expr, GlobalConstraint) and cpm_expr.name not in supported:
            raise ValueError(f"Linearization of global constraint {cpm_expr} not supported, run "
                             f"`cpmpy.transformations.decompose_global.decompose_global() first")

    return newlist


def only_positive_bv(lst_of_expr):
    """
        Replaces constraints containing NegBoolView with equivalent expression using only BoolVar.
        cpm_expr is expected to be linearized. Only apply after applying linearize_constraint(cpm_expr)

        Resulting expression is linear.
    """
    newlist = []
    for cpm_expr in lst_of_expr:

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
                lhs = Operator("wsum", [list(nw), list(na)]) # force making wsum, even for arity = 1
                rhs -= sum(weights[i] for i in idxes)

            if isinstance(lhs, Operator) and lhs.name not in {"sum","wsum"}:
            # other operators in comparison such as "min", "max"
                lhs = copy.copy(lhs)
                for i,arg in enumerate(list(lhs.args)):
                    if isinstance(arg, NegBoolView):
                        new_arg, cons = get_or_make_var(1 - arg)
                        lhs.args[i] = new_arg
                        new_cons += cons

            newlist.append(eval_comparison(cpm_expr.name, lhs, rhs))
            newlist += linearize_constraint(new_cons)

        # reification
        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            cond, subexpr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"{cpm_expr} is not a supported linear expression. Apply " \
                                                   f"`linearize_constraint` before calling `only_positive_bv` "
            if isinstance(cond, _BoolVarImpl): # BV -> Expr
                subexpr = only_positive_bv([subexpr])
                newlist += [cond.implies(expr) for expr in subexpr]


        elif isinstance(cpm_expr, (GlobalConstraint, BoolVal, DirectConstraint)):
            newlist.append(cpm_expr)

        else:
            raise Exception(f"{cpm_expr} is not linear or is not supported. Please report on github")

    return newlist

def canonical_comparison(lst_of_expr):

    lst_of_expr = toplevel_list(lst_of_expr)               # ensure it is a list

    newlist = []
    for cpm_expr in lst_of_expr:

        if isinstance(cpm_expr, Operator) and cpm_expr.name == '->':    # half reification of comparison
            lhs, rhs = cpm_expr.args
            if isinstance(rhs, Comparison):
                rhs = canonical_comparison(rhs)[0]
                newlist.append(lhs.implies(rhs))
            elif isinstance(lhs, Comparison):
                lhs = canonical_comparison(lhs)[0]
                newlist.append(lhs.implies(rhs))
            else:
                newlist.append(cpm_expr)

        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            if isinstance(lhs, Comparison) and cpm_expr.name == "==":  # reification of comparison
                lhs = canonical_comparison(lhs)[0]
            elif is_num(lhs) or isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name in {"sum", "wsum"}):
                # Bring all vars from rhs to lhs
                # 1) collect the variables to bring over
                lhs2 = []
                if isinstance(rhs, _NumVarImpl):
                    lhs2, rhs = [-1 * rhs], 0
                elif isinstance(rhs, Operator) and rhs.name == "-":
                    lhs2, rhs = [rhs.args[0]], 0
                elif isinstance(rhs, Operator) and rhs.name == "sum":
                    lhs2, rhs = [-1 * b if isinstance(b, _NumVarImpl) else 1 * b.args[0] for b in rhs.args
                                 if isinstance(b, _NumVarImpl) or isinstance(b, Operator)], \
                                 sum(b for b in rhs.args if is_num(b))
                elif isinstance(rhs, Operator) and rhs.name == "wsum":
                    lhs2, rhs = [-a * b for a, b in zip(rhs.args[0], rhs.args[1])
                                    if isinstance(b, _NumVarImpl)], \
                                    sum(-a * b for a, b in zip(rhs.args[0], rhs.args[1])
                                    if not isinstance(b, _NumVarImpl))
                # 2) add collected variables to lhs
                if isinstance(lhs, Operator) and lhs.name == "sum":
                    lhs, rhs = sum([1 * a for a in lhs.args] + lhs2), rhs
                elif isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name == "wsum"):
                    lhs = lhs + lhs2
                else:
                    raise ValueError(
                        f"unexpected expression on lhs of expression, should be sum, wsum or intvar but got {lhs}")

                assert not is_num(lhs), "lhs cannot be an integer at this point!"

                # bring all const to rhs
                if isinstance(lhs, Operator):
                    if lhs.name == "sum":
                        new_args = []
                        for i, arg in enumerate(lhs.args):
                            if is_num(arg):
                                rhs -= arg
                            else:
                                new_args.append(arg)
                        lhs = Operator("sum", new_args)

                    elif lhs.name == "wsum":
                        new_weights, new_args = [], []
                        for i, (w, arg) in enumerate(zip(*lhs.args)):
                            if is_num(arg):
                                rhs -= w * arg
                            else:
                                new_weights.append(w)
                                new_args.append(arg)
                        lhs = Operator("wsum", [new_weights, new_args])
                    else:
                        raise ValueError(f"lhs should be sum or wsum, but got {lhs}")
                else:
                    assert isinstance(lhs, _NumVarImpl)
                    lhs = Operator("sum", [lhs])

            newlist.append(eval_comparison(cpm_expr.name, lhs, rhs))
        else:   # rest of expressions
            newlist.append(cpm_expr)

    return newlist

def only_positive_coefficients(lst_of_expr):
    """
        Replaces Boolean terms with negative coefficients in linear constraints with terms with positive coefficients by negating its literal.
        This can simplify a wsum into sum.
        cpm_expr is expected to be a canonical comparison.
        Only apply after applying canonical_comparison(cpm_expr)

        Resulting expression is linear.
    """
    newlist = []
    for cpm_expr in lst_of_expr:
        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args

            #    ... -c*b + ... <= k
            # :: ... -c*(1 - ~b) + ... <= k
            # :: ... -c + c* ~b + ... <= k
            # :: ... + c*~b + ... <= k+c
            if lhs.name == "wsum":
                weights, args = lhs.args
                idxes = {i for i, (w, a) in enumerate(zip(weights, args)) if w < 0 and isinstance(a, _BoolVarImpl)}
                nw, na = zip(*[(-w, ~a) if i in idxes else (w, a) for i, (w, a) in enumerate(zip(weights, args))])
                rhs += sum(-weights[i] for i in idxes)

                # Simplify wsum to sum if all weights are 1
                if all(w == 1 for w in nw):
                    lhs = Operator("sum", [list(na)])
                else:
                    lhs = Operator("wsum", [list(nw), list(na)])

            newlist.append(eval_comparison(cpm_expr.name, lhs, rhs))

        # reification
        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            cond, subexpr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"{cpm_expr} is not a supported linear expression. Apply " \
                                                   f"`linearize_constraint` before calling `only_positive_coefficients` "
            subexpr = only_positive_coefficients([subexpr])
            newlist += [cond.implies(expr) for expr in subexpr]

        else:
            newlist.append(cpm_expr)

    return newlist

