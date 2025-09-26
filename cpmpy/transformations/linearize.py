"""
Transforms flat constraints into linear constraints.

Linearized constraints have one of the following forms:

Linear comparison:
------------------
- ``LinExpr == Constant``
- ``LinExpr >= Constant``
- ``LinExpr <= Constant``

LinExpr can be any of:

- `NumVar`
- `sum`
- `wsum`

Indicator constraints:
----------------------

+------------------------------------+
| ``BoolVar -> LinExpr == Constant`` |
+------------------------------------+
| ``BoolVar -> LinExpr >= Constant`` |
+------------------------------------+
| ``BoolVar -> LinExpr <= Constant`` |
+------------------------------------+

========================================   ==============================================
``BoolVar -> GenExpr``                     (GenExpr.name in supported, GenExpr.is_bool()) 
``BoolVar -> GenExpr >= Var/Constant``     (GenExpr.name in supported, GenExpr.is_num())  
``BoolVar -> GenExpr <= Var/Constant``     (GenExpr.name in supported, GenExpr.is_num())  
``BoolVar -> GenExpr == Var/Constant``     (GenExpr.name in supported, GenExpr.is_num())  
========================================   ==============================================

Where ``BoolVar`` is a boolean variable or its negation.

General comparisons or expressions
-----------------------------------

============================  ==============================================
``GenExpr``                   (GenExpr.name in supported, GenExpr.is_bool())  
``GenExpr == Var/Constant``   (GenExpr.name in supported, GenExpr.is_num())  
``GenExpr <= Var/Constant``   (GenExpr.name in supported, GenExpr.is_num())  
``GenExpr >= Var/Constant``   (GenExpr.name in supported, GenExpr.is_num()) 
============================  ============================================== 



"""
import copy
import numpy as np
import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables

from cpmpy.transformations.reification import only_implies


from .decompose_global import decompose_in_tree

from .flatten_model import flatten_constraint, get_or_make_var
from .normalize import toplevel_list, simplify_boolean
from .. import Abs
from ..exceptions import TransformationNotImplementedError

from ..expressions.core import Comparison, Expression, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.utils import is_bool, is_num, eval_comparison, get_bounds, is_true_cst, is_false_cst, is_int

from ..expressions.variables import _BoolVarImpl, boolvar, NegBoolView, _NumVarImpl, intvar

def linearize_constraint(lst_of_expr, supported={"sum","wsum"}, reified=False, csemap=None):
    """
    Transforms all constraints to a linear form.
    This function assumes all constraints are in 'flat normal form' with only boolean variables on the lhs of an implication.
    Only apply after :func:'cpmpy.transformations.flatten_model.flatten_constraint()' and :func:'cpmpy.transformations.reification.only_implies()'.

    Arguments:
        supported: which constraint and variable types are supported, i.e. `sum`, `and`, `or`, `alldifferent`
            :class:`~cpmpy.expressions.globalconstraints.AllDifferent` has a special linearization and is decomposed as such if not in `supported`.
            Any other unsupported global constraint should be decomposed using :func:`cpmpy.transformations.decompose_global.decompose_in_tree()`
        reified: whether the constraint is fully reified
    """

    newlist = []
    for cpm_expr in lst_of_expr:
        # Boolean literals are handled as trivial linears or unit clauses depending on `supported`
        if isinstance(cpm_expr, _BoolVarImpl):
            if "or" in supported:
                # post clause explicitly (don't use cp.any, which will just return the BoolVar)
                newlist.append(Operator("or", [cpm_expr]))
            elif isinstance(cpm_expr, NegBoolView):
                # might as well remove the negation
                newlist.append(sum([~cpm_expr]) <= 0)
            else: # positive literal
                newlist.append(sum([cpm_expr]) >= 1)

        # Boolean operators
        elif isinstance(cpm_expr, Operator) and cpm_expr.is_bool():
            # conjunction
            if cpm_expr.name == "and" and cpm_expr.name not in supported:
                newlist.append(sum(cpm_expr.args) >= len(cpm_expr.args))

            # disjunction
            elif cpm_expr.name == "or" and cpm_expr.name not in supported:
                newlist.append(sum(cpm_expr.args) >= 1)

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
                    lin_sub = linearize_constraint([sub_expr], supported=supported, reified=True, csemap=csemap)
                    # BV -> (C1 and ... and Cn) == (BV -> C1) and ... and (BV -> Cn)
                    indicator_constraints=[]
                    for lin in lin_sub:
                        if is_true_cst(lin):
                            continue
                        elif is_false_cst(lin):
                            indicator_constraints=[] # do not add any constraints
                            newlist+=linearize_constraint([~cond], supported=supported, csemap=csemap) # post linear version of unary constraint
                            break # do not need to add other
                        else:
                            indicator_constraints.append(cond.implies(lin)) # Add indicator constraint
                    newlist+=indicator_constraints

                    # ensure no new solutions are created
                    new_vars = set(get_variables(lin_sub)) - set(get_variables(sub_expr))
                    newlist += linearize_constraint([(~cond).implies(nv == nv.lb) for nv in new_vars], supported=supported, reified=reified, csemap=csemap)

            else: # supported operator
                newlist.append(cpm_expr)


        # comparisons
        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args

            if lhs.name == "sub":
                # convert to wsum
                lhs = Operator("wsum", [[1, -1], [lhs.args[0], lhs.args[1]]])
                cpm_expr = eval_comparison(cpm_expr.name, lhs, rhs)

            if lhs.name == "-":
                lhs = Operator("wsum", [[-1], [lhs.args[0]]])
                cpm_expr = eval_comparison(cpm_expr.name, lhs, rhs)

            # linearize unsupported operators
            elif isinstance(lhs, Operator) and lhs.name not in supported:

                if lhs.name == "mul" and is_num(lhs.args[0]):
                    lhs = Operator("wsum",[[lhs.args[0]], [lhs.args[1]]])
                    cpm_expr = eval_comparison(cpm_expr.name, lhs, rhs)

                elif lhs.name == "pow" and "pow" not in supported:
                    if "mul" not in supported:
                        raise NotImplementedError("Cannot linearize power without multiplication")
                    if not is_int(lhs.args[1]) or lhs.args[1] < 0:
                        raise NotImplementedError("Cannot linearize power with non-integer or negative exponent")
                    # only `POW(b,n) == IV` supported, with n being a non-negative integer, post as b*b*...*b (n times) == IV
                    x, n = lhs.args
                    new_lhs = 1
                    for exp in range(n):
                        new_lhs, new_cons = get_or_make_var(x * new_lhs, csemap=csemap)
                        newlist.extend(new_cons)
                    cpm_expr = eval_comparison(cpm_expr.name, new_lhs, rhs)


                elif lhs.name == "mod" and "mod" not in supported:
                    if "mul" not in supported and not is_num(lhs.args[1]):
                        raise NotImplementedError("Cannot linearize modulo without multiplication")


                    if cpm_expr.name != "==":
                        new_rhs, newcons = get_or_make_var(lhs, csemap=csemap)
                        newlist.append(eval_comparison(cpm_expr.name, new_rhs, rhs))
                        newlist += linearize_constraint(newcons, supported=supported, reified=reified, csemap=csemap)
                        continue
                    else:
                        # mod != remainder after division because defined on integer div (rounding towards 0)
                        #   e.g., 7 % -5 = 2 and -7 % 5 = -2
                        # implement x % y == z as k * y + z == x with |z| < |y| and sign(x) = sign(z)
                        # https://marcelkliemannel.com/articles/2021/dont-confuse-integer-division-with-floor-division/
                        x, y = lhs.args
                        lby, uby = get_bounds(y)
                        if lby <= 0 <= uby:
                            raise ValueError("Attempting linearization of unsafe modulo, safen expression first (cpmpy/transformations/safen.py)")

                        # k * y + z == x
                        k = intvar(*get_bounds((x - rhs) // y))
                        mult_res, side_cons = get_or_make_var(k * y, csemap=csemap)
                        cpm_expr = (mult_res + rhs) == x
                        # |z| < |y|
                        abs_of_z, new_cons = get_or_make_var(abs(rhs), csemap=csemap)
                        side_cons += new_cons
                        # TODO: do the following in constructor of abs instead?
                        # we know y is strictly positive or negative due to safening.
                        if lby >= 0:
                            side_cons.append(abs_of_z < y)
                        if uby <= 0:
                            side_cons.append(abs_of_z < -y)
                        # sign(x) = sign(z)
                        lbx, ubx = get_bounds(x)
                        if lbx >= 0:
                            side_cons.append(rhs >= 0)
                        elif ubx <= 0:
                            side_cons.append(rhs <= 0)
                        else: # x can be pos or neg
                            x_is_pos = cp.boolvar()
                            x_is_neg = ~x_is_pos
                            side_cons += [
                                x_is_pos.implies(x >= 0), x_is_neg.implies(x < 0),
                                x_is_pos.implies(rhs >= 0), x_is_neg.implies(rhs <= 0)
                            ]

                        side_cons = toplevel_list(side_cons) # get rid of bools that may result from the above
                        newlist += linearize_constraint(side_cons, supported, reified=reified, csemap=csemap)

                elif lhs.name == 'div' and 'div' not in supported:
                    if "mul" not in supported:
                        raise NotImplementedError("Cannot linearize division without multiplication")

                    if cpm_expr.name != "==":
                        new_rhs, newcons = get_or_make_var(lhs, csemap=csemap)
                        newlist.append(eval_comparison(cpm_expr.name, new_rhs, rhs))
                        newlist += linearize_constraint(newcons, supported=supported, reified=reified, csemap=csemap)
                        continue

                    else:
                        # integer division, rounding towards zero
                        # x / y = z implemented as x = y * z + r with r the remainder and |r| < |y|
                        #      r can be positive or negative, so also ensure that |y| * |z| <= |x|
                        a, b = lhs.args
                        lb, ub = get_bounds(b)
                        if lb <= 0 <= ub:
                            raise ValueError("Attempting linearization of unsafe division, safen expression first (cpmpy/transformations/safen.py)")

                        r = intvar(*get_bounds(a % b)) # r is the remainder, reuse our bound calculations
                        mult_res, side_cons = get_or_make_var(b * rhs, csemap=csemap)
                        cpm_expr = eval_comparison(cpm_expr.name, a, mult_res + r)

                        # need absolute values of variables later
                        abs_of_a, side_cons_a = get_or_make_var(abs(a), csemap=csemap)
                        abs_of_b, side_cons_b = get_or_make_var(abs(b), csemap=csemap)
                        abs_of_rhs, side_cons_rhs = get_or_make_var(abs(rhs), csemap=csemap)
                        abs_of_r, side_cons_r = get_or_make_var(abs(r), csemap=csemap)
                        side_cons += side_cons_a + side_cons_b + side_cons_rhs + side_cons_r
                        # |r| < |b|
                        side_cons.append(abs_of_r < abs_of_b)

                        # ensure we round towards zero
                        mul_abs, extra_cons = get_or_make_var(abs_of_b * abs_of_rhs, csemap=csemap)
                        side_cons += extra_cons + [mul_abs <= abs_of_a]
                        newlist += linearize_constraint(side_cons, supported=supported, reified=reified, csemap=csemap)

                else:
                    raise TransformationNotImplementedError(f"lhs of constraint {cpm_expr} cannot be linearized, should"
                                                            f" be any of {supported | {'sub'} } but is {lhs}. "
                                                            f"Please report on github")

            elif isinstance(lhs, GlobalFunction) and lhs.name == "abs" and "abs" not in supported:
                if cpm_expr.name != "==": # TODO: remove this restriction, requires comparison flipping
                    newvar, newcons = get_or_make_var(lhs, csemap=csemap)
                    newlist += linearize_constraint(newcons, supported=supported, reified=reified, csemap=csemap)
                    cpm_expr = eval_comparison(cpm_expr.name, newvar, rhs)
                else:
                    x = lhs.args[0]
                    lb, ub = get_bounds(x)
                    if lb >= 0:  # always positive
                        newlist.append(x == rhs)
                    elif ub <= 0:  # always negative
                        newlist.append(x + rhs == 0)
                    else:
                        lhs_is_pos = cp.boolvar()
                        newcons = [lhs_is_pos.implies(x >= 0), (~lhs_is_pos).implies(x <= -1),
                                   lhs_is_pos.implies(x == rhs), (~lhs_is_pos).implies(x + rhs == 0)]
                        newlist += linearize_constraint(newcons, supported=supported, reified=reified, csemap=csemap)
                    continue # all should be linear now


            elif isinstance(lhs, GlobalFunction) and lhs.name not in supported:
                raise ValueError(f"Linearization of `lhs` ({lhs}) not supported, run "
                                 "`cpmpy.transformations.decompose_global.decompose_global() first")

            [cpm_expr] = canonical_comparison([cpm_expr])  # just transforms the constraint, not introducing new ones
            lhs, rhs = cpm_expr.args

            if lhs.name == "sum" and len(lhs.args) == 1 and isinstance(lhs.args[0], _BoolVarImpl) and "or" in supported:
                # very special case, avoid writing as sum of 1 argument
                new_expr = simplify_boolean([eval_comparison(cpm_expr.name,lhs.args[0], rhs)])
                assert len(new_expr) == 1
                newlist.append(Operator("or", new_expr))
                continue



            # check trivially true/false (not allowed by PySAT Card/PB)
            if cpm_expr.name in ('<', '<=', '>', '>=') and is_num(rhs):
                lb,ub = lhs.get_bounds()
                t_lb = eval_comparison(cpm_expr.name, lb, rhs)
                t_ub = eval_comparison(cpm_expr.name, ub, rhs)
                if t_lb and t_ub:
                    continue
                elif not t_lb and not t_ub:
                    newlist += linearize_constraint([BoolVal(False)], supported=supported, csemap=csemap) # post the linear version of False
                    break

            # now fix the comparisons themselves
            if cpm_expr.name == "<":
                new_rhs, cons = get_or_make_var(rhs - 1, csemap=csemap) # if rhs is constant, will return new constant
                newlist.append(lhs <= new_rhs)
                newlist += linearize_constraint(cons, csemap=csemap)
            elif cpm_expr.name == ">":
                new_rhs, cons = get_or_make_var(rhs + 1, csemap=csemap) # if rhs is constant, will return new constant
                newlist.append(lhs >= new_rhs)
                newlist += linearize_constraint(cons, csemap=csemap)
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
                    newlist += linearize_constraint(flatten_constraint(cons, csemap=csemap), supported=supported, reified=reified, csemap=csemap)

                else:
                    # introduce new indicator constraints
                    z = boolvar()
                    constraints = [z.implies(lhs < rhs), (~z).implies(lhs > rhs)]
                    newlist += linearize_constraint(constraints, supported=supported, reified=reified, csemap=csemap)
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
                    newlist.append(Operator("sum", [row[arg-lb]]) == 1) # ensure it is linear
                else: # ensure result is canonical
                    newlist.append(sum(np.arange(lb, ub + 1) * row) + -1 * arg == 0)

        elif isinstance(cpm_expr, (DirectConstraint, BoolVal)):
            newlist.append(cpm_expr)

        elif isinstance(cpm_expr, GlobalConstraint) and cpm_expr.name not in supported:
            raise ValueError(f"Linearization of global constraint {cpm_expr} not supported, run "
                             f"`cpmpy.transformations.decompose_global.decompose_global() first")

    return newlist

def only_positive_bv(lst_of_expr, csemap=None):
    """
        Replaces :class:`~cpmpy.expressions.comparison.Comparison` containing :class:`~cpmpy.expressions.variables.NegBoolView` with equivalent expression using only :class:`~cpmpy.expressions.variables.BoolVar`.
        Comparisons are expected to be linearized. Only apply after applying :func:`linearize_constraint(cpm_expr) <linearize_constraint>`.

        Resulting expression is linear if the original expression was linear.
    """
    newlist = []
    for cpm_expr in lst_of_expr:

        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            new_lhs = lhs
            new_cons = []

            if isinstance(lhs, _NumVarImpl) or lhs.name in {"sum","wsum"}:
                new_lhs, const = only_positive_bv_wsum_const(lhs)
                rhs -= const
            else:
                # other operators in comparison such as "min", "max"
                nbv_sel = [isinstance(a, NegBoolView) for a in lhs.args]
                if any(nbv_sel):
                    new_args = []
                    for i, nbv in enumerate(nbv_sel):
                        if nbv:
                            aux = cp.boolvar()
                            new_args.append(aux)
                            new_cons += [aux + lhs.args[i]._bv == 1]  # aux == 1 - arg._bv
                        else:
                            new_args.append(lhs.args[i])

                    new_lhs = copy.copy(lhs)
                    new_lhs.update_args(new_args)

            if new_lhs is not lhs:
                newlist.append(eval_comparison(cpm_expr.name, new_lhs, rhs))
                newlist += new_cons  # already linear
            else:
                newlist.append(cpm_expr)

        # reification
        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            cond, subexpr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"{cpm_expr} is not a supported linear expression. Apply " \
                                                   f"`linearize_constraint` before calling `only_positive_bv` "
            # BV -> Expr
            subexpr = only_positive_bv([subexpr], csemap=csemap)
            newlist += [cond.implies(expr) for expr in subexpr]

        elif isinstance(cpm_expr, _BoolVarImpl):
            raise ValueError(f"Unreachable: unexpected Boolean literal (`_BoolVarImpl`) in expression {cpm_expr}, perhaps `linearize_constraint` was not called before this `only_positive_bv `call")
        elif isinstance(cpm_expr, (GlobalConstraint, BoolVal, DirectConstraint)):
            newlist.append(cpm_expr)
        else:
            raise Exception(f"{cpm_expr} is not linear or is not supported. Please report on github")

    return newlist

def only_positive_bv_wsum(expr):
    """
        Replaces a var/sum/wsum expression containing :class:`~cpmpy.expressions.variables.NegBoolView` with an equivalent expression 
        using only :class:`~cpmpy.expressions.variables.BoolVar`. 

        It might add a constant term to the expression, if you want the constant separately, use :func:`only_positive_bv_wsum_const`.
        
        Arguments:
        - `cpm_expr`: linear expression (sum, wsum, var)
        
        Returns tuple of:
        - `pos_expr`: linear expression (sum, wsum, var) without NegBoolView
    """
    if isinstance(expr, _NumVarImpl) or expr.name in {"sum","wsum"}:
        pos_expr, const = only_positive_bv_wsum_const(expr)
        if const == 0:
            return pos_expr
        else:
            assert isinstance(pos_expr, Operator) and pos_expr.name == "wsum", f"unexpected expression, should be wsum but got {pos_expr}"
            # should we check if it already has a constant term?
            return Operator("wsum", [pos_expr.args[0]+[1], pos_expr.args[1]+[const]])
    else:
        return expr

def only_positive_bv_wsum_const(cpm_expr):
    """
        Replaces a var/sum/wsum expression containing :class:`~cpmpy.expressions.variables.NegBoolView` with an equivalent expression 
        using only :class:`~cpmpy.expressions.variables.BoolVar` as well as a constant term that must be added to the new expression to be equivalent.

        If you want the expression where the constant term is part of the wsum returned, use :func:`only_positive_bv_wsum`.
        
        Arguments:
        - `cpm_expr`: linear expression (sum, wsum, var)
        
        Returns tuple of:
        - `pos_expr`: linear expression (sum, wsum, var) without NegBoolView
        - `const`: The difference between the original expression and the new expression, 
                   i.e. a constant term that must be added to pos_expr to be an equivalent linear expression.
    """
    if isinstance(cpm_expr, _NumVarImpl):
        if isinstance(cpm_expr,NegBoolView):
            return Operator("wsum",[[-1], [cpm_expr._bv]]), 1
        else:
            return cpm_expr, 0

    elif cpm_expr.name == "sum":
        # indicator on arguments being negboolviews
        nbv_sel = [isinstance(a, NegBoolView) for a in cpm_expr.args]
        if any(nbv_sel):
            const = 0
            weights = []
            variables = []
            for i, nbv in enumerate(nbv_sel):
                if nbv:
                    const += 1
                    weights.append(-1)
                    variables.append(cpm_expr.args[i]._bv)
                else:
                    weights.append(1)
                    variables.append(cpm_expr.args[i])
            return Operator("wsum", [weights, variables]), const
        else:
            return cpm_expr, 0

    elif cpm_expr.name == "wsum":
        # indicator on arguments of the wsum variable being negboolviews
        nbv_sel = [isinstance(a, NegBoolView) for a in cpm_expr.args[1]]
        if any(nbv_sel):
            # copy weights and variables
            weights = [w for w in cpm_expr.args[0]]
            variables = [v for v in cpm_expr.args[1]]
            const = 0
            for i, nbv in enumerate(nbv_sel):
                if nbv:
                    const += weights[i]
                    weights[i] = -weights[i]
                    variables[i] = variables[i]._bv
            return Operator("wsum", [weights, variables]), const
        else:
            return cpm_expr, 0

    else:
        raise ValueError(f"unexpected expression, should be sum, wsum or var but got {cpm_expr}")


def canonical_comparison(lst_of_expr):
    """
        Canonicalize a comparison expression.
        Transforms linear expressions, or a reification thereof into canonical form by:
            - moving all variables to the left-hand side
            - moving constants to the right-hand side

        Expects the input constraints to be flat. Only apply after applying :func:`flatten_constraint`
    """

    lst_of_expr = toplevel_list(lst_of_expr) # ensure it is a list

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
            if isinstance(lhs, Comparison) and (is_bool(rhs) or isinstance(rhs, Expression) and rhs.is_bool()):
                assert cpm_expr.name == "==", "Expected a reification of a comparison here, but got {}".format(cpm_expr.name)
                lhs = canonical_comparison(lhs)[0]
            elif is_num(lhs) or isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name in {"sum", "wsum", "sub"}):
                if lhs.name == "sub":
                    lhs = Operator("wsum", [[1,-1],lhs.args])
                # bring all vars to lhs
                lhs2 = []
                if isinstance(rhs, _NumVarImpl):
                    lhs2, rhs = [-1 * rhs], 0
                elif isinstance(rhs, BoolVal):
                    lhs2, rhs = [-1] if rhs.value() else [], 0 
                elif isinstance(rhs, Operator) and rhs.name == "-":
                    lhs2, rhs = [rhs.args[0]], 0
                elif isinstance(rhs, Operator) and rhs.name == "sum":
                    lhs2, rhs = [-1 * b if isinstance(b, _NumVarImpl) else -1 * b.args[0] for b in rhs.args
                                 if isinstance(b, _NumVarImpl) or isinstance(b, Operator)], \
                                 sum(b for b in rhs.args if is_num(b))
                elif isinstance(rhs, Operator) and rhs.name == "wsum":
                    lhs2, rhs = [-a * b for a, b in zip(rhs.args[0], rhs.args[1])
                                    if isinstance(b, _NumVarImpl)], \
                                    sum(-a * b for a, b in zip(rhs.args[0], rhs.args[1])
                                    if not isinstance(b, _NumVarImpl))
                
                # 2) add collected variables to lhs
                if isinstance(lhs, Operator) and lhs.name == "sum":
                    lhs = sum([1 * a for a in lhs.args] + lhs2)
                elif isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name == "wsum"):
                    lhs = lhs + lhs2
                else:
                    raise ValueError(f"unexpected expression on lhs of expression, should be sum, wsum or intvar but got {lhs}")

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
                    assert isinstance(lhs, _NumVarImpl), f"Expected variable here, but got {lhs} in expression {cpm_expr}"
                    lhs = Operator("sum", [lhs])

            newlist.append(eval_comparison(cpm_expr.name, lhs, rhs))
        else:   # rest of expressions
            newlist.append(cpm_expr)

    return newlist

def only_positive_coefficients(lst_of_expr):
    """
        Replaces Boolean terms with negative coefficients in linear constraints with terms with positive coefficients by negating its literal.
        This can simplify a `wsum` into `sum`.
        `cpm_expr` is expected to be a canonical comparison.
        Only apply after applying :func:`canonical_comparison(cpm_expr) <canonical_comparison>`

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
