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
import warnings
from typing import Set, Sequence, Optional

import numpy as np
import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables

from .flatten_model import flatten_constraint, get_or_make_var
from .decompose_global import decompose_in_tree, decompose_objective
from .normalize import toplevel_list, simplify_boolean
from ..exceptions import TransformationNotImplementedError

from ..expressions.core import Comparison, Expression, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint, AllDifferent, Table, NegativeTable
from ..expressions.globalfunctions import GlobalFunction, Element, NValue, Count
from ..expressions.utils import is_bool, is_num, is_int, eval_comparison, get_bounds, is_true_cst, is_false_cst
from ..expressions.variables import _BoolVarImpl, boolvar, NegBoolView, _NumVarImpl
from .int2bool import _encode_int_var



def linearize_constraint(lst_of_expr, supported={"sum","wsum","->"}, reified=False, csemap=None):
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
                            newlist += linearize_constraint([~cond], supported=supported, csemap=csemap, reified=reified) # post linear version of unary constraint
                            break # do not need to add other
                        elif "->" in supported and not reified:
                            indicator_constraints.append(cond.implies(lin)) # Add indicator constraint
                        else: # need to linearize the implication constraint itself
                            # either -> is not supported, or we are in a reified context (nested -> constraints are not linear)
                            assert isinstance(lin, Comparison), f"Expected a comparison as rhs of implication constraint, got {lin}"
                            if lin.args[0].name not in {"sum", "wsum"}:
                                assert lin.args[0].name in supported, f"Unexpected rhs of implication: {lin}, it is not supported ({supported})"
                                indicator_constraints.append(cond.implies(lin))
                                continue

                            # need to write as big-M
                            assert lin.args[0].name in frozenset({'sum', 'wsum'}), f"Expected sum or wsum as rhs of implication constraint, but got {lin}"
                            assert is_num(lin.args[1])
                            lb, ub = get_bounds(lin.args[0])
                            if lin.name == "<=":
                                M = lin.args[1] - ub # subtracting M from lhs will always satisfy the implied constraint
                                lin.args[0] += M * ~cond
                                indicator_constraints.append(lin)
                            elif lin.name == ">=":
                                M = lin.args[1] - lb # adding M to lhs will always satisfy the implied constraint
                                lin.args[0] += M * ~cond
                                indicator_constraints.append(lin)
                            elif lin.name == "==":
                                indicator_constraints += linearize_constraint([cond.implies(lin.args[0] <= lin.args[1]),
                                                                               cond.implies(lin.args[0] >= lin.args[1])],
                                                                              supported=supported, reified=reified, csemap=csemap)
                            else:
                                raise ValueError(f"Unexpected linearized rhs of implication {lin} in {cpm_expr}")
                    newlist+=indicator_constraints

                    # ensure no new solutions are created
                    new_vars = set(get_variables(lin_sub)) - set(get_variables(sub_expr)) - {cond, ~cond}
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

                if lhs.name == "mul":
                    bv_idx = None
                    if is_num(lhs.args[0]): # const * iv <comp> rhs
                        lhs = Operator("wsum",[[lhs.args[0]], [lhs.args[1]]])
                        newlist += linearize_constraint([eval_comparison(cpm_expr.name, lhs, rhs)], supported=supported, reified=reified, csemap=csemap)
                        continue
                    elif isinstance(lhs.args[0], _BoolVarImpl):
                        bv_idx = 0
                    elif isinstance(lhs.args[1], _BoolVarImpl):
                        bv_idx = 1

                    if bv_idx is not None:
                        # bv * iv <comp> rhs, rewrite to (bv -> iv <comp> rhs) & (~bv -> 0 <comp> rhs)
                        bv, iv = lhs.args[bv_idx], lhs.args[1-bv_idx]
                        bv_true = bv.implies(eval_comparison(cpm_expr.name, iv, rhs))
                        bv_false = (~bv).implies(eval_comparison(cpm_expr.name, 0, rhs))
                        newlist += linearize_constraint(simplify_boolean([bv_true, bv_false]), supported=supported, reified=reified, csemap=csemap)
                        continue
                    else:
                        raise NotImplementedError(f"Linearization of integer multiplication {cpm_expr} is not supported")

                else:
                    raise TransformationNotImplementedError(f"lhs of constraint {cpm_expr} cannot be linearized, should"
                                                            f" be any of {supported | {'sub'} } but is {lhs}. "
                                                            f"Please report on github")

            elif isinstance(lhs, GlobalFunction) and lhs.name not in supported:
                raise ValueError(f"Linearization of `lhs` ({lhs}) not supported, run "
                                 "`cpmpy.transformations.decompose_global.decompose_in_tree() first")

            [cpm_expr] = canonical_comparison([cpm_expr])  # just transforms the constraint, not introducing new ones
            lhs, rhs = cpm_expr.args

            if lhs.name == "sum" and len(lhs.args) == 1 and isinstance(lhs.args[0], _BoolVarImpl) and "or" in supported:
                # very special case, avoid writing as sum of 1 argument
                new_expr = simplify_boolean([eval_comparison(cpm_expr.name,lhs.args[0], rhs)])
                assert len(new_expr) == 1
                if isinstance(new_expr[0], BoolVal) and  new_expr[0].value() is True:
                    continue # skip or([BoolVal(True)])
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

        elif isinstance(cpm_expr, (DirectConstraint, BoolVal)):
            newlist.append(cpm_expr)

        elif isinstance(cpm_expr, GlobalConstraint):
            if cpm_expr.name not in supported:
                raise ValueError(f"Linearization of global constraint {cpm_expr} not supported, run "
                                 f"`cpmpy.transformations.linearize.decompose_linear() first")
            else:
                newlist.append(cpm_expr)
        else:
            raise ValueError(f"Unexpected expression {cpm_expr}, if you reach this, please report on github.")

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
                if isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and (lhs.name == "sum" or lhs.name == "wsum")):
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

def only_positive_coefficients_(ws, xs):
    """
    Helper function to replace Boolean terms with negative coefficients with terms with positive coefficients (including 0) in Boolean linear expressions, given as a list of coefficients `ws` and a list of Boolean variables `xs`. Returns new non-negative coefficients and variables, and a constant term to be added.
    """
    indices = {i for i, (w, x) in enumerate(zip(ws, xs)) if w < 0 and isinstance(x, _BoolVarImpl)}
    nw, na = zip(*[(-w, ~x) if i in indices else (w, x) for i, (w, x) in enumerate(zip(ws, xs))])
    constant = sum(ws[i] for i in indices)
    return nw, na, constant

def only_positive_coefficients(lst_of_expr):
    """
        Replaces Boolean terms with negative coefficients in linear constraints with terms with positive coefficients (including 0) by negating its literal.
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
                nw, na, k = only_positive_coefficients_(weights, args)
                rhs -= k

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


def decompose_linear(lst_of_expr: Sequence[Expression],
                     supported: Set[str]=frozenset(),
                     supported_reified:Set[str]=frozenset(),
                     csemap:Optional[dict[Expression,Expression]]=None):
    """
        Decompose unsupported global constraints in a linear-friendly way using (var == val) in sums.

        args:
            lst_of_expr: list of expressions to decompose
            supported: set of supported global constraints and global functions
            supported_reified: set of supported reified global constraints
            csemap: map of expressions to an auxiliary variable

        returns:
            list of expressions
    """
    decompose_custom = get_linear_decompositions()

    return decompose_in_tree(lst_of_expr, supported, supported_reified, csemap=csemap, decompose_custom=decompose_custom)

def decompose_linear_objective(obj: Sequence[Expression],
                               supported: Set[str] = frozenset(),
                               supported_reified: Set[str] = frozenset(),
                               csemap: Optional[dict[Expression, Expression]] = None):
    """Decompose objective using linear-friendly (var == val) decompositions."""
    decompose_custom = get_linear_decompositions()

    return decompose_objective(obj, supported, supported_reified, csemap=csemap, decompose_custom=decompose_custom)

def get_linear_decompositions():
    """
        Implementation of custom linear decompositions for some global constraints.
        Uses (var == val) in sums; no integer encoding.

        returns:
            dict: a dictionary mapping expression names to a function, taking as argument the expression to decompose
    """
    return dict(
        alldifferent=AllDifferent.decompose_linear,
        element=Element.decompose_linear,
    )
    # Should we add Gleb's table decomposition? or is it not non-reifiable?


def linearize_reified_variables(constraints, min_values=3, csemap=None, ivarmap=None):
    """
    Replace reified (BV <-> (x == val)) implications with direct encoding when a variable
    has at least min_values such reifications: remove those implications and add
    the 'direct' encoding of x.

    If ivarmap is None, both sum(bvs)==1 and wsum(values, bvs)==var are posted.
    If ivarmap is not None, the encoding is added to ivarmap and only sum(bvs)==1
    (the domain constraint) is posted; the solver can then choose to eliminate the
    vars, or post the wsums itself anyway.

    Apply AFTER flatten_constraint and BEFORE only_implies and linearize_constraint.
    """
    # this transformation can only be done if there is a csemap
    if csemap is None:
        return constraints

    # Collect bv -> (var == val)'s in csemap
    var_vals = {}  # var: [val, bv]
    for expr, bv in csemap.items():
        if expr.name == '==':
            var,val = expr.args
            if isinstance(var, _NumVarImpl) and is_int(val):
                var_vals.setdefault(var, []).append((val, bv))
    
    # Make the integer encodings in integer linear friendly way
    my_ivarmap = ivarmap if ivarmap is not None else {}
    toplevel = []
    bv_map = {}  # bv -> (var, val)
    for var, vals in var_vals.items():
        # check if we should linearize the reified variables
        lb, ub = var.lb, var.ub
        vals = [(val, bv) for val, bv in vals if lb <= val <= ub]  # only the valid values, in bounds!
        if len(vals) < min_values:
            continue  # do not encode

        # encode the values
        enc, _ = _encode_int_var(my_ivarmap, var, "direct", csemap=csemap)
        
        # domain and channeling constraints
        toplevel.extend(enc.encode_domain_constraint()) # with the overwritten Bools
        if ivarmap is None:
            # also post the var=wsum mapping
            terms, k = enc.encode_term()
            # var == wsum + k :: var - wsum == k
            ws = [1] + [-w for (w, _) in terms]
            bs = [var] + [b for (_, b) in terms]
            toplevel.append(Operator("wsum", (ws, bs)) == k)  
        
        # store the bvs that no longer need to be reified
        for val, bv in vals:
            bv_map[bv] = (var, val)

    if len(bv_map) > 0:
        # Now clean up and remove the '(var == val) == bv' constraints:
        newcons = []
        for con in constraints:
            if con.name == '==' and con.args[0].name == '==':
                # potential '(var == val) == bv'
                lhs,bv = con.args
                if bv in bv_map:
                    (var, val) = bv_map[bv]
                    (lhs_var, lhs_val) = lhs.args
                    if lhs_val == val and lhs_var == var:
                        continue  # do not keep
            newcons.append(con)
        constraints = newcons

    return constraints + toplevel


def _extract_var_from_lhs(lhs):
    """Extract integer variable from lhs of (x == val) or (x != val). Returns None if not applicable."""
    if isinstance(lhs, _NumVarImpl) and not lhs.is_bool():
        return lhs
    if isinstance(lhs, Operator) and lhs.name == "sum" and len(lhs.args) == 1:
        arg = lhs.args[0]
        if isinstance(arg, _NumVarImpl) and not arg.is_bool():
            return arg
    return None
