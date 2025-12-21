"""
Transform constraints to a linear form.

This transformation is necessary for Integer Linear Programming (ILP) solvers, and also
for translating to Pseudo-Boolean or CNF formats.

There are a number of components to getting a good linearisation:
- **Decomposing global constraints/functions** in a 'linear friendly' way.
  A common pattern is that constraints that enforce domain consistency on integer variables,
  that they should be decomposed over a Boolean representation of the domain. This is the
  case for :class:`~cpmpy.expressions.globalconstraints.AllDifferent`, :class:`~cpmpy.expressions.globalfunctions.Element`
  and possibly others.
  Their default decomposition might not do it this way, in which case we want to use different
  decompositions.

- **Linearising multiplication** of variables, e.g. bool*bool, bool*int and int*int.
  Because '*' is a core Operator and not a global constraint, we need to do it with a helper function here.

- **Disequalities** e.g. `sum(X) != 5` should be rewritten as `(sum(X) < 5) | (sum(X) > 5)`
  and further flattened into implications and linearised.

- **Implications** e.g. `B -> sum(X) <= 4` should be linearised with the Big-M method.

- **Domain encodings** e.g. of `A=cp.intvar(0,4)` would create binary variables and constraints
  like `B0 -> A=0`, `B1 -> A=1`, `B2 -> A=2`, etc and `sum(B0..4) = 1`.
  However each `B0 -> A=0` would require 2 Big-M constraints. Instead we can linearise the entire
  domain encoding with two constraints: `sum(B0..4) = 1` and `A = sum(B0..4 * [0,1,2,3,4])`.
  We could even go as far as eliminate the original integer decision variable.

After linearisation, the output can be further transformed:
- Remove negated boolean variables (:class:`~cpmpy.expressions.variables.NegBoolView`) so only
  positive boolean variables (:class:`~cpmpy.expressions.variables.BoolVar`) appear
- Ensure only positive coefficients appear in linear constraints

Module functions
----------------

Main transformations:
- :func:`linearize_constraint`: Transforms a list of constraints to a linear form.

Helper functions:
- :func:`canonical_comparison`: Canonicalizes comparison expressions by moving variables to the
  left-hand side and constants to the right-hand side.
- :func:`decompose_mul_linear`: Decomposes multiplication operations (const*v, bool*bool, bool*int, int*int) into linear form.

Post-linearisation transformations:
- :func:`only_positive_bv`: Transforms constraints so only boolean variables appear positively
  (no :class:`~cpmpy.expressions.variables.NegBoolView`).
- :func:`only_positive_bv_wsum`: Helper function that replaces :class:`~cpmpy.expressions.variables.NegBoolView`
  in var/sum/wsum expressions with equivalent expressions using only :class:`~cpmpy.expressions.variables.BoolVar`.
- :func:`only_positive_bv_wsum_const`: Same as :func:`only_positive_bv_wsum` but returns the constant
  term separately.
- :func:`only_positive_coefficients`: Transforms constraints so only positive coefficients appear
  in linear constraints.
"""

import copy
import numpy as np
import cpmpy as cp
from typing import List, Set, Optional, Dict, Any, Tuple, Union
from cpmpy.transformations.get_variables import get_variables

from .flatten_model import flatten_constraint, get_or_make_var
from .normalize import toplevel_list, simplify_boolean
from ..exceptions import TransformationNotImplementedError

from ..expressions.core import Comparison, Expression, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.utils import is_bool, is_num, eval_comparison, get_bounds, is_true_cst, is_false_cst

from ..expressions.variables import intvar, _BoolVarImpl, boolvar, NegBoolView, _NumVarImpl, _IntVarImpl
from ..transformations.int2bool import _encode_int_var

def decompose_mul_linear(mul: Operator, supported: Set[str], reified: bool = False, csemap: Optional[Dict[Any, Any]] = None) -> Tuple[Expression, List[Expression]]:
    """
    Linearize a multiplication operation.

    Acts like a decomposition function: returns an expression and a list of defining constraints.

    Handles the following cases:
    - `c * v` with a constant  (rewritten to `wsum([c], [v])`)
    - `b1 * b2`  (rewritten to the linearisation of `b1 & b2`)
    - `b * i` or `i * b`  (rewritten to the linearisation of `aux, [b -> aux=i, ~b -> aux=0]`)
    - `i1 * i2`  (rewritten with Boolean expansion of the smallest integer, resulting in a sum of `b*i` cases)

    Arguments:
        mul: Multiplication operator to linearize.
        supported: Set of supported constraint names.
        csemap: Optional dictionary for common subexpression elimination.

    Returns:
        Tuple containing an expression representing the multiplication, and a list of defining constraints
    """
    assert mul.name == "mul", "Expected a multiplication operator, got {mul}"

    mul0,mul1 = mul.args

    if is_num(mul0):
        return Operator("wsum",[[mul0], [mul1]]), []

    # check if common subexpression
    if csemap is not None and mul in csemap:
        return csemap[mul], []

    bv_mul0 = isinstance(mul0, _BoolVarImpl)
    bv_mul1 = isinstance(mul1, _BoolVarImpl)

    if bv_mul0 and bv_mul1:
        # Boolean mul0 * mul1 = (mul0 & mul1)
        # equiv: aux, [aux -> (mul0 & mul1), ~aux -> (~mul0 | ~mul1)]
        # equiv: aux, [aux >= mul0+mul1-1, aux + ~mul0 + ~mul1 >= 1]
        aux = boolvar()
        if csemap is not None:
            csemap[mul] = aux

        return aux, linearize_constraint([aux.implies(mul0 & mul1), (~aux).implies(~mul0 | ~mul1)], supported=supported, reified=reified, csemap=csemap)

    if bv_mul0 or bv_mul1:
        # b * i
        # equiv: aux, [b -> aux=i, ~b -> aux=0]
        bv_idx = 0 if bv_mul0 else 1
        bv, iv = mul.args[bv_idx], mul.args[1-bv_idx]

        aux = intvar(*mul.get_bounds())
        if csemap is not None:
            csemap[mul] = aux

        return aux, linearize_constraint([bv.implies(aux == iv), (~bv).implies(aux == 0)], supported=supported, reified=reified, csemap=csemap)

    else:
        # i1 * i2
        # equiv: aux, [aux = sum(v*(b*i2) for v,b in int2bool(i1))]
        # choose smallest integer as i1 (to minimize the number of boolean variables)
        leni1 = (mul0.ub - mul0.lb + 1)
        leni2 = (mul1.ub - mul1.lb + 1)
        i1 = mul0 if leni1 <= leni2 else mul1
        i2 = mul1 if leni1 <= leni2 else mul0

        # Encode i1 with temporary ivarmap
        ivarmap = {}
        encoding = "direct"
        if min(leni1, leni2) >= 8:  # arbitrary heuristic
            encoding = "binary"
        i1_enc, cons = _encode_int_var(ivarmap, i1, encoding)

        # Build the sum: aux = sum(v * (b_v * i2) for v,bv in encoding)
        (encpairs, offset) = i1_enc.encode_term()
        terms = []
        for v, b_v in encpairs:
            # Create b_v * i2, which we can handle recursively
            bv_i2_expr, bv_i2_cons = decompose_mul_linear(b_v*i2, supported=supported, reified=reified, csemap=csemap)
            cons += bv_i2_cons
            terms.append((offset + v) * bv_i2_expr)
        assert len(terms) > 0, f"Expected at least one term, got {terms} for {mul} with encoding {i1_encoding} of {i1}"

        # the multiplication value is the sum of the terms, other are defining constraints
        return sum(terms), cons


def linearize_constraint(lst_of_expr: List[Expression], supported: Set[str] = {"sum","wsum","->"}, reified: bool = False, csemap: Optional[Dict[Any, Any]] = None) -> List[Expression]:
    """
    Transforms all constraints to a linear form.
    This function assumes all constraints are in 'flat normal form' with only boolean variables on the lhs of an implication.
    Only apply after :func:'cpmpy.transformations.flatten_model.flatten_constraint()' and :func:'cpmpy.transformations.reification.only_implies()'.

    After linearize, the following constraints remain:

    **Linear comparisons:**
        Equality and non-strict inequality comparison where the left-hand side is
        a linear expression and the right-hand side is a constant:

        - ``LinExpr == Constant``
        - ``LinExpr >= Constant``
        - ``LinExpr <= Constant``

        The linear expression (LinExpr) can be:
        - A numeric variable (``NumVar``)
        - A sum expression (``sum([...])``)
        - A weighted sum expression (``wsum([weights], [vars])``)

    **General comparisons or expressions, if their name is in `supported`:**

        * Boolean expressions:
          - ``GenExpr`` (when :func:`~cpmpy.expressions.core.Expression.is_bool()`)

        * Numeric comparisons:
          - ``GenExpr == Var/Constant`` (when GenExpr is numeric)
          - ``GenExpr <= Var/Constant`` (when GenExpr is numeric)
          - ``GenExpr >= Var/Constant`` (when GenExpr is numeric) 

    **Indicator constraints, if '->' is in 'supported':**
        The left-hand side is always a boolean variable or its negation, and
        the right-hand side can be a linear comparison or a general expression.

        * Linear comparisons:
          - ``BoolVar -> LinExpr == Constant``
          - ``BoolVar -> LinExpr >= Constant``
          - ``BoolVar -> LinExpr <= Constant``

        * General expressions (when the expression name is in `supported`):
          - ``BoolVar -> GenExpr`` (when GenExpr is boolean)
          - ``BoolVar -> GenExpr == Var/Constant`` (when GenExpr is numeric)
          - ``BoolVar -> GenExpr >= Var/Constant`` (when GenExpr is numeric)
          - ``BoolVar -> GenExpr <= Var/Constant`` (when GenExpr is numeric)

    Arguments:
        lst_of_expr: List of CPMpy expressions to linearize. Must be in 'flat normal form'
            with only boolean variables on the left-hand side of implications.
        supported: Set of constraint and variable type names that are supported by the target
            solver, e.g. `{"sum", "wsum", "->", "and", "or", "alldifferent"}`.
            :class:`~cpmpy.expressions.globalconstraints.AllDifferent` has a special linearization
            and is decomposed as such if not in `supported`.
            Any other unsupported global constraint should be decomposed using
            :func:`cpmpy.transformations.decompose_global.decompose_in_tree()`.
        reified: Whether the constraint is fully reified. When True, nested implications
            are linearized using Big-M method instead of indicator constraints.
        csemap: Optional dictionary for common subexpression elimination, mapping expressions
            to their flattened variable representations. Used to avoid creating duplicate
            variables for the same subexpression.

    Returns:
        List of linearized CPMpy expressions. The constraints are in one of the forms
        described above (linear comparisons, general expressions, or indicator constraints).
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
                    # use the specialized linearization for multiplication
                    mul_expr, mul_cons = decompose_mul_linear(lhs, supported=supported, reified=reified, csemap=csemap)
                    newlist += mul_cons
                    cpm_expr = eval_comparison(cpm_expr.name, mul_expr, rhs)

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

def only_positive_bv(lst_of_expr: List[Expression], csemap: Optional[Dict[Any, Any]] = None) -> List[Expression]:
    """
    Replaces :class:`~cpmpy.expressions.comparison.Comparison` containing :class:`~cpmpy.expressions.variables.NegBoolView`
    with equivalent expression using only :class:`~cpmpy.expressions.variables.BoolVar`.
    
    Comparisons are expected to be linearized. Only apply after applying :func:`linearize_constraint(cpm_expr) <linearize_constraint>`.
    The resulting expression is linear if the original expression was linear.

    Arguments:
        lst_of_expr: List of linearized CPMpy expressions that may contain NegBoolView.
        csemap: Optional dictionary for common subexpression elimination, mapping expressions
            to an equivalent decision variable.

    Returns:
        List of CPMpy expressions where all boolean variables appear positively (no NegBoolView).
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

def only_positive_bv_wsum(expr: Expression) -> Expression:
    """
    Replaces a var/sum/wsum expression containing :class:`~cpmpy.expressions.variables.NegBoolView`
    with an equivalent expression using only :class:`~cpmpy.expressions.variables.BoolVar`.
    
    It might add a constant term to the expression. If you want the constant separately,
    use :func:`only_positive_bv_wsum_const`.

    Arguments:
        expr: Linear expression (NumVar, sum, or wsum) that may contain NegBoolView.

    Returns:
        Linear expression (NumVar, sum, or wsum) without NegBoolView. The constant term
        (if any) is incorporated into the expression.
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

def only_positive_bv_wsum_const(cpm_expr: Expression) -> Tuple[Expression, int]:
    """
    Replaces a var/sum/wsum expression containing :class:`~cpmpy.expressions.variables.NegBoolView`
    with an equivalent expression using only :class:`~cpmpy.expressions.variables.BoolVar`,
    and returns the constant term separately.
    
    If you want the expression where the constant term is part of the wsum returned,
    use :func:`only_positive_bv_wsum`.

    Arguments:
        cpm_expr: Linear expression (NumVar, sum, or wsum) that may contain NegBoolView.

    Returns:
        Tuple of:
        - pos_expr: Linear expression (NumVar, sum, or wsum) without NegBoolView.
        - const: The constant term that must be added to pos_expr to make it equivalent
                 to the original expression.
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


def canonical_comparison(lst_of_expr: Union[Expression, List[Expression]]) -> List[Expression]:
    """
    Canonicalize comparison expressions.
    
    Transforms linear expressions, or a reification thereof, into canonical form by:
    - moving all variables to the left-hand side
    - moving constants to the right-hand side

    Expects the input constraints to be flat. Only apply after applying :func:`flatten_constraint`.

    Arguments:
        lst_of_expr: Single expression or list of CPMpy expressions to canonicalize.
                     Can be a single :class:`~cpmpy.expressions.core.Expression` or a list.

    Returns:
        List of canonicalized CPMpy expressions. All variables are on the left-hand side
        and all constants are on the right-hand side of comparisons.
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

def only_positive_coefficients(lst_of_expr: List[Expression]) -> List[Expression]:
    """
    Replaces Boolean terms with negative coefficients in linear constraints with terms
    with positive coefficients by negating the literal.
    
    This can simplify a `wsum` into `sum` when all coefficients become 1.
    Input expressions are expected to be canonical comparisons. Only apply after
    applying :func:`canonical_comparison(cpm_expr) <canonical_comparison>`.

    Arguments:
        lst_of_expr: List of canonical CPMpy expressions (comparisons) that may contain
                     boolean variables with negative coefficients.

    Returns:
        List of CPMpy expressions where all boolean variables have positive coefficients.
        The resulting expression is linear if the original expression was linear.
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
