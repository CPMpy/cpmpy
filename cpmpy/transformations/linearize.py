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
from cpmpy.transformations.normalize import toplevel_list
from .decompose_global import decompose_in_tree

from .flatten_model import flatten_constraint, get_or_make_var
from .get_variables import get_variables
from .. import Abs
from ..exceptions import TransformationNotImplementedError

from ..expressions.core import Comparison, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.utils import ExprStore, get_store, is_any_list, is_num, eval_comparison, is_bool, get_bounds

from ..expressions.variables import _BoolVarImpl, boolvar, NegBoolView, _NumVarImpl, intvar


def linearize_constraint(lst_of_expr, supported={"sum","wsum"}, expr_store:ExprStore=None, reified=False):
    """
    Transforms all constraints to a linear form.
    This function assumes all constraints are in 'flat normal form' with only boolean variables on the lhs of an implication.
    Only apply after 'cpmpy.transformations.flatten_model.flatten_constraint()' 'and only_bv_implies()'.

    `AllDifferent` has a special linearization and is decomposed as such if not in `supported`.
    Any other unsupported global constraint should be decomposed using `cpmpy.transformations.decompose_global.decompose_global()`

    """
    return _linearize_constraint_helper(lst_of_expr, supported, expr_store, reified)[0]


def _linearize_constraint_helper(lst_of_expr, supported={"sum","wsum"}, expr_store:ExprStore=None, reified=False):
    """
    Helper function for linearize_constraint.
    Besides a list of linearised expressions, also keeps track for the newly introduced variables as to prevent unwanted symmetries.
    """

    if expr_store is None:
        expr_store = get_store()

    newlist = [] # to collect the linearized constraints
    newvars = [] # to collect the newly introduced variables
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
                assert isinstance(cond, _BoolVarImpl), f"Linearization of {cpm_expr} is not supported, lhs of implication must be boolvar. Apply `only_bv_implies` before calling `linearize_constraint`"

                if isinstance(cond, _BoolVarImpl) and isinstance(sub_expr, _BoolVarImpl):
                    # shortcut for BV -> BV, convert to disjunction and apply linearize on it
                    newlist.append(1 * cond + -1 * sub_expr <= 0)

                # BV -> LinExpr
                elif isinstance(cond, _BoolVarImpl):
                    lin_sub, new_vars = _linearize_constraint_helper([sub_expr], supported=supported, reified=True, expr_store=expr_store)
                    newlist += [cond.implies(lin) for lin in lin_sub]
                    a, b = _linearize_constraint_helper([(~cond).implies(nv == nv.lb) for nv in new_vars], reified=reified, expr_store=expr_store)
                    newlist += a
                    newvars += b

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
                elif lhs.name == 'div':
                    a, b = lhs.args
                    # if division is total, b is either strictly negative or strictly positive!
                    lb, ub = get_bounds(b)
                    if not ((lb < 0 and ub < 0) or (lb > 0 and ub > 0)):
                        raise TypeError(
                            f"Can't divide by a domain containing 0, safen the expression first")
                    r = intvar(0, max(abs(lb) - 1, abs(ub) - 1))  # remainder is always positive for floordivision.
                    cpm_expr = [eval_comparison(cpm_expr.name, a, b * rhs + r)]
                    cond = [r < Abs(b)]
                    decomp = toplevel_list(decompose_in_tree(cond))  # decompose abs
                    cpm_exprs = toplevel_list(decomp + cpm_expr)
                    exprs = linearize_constraint(flatten_constraint(cpm_exprs, expr_store=expr_store, skip_simplify_bool=True), supported=supported, expr_store=expr_store)
                    newlist.extend(exprs)
                    continue
                    #newrhs = lhs.args[0]
                    #lhs = lhs.args[1] * rhs #operator is actually always '==' here due to only_numexpr_equality
                    #cpm_expr = eval_comparison(cpm_expr.name, lhs, newrhs)
                elif lhs.name == 'idiv':
                    a, b = lhs.args
                    # if division is total, b is either strictly negative or strictly positive!
                    lb, ub = get_bounds(b)
                    if not ((lb < 0 and ub < 0) or (lb > 0 and ub > 0)):
                        raise TypeError(
                            f"Can't divide by a domain containing 0, safen the expression first")
                    r = intvar(-(max(abs(lb) - 1, abs(ub) - 1)), max(abs(lb) - 1, abs(ub) - 1)) # remainder can be both positive and negative
                    cpm_expr = [eval_comparison(cpm_expr.name, a, b * rhs + r)]
                    cond = [Abs(r) < Abs(b), Abs(b * rhs) < Abs(a)]
                    decomp = toplevel_list(decompose_in_tree(cond))  # decompose abs
                    cpm_exprs = toplevel_list(decomp + cpm_expr)
                    exprs = linearize_constraint(flatten_constraint(cpm_exprs, expr_store=expr_store, skip_simplify_bool=True), supported=supported, expr_store=expr_store)
                    newlist.extend(exprs)
                    continue

                    
                elif lhs.name == 'mod':  # x mod y == x - (x//y) * y
                    # gets handles in the solver interface
                    # We should never get here, since both Gurobi and Exact have "faked support" for "Mod"
                    newlist.append(cpm_expr)
                    continue
                else:
                    raise TransformationNotImplementedError(f"lhs of constraint {cpm_expr} cannot be linearized, should be any of {supported | set(['sub'])} but is {lhs}. Please report on github")

            elif isinstance(lhs, GlobalConstraint) and lhs.name not in supported:
                raise ValueError("Linearization of `lhs` not supported, run `cpmpy.transformations.decompose_global.decompose_global() first")

            [cpm_expr] = canonical_comparison([cpm_expr])  # just transforms the constraint, not introducing new ones
            lhs, rhs = cpm_expr.args

            # now fix the comparisons themselves
            if cpm_expr.name == "<":
                new_rhs, cons = get_or_make_var(rhs - 1, expr_store=expr_store) # if rhs is constant, will return new constant
                newlist.append(lhs <= new_rhs)
                if isinstance(new_rhs, _NumVarImpl):
                    newvars.append(new_rhs)
                a,b = _linearize_constraint_helper(cons, expr_store=expr_store)
                newlist += a
                newvars += b
            elif cpm_expr.name == ">":
                new_rhs, cons = get_or_make_var(rhs + 1, expr_store=expr_store) # if rhs is constant, will return new constant
                newlist.append(lhs >= new_rhs)
                if isinstance(new_rhs, _NumVarImpl):
                    newvars.append(new_rhs)
                a,b = _linearize_constraint_helper(cons, expr_store=expr_store)
                newlist += a
                newvars += b
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
                    a,b = _linearize_constraint_helper(flatten_constraint(cons, expr_store=expr_store), supported=supported, reified=reified, expr_store=expr_store)
                    newlist += a
                    newvars += b + [z]
                else:
                    # introduce new indicator constraints
                    z = boolvar()
                    constraints = [z.implies(lhs < rhs), (~z).implies(lhs > rhs)]
                    a,b = _linearize_constraint_helper(constraints, supported=supported, reified=reified, expr_store=expr_store)
                    newlist += a
                    newvars += b + [z]
            else:
                # supported comparison
                newlist.append(eval_comparison(cpm_expr.name, lhs, rhs))

        elif cpm_expr.name == "alldifferent" and cpm_expr.name in supported:
            newlist.append(cpm_expr)
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

            newlist += constraints
            newvars += [sigma]

        elif isinstance(cpm_expr, (DirectConstraint, BoolVal)):
            newlist.append(cpm_expr)

        elif isinstance(cpm_expr, GlobalConstraint) and cpm_expr.name not in supported:
            raise ValueError(f"Linearization of global constraint {cpm_expr} not supported, run `cpmpy.transformations.decompose_global.decompose_global() first")
    
    return (newlist, newvars)


def only_positive_bv(lst_of_expr, expr_store:ExprStore=None):
    """
        Replaces constraints containing NegBoolView with equivalent expression using only BoolVar.
        cpm_expr is expected to be linearized. Only apply after applying linearize_constraint(cpm_expr)

        Resulting expression is linear.
    """
    if expr_store is None:
        expr_store = get_store()

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
                        new_arg, cons = get_or_make_var(1 - arg, expr_store=expr_store)
                        lhs.args[i] = new_arg
                        new_cons += cons

            newlist.append(eval_comparison(cpm_expr.name, lhs, rhs))
            newlist += linearize_constraint(new_cons, expr_store=expr_store)

        # reification
        elif cpm_expr.name == "->":
            cond, subexpr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"{cpm_expr} is not a supported linear expression. Apply `linearize_constraint` before calling `only_positive_bv`"
            if isinstance(cond, _BoolVarImpl): # BV -> Expr
                subexpr = only_positive_bv([subexpr], expr_store=expr_store)
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
                    if len(lhs2) != 0:
                        lhs, rhs = lhs + lhs2, rhs
                else:
                    raise ValueError(
                        f"unexpected expression on lhs of expression, should be sum,wsum or intvar but got {lhs}")

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
