"""
Transform CPMpy expressions into expression trees suitable for solvers that support non-linear expression trees. This transformation is currently quite specific towards Gurobi's new non-linear expression tree support.

Unlike the flatten/linearize pipeline, this transformation preserves
non-linear structure (e.g. mul, pow) as expression tree nodes, only flattening
what the solver cannot handle natively:
- Unsupported constraints (e.g. and, or, min, max, abs) are reified into y=f(x) form

currently, there are gurobi specific transformations:

- Indicator constraints require linear bodies, so non-linear parts are reified
- Reification of boolean expressions uses bi-implication with indicator constraints
- != is decomposed into a disjunction of strict inequalities


There are two functions: `into_tree` (for use in transformations, returns constraints as expression trees), and `into_tree_expr`, which returns both the root node and new top-level (defining) constraints
"""

import copy
import numpy as np

# TODO mainly, unduplicate the added ad-hoc transformations done by the other transformation
# TODO specific to gurobi: handling neq, linearize for indicator

import cpmpy as cp
from ..expressions.core import Comparison, Operator, is_boolexpr
from ..exceptions import NotSupportedError
from ..expressions.utils import is_num
from ..expressions.variables import _BoolVarImpl, NegBoolView, _NumVarImpl
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.globalfunctions import GlobalFunction
from .negation import recurse_negation, push_down_negation


def into_tree_expr(cpm_expr, csemap=None, verbose=False, supported={}, general_constraints={}, reified=False):
    """Same as into_tree, but returns a tuple of the root node of the tree and the new top-level (definining) constraints."""

    if csemap is None:
        csemap = {}

    # constraints will be added to this list
    cpm_cons = []

    def add(cpm_expr):
        """Add this expression to the tree, flattening if unsupported."""
        cpm_cons.append(into_tree_expr_(cpm_expr, 0))

    def get_or_make_var(cpm_expr, define=True):
        """Get or make (Boolean/integer) var `b` which represent the expression. Add defining constraints b == cpm_expr if `define=True`."""
        if cpm_expr in csemap:
            return csemap[cpm_expr]

        r = cp.boolvar() if is_boolexpr(cpm_expr) else cp.intvar(*cpm_expr.get_bounds())
        csemap[cpm_expr] = r
        if define:
            add(r == cpm_expr)
        return r

    def with_args(cpm_expr, args):
        """Copy expression and replace args"""
        cpm_expr = copy.copy(cpm_expr)
        cpm_expr.update_args(args)
        return cpm_expr

    def propagate_boolconst(name, args):
        """Propagate boolean constants in and/or: and(0,...)=0, or(1,...)=1, filter neutral elements."""
        match name:
            case "and" | "or":
                # TODO single loop should be possible
                absorb = 0 if name == "and" else 1  # absorbing element
                args = [a for a in args if not (is_num(a) and a == 1 - absorb)]
                if any(is_num(a) and a == absorb for a in args):
                    return absorb
                return args
            case _:
                return args

    def add_general_constraint(f, depth, reified=False, y=None):
        """Add the general constraint `f(x)` in gurobi's required form, `y=f(x)` with `x` a list of variables (Boolean/integer, depending on the general constraint type)"""
        # require only variables
        # f(x1, x2, ..) === f(y1, y2, ..), y1=x1, y2=x2, ..
        args = [reify(into_tree_expr_(arg, depth, reified=True), depth) for arg in f.args]
        # require non-constants
        args = propagate_boolconst(f.name, args)
        if is_num(args):  # may have become fixed (e.g. `and(x1, 0, x2) === 0`)
            return args
        else:
            # require the form: y = f(x)
            if y is None:
                # assert reified or f.name in {"and", "or"}, f"Unexpected numexpr {f} encountered at root level"
                y = get_or_make_var(f, define=False) if reified else 1

            # y, y = f(x)
            f = with_args(f, args)
            cpm_cons.append(y == f)  # add directly so that Comparison does not have to deal with it
            # TODO could be done by add(y == f)?
            return y

    def reify(cpm_expr, depth):
        """Return a variable representing the expression, for use as argument in general/indicator constraints."""
        if verbose:
            print(f"{'  ' * depth}reify", cpm_expr, getattr(cpm_expr, "name", None))

        if is_num(cpm_expr):
            return cpm_expr
        elif isinstance(cpm_expr, _NumVarImpl) and not isinstance(cpm_expr, NegBoolView):
            return cpm_expr
        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            # Convert p -> q to or(~p, q) to avoid circular reification
            a, b = cpm_expr.args
            return reify((~a) | b, depth)
        else:
            return get_or_make_var(cpm_expr)

    def add_comparison(cpm_expr, depth, reified=False, _supported=supported):
        """Process a Comparison expression. Returns a Comparison or True (posted as side effect).
        If reified=True, returns a BoolVar representing the truth value of the comparison."""
        a, b = cpm_expr.args

        match cpm_expr.name:
            case "==" if (
                not reified
                and isinstance(a, (int, _NumVarImpl))
                and isinstance(b, (GlobalFunction, Operator))
                and b.name in general_constraints
            ):
                # already of the form: y = f(x)
                add_general_constraint(b, depth, reified=reified, y=a)
                return True
            case "==" if isinstance(a, _BoolVarImpl) and is_boolexpr(b) and not isinstance(b, _NumVarImpl):
                # BV == boolexpr === BV <-> boolexpr: post as bi-implications
                add(a.implies(b))
                add((~a).implies(recurse_negation(b)))
                con = True
            case "==" | "<=" | ">=":
                a, b = into_tree_expr_(a, depth, reified=True, _supported=_supported), into_tree_expr_(b, depth, reified=True, _supported=_supported)
                con = with_args(cpm_expr, [a, b])
            case "!=":
                # One-directional indicator split: d=1 -> a>b, e=1 -> a<b
                (cpm_expr,) = push_down_negation([cpm_expr], toplevel=not reified)
                if cpm_expr.name == "!=":
                    a, b = cpm_expr.args
                    if reified:
                        return into_tree_expr_((a > b) | (a < b), depth, reified=reified, _supported=_supported)
                    else:
                        z = cp.boolvar()
                        add(z.implies(a > b))
                        add((~z).implies(a < b))
                        return True
                else:  # push_down_negation may have changed e.g. != into ==
                    return into_tree_expr_(cpm_expr, depth, reified=reified, _supported=_supported)
            case ">":
                return into_tree_expr_(a >= b + 1, depth, reified=reified, _supported=_supported)
            case "<":
                return into_tree_expr_(a <= b - 1, depth, reified=reified, _supported=_supported)
            case _:
                raise Exception(
                    f"Expected comparator to be ==,<=,>= in Comparison expression {cpm_expr}, but was {cpm_expr.name}"
                )

        return reify(con, depth) if reified else con

    # Linear operators that can remain as tree nodes in indicator bodies
    linear_supported = supported & {"sum", "wsum", "sub", "-"}

    def raise_unexpected_expr(cpm_expr):
        raise NotSupportedError(f"Unexpected constraint {cpm_expr} in `into_tree`")

    def into_tree_expr_(cpm_expr, depth, reified=False, _supported=supported):
        indent = "  " * depth
        depth += 1
        if verbose:
            print(f"{indent}Con:", cpm_expr, type(cpm_expr), "reif" if reified else "root")

        if is_num(cpm_expr):
            # TODO clean this up ; currently overly complicated due to float expressions
            return int(cpm_expr) if isinstance(cpm_expr, (bool, np.bool_)) or not isinstance(cpm_expr, (int, float)) else cpm_expr
        elif isinstance(cpm_expr, _NumVarImpl):
            return cpm_expr
        elif isinstance(cpm_expr, (Operator, GlobalFunction)):
            match cpm_expr.name:
                case "->":
                    a, b = cpm_expr.args
                    # if isinstance(b, (_NumVarImpl, Comparison)):  # TODO requires reification?
                    if isinstance(b, Comparison):
                        # Gurobi indicator constraint: (Var == 0|1) >> (LinExpr sense LinExpr) (note: not required to be a canonical comparison)
                        p = a if isinstance(a, NegBoolView) else reify(a, depth)
                        if is_num(p):  # propagate fixed antecedent
                            return into_tree_expr_(b, depth, reified=reified) if a else True
                        assert isinstance(p, _BoolVarImpl)
                        # Indicator body must be linear; restrict supported to linear operators
                        can_be_linear = isinstance(b, Comparison) and b.name != "!="
                        q = into_tree_expr_(b, depth, reified=not can_be_linear, _supported=linear_supported)
                        # Ensure result is a Comparison (indicator body requirement)
                        if isinstance(q, _NumVarImpl):
                            q = q >= 1
                        elif not isinstance(q, Comparison):
                            if is_num(q):
                                return True if q else into_tree_expr_(p, depth, reified=reified)
                            q = reify(q, depth) >= 1
                        assert isinstance(q, Comparison), f"Expected linear constraint, but got {q}"
                        return with_args(cpm_expr, [p, q])
                    else:
                        # Other constraint will have to be handled as logical constraints
                        return into_tree_expr_((~a) | b, depth, reified=reified)
                case "not":  # not is not handled by gurobi
                    (a,) = cpm_expr.args
                    return into_tree_expr_(recurse_negation(a), depth, reified=True)
                case name if name in _supported:  # Expression tree nodes (w/ args)
                    assert cpm_expr.name != "div", "TODO"
                    if name == "wsum":  # Just for efficiency, don't call add on the weights
                        ws, xs = cpm_expr.args
                        return with_args(cpm_expr, [ws, [into_tree_expr_(x, depth, reified=True, _supported=_supported) for x in xs]])
                    else:
                        return with_args(cpm_expr, [into_tree_expr_(a, depth, reified=True, _supported=_supported) for a in cpm_expr.args])
                case name if name in general_constraints:
                    return add_general_constraint(cpm_expr, depth, reified=reified)
                case _:
                    # Operator not in current _supported set (e.g. non-linear op in linear context): reify
                    if cpm_expr.name in supported:
                        return reify(cpm_expr, depth)
                    raise_unexpected_expr(cpm_expr)
        elif isinstance(cpm_expr, Comparison):
            return add_comparison(cpm_expr, depth, reified=reified, _supported=_supported)
        elif isinstance(cpm_expr, DirectConstraint):
            return cpm_expr  # pass through to solver-specific posting
        else:
            raise_unexpected_expr(cpm_expr)

    return into_tree_expr_(cpm_expr, 0, reified=reified), cpm_cons


def into_tree(cpm_expr, csemap=None, verbose=False, supported={}, general_constraints={}):
    """Transform CPMpy expressions into an expression tree, flattening unsupported expressions but keeping supported ones as tree nodes.

    Recursively processes expressions, keeping mul/pow/sum as tree nodes and reifying only what the solver cannot handle natively (general constraints, non-linear indicator bodies, !=). Side-effect constraints (from reification and decomposition) are collected and returned alongside the main constraints.

    Args:
        cpm_expr: CPMpy expression or list of expressions to transform
        csemap: dict for common subexpression elimination (shared across calls)
        supported: set of supported constraints (will not be flattened)
        general_constraints: set of Gurobi general constraints (will be left as y = f(x1, x2, ...) with xi being integer/Boolean variables).
        verbose: if True, print debug trace of the transformation

    Returns:
        list of transformed CPMpy constraints ready for solver-specific posting
    """

    cpm_cons = []
    for c in cpm_expr:
        root, cons = into_tree_expr(
            c,
            csemap=csemap,
            verbose=verbose,
            supported=supported,
            general_constraints=general_constraints,
        )
        cpm_cons += [root] + cons
    return cpm_cons
