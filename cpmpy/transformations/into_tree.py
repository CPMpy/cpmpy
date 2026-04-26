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
from ..expressions.utils import is_num, is_int
from ..expressions.variables import _BoolVarImpl, _IntVarImpl, NegBoolView, _NumVarImpl
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.globalfunctions import GlobalFunction
from .negation import recurse_negation, push_down_negation
from .int2bool import _encode_int_var


def handle_implication(cpm_expr, depth, reified, handlers, ctx):
    """Handle implication (indicator) constraints for solvers like Gurobi.

    When the body is a Comparison, produces an indicator constraint with a linearized body.
    Otherwise, converts p -> q to or(~p, q) for handling as a logical constraint.
    """
    a, b = cpm_expr.args
    if isinstance(b, Comparison):
        # Indicator constraint: (Var == 0|1) >> (LinExpr sense LinExpr)
        p = a if isinstance(a, NegBoolView) else ctx.reify(a, depth)
        if is_num(p):  # propagate fixed antecedent
            return ctx.recurse(b, depth, reified=reified) if a else True
        assert isinstance(p, _BoolVarImpl)
        # Indicator body must be linear; restrict tree-node handlers to linear operators
        can_be_linear = isinstance(b, Comparison) and b.name != "!="
        linear_supported = {
            n: h
            for n, h in handlers.items()
            if n in {"sum", "wsum", "sub", "-"} or h not in (handle_supported_expr, handle_wsum)
        }
        q = ctx.recurse(b, depth, reified=not can_be_linear, handlers=linear_supported)
        # Ensure result is a Comparison (indicator body requirement)
        if isinstance(q, _NumVarImpl):
            q = q >= 1
        elif not isinstance(q, Comparison):
            if is_num(q):
                return True if q else ctx.recurse(p, depth, reified=reified)
            q = ctx.reify(q, depth) >= 1
        assert isinstance(q, Comparison), f"Expected linear constraint, but got {q}"
        return _with_args(cpm_expr, [p, q])
    else:
        # Other constraint will have to be handled as logical constraints
        return ctx.recurse((~a) | b, depth, reified=reified)


def handle_strict_ineq(cpm_expr, depth, reified, handlers, ctx):
    """Convert strict inequalities to non-strict for integer solvers: a > b becomes a >= b+1, a < b becomes a <= b-1."""
    a, b = cpm_expr.args
    if cpm_expr.name == ">":
        return ctx.recurse(a >= b + 1, depth, reified=reified, handlers=handlers)
    else:  # "<"
        return ctx.recurse(a <= b - 1, depth, reified=reified, handlers=handlers)


def handle_neq(cpm_expr, depth, reified, handlers, ctx):
    """Decompose != into a disjunction of strict inequalities using one-directional indicators."""
    result = push_down_negation([cpm_expr], toplevel=not reified)
    if len(result) == 1 and result[0].name == "!=":
        a, b = result[0].args
        if reified:
            return ctx.recurse((a > b) | (a < b), depth, reified=reified, handlers=handlers)
        else:
            z = cp.boolvar()
            ctx.add(z.implies(a > b))
            ctx.add((~z).implies(a < b))
            return True
    elif len(result) == 1:  # push_down_negation changed the expression (e.g. bool != to ==)
        return ctx.recurse(result[0], depth, reified=reified, handlers=handlers)
    else:  # push_down_negation decomposed into multiple constraints (toplevel only)
        for c in result:
            ctx.add(c)
        return True


def handle_supported_expr(cpm_expr, depth, reified, handlers, ctx):
    """Identity handler: recurse on args, keeping the expression as a tree node.

    When the expression is not in the current handlers context (e.g. non-linear op
    in a linear indicator body), reify it instead.
    """
    if cpm_expr.name not in handlers:
        return ctx.reify(cpm_expr, depth)
    if cpm_expr.name == "wsum":
        ws, xs = cpm_expr.args
        xs = [ctx.recurse(x, depth, reified=True, handlers=handlers) for x in xs]
        return _with_args(cpm_expr, [ws, xs])
    return _with_args(
        cpm_expr, [ctx.recurse(a, depth, reified=True, handlers=handlers) for a in cpm_expr.args]
    )


handle_wsum = handle_supported_expr


def handle_general_constraint(cpm_expr, depth, reified, handlers, ctx, y=None):
    """Handle general constraints in y=f(x) form, with x a list of variables.

    Used for constraints like max, min, abs, and, or that solvers like Gurobi
    require in the form y = f(x1, x2, ...) with all xi being variables.
    """
    # require only variables: f(x1, x2, ..) === f(y1, y2, ..), y1=x1, y2=x2, ..
    args = [ctx.reify(ctx.recurse(arg, depth, reified=True), depth) for arg in cpm_expr.args]
    # require non-constants
    args = _propagate_boolconst(cpm_expr.name, args)
    if is_num(args):  # may have become fixed (e.g. `and(x1, 0, x2) === 0`)
        return args
    # require the form: y = f(x)
    if y is None:
        y = ctx.get_or_make_var(cpm_expr, define=False) if reified else 1
    f = _with_args(cpm_expr, args)
    ctx.post(y == f)  # add directly so that Comparison does not have to deal with it
    return y


def _with_args(cpm_expr, args):
    """Copy expression and replace args."""
    cpm_expr = copy.copy(cpm_expr)
    cpm_expr.update_args(args)
    return cpm_expr


def _propagate_boolconst(name, args):
    """Propagate boolean constants in and/or: and(0,...)=0, or(1,...)=1, filter neutral elements."""
    match name:
        case "and" | "or":
            absorb = 0 if name == "and" else 1  # absorbing element
            args = [a for a in args if not (is_num(a) and a == 1 - absorb)]
            if any(is_num(a) and a == absorb for a in args):
                return absorb
            return args
        case _:
            return args


def into_tree_expr(cpm_expr, csemap=None, verbose=False, reified=False, handlers=None):
    """Same as into_tree, but returns a tuple of the root node of the tree and the new top-level (definining) constraints."""

    if csemap is None:
        from ..transformations.cse import CSEMap

        csemap = CSEMap()
    if handlers is None:
        handlers = {}

    # constraints will be added to this list
    cpm_cons = []
    ivarmap = {}  # int2bool encoding map for direct encoding of integer variables

    def add(cpm_expr):
        """Add this expression to the tree, flattening if unsupported."""
        cpm_cons.append(into_tree_expr_(cpm_expr, 0))

    def get_or_make_var(cpm_expr, define=True):
        """Get or make (Boolean/integer) var `b` which represent the expression. Add defining constraints b == cpm_expr if `define=True`."""
        cached = csemap.get(cpm_expr)
        if cached is not None:
            return cached

        # For x ==/!= val with integer x, eagerly create direct encoding
        if (
            isinstance(cpm_expr, Comparison)
            and cpm_expr.name in ("==", "!=")
            and isinstance(cpm_expr.args[0], _IntVarImpl)
            and not isinstance(cpm_expr.args[0], _BoolVarImpl)
            and is_int(cpm_expr.args[1])
        ):
            var, val = cpm_expr.args
            enc, domain_cons = _encode_int_var(ivarmap, var, "direct", csemap=csemap)
            if domain_cons:  # first time encoding this variable
                for dc in domain_cons:
                    add(dc)
                # channeling: var == wsum(vals, bvs) + k
                terms, k = enc.encode_term()
                add(cp.sum(w * b for w, b in terms) + k == var)
            bv = enc.eq(val)
            return ~bv if cpm_expr.name == "!=" else bv

        r = cp.boolvar() if is_boolexpr(cpm_expr) else cp.intvar(*cpm_expr.get_bounds())
        csemap.flat_map[cpm_expr] = r
        if define:
            add(r == cpm_expr)
        return r

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

    def raise_unexpected_expr(cpm_expr):
        raise NotSupportedError(f"Unexpected constraint {cpm_expr} in `into_tree`")

    def into_tree_expr_(cpm_expr, depth, reified=False, handlers=handlers):
        indent = "  " * depth
        depth += 1
        if verbose:
            print(f"{indent}Con:", cpm_expr, type(cpm_expr), "reif" if reified else "root")

        if is_num(cpm_expr):
            # TODO clean this up ; currently overly complicated due to float expressions
            return int(cpm_expr) if isinstance(cpm_expr, (bool, np.bool_)) or not isinstance(cpm_expr, (int, float)) else cpm_expr
        elif isinstance(cpm_expr, _NumVarImpl):
            return cpm_expr
        elif isinstance(cpm_expr, (Operator, GlobalFunction, Comparison)):
            match cpm_expr.name:
                case name if name in handlers:
                    return handlers[name](cpm_expr, depth, reified, handlers, ctx)
                case "not":
                    (a,) = cpm_expr.args
                    return into_tree_expr_(recurse_negation(a), depth, reified=True)
                case "==" if (
                    isinstance(cpm_expr.args[0], _BoolVarImpl)
                    and is_boolexpr(cpm_expr.args[1])
                    and not isinstance(cpm_expr.args[1], _NumVarImpl)
                ):
                    # BV == boolexpr === BV <-> boolexpr: post as bi-implications
                    a, b = cpm_expr.args
                    add(a.implies(b))
                    add((~a).implies(recurse_negation(b)))
                    return True
                case "==" | "<=" | ">=":
                    a, b = (
                        into_tree_expr_(cpm_expr.args[0], depth, reified=True, handlers=handlers),
                        into_tree_expr_(cpm_expr.args[1], depth, reified=True, handlers=handlers),
                    )
                    con = _with_args(cpm_expr, [a, b])
                    return reify(con, depth) if reified else con
                case _:
                    # Expression not in current handlers: reify into a variable
                    return reify(cpm_expr, depth)
        elif isinstance(cpm_expr, DirectConstraint):
            return cpm_expr  # pass through to solver-specific posting
        else:
            raise_unexpected_expr(cpm_expr)

    # Context object providing helpers to external handlers
    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.recurse = into_tree_expr_
    ctx.reify = reify
    ctx.add = add
    ctx.post = cpm_cons.append
    ctx.get_or_make_var = get_or_make_var

    return into_tree_expr_(cpm_expr, 0, reified=reified), cpm_cons


def into_tree(cpm_expr, csemap=None, verbose=False, handlers=None):
    """Transform CPMpy expressions into an expression tree, flattening unsupported expressions but keeping supported ones as tree nodes.

    Recursively processes expressions, keeping mul/pow/sum as tree nodes and reifying only what the solver cannot handle natively. Side-effect constraints (from reification and decomposition) are collected and returned alongside the main constraints.

    Args:
        cpm_expr: CPMpy expression or list of expressions to transform
        csemap: dict for common subexpression elimination (shared across calls)
        handlers: dict mapping expression names to handler functions.
            Each handler receives (cpm_expr, depth, reified, handlers, ctx).
            Use handle_supported_expr/handle_wsum for operators that should stay as tree nodes.
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
            handlers=handlers,
        )
        cpm_cons += [root] + cons
    return cpm_cons
