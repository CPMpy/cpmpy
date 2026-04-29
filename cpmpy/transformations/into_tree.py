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
from ..expressions.utils import is_num, is_int, get_bounds
from ..expressions.variables import _BoolVarImpl, _IntVarImpl, NegBoolView, _NumVarImpl
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.globalfunctions import GlobalFunction
from .negation import recurse_negation, push_down_negation
from .int2bool import _encode_int_var

BIG_M_NEQ = True  # Use Big-M formulation for != instead of indicator-based OR decomposition
QUAD_AND = True  # Use quadratic p == a * b for binary AND of 2 vars instead of GenConstr AND
NAMED = False


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
            return ctx.recurse(b, depth, reified=reified) if p else True
        assert isinstance(p, _BoolVarImpl)
        # p -> (a != b): decompose into (p & z) -> (lhs > rhs), (p & ~z) -> (lhs < rhs)
        # using Big-M to linearize the conjunction in the antecedent (2 indicators vs 6 + 1 OR)
        if BIG_M_NEQ and b.name == "!=":
            lhs, rhs = b.args
            z = cp.boolvar()
            _, M1 = (lhs - rhs + 1).get_bounds()
            _, M2 = (rhs - lhs + 1).get_bounds()
            lhs_lt_rhs = p.implies(lhs - M1 * z <= rhs - 1)  # z=1: True; z=0: lhs < rhs
            lhs_gt_rhs = p.implies(lhs - M2 * z >= rhs - M2 + 1)  # z=0: True; z=1: lhs > rhs
            # ctx.add((~p).implies(z <= 0))  # redundant: z only appears in p's indicator bodies
            return ctx.recurse(lhs_lt_rhs & lhs_gt_rhs, depth, reified=reified)

        # Indicator body must be linear; restrict tree-node handlers to linear operators
        def _contains_general(expr):
            if hasattr(expr, "name") and expr.name in ("and", "or", "abs", "min", "max", "->"):
                return True
            return hasattr(expr, "args") and any(_contains_general(a) for a in expr.args)

        def handle_lin_con(cpm_expr, depth, reified, handlers, ctx):
            def handle_lin_exp(cpm_expr, depth, reified, handlers, ctx):
                def handle_lin_term(cpm_expr, depth, reified, handlers, ctx):
                    def handle_mul_term(cpm_expr, depth, reified, handlers, ctx):
                        if is_num(cpm_expr.args[0]) or is_num(cpm_expr.args[1]):
                            return cpm_expr
                        else:
                            return ctx.reify(cpm_expr, depth)

                    return handle_supported_expr(
                        cpm_expr, depth, reified, {"mul": handle_mul_term, "sum": handle_lin_term, "wsum": handle_wsum}, ctx
                    )

                return ctx.recurse(
                    cpm_expr,
                    depth,
                    True,
                    {k: handle_lin_term for k in {"sum", "wsum", "sub", "-"}},
                )

            a, b = cpm_expr.args
            return _with_args(
                cpm_expr, [handle_lin_exp(a, depth, reified, handlers, ctx), handle_lin_exp(b, depth, reified, handlers, ctx)]
            )

        lin_con_handlers = {k: v for k, v in handlers.items() if k not in {"pow", "->"}} | {
            k: handle_lin_con for k in (">=", "<=", "==")
        }
        q = ctx.recurse(b, depth, handlers=lin_con_handlers)

        # Ensure result is a Comparison (indicator body requirement)
        if isinstance(q, _NumVarImpl):
            q = q >= 1
        elif not isinstance(q, Comparison):
            if is_num(q):
                return True if q else ctx.recurse(~p, depth, reified=reified)
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
        if BIG_M_NEQ and not reified:
            # Big-M formulation: z selects a > b (z=1) or a < b (z=0)
            # a - b + 1 <= M1*z  (z=0 forces a < b; z=1 is trivially satisfied)
            # b - a + 1 <= M2*(1-z)  (z=1 forces a > b; z=0 is trivially satisfied)
            z = cp.boolvar()
            _, M1 = (a - b + 1).get_bounds()
            _, M2 = (b - a + 1).get_bounds()
            lhs_lt_rhs = a - M1 * z <= b - 1  # z=1: True; z=0: lhs < rhs
            lhs_gt_rhs = a - M2 * z >= b - M2 + 1  # z=0: True; z=1: lhs > rhs
            return ctx.recurse(lhs_lt_rhs & lhs_gt_rhs, depth, reified=reified)
        return ctx.recurse((a > b) | (a < b), depth, reified=reified, handlers=handlers)
    else:  # push_down_negation changed/decomposed the expression
        return ctx.recurse(cp.all(result), depth, reified=reified, handlers=handlers)


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
    return _with_args(cpm_expr, [ctx.recurse(a, depth, reified=True, handlers=handlers) for a in cpm_expr.args])


handle_wsum = handle_supported_expr


def handle_general_constraint(cpm_expr, depth, reified, handlers, ctx, y=None):
    """Handle general constraints in y=f(x) form, with x a list of variables.

    Used for constraints like max, min, abs, and, or that solvers like Gurobi
    require in the form y = f(x1, x2, ...) with all xi being variables.
    """
    # at root level, and-constraints can be split into individual constraints
    # (before reifying args, to preserve indicator structure)
    # if not reified and y is None and cpm_expr.name == "and":
    #     print("root and")
    #     for arg in cpm_expr.args:
    #         ctx.post(ctx.recurse(arg, depth, reified=False, handlers=handlers))
    #     return cpm_expr
    # or(comp1, ..., compN) at root level: use a selector variable instead of reifying each disjunct.
    # Only valid at root level (not reified), since the selector always enforces one branch.
    # n=2: BV -> comp1, ~BV -> comp2 (2 indicators instead of 4 + 1 OR)
    # n>2: z=intvar(0,n-1), (z==i) -> comp_i (n indicators instead of 2n + 1 OR)
    if not reified and y is None and cpm_expr.name == "or":
        args = cpm_expr.args
        if all(isinstance(a, Comparison) for a in args):
            if len(args) == 2:
                z = cp.boolvar()
                return ctx.recurse(z.implies(args[0]) & (~z).implies(args[1]), depth, reified=reified)
            else:
                z = cp.intvar(0, len(args) - 1)
                return ctx.recurse(cp.all((z == i).implies(arg) for i, arg in enumerate(args)), depth, reified=reified)
    # require only variables: f(x1, x2, ..) === f(y1, y2, ..), y1=x1, y2=x2, ..

    args = [ctx.recurse(arg, depth, reified=True) for arg in cpm_expr.args]

    # assert not (cpm_expr.name == "and" and not reified)
    if reified or cpm_expr.name != "and" or y is not None:
        args = [ctx.reify(arg, depth) for arg in args]

    # root-level is better posted as `sum(args) >= 1` (rather than `1 = OR ( ... )`)
    if not reified and y is None and cpm_expr.name == "or":
        return cp.sum(args) >= 1

    # require the form: y = f(x)
    f = _with_args(cpm_expr, args)
    f = _propagate_boolconst(f)
    if (
        is_num(f) or isinstance(f, (_BoolVarImpl, _IntVarImpl)) and not isinstance(f, NegBoolView)
    ):  # may have become fixed (e.g. `and(x1, 0, x2) === 0`), or changed into e.g. singleton and
        # TODO possibility to add to CSE ..
        return f

    # y = and(a, b) with 2 boolean args: post as quadratic y == a * b
    # This gives Gurobi bilinear constraints, enabling spatial branching (NonConvex=2)
    if QUAD_AND and cpm_expr.name == "and" and len(args) == 2:
        a, b = args
        if y is not None:
            return y == a * b
        if reified:
            return ctx.get_or_make_var(a * b, depth)
        return a * b

    if y is not None:
        return y == f

    y = ctx.get_or_make_var(f, depth) if reified else 1
    return y


def _simplify_comparison(name, a, b):
    """Check if a comparison is trivially True/False based on bounds. Returns True/False or None."""

    def _get_bounds(x):
        return get_bounds(x)

    bounds_a = _get_bounds(a)
    bounds_b = _get_bounds(b)
    if bounds_a is None or bounds_b is None:
        return None

    a_lb, a_ub = bounds_a
    b_lb, b_ub = bounds_b

    if name == "<=":
        if a_ub <= b_lb:
            return True
        if a_lb > b_ub:
            return False
    elif name == ">=":
        if a_lb >= b_ub:
            return True
        if a_ub < b_lb:
            return False
    elif name == "==":
        if a_lb > b_ub or a_ub < b_lb:
            return False
        if a_lb == a_ub == b_lb == b_ub:
            return True

    return None


def _with_args(cpm_expr, args):
    """Copy expression and replace args."""
    cpm_expr = copy.copy(cpm_expr)
    cpm_expr.update_args(args)
    return cpm_expr


def _propagate_boolconst(cpm_expr):
    """Propagate boolean constants in and/or: and(0,...)=0, or(1,...)=1, filter neutral elements."""
    if is_num(cpm_expr):
        return cpm_expr
    match cpm_expr.name:
        case "and" | "or":
            absorb = 0 if cpm_expr.name == "and" else 1  # absorbing element
            args = [a for a in cpm_expr.args if not (is_num(a) and a == 1 - absorb)]
            if len(args) == 0:
                return 1 if cpm_expr.name == "and" else 0
            elif len(args) == 1:
                return args[0]
            elif any(is_num(a) and a == absorb for a in args):
                return absorb

            # TODO avoid copy
            return _with_args(cpm_expr, args)
        case _:
            return cpm_expr


def into_tree_expr(cpm_expr, csemap=None, verbose=False, reified=False, handlers=None):
    """Same as into_tree, but returns a tuple of the root node of the tree and the new top-level (defining) constraints."""

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

    def get_or_make_var(cpm_expr, depth, define=True):
        """Get or make (Boolean/integer) var `b` which represent the expression. Add defining constraints b == cpm_expr if `define=True`."""
        if verbose:
            print(f"{'  ' * depth}get_or_make_var", cpm_expr)
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

        if not isinstance(cpm_expr, NegBoolView):
            cpm_expr = _with_args(cpm_expr, [ctx.recurse(arg, depth, reified=True) for arg in cpm_expr.args])

        cpm_expr_ = _propagate_boolconst(cpm_expr)
        # if is_num(cpm_expr) or isinstance(cpm_expr, (_BoolVarImpl, _IntVarImpl)):  # may have become fixed (e.g. `and(x1, 0, x2) === 0`)
        if (
            is_num(cpm_expr_) or isinstance(cpm_expr_, (_BoolVarImpl, _IntVarImpl)) and not isinstance(cpm_expr_, NegBoolView)
        ):  # may have become fixed (e.g. `and(x1, 0, x2) === 0`), or changed into e.g. singleton and
            # TODO possibility to add to CSE ..
            return cpm_expr_

        # if is_num(cpm_expr) or isinstance(cpm_expr, (_BoolVarImpl, _IntVarImpl)) and not isinstance(cpm_expr, NegBoolView):
        if cpm_expr_.name != cpm_expr.name:
            return get_or_make_var(cpm_expr_, depth)

        r = cp.boolvar() if is_boolexpr(cpm_expr_) else cp.intvar(*cpm_expr_.get_bounds())
        if NAMED:
            r.name = f"{'BV' if is_boolexpr(cpm_expr_) else 'IV'}[{cpm_expr_}]"
            if verbose:
                print(f"{'  ' * depth}intro", r)
        csemap.flat_map[cpm_expr_] = r
        if define:
            add(r == cpm_expr_)

        return r

    def reify(cpm_expr, depth):
        """Return a variable representing the expression, for use as argument in general/indicator constraints."""
        if verbose:
            print(f"{'  ' * depth}reify", cpm_expr, "name =", getattr(cpm_expr, "name", None))

        if is_num(cpm_expr):
            return cpm_expr
        elif isinstance(cpm_expr, _NumVarImpl) and not isinstance(cpm_expr, NegBoolView):
            return cpm_expr
        # elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
        #     # Convert p -> q to or(~p, q) to avoid circular reification
        #     a, b = cpm_expr.args
        #     return reify((~a) | b, depth)
        else:
            return get_or_make_var(cpm_expr, depth)

    def raise_unexpected_expr(cpm_expr):
        raise NotSupportedError(f"Unexpected constraint {cpm_expr} in `into_tree`")

    def into_tree_expr_(cpm_expr, depth, reified=False, handlers=handlers):
        indent = "  " * depth
        depth += 1
        # assert depth < 10
        if verbose:
            print(f"{indent}Con:", cpm_expr, type(cpm_expr), "reif" if reified else "root")

        if is_num(cpm_expr):
            # TODO clean this up ; currently overly complicated due to float expressions
            return int(cpm_expr) if isinstance(cpm_expr, (bool, np.bool_)) or not isinstance(cpm_expr, (int, float)) else cpm_expr
        elif isinstance(cpm_expr, _NumVarImpl):
            return cpm_expr
        elif isinstance(cpm_expr, (Operator, GlobalFunction, Comparison)):
            match cpm_expr.name:
                case "and" if not reified:
                    for arg in cpm_expr.args:
                        ctx.add(arg)
                    return True
                case name if name in handlers:
                    return handlers[name](cpm_expr, depth, reified, handlers, ctx)
                case "not":
                    (a,) = cpm_expr.args
                    # TODO reified = True ?
                    return ctx.recurse(recurse_negation(a), depth, reified=reified)
                case "==" if (
                    isinstance(cpm_expr.args[0], _NumVarImpl)
                    and not isinstance(cpm_expr.args[0], NegBoolView)
                    and isinstance(cpm_expr.args[1], (Operator, GlobalFunction))
                    and cpm_expr.args[1].name in handlers
                    and handlers[cpm_expr.args[1].name] is handle_general_constraint
                    and (not is_boolexpr(cpm_expr.args[1]) or isinstance(cpm_expr.args[0], _BoolVarImpl))
                ):
                    # y == f(...) already in normal form: use y directly as defining variable
                    y, f = cpm_expr.args
                    return handlers[f.name](f, depth, reified, handlers, ctx, y=y)
                case "==" if (
                    isinstance(cpm_expr.args[0], _BoolVarImpl)
                    and is_boolexpr(cpm_expr.args[1])
                    and not isinstance(cpm_expr.args[1], _NumVarImpl)
                ) or (
                    isinstance(cpm_expr.args[1], _BoolVarImpl)
                    and is_boolexpr(cpm_expr.args[0])
                    and not isinstance(cpm_expr.args[0], _NumVarImpl)
                ):
                    # BV == boolexpr === BV <-> boolexpr: post as bi-implications
                    a, b = cpm_expr.args
                    if not isinstance(a, _BoolVarImpl):
                        a, b = b, a  # ensure a is the BoolVar
                    # TODO remove recurse_negation for ~
                    return ctx.recurse(a.implies(b) & (~a).implies(recurse_negation(b)), depth, reified=reified)
                case "==" | "<=" | ">=":
                    # Bounds-based simplification: check if comparison is trivially True/False
                    simp = _simplify_comparison(cpm_expr.name, cpm_expr.args[0], cpm_expr.args[1])
                    if simp is not None:
                        if verbose:
                            print(f"{indent}Simp:", cpm_expr, simp)
                        return simp

                    # con = _with_args(cpm_expr, [ctx.recurse(arg, depth, reified=True) for arg in cpm_expr.args])
                    con = _with_args(
                        cpm_expr,
                        [
                            # TODO refactor to use recurse.. but will mess up the linearization
                            into_tree_expr_(cpm_expr.args[0], depth, reified=True, handlers=handlers),
                            into_tree_expr_(cpm_expr.args[1], depth, reified=True, handlers=handlers),
                        ],
                    )
                    # return con
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
