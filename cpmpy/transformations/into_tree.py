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
"""
import copy

# TODO mainly, unduplicate the added ad-hoc transformations done by the other transformation
# TODO specific to gurobi: handling neq, linearize for indicator

import cpmpy as cp
from ..expressions.core import Comparison, Operator, is_boolexpr

from ..exceptions import NotSupportedError
from ..expressions.utils import is_any_list, is_num
from ..expressions.variables import _BoolVarImpl, NegBoolView, _NumVarImpl
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.globalfunctions import GlobalFunction
from .negation import recurse_negation, push_down_negation

# TODO change to supported constraints
GENERAL_CONSTRAINTS = {"max", "min", "abs", "and", "or"}


def into_tree(cpm_expr, csemap=None, verbose=False):
    """Transform CPMpy expressions into an expression tree with non-linear nodes preserved.

    Recursively processes expressions, keeping mul/pow/sum as tree nodes and
    reifying only what the solver cannot handle natively (general constraints,
    non-linear indicator bodies, !=). Side-effect constraints (from reification
    and decomposition) are collected and returned alongside the main constraints.

    Args:
        cpm_expr: CPMpy expression or list of expressions to transform
        csemap: dict for common subexpression elimination (shared across calls)
        verbose: if True, print debug trace of the transformation

    Returns:
        list of transformed CPMpy constraints ready for solver-specific posting
    """

    # constraints will be added to this list
    cpm_cons = []

    if csemap is None:
        csemap = {}

    def add(cpm_expr):
        """Add this expression to the tree, flattening if unsupported."""

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
            args = [reify(to_expr(arg, depth, reified=True), depth) for arg in f.args]
            # require non-constants
            args = propagate_boolconst(f.name, args)
            if is_num(args):  # may have become fixed (e.g. `and(x1, 0, x2) === 0`)
                return args
            else:
                # require the form: y = f(x)
                if y is None:
                    assert reified or f.name in {"and", "or"}, f"Unexpected numexpr {f} encountered at root level"
                    y = get_or_make_var(f, define=False) if reified else 1

                # y, y = f(x)
                f = with_args(f, args)
                cpm_cons.append(y == f)  # add directly so that Comparison does not have to deal with it
                # TODO could be done by add(y == f)?
                return y

        def reify(cpm_expr, depth):
            """Return a variable representing the expression, for use as argument in general/indicator constraints."""
            if verbose: print(f"{'  ' * depth}reify", cpm_expr, getattr(cpm_expr, 'name', None))

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

        def add_comparison(cpm_expr, depth, reified=False):
            """Process a Comparison expression. Returns a Comparison or True (posted as side effect).
            If reified=True, returns a BoolVar representing the truth value of the comparison."""
            a, b = cpm_expr.args

            match cpm_expr.name:
                case "==" if not reified and isinstance(a, (int, _NumVarImpl)) and isinstance(b, Operator) and b.name in GENERAL_CONSTRAINTS:
                    # already of the form: y = f(x)
                    add_general_constraint(b, depth, reified=reified, y=a)
                    return True
                case "==" if isinstance(a, _BoolVarImpl) and is_boolexpr(b) and not isinstance(b, _NumVarImpl):
                    # BV == boolexpr === BV <-> boolexpr: post as bi-implications
                    add(a.implies(b))
                    add((~a).implies(recurse_negation(b)))
                    con = True
                case "==" | "<=" | ">=":
                    a, b = to_expr(a, depth, reified=True), to_expr(b, depth, reified=True)
                    con = with_args(cpm_expr, [a, b])
                case "!=":
                    # One-directional indicator split: d=1 -> a>b, e=1 -> a<b
                    cpm_expr, = push_down_negation([cpm_expr], toplevel=not reified)
                    if cpm_expr.name == "!=":
                        a, b = cpm_expr.args
                        if reified:
                            return to_expr((a > b) | (a < b), depth, reified=reified)
                        else:
                            z = cp.boolvar()
                            # return add_(z.implies(a > b) & (~z).implies(a < b), depth, reified=reified)
                            add(z.implies(a > b))
                            add((~z).implies(a < b))
                            return True
                    else:  # push_down_negation may have changed e.g. != into ==
                        return to_expr(cpm_expr, depth, reified=reified)
                case ">":
                    return to_expr(a >= b + 1, depth, reified=reified)
                case "<":
                    return to_expr(a <= b - 1, depth, reified=reified)
                case _:
                    raise Exception(f"Expected comparator to be ==,<=,>= in Comparison expression {cpm_expr}, but was {cpm_expr.name}")

            return reify(con, depth) if reified else con

        def linearize(cpm_expr, depth):
            """Ensure expression is linear (no mul/pow) by reifying non-linear parts into aux vars."""
            if verbose: print(f"{'  ' * depth}lin", cpm_expr)

            # Only comparisons (except !=) can be indicator bodies directly;
            # everything else (!=, BoolVars, and/or/etc.) needs reification into a BV
            can_be_linear = isinstance(cpm_expr, Comparison) and cpm_expr.name != "!="
            cpm_expr = to_expr(cpm_expr, depth, reified=not can_be_linear)
            if isinstance(cpm_expr, _NumVarImpl):
                return cpm_expr >= 1
            elif isinstance(cpm_expr, Comparison):
                def linearize_expr(expr):
                    if is_num(expr) or isinstance(expr, _NumVarImpl):
                        return expr
                    elif isinstance(expr, Operator):
                        match expr.name:
                            case "sum":
                                return with_args(expr, [reify(a_i, depth) for a_i in expr.args])
                            case "wsum":
                                w, x = expr.args
                                return with_args(expr, [w, [reify(x_i, depth) for x_i in x]])
                            case _:
                                return reify(expr, depth)
                    else:
                        return reify(expr, depth)

                a, b = cpm_expr.args
                return with_args(cpm_expr, [linearize_expr(a), linearize_expr(b)])
            else:
                return reify(cpm_expr, depth) >= 1

        def raise_unexpected_expr(cpm_expr):
            raise NotSupportedError("CPM_gurobi: Unsupported constraint", cpm_expr)
          
        def to_expr(cpm_expr, depth, reified=False):
            """Create an supported expression tree node."""

            indent = "  " * depth
            depth += 1
            if verbose: print(f"{indent}Con:", cpm_expr, type(cpm_expr), "reif" if reified else "root")

            if is_num(cpm_expr):
                return int(cpm_expr)
            elif isinstance(cpm_expr, _NumVarImpl):
                return cpm_expr
            elif isinstance(cpm_expr, (Operator, GlobalFunction)):
                match cpm_expr.name:
                    case "->":  # Gurobi indicator constraint: (Var == 0|1) >> (LinExpr sense LinExpr)
                        a, b = cpm_expr.args
                        # if isinstance(b, (_NumVarImpl, Comparison)):  # TODO requires reification?
                        if isinstance(b, Comparison):
                            p = a if isinstance(a, NegBoolView) else reify(a, depth)
                            if is_num(p):  # propagate fixed antecedent
                                return to_expr(b, depth, reified=reified) if a else True
                            assert isinstance(p, _BoolVarImpl)
                            q = linearize(b, depth)
                            if is_num(q):
                                return True if q else to_expr(p, depth, reified=reified)
                            assert isinstance(q, Comparison), f"Expected linear constraint, but got {q}"  # not required to be a canonical comparison
                            return with_args(cpm_expr, [p, q])
                        else:
                            return to_expr((~a) | b, depth, reified=reified)
                    case "not":  # not is not handled by gurobi
                        a, = cpm_expr.args
                        return to_expr(recurse_negation(a), depth, reified=True)
                    case "-" | "sub" | "sum" | "mul" | "pow" | "div":  # Expression tree nodes (w/ args)
                        assert cpm_expr.name != "div", "TODO"
                        return with_args(cpm_expr, [to_expr(a, depth, reified=True) for a in cpm_expr.args])
                    case "wsum":  # Just for efficiency, don't call add on the weights
                        ws, xs = cpm_expr.args
                        return with_args(cpm_expr, [ws, [to_expr(x, depth, reified=True) for x in xs]])
                    case _:
                        return add_general_constraint(cpm_expr, depth, reified=reified)
                    # case name if name in self.general_constraints:  # general constraints are not handled by the expression tree, so they will be reified
                    #     return add_general_constraint(cpm_expr, depth, reified=reified)
                    # case _:
                    #     raise_unexpected_expr(cpm_expr)
            elif isinstance(cpm_expr, Comparison):
                return add_comparison(cpm_expr, depth, reified=reified)
            elif isinstance(cpm_expr, DirectConstraint):
                return cpm_expr  # pass through to solver-specific posting
            else:
                raise_unexpected_expr(cpm_expr)

        cpm_cons.append(to_expr(cpm_expr, 0))

    if is_any_list(cpm_expr):
        for c in cpm_expr:
            add(c)
    else:
        add(cpm_expr)

    return cpm_cons
