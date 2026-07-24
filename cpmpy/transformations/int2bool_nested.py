"""
Convert nested Boolean expressions with integer comparisons to nested
pseudo-Boolean constraints.

Unlike :func:`~cpmpy.transformations.int2bool.int2bool`, this keeps Boolean
nesting (``and`` / ``or`` / ``not`` / ``->``) and does not flatten.

Integer comparisons are rewritten as follows:
- ``IntVar[+offset]`` compared to a constant uses :func:`_encode_comparison`
- optionally, two-variable ``==`` / ``!=`` of the form ``x - y == k`` or
  ``x + y == c`` (including ``x == y`` and ``x == c - y``) are encoded
  pairwise over the Boolean encodings of ``x`` and ``y``
  (controlled by ``pairwise_eq``)
- other linear comparisons become ``sum`` / ``wsum`` compared to a constant,
  with integer variables expanded to their Boolean encoding terms
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import cpmpy as cp

from ..exceptions import NotSupportedError
from ..expressions.core import BoolVal, Comparison, Expression, Operator
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.utils import eval_comparison, is_boolexpr, is_int, is_num, is_true_cst
from ..expressions.variables import _BoolVarImpl, _IntVarImpl
from .int2bool import IntVarEnc, _decide_encoding, _encode_comparison, _encode_int_var

if TYPE_CHECKING:
    from .cse import CSEMap


def int2bool_nested(
    cpm_lst: list[Expression],
    ivarmap: dict[str, IntVarEnc],
    encoding: str = "auto",
    csemap: CSEMap | None = None,
    pairwise_eq: bool = True,
) -> list[Expression]:
    """
    Encode integer comparisons inside nested Boolean expressions to pseudo-Boolean.

    Arguments:
        cpm_lst: list of (possibly nested) Boolean constraints
        ivarmap: dictionary mapping integer variable names to their encoding
        encoding: ``"auto"``, ``"direct"``, ``"order"``, or ``"binary"``
        csemap: optional CSE map passed through to int2bool encodings
        pairwise_eq: if True (default), encode two-variable ``==`` / ``!=``
            pairwise over Boolean encodings; if False, use the general
            linear ``sum`` / ``wsum`` encoding instead

    Returns:
        list of constraints: domain constraints first, then transformed exprs
    """
    assert encoding in ("auto", "direct", "order", "binary"), (
        "Only auto, direct, order, and binary encoding are supported"
    )

    newlist: list[Expression] = []
    for expr in cpm_lst:
        newexpr, domain_constraints = _encode_nested_expr(
            expr, ivarmap, encoding, csemap, pairwise_eq
        )
        newlist.extend(domain_constraints)  # domain cons first (needed before use)
        newlist.append(newexpr)
    return newlist


def _encode_nested_expr(
    expr: Expression | int | bool,
    ivarmap: dict[str, IntVarEnc],
    encoding: str,
    csemap: CSEMap | None,
    pairwise_eq: bool,
) -> tuple[Expression, list[Expression]]:
    """Return ``(newexpr, domain_constraints)`` — rewritten expr + domain side-constraints."""
    if isinstance(expr, Operator) and expr.name in ("and", "or", "not", "->"):
        domain_constraints: list[Expression] = []
        newargs: list[Expression] = []
        changed = False
        for a in expr.args:
            newarg, dc = _encode_nested_expr(a, ivarmap, encoding, csemap, pairwise_eq)
            newargs.append(newarg)
            domain_constraints.extend(dc)
            if newarg is not a:
                changed = True
        if not changed:
            return expr, domain_constraints
        newexpr = copy.copy(expr)
        newexpr.update_args(newargs)
        return newexpr, domain_constraints

    elif isinstance(expr, Comparison):
        lhs, rhs = expr.args
        # Boolean reification / xor-style: recurse into both sides, keep structure
        if expr.name in ("==", "!=") and is_boolexpr(lhs) and is_boolexpr(rhs):
            nl, dc1 = _encode_nested_expr(lhs, ivarmap, encoding, csemap, pairwise_eq)
            nr, dc2 = _encode_nested_expr(rhs, ivarmap, encoding, csemap, pairwise_eq)
            domain_constraints = dc1 + dc2
            if nl is lhs and nr is rhs:
                return expr, domain_constraints
            return Comparison(expr.name, nl, nr), domain_constraints

        name = expr.name

        # Special cases: one IntVar, or two-IntVar ==/!= with coeffs ±1
        lhs_c, rhs_c = _get_unit_coeffs(lhs), _get_unit_coeffs(rhs)
        if lhs_c is not None and rhs_c is not None:
            coeffs: dict[_IntVarImpl, int] = dict(lhs_c[0])
            k = lhs_c[1] - rhs_c[1]
            ok = True
            for v, c in rhs_c[0].items():
                nv = coeffs.get(v, 0) - c
                if abs(nv) > 1:
                    ok = False
                    break
                if nv == 0:
                    coeffs.pop(v, None)
                else:
                    coeffs[v] = nv
            if ok and len(coeffs) == 1:
                (iv, c), = coeffs.items()
                d = -k if c == 1 else k
                if c == -1:
                    name = {
                        "==": "==", "!=": "!=",
                        "<": ">", "<=": ">=", ">": "<", ">=": "<=",
                    }[name]
                if name == "<":
                    name, d = "<=", d - 1
                elif name == ">":
                    name, d = ">=", d + 1
                constraints, domain_constraints = _encode_comparison(
                    ivarmap, iv, name, d, encoding, csemap=csemap
                )
                constraints = [con for con in constraints if not is_true_cst(con)]
                if len(constraints) == 0:
                    return BoolVal(True), domain_constraints
                if len(constraints) == 1:
                    return constraints[0], domain_constraints
                return Operator("and", constraints), domain_constraints
            if pairwise_eq and ok and len(coeffs) == 2 and name in ("==", "!="):
                (x, cx), (y, cy) = list(coeffs.items())
                if {cx, cy} == {1, -1}:
                    if cx == -1:
                        x, y = y, x
                    # x - y + k == 0  =>  y = x + k
                    return _encode_eq_pairwise(
                        ivarmap, x, y, lambda d: d + k, name, encoding, csemap
                    )
                if cx == 1 and cy == 1:
                    # x + y + k == 0  =>  y = -k - x
                    return _encode_eq_pairwise(
                        ivarmap, x, y, lambda d: -k - d, name, encoding, csemap
                    )
                if cx == -1 and cy == -1:
                    # -x - y + k == 0  =>  y = k - x
                    return _encode_eq_pairwise(
                        ivarmap, x, y, lambda d: k - d, name, encoding, csemap
                    )

        # General linear: fold to PB sum/wsum (IntVars via encode_term)
        domain_constraints = []
        ws, xs, k = _to_wsum_terms(
            lhs, ivarmap, encoding, csemap, domain_constraints, name, pairwise_eq
        )
        w2, x2, k2 = _to_wsum_terms(
            rhs, ivarmap, encoding, csemap, domain_constraints, name, pairwise_eq
        )

        # normalize comparison to >= CONST or <= CONST
        ws = ws + [-w for w in w2]
        xs = xs + x2
        rhs_const = -(k - k2)
        if name == "<":
            name, rhs_const = "<=", rhs_const - 1
        elif name == ">":
            name, rhs_const = ">=", rhs_const + 1

        if len(xs) == 0:
            return BoolVal(bool(eval_comparison(name, 0, rhs_const))), domain_constraints
        if all(w == 1 for w in ws):
            return Comparison(name, Operator("sum", xs), rhs_const), domain_constraints
        return Comparison(name, Operator("wsum", (ws, xs)), rhs_const), domain_constraints

    elif isinstance(expr, (BoolVal, _BoolVarImpl, DirectConstraint)) or is_int(expr):
        return expr, []  # type: ignore[return-value]

    else:
        raise NotSupportedError(f"int2bool_nested: unsupported expression {expr}")


def _to_wsum_terms(
    expr: object,
    ivarmap: dict[str, IntVarEnc],
    encoding: str,
    csemap: CSEMap | None,
    domain_constraints: list[Expression],
    cmp: str,
    pairwise_eq: bool,
) -> tuple[list[int], list[Expression], int]:
    """Return ``(weights, bool_terms, const)``; expand IntVars to encoding BVs."""
    if is_num(expr):
        return [], [], int(cast(int, expr))
    elif isinstance(expr, _BoolVarImpl):
        return [1], [expr], 0
    elif isinstance(expr, _IntVarImpl):
        x_enc, x_cons = _encode_int_var(
            ivarmap, expr, _decide_encoding(expr, cmp, encoding), csemap=csemap
        )
        domain_constraints.extend(x_cons)
        terms, k = x_enc.encode_term(1)
        if len(terms) == 0:
            return [], [], k
        ws_t, xs_t = zip(*terms)
        return list(ws_t), list(xs_t), k

    # Bool op / comparison as 0/1 term
    elif (isinstance(expr, Operator) and expr.name in ("and", "or", "not", "->")) \
         or isinstance(expr, Comparison):
        newexpr, dc = _encode_nested_expr(expr, ivarmap, encoding, csemap, pairwise_eq)
        domain_constraints.extend(dc)
        if isinstance(newexpr, BoolVal):
            return [], [], int(bool(newexpr.value()))
        return [1], [newexpr], 0

    elif isinstance(expr, Operator):
        if expr.name == "-":
            ws, xs, k = _to_wsum_terms(
                expr.args[0], ivarmap, encoding, csemap, domain_constraints, cmp, pairwise_eq
            )
            return [-w for w in ws], xs, -k
        elif expr.name == "sub":
            w1, x1, k1 = _to_wsum_terms(
                expr.args[0], ivarmap, encoding, csemap, domain_constraints, cmp, pairwise_eq
            )
            w2, x2, k2 = _to_wsum_terms(
                expr.args[1], ivarmap, encoding, csemap, domain_constraints, cmp, pairwise_eq
            )
            return w1 + [-w for w in w2], x1 + x2, k1 - k2
        elif expr.name == "sum":
            ws_l: list[int] = []
            xs_l: list[Expression] = []
            k = 0
            for a in expr.args:
                w, x, kk = _to_wsum_terms(
                    a, ivarmap, encoding, csemap, domain_constraints, cmp, pairwise_eq
                )
                ws_l += w
                xs_l += x
                k += kk
            return ws_l, xs_l, k
        elif expr.name == "wsum":
            ws_l = []
            xs_l = []
            k = 0
            for w, a in zip(expr.args[0], expr.args[1]):
                ww, xx, kk = _to_wsum_terms(
                    a, ivarmap, encoding, csemap, domain_constraints, cmp, pairwise_eq
                )
                ws_l += [w * wi for wi in ww]
                xs_l += xx
                k += w * kk
            return ws_l, xs_l, k
        elif expr.name == "mul" and is_num(expr.args[0]):
            ws, xs, k = _to_wsum_terms(
                expr.args[1], ivarmap, encoding, csemap, domain_constraints, cmp, pairwise_eq
            )
            c = int(cast(int, expr.args[0]))
            return [c * w for w in ws], xs, c * k
    raise NotSupportedError(f"int2bool_nested: non-linear or unsupported numexpr {expr}")


def _get_unit_coeffs(expr: object) -> tuple[dict[_IntVarImpl, int], int] | None:
    """Return ``({iv: ±1}, const)`` for IntVars and ints with unary -, sum, sub; else None.

    BoolVars are rejected (also ``_IntVarImpl``) so they stay 0/1 terms.
    """
    if is_num(expr):
        return {}, int(cast(int, expr))
    if isinstance(expr, _IntVarImpl) and not isinstance(expr, _BoolVarImpl):
        return {expr: 1}, 0
    if isinstance(expr, Operator):
        if expr.name == "-" and len(expr.args) == 1:
            r = _get_unit_coeffs(expr.args[0])
            if r is None:
                return None
            cfs, kk = r
            return {v: -c for v, c in cfs.items()}, -kk
        if expr.name == "sum":
            coeffs: dict[_IntVarImpl, int] = {}
            k = 0
            for a in expr.args:
                r = _get_unit_coeffs(a)
                if r is None:
                    return None
                ca, ka = r
                k += ka
                for v, c in ca.items():
                    nv = coeffs.get(v, 0) + c
                    if abs(nv) > 1:
                        return None
                    if nv == 0:
                        coeffs.pop(v, None)
                    else:
                        coeffs[v] = nv
            return coeffs, k
        if expr.name == "sub":
            a = _get_unit_coeffs(expr.args[0])
            b = _get_unit_coeffs(expr.args[1])
            if a is None or b is None:
                return None
            cfs = dict(a[0])
            k = a[1] - b[1]
            for v, c in b[0].items():
                nv = cfs.get(v, 0) - c
                if abs(nv) > 1:
                    return None
                if nv == 0:
                    cfs.pop(v, None)
                else:
                    cfs[v] = nv
            return cfs, k
    return None


def _encode_eq_pairwise(
    ivarmap: dict[str, IntVarEnc],
    x: _IntVarImpl,
    y: _IntVarImpl,
    y_of_x: Callable[[int], int],
    name: str,
    encoding: str,
    csemap: CSEMap | None,
) -> tuple[Expression, list[Expression]]:
    """Encode ``x == y_of_x(x)`` (or its negation) over the Boolean encodings."""
    x_enc, x_cons = _encode_int_var(
        ivarmap, x, _decide_encoding(x, "==", encoding), csemap=csemap
    )
    y_enc, y_cons = _encode_int_var(
        ivarmap, y, _decide_encoding(y, "==", encoding), csemap=csemap
    )
    pairs = []
    for d in range(x.lb, x.ub + 1):
        e = y_of_x(d)
        if y.lb <= e <= y.ub:
            xd, yd = x_enc.eq(d), y_enc.eq(e)
            if isinstance(xd, list):
                xd = cp.all(xd)
            if isinstance(yd, list):
                yd = cp.all(yd)
            pairs.append(xd & yd)
    eq = cp.any(pairs) if pairs else BoolVal(False)
    return (eq if name == "==" else ~eq), x_cons + y_cons
