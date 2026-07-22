"""
Convert nested Boolean expressions with integer linear comparisons to nested PB.

Walks ``and`` / ``or`` / ``not`` / ``->`` (and Boolean reifications). Numeric
comparison leaves are folded to ``sum``/``wsum`` ⋈ const while integer variables
are expanded to their Boolean encoding terms (domain side-constraints lifted).

Unlike :func:`~cpmpy.transformations.int2bool.int2bool`, this preserves Boolean
nesting (no flatten).
"""

import copy
from typing import List

from ..exceptions import NotSupportedError
from ..expressions.core import BoolVal, Comparison, Expression, Operator
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.utils import eval_comparison, is_boolexpr, is_int, is_num
from ..expressions.variables import _BoolVarImpl, _IntVarImpl
from .int2bool import _decide_encoding, _encode_int_var


def int2bool_nested(cpm_lst: List[Expression], ivarmap, encoding="auto", csemap=None):
    """
    Encode integer comparisons inside nested Boolean expressions to pseudo-Boolean.

    Arguments:
        cpm_lst: list of (possibly nested) Boolean constraints
        ivarmap: dictionary mapping integer variable names to their encoding
        encoding: ``"auto"``, ``"direct"``, ``"order"``, or ``"binary"``
        csemap: optional CSE map passed through to int2bool encodings

    Returns:
        list of constraints: domain side-constraints first, then transformed exprs
    """
    assert encoding in ("auto", "direct", "order", "binary"), (
        "Only auto, direct, order, and binary encoding are supported"
    )

    newlist = []
    for expr in cpm_lst:
        newexpr, toplevel = _int2bool_nested_expr(expr, ivarmap, encoding, csemap)
        newlist.extend(toplevel)  # encoding domain cons first (needed before use)
        newlist.append(newexpr)
    return newlist


def _int2bool_nested_expr(expr, ivarmap, encoding, csemap):
    """Return ``(newexpr, toplevel)`` — rewritten expr + lifted domain side-constraints."""
    if isinstance(expr, Operator) and expr.name in ("and", "or", "not", "->"):
        toplevel = []
        newargs = []
        changed = False
        for a in expr.args:
            newarg, tl = _int2bool_nested_expr(a, ivarmap, encoding, csemap)
            newargs.append(newarg)
            toplevel.extend(tl)
            if newarg is not a:
                changed = True
        if not changed:
            return expr, toplevel
        newexpr = copy.copy(expr)
        newexpr.update_args(newargs)
        return newexpr, toplevel

    elif isinstance(expr, Comparison):
        lhs, rhs = expr.args
        # Boolean reification / xor-style: recurse into both sides, keep structure
        if expr.name in ("==", "!=") and is_boolexpr(lhs) and is_boolexpr(rhs):
            nl, tl1 = _int2bool_nested_expr(lhs, ivarmap, encoding, csemap)
            nr, tl2 = _int2bool_nested_expr(rhs, ivarmap, encoding, csemap)
            toplevel = tl1 + tl2
            if nl is lhs and nr is rhs:
                return expr, toplevel
            return Comparison(expr.name, nl, nr), toplevel

        # Numeric comparison: fold to PB sum/wsum (IntVars expanded in _to_wsum_terms)
        name = expr.name
        toplevel = []
        ws, xs, k = _to_wsum_terms(lhs, ivarmap, encoding, csemap, toplevel, name)
        w2, x2, k2 = _to_wsum_terms(rhs, ivarmap, encoding, csemap, toplevel, name)

        # normalize comparison to >= CONST or <= CONST
        ws = ws + [-w for w in w2]
        xs = xs + x2
        rhs_c = -(k - k2)
        if name == "<":
            name, rhs_c = "<=", rhs_c - 1
        elif name == ">":
            name, rhs_c = ">=", rhs_c + 1

        if len(xs) == 0:
            return BoolVal(bool(eval_comparison(name, 0, rhs_c))), toplevel
        if all(w == 1 for w in ws):
            return Comparison(name, Operator("sum", xs), rhs_c), toplevel
        return Comparison(name, Operator("wsum", (ws, xs)), rhs_c), toplevel

    elif isinstance(expr, (BoolVal, _BoolVarImpl, DirectConstraint)) or is_int(expr):
        return expr, []

    else:
        raise NotSupportedError(f"int2bool_nested: unsupported expression {expr}")


def _to_wsum_terms(expr, ivarmap, encoding, csemap, toplevel, cmp):
    """Return ``(weights, bool_terms, const)``; expand IntVars to encoding BVs."""
    if is_num(expr):
        return [], [], int(expr)
    elif isinstance(expr, _BoolVarImpl):
        return [1], [expr], 0
    elif isinstance(expr, _IntVarImpl):
        x_enc, x_cons = _encode_int_var(
            ivarmap, expr, _decide_encoding(expr, cmp, encoding), csemap=csemap
        )
        toplevel.extend(x_cons)
        terms, k = x_enc.encode_term(1)
        if len(terms) == 0:
            return [], [], k
        ws, xs = zip(*terms)
        return list(ws), list(xs), k

    # Bool op / comparison as 0/1 term
    elif (isinstance(expr, Operator) and expr.name in ("and", "or", "not", "->")) \
         or isinstance(expr, Comparison): 
        newexpr, tl = _int2bool_nested_expr(expr, ivarmap, encoding, csemap)
        toplevel.extend(tl)
        if isinstance(newexpr, BoolVal):
            return [], [], int(bool(newexpr.value()))
        return [1], [newexpr], 0

    elif isinstance(expr, Operator):
        if expr.name == "-":
            ws, xs, k = _to_wsum_terms(expr.args[0], ivarmap, encoding, csemap, toplevel, cmp)
            return [-w for w in ws], xs, -k
        elif expr.name == "sub":
            w1, x1, k1 = _to_wsum_terms(expr.args[0], ivarmap, encoding, csemap, toplevel, cmp)
            w2, x2, k2 = _to_wsum_terms(expr.args[1], ivarmap, encoding, csemap, toplevel, cmp)
            return w1 + [-w for w in w2], x1 + x2, k1 - k2
        elif expr.name == "sum":
            ws, xs, k = [], [], 0
            for a in expr.args:
                w, x, kk = _to_wsum_terms(a, ivarmap, encoding, csemap, toplevel, cmp)
                ws += w
                xs += x
                k += kk
            return ws, xs, k
        elif expr.name == "wsum":
            ws, xs, k = [], [], 0
            for w, a in zip(expr.args[0], expr.args[1]):
                ww, xx, kk = _to_wsum_terms(a, ivarmap, encoding, csemap, toplevel, cmp)
                ws += [w * wi for wi in ww]
                xs += xx
                k += w * kk
            return ws, xs, k
        elif expr.name == "mul" and is_num(expr.args[0]):
            ws, xs, k = _to_wsum_terms(expr.args[1], ivarmap, encoding, csemap, toplevel, cmp)
            c = int(expr.args[0])
            return [c * w for w in ws], xs, c * k
    raise NotSupportedError(f"int2bool_nested: non-linear or unsupported numexpr {expr}")
