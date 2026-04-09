from typing import Optional
from ..expressions.core import Expression, Comparison
from ..expressions.utils import is_int
from ..expressions.variables import boolvar, intvar, _IntVarImpl


class CSEMap:

    def __init__(self):
        self.csemap = dict[Expression, _IntVarImpl]()
        self.decomp_map = dict[Expression, Expression]()

    # pass special methods to internal csemap
    def __len__(self):
        return len(self.csemap)

    def __getitem__(self, expr: Expression) -> _IntVarImpl:
        return self.csemap[expr]

    def __setitem__(self, attr, val):
        raise ValueError("__setitem__ is not supported for csemap, use get_or_make_var instead")

    def get(self, expr: Expression) -> Optional[_IntVarImpl]:
        try:
            return self.csemap[expr]
        except KeyError:
            return None

    def save_decomposition(self, expr: Expression, newexpr: Expression):
        """Save the decomposition of the given global constraint or global function."""
        self.decomp_map[expr] = newexpr

    def get_decomposition(self, expr: Expression) -> Optional[Expression]:
        """Get the decomposition of the given global constraint or global function."""
        return self.decomp_map.get(expr)

    def get_reified_equalities(self) -> dict[_IntVarImpl, list[tuple[int, _IntVarImpl]]]:
        """collect all bv <-> var == val expressions in csemap"""
        var_vals = dict[_IntVarImpl, list[tuple[int, _IntVarImpl]]]()  # var: [val, bv]
        for expr, bv in self.csemap.items():
            if expr.name == "==":
                var, val = expr.args
                if isinstance(var, _IntVarImpl) and is_int(val):
                    var_vals.setdefault(var, []).append((val, bv))

        return var_vals

    def get_or_make_var(self, expr: Expression) -> tuple[_IntVarImpl, list[Expression]]:

        if isinstance(expr, _IntVarImpl):
            return expr, []

        if expr in self.csemap:
            return self.csemap[expr], []

        elif expr.is_bool():
            bv = boolvar()
            self.csemap[expr] = bv
            return bv, [Comparison("==", expr, bv)]
        else:
            iv = intvar(*expr.get_bounds())
            self.csemap[expr] = iv
            return iv, [Comparison("==", expr, iv)]
