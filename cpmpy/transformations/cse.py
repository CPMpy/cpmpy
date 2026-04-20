from typing import Optional
from ..expressions.core import Expression, Comparison
from ..expressions.utils import is_int
from ..expressions.variables import boolvar, intvar, _IntVarImpl


class CSEMap:

    def __init__(self):
        self.flat_map = dict[Expression, _IntVarImpl]()   # map expression to variable filled during flattening
        self.decomp_map = dict[Expression, Expression]()  # map global constraint/function to its decomposition

    # pass special methods to internal flat_map
    def __len__(self):
        return len(self.flat_map)

    def __getitem__(self, expr: Expression) -> _IntVarImpl:
        return self.flat_map[expr]

    def __setitem__(self, attr, val):
        raise ValueError("__setitem__ is not supported for flat_map, use get_or_make_var instead")

    def get(self, expr: Expression) -> Optional[_IntVarImpl]:
        try:
            return self.flat_map[expr]
        except KeyError:
            return None

    def save_decomposition(self, expr: Expression, newexpr: Expression):
        """Save the decomposition of the given global constraint or global function."""
        self.decomp_map[expr] = newexpr

    def get_decomposition(self, expr: Expression) -> Optional[Expression]:
        """Get the decomposition of the given global constraint or global function."""
        return self.decomp_map.get(expr)

    def get_reified_varvals(self) -> dict[_IntVarImpl, list[tuple[int, _IntVarImpl]]]:
        """collect all bv <-> var == val expressions in flat_map"""
        
        var_vals = dict[_IntVarImpl, list[tuple[int, _IntVarImpl]]]()  # var: [val, bv]
        for expr, bv in self.flat_map.items():
            if expr.name == "==":
                var, val = expr.args
                if isinstance(var, _IntVarImpl) and is_int(val):
                    var_vals.setdefault(var, []).append((val, bv))

        return var_vals

    def get_or_make_var(self, expr: Expression) -> tuple[_IntVarImpl, Optional[Expression]]:
        """
        Make an auxiliary variable for the given expression

        Arguments:
            expr: Expression to make an auxiliary variable for

        Returns:
            tuple[_IntVarImpl, Optional[Expression]]: (variable, equality constraint)
            - variable: the auxiliary variable for the expression
            - equality constraint: the equality constraint between the expression and the variable, or None if the expression is already in the flat_map
        """

        if isinstance(expr, _IntVarImpl):
            return expr, None

        if expr in self.flat_map:
            return self.flat_map[expr], None

        elif expr.is_bool():
            bv = boolvar()
            self.flat_map[expr] = bv
            return bv, Comparison("==", expr, bv)
        else:
            iv = intvar(*expr.get_bounds())
            self.flat_map[expr] = iv
            return iv, Comparison("==", expr, iv)
