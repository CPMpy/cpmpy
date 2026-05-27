import copy
from typing import Optional, Any, cast, overload, Literal
from ..expressions.core import Expression, Comparison
from ..expressions.utils import is_int
from ..expressions.variables import NegBoolView, boolvar, intvar, _IntVarImpl, _BoolVarImpl


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

    @overload
    def get(self, expr: Expression) -> Optional[_IntVarImpl]: ...
    @overload
    def get(self, expr: Expression, default: Literal[None]) -> Optional[_IntVarImpl]: ...
    @overload
    def get(self, expr: Expression, default: Any) -> Any: ...
    def get(self, expr: Expression, default: Any = None) -> Any:
        if expr.is_bool():
            expr, negate = self._canonicalize_boolexpr(expr)
            res = self.flat_map.get(expr, default)
            if res is not None and negate: # return negated Boolean variable, stored negation in flat_map
                return ~res
            return res
            
        return self.flat_map.get(expr, default)

    def save_decomposition(self, expr: Expression, newexpr: Expression):
        """Save the decomposition of the given global constraint or global function."""
        self.decomp_map[expr] = newexpr

    @overload   
    def get_decomposition(self, expr: Expression) -> Optional[Expression]: ...
    @overload
    def get_decomposition(self, expr: Expression, default: Literal[None]) -> Optional[Expression]: ...
    @overload
    def get_decomposition(self, expr: Expression, default: Any) -> Any: ...
    def get_decomposition(self, expr: Expression, default: Any = None) -> Any:
        """Get the decomposition of the given global constraint or global function."""
        return self.decomp_map.get(expr, default)

    def get_reified_comparisons(self, cmp) -> dict[_IntVarImpl, list[tuple[int, _BoolVarImpl]]]:
        """collect all bv <-> var cmp val expressions in flat_map, where cmp can be `==` or `>=`"""
        
        var_vals = dict[_IntVarImpl, list[tuple[int, _BoolVarImpl]]]()  # var: [val, bv]
        for expr, bv in self.flat_map.items():
            if expr.name == cmp:
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

        res = self.get(expr)
        if res is not None:
            return res, None # already made a variable for expr

        if expr.is_bool():
            bv = boolvar()
            expr, negate = self._canonicalize_boolexpr(expr)

            self.flat_map[expr] = bv
            if negate: # return the negated variable to get the original expression back
                neg_bv = NegBoolView(bv) # invert
                return neg_bv, Comparison("==", expr, bv)
            return bv, Comparison("==", expr, bv)
        else:
            iv = intvar(*expr.get_bounds())
            self.flat_map[expr] = iv
            return iv, Comparison("==", expr, iv)


    def _canonicalize_boolexpr(self, expr: Expression) -> tuple[Expression, bool]:
        """canonicalize a Boolean expression, results in more hits in the flat_map"""

        if isinstance(expr, Comparison):
            lhs, rhs = expr.args
            if expr.name == "!=" and is_int(rhs):
                # b <-> (expr != val) :: (~b) <-> (expr == val)
                new_expr = Comparison("==", lhs, rhs)
                new_expr._has_subexpr = expr._has_subexpr
                return new_expr, True

        return expr, False
