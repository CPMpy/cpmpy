import warnings
from math import floor, ceil

from ..expressions.core import Expression
from ..expressions.variables import _NumVarImpl, boolvar, intvar
from ..expressions.utils import is_int


class CSEMap:
    """
        Class implementing a mapping from cpmpy Expressions to auxiliary variables.
    """

    def __init__(self):
        self._int_map = dict()
        self._bool_map = dict()

    def get(self, expr):
        if expr.is_bool():
            return self._bool_map.get(expr, None)
        return self._int_map.get(expr, None)
    
    def set(self, expr, value):
        if expr.is_bool():
            self._bool_map[expr] = value
        else:
            self._int_map[expr] = value

    def __len__(self):
        return len(self._int_map) + len(self._bool_map)

    def __contains__(self, expr):
        return expr in self._int_map or expr in self._bool_map

    def __getitem__(self, expr):
        return self.get(expr)

    def __setitem__(self, expr, value):
        if expr.is_bool():
            self._bool_map[expr] = value
        else:
            self._int_map[expr] = value


    def get_or_make_var(self, expr:Expression) -> tuple[_NumVarImpl, list[Expression]]:
        """
            Get or make an auxiliary variable for the given expression.
            
            args:
                expr (Expression): the expression to get or make a variable for

            returns:
                a tuple containing the auxiliary variable and the constraints to enforce the auxiliary variable to be equal to the expression
        """

        if expr.is_bool():
            if expr in self._bool_map:                
                return self._bool_map[expr]
            var = boolvar()
            self._bool_map[expr] = var
            return var, [expr == var]
        
        else:
            if expr in self._int_map:
                return self._int_map[expr]
            lb, ub = expr.get_bounds()
            if not is_int(lb) or not is_int(ub):
                warnings.warn(f"CPMpy only uses integer variables, but found expression ({expr}) with domain {lb}({type(lb)}"
                            f" - {ub}({type(ub)}. CPMpy will rewrite this constriants with integer bounds instead.")
                lb, ub = floor(lb), ceil(ub)            
            var = intvar(lb, ub)
            self._int_map[expr] = var
            return var, [expr == var]