
from cpmpy.expressions.utils import is_any_list
from cpmpy.expressions.core import Operator, BoolVal

def make_cpm_expr(cpm_expr):
    """
        unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
        """

    if is_any_list(cpm_expr):
        expr = [make_cpm_expr(e) for e in cpm_expr]
        return [e for lst in expr for e in lst]
    if cpm_expr is True:
        return []
    if cpm_expr is False:
        return [BoolVal(cpm_expr)]
    if isinstance(cpm_expr, Operator) and cpm_expr.name == "and":
        return make_cpm_expr(cpm_expr.args)
    return [cpm_expr]

