from copy import copy

from ..expressions.variables import _NumVarImpl, boolvar, intvar, NDVarArray
from ..expressions.core import Operator, BoolVal
from ..expressions.utils import get_bounds, is_num
from ..expressions.globalfunctions import GlobalFunction, Element
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.python_builtins import all as cpm_all

def no_partial_functions(lst_of_expr, _toplevel=None, nbc=None):

    if _toplevel is None:
        toplevel_call = True
        _toplevel = []
    else:
        assert isinstance(_toplevel, list), f"_toplevel argument must be of type list but got {type(_toplevel)}"
        toplevel_call = False

    new_lst = []
    for cpm_expr in lst_of_expr:

        if is_num(cpm_expr) or isinstance(cpm_expr, _NumVarImpl):
            new_lst.append(cpm_expr)

        elif isinstance(cpm_expr, list):
            new_lst.append(no_partial_functions(cpm_expr, _toplevel, nbc))

        elif isinstance(cpm_expr, NDVarArray):
            safened = no_partial_functions(cpm_expr.tolist(), _toplevel, nbc)
            new_lst.append(safened)


        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "div":
            safe_args = no_partial_functions(cpm_expr.args, _toplevel, nbc=nbc)
            x, y = safe_args
            lb, ub = get_bounds(y)

            if lb <= 0 <= ub:
                assert lb != 0 or ub != 0, "domain of divisor contains only 0" # TODO, I guess we can fix this by making nbc = False?

                is_safe = boolvar()
                nbc.append(is_safe)
                _toplevel += [(y != 0).implies(is_safe)]
                # case-by case analysis on exact domain of divisor
                if lb < 0 < ub: # proper hole... need to split up into two and do some flattening
                    y_neg, y_pos = intvar(lb, -1), intvar(1,ub)
                    new_rhs = intvar(*get_bounds(cpm_expr))
                    _toplevel += [is_safe == ((y == y_neg) | (y == y_pos))]
                    _toplevel += [(y < 0).implies((x // y_neg) == new_rhs),
                                  (y > 0).implies((x // y_pos) == new_rhs),
                                  # avoid new solutions in aux vars (is this dangerous?)
                                  (y >= 0).implies(y_neg == -1),
                                  (y <= 0).implies(y_pos == 1),
                                  (~is_safe).implies(new_rhs == new_rhs.lb)
                                  ]
                    new_lst.append(new_rhs)
                else: # just a range, exclude 0 from domain
                    assert lb == 0 or ub == 0, "domain of divisor should contain a 0, otherwise it's safe..."
                    if lb == 0:
                        safe_y = intvar(1,ub)
                    elif ub == 0:
                        safe_y = intvar(lb, -1)
                    _toplevel += [is_safe == (y == safe_y)]
                    _toplevel += [(~is_safe).implies(safe_y == safe_y.lb)] # avoid new solutions in aux vars (is this dangerous?)
                    new_lst.append(x // safe_y)

            else: # already safe
                new_lst.append(x // y)


        elif isinstance(cpm_expr, GlobalFunction) and cpm_expr.name == "element":
            safe_args = no_partial_functions(cpm_expr.args, _toplevel, nbc=nbc)
            arr, idx = safe_args
            lb, ub = get_bounds(idx)
            if lb < 0 or ub >= len(arr): # index can be out of bounds
                is_safe, safe_idx = boolvar(), intvar(0,len(arr)-1)
                _toplevel += [((idx >= 0) & (idx < len(arr))).implies(is_safe)]
                _toplevel += [(is_safe == (safe_idx == idx))]
                _toplevel += [(~is_safe).implies(safe_idx == 0)] # avoid new solutions in aux vars (is this dangerous?)
                nbc.append(is_safe)
                new_lst.append(Element(arr, safe_idx))

            else:
                new_lst.append(Element(arr, idx))


        elif isinstance(cpm_expr, DirectConstraint): # do nothing
            new_lst.append(cpm_expr)

        elif cpm_expr.is_bool(): # reached Boolean (sub)expression
            new_exprs = []
            safe_args = no_partial_functions(cpm_expr.args, _toplevel, nbc=new_exprs)
            cpm_expr = copy(cpm_expr)
            cpm_expr.args = safe_args
            new_lst.append(cpm_expr & cpm_all(new_exprs))

        else: # numerical subexpression or toplevel, just recurse
            safe_args = no_partial_functions(cpm_expr.args, _toplevel, nbc=nbc)
            cpm_expr = copy(cpm_expr)
            cpm_expr.args = safe_args
            new_lst.append(cpm_expr)


    if toplevel_call is True:
        return new_lst + _toplevel
    else:
        return new_lst


