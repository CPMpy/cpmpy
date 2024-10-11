from copy import copy

from ..expressions.variables import _NumVarImpl, boolvar, intvar, NDVarArray, cpm_array
from ..expressions.core import Expression, Operator, BoolVal
from ..expressions.utils import get_bounds, is_num
from ..expressions.globalfunctions import GlobalFunction, Element
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.python_builtins import all as cpm_all

def no_partial_functions(lst_of_expr, _toplevel=None, _nbc=None):
    """
        A partial function is a function whose output is not defined for all possible inputs.

        In CPMpy, this is the case for the following 3 numeric functions:
            - (Integer) division 'x // y': undefined when y=0
            - Modulo 'x mod y': undefined when y=0
            - Element 'Arr[idx]': undefined when idx is not in the range of Arr

        A toplevel constraint must always be true, so constraint solvers simply propagate the 'unsafe'
        value(s) away. However CPMpy allows arbitrary nesting and reification of constraints, so an
        expression like `b <-> (Arr[idx] == 5)` is allowed, even when variable `idx` goes outside the bounds of `Arr`.
        Should idx be restricted to be in-bounds? and what value should 'b' take if it is out-of-bounds?

        This transformation will transform a partial function (e.g. Arr[idx]) into a total function
        following the **relational semantics** as discussed in:
            Frisch, Alan M., and Peter J. Stuckey. "The proper treatment of undefinedness in
            constraint languages." International Conference on Principles and Practice of Constraint
             Programming. Berlin, Heidelberg: Springer Berlin Heidelberg, 2009.

        Under the relational semantic, an 'undefined' output for a (numerical) expression should
        propagate to `False` in the nearest Boolean parent expression. In the above example: `idx` should
        be allowed to take a value outside the bounds of `Arr`, and `b` should be False in that case.

        To enable this, we can add a Boolean 'guard' to the nearest Boolean parent that represents whether
        the input is 'safe' (has a defined output). We can then use a standard (total function) constraint
        to compute the output for safe ranges of inputs, and state that if the guard is true, the output
        should match the output of the total function (if not, the output can take an arbitrary value).

        A key observation of the implementation below is that for the above 3 expressions, the 'safe'
        domain of a potentially unsafe argument (y or idx) is either one 'trimmed' continuous domain
        (for idx and in case y = [0..n] or [-n..0]), or two 'trimmed' continuous domains (for y=[-n..m]).
        Furthermore, the reformulation for these two cases can be done generically, without needing
        to know the specifics of the partial function being made total.


        :param: list_of_expr: list of CPMpy expressions
        :param: _toplevel: list of new expressions to put toplevel (used internally)
        :param: _nbc: list of new expressions to put in nearest Boolean context (used internally)
    """

    if _toplevel is None:
        toplevel_call = True
        _toplevel = []
    else:
        toplevel_call = False
        assert isinstance(_toplevel, list), f"_toplevel argument must be of type list but got {type(_toplevel)}"

    new_lst = []
    for cpm_expr in lst_of_expr:

        if is_num(cpm_expr) or isinstance(cpm_expr, _NumVarImpl):
            new_lst.append(cpm_expr)

        elif isinstance(cpm_expr, (list,tuple)):
            new_lst.append(no_partial_functions(cpm_expr, _toplevel, _nbc))

        elif isinstance(cpm_expr, NDVarArray):  # TODO efficiency: and cpm_expr.has_subexpr()
            # efficiency: flatten into single iterator, then reshape to n-dimensional again
            new_cpm_expr = cpm_array(no_partial_functions(cpm_expr.flat, _toplevel, _nbc)).reshape(cpm_expr.shape)
            new_lst.append(new_cpm_expr)

        elif isinstance(cpm_expr, DirectConstraint):  # do not recurse into args
            new_lst.append(cpm_expr)

        else:
            assert isinstance(cpm_expr, Expression), f"each `cpm_expr` should be an Expression at this point, not {type(cpm_expr)}"
            args = cpm_expr.args

            if cpm_expr.is_bool():  # a Boolean context, create a new _nbc
                _nbc = []

            # recurse over the arguments of the expression
            # TODO efficiency: if cpm_expr.has_subexpr()
            new_args = no_partial_functions(args, _toplevel, _nbc=_nbc)
            if any((a1 is not a2) for a1,a2 in zip(new_args,args)):  # efficiency (hopefully): only copy if an arg changed
                cpm_expr = copy(cpm_expr)
                cpm_expr.args = new_args
                args = new_args

            if cpm_expr.is_bool() and len(_nbc) != 0:
                # a Boolean context, conjoin my Boolean expression with _nbc
                # in `b <-> (Arr[idx] == 5)`, this would trigger for cpm_expr (Arr[idx] == 5)
                cpm_expr = cpm_all(_nbc) & cpm_expr

            elif isinstance(cpm_expr, GlobalFunction) and cpm_expr.name == "element":
                arr, idx = args
                lb, ub = get_bounds(idx)

                if lb < 0 or ub >= len(arr): # index can be out of bounds
                    guard, output_expr, extra_cons = _safen_range(cpm_expr, safe_range=(0, len(arr)-1), idx_to_safen=1)

                    _nbc.append(guard)  # guard must be added to nearest Boolean context
                    _toplevel += extra_cons  # any additional constraint that must be true
                    cpm_expr = output_expr  # replace partial function by this (total) new output expression

            elif cpm_expr.name == "div" or cpm_expr.name == "mod":
                idx_to_safen = 1
                lb, ub = get_bounds(args[idx_to_safen])

                if lb <= 0 <= ub:
                    assert lb != 0 or ub != 0, "domain of divisor contains only 0" # TODO, I guess we can fix this by making nbc = False?

                    if lb == 0:
                        guard, output_expr, extra_cons = _safen_range(cpm_expr, safe_range=(1, ub), idx_to_safen=idx_to_safen)
                    elif ub == 0:
                        guard, output_expr, extra_cons = _safen_range(cpm_expr, safe_range=(lb, -1), idx_to_safen=idx_to_safen)
                    else:  # proper hole
                        guard, output_expr, extra_cons = _safen_hole(cpm_expr, exclude=0, idx_to_safen=idx_to_safen)

                    _nbc.append(guard)  # guard must be added to nearest Boolean context
                    _toplevel += extra_cons  # any additional constraint that must be true
                    cpm_expr = output_expr  # replace partial function by this (total) new output expression

            # add the (potentially new) cpm_expr to the list
            new_lst.append(cpm_expr)

    if toplevel_call is True:
        return new_lst + _toplevel
    else:
        return new_lst



def _safen_range(cpm_expr, safe_range, idx_to_safen):
    """
        Safen expression where only a continuous of values is considered safe.
        An example is `Element` where the index should be within the array's range.

        Constructs an equivalent expression with safe values,
            and asserts the new expression to be equal to the original one when defined.

        :param cpm_expr: The numerical expression to safen
        :param safe_range: The range of safe values
        :param idx_to_safen: The index of unsafe argument in the expression

    """
    lb, ub = safe_range
    is_safe, safe_arg = boolvar(), intvar(lb, ub)

    unsafe_arg = cpm_expr.args[idx_to_safen]
    safe_expr = copy(cpm_expr)
    safe_expr.args = [safe_arg if i == idx_to_safen else a for i,a in enumerate(safe_expr.args)]

    toplevel = [((unsafe_arg >= lb) & (unsafe_arg <= ub)).implies(is_safe),
                 (is_safe == (safe_arg == unsafe_arg)),
                 # avoid additional solutions when new var is unconstrained (should always work because newly defined?)
                 (~is_safe).implies(safe_arg == lb)]

    return is_safe, safe_expr, toplevel


def _safen_hole(cpm_expr, exclude, idx_to_safen):
    """
        Safen expression where a single value of an argument can cause undefinedness.
        Examples include `div` where 0 has to be removed from the denominator

        Constructs an expression for each interval of safe values, and
            introduces a new `output_expr` variable

        :param cpm_expr: The numerical expression to safen
        :param exclude: The domain value to exclude
        :param idx_to_safen: The index of unsafe argument in the expression
    """
    unsafe_arg = cpm_expr.args[idx_to_safen]
    unsafe_lb, unsafe_ub = get_bounds(unsafe_arg)

    # output when arg in [unsafe_lb..exclude-1]
    safe_arg_lower = intvar(unsafe_lb, exclude-1)
    output_expr_lower = copy(cpm_expr)
    output_expr_lower.args = [safe_arg_lower if i == idx_to_safen else a for i,a in enumerate(cpm_expr.args)]
    # output when arg in [exclude+1..unsafe_ub]
    safe_arg_upper = intvar(exclude+1, unsafe_ub)
    output_expr_upper = copy(cpm_expr)
    output_expr_upper.args = [safe_arg_upper if i == idx_to_safen else a for i, a in enumerate(cpm_expr.args)]

    is_safe = boolvar()
    output_expr = intvar(*get_bounds(cpm_expr))

    toplevel  =[
        (unsafe_arg < exclude).implies(output_expr_lower == output_expr),
        (unsafe_arg > exclude).implies(output_expr_upper == output_expr),
        is_safe == ((unsafe_arg == safe_arg_lower) | (unsafe_arg == safe_arg_upper)),
        # avoid additional solutions when new vars are unconstrained (should always work because newly defined?)
        (unsafe_arg >= exclude).implies(safe_arg_lower == exclude-1),
        (unsafe_arg <= exclude).implies(safe_arg_upper == exclude+1),
        (~is_safe).implies(output_expr == output_expr.lb)
    ]

    return is_safe, output_expr, toplevel


