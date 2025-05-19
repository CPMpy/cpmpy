"""
    Transforms partial functions into total functions.
"""

from copy import copy

from ..expressions.variables import _NumVarImpl, boolvar, intvar, NDVarArray, cpm_array
from ..expressions.core import Expression, Operator, BoolVal
from ..expressions.utils import get_bounds, is_num
from ..expressions.globalfunctions import GlobalFunction, Element
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.python_builtins import all as cpm_all

def no_partial_functions(lst_of_expr, _toplevel=None, _nbc=None, safen_toplevel=frozenset()):
    """
        A partial function is a function whose output is not defined for all possible inputs.

        In CPMpy, this is the case for the following 3 numeric functions:

        - (Integer) division ``x // y``: undefined when y=0
        - Modulo ``x mod y``: undefined when y=0
        - Element ``Arr[idx]``: undefined when idx is not in the range of Arr

        A toplevel constraint must always be true, so constraint solvers simply propagate the 'unsafe'
        value(s) away. However, CPMpy allows arbitrary nesting and reification of constraints, so an
        expression like ``b <-> (Arr[idx] == 5)`` is allowed, even when variable `idx` goes outside the bounds of `Arr`.
        Should `idx` be restricted to be in-bounds? and what value should 'b' take if it is out-of-bounds?

        This transformation will transform a partial function (e.g. Arr[idx]) into a total function
        following the **relational semantics** as discussed in:
            Frisch, Alan M., and Peter J. Stuckey. "The proper treatment of undefinedness in
            constraint languages." International Conference on Principles and Practice of Constraint
            Programming. Berlin, Heidelberg: Springer Berlin Heidelberg, 2009.

        Under the relational semantic, an 'undefined' output for a (numerical) expression should
        propagate to `False` in the nearest Boolean parent expression. In the above example: `idx` should
        be allowed to take a value outside the bounds of `Arr`, and `b` should be False in that case.

        To enable this, we want to rewrite an expression like ``b <-> (partial == 5)`` to something like
        ``b <-> (is_defined & (total == 5))``. The key idea is to create a copy of the potentially unsafe
        argument, that can only take 'safe' values. Using this new argument in the original expression results
        in a total function. We now have the original argument variable, which is decoupled from the expression,
        and a new 'safe' argument variable which is used in the expression. The `is_defined` flag serves two
        purposes: it represents whether the original argument has a safe value; and if it is true the new
        argument must equal the original argument so the two are coupled again. If `is_defined` is false, the new
        argument remains decoupled (can take any value, as will the function's output).

        .. warning::
            Under the relational semantics, ``b <-> ~(partial==5)`` and ``b <-> (partial!=5)`` mean
            different things! The second is ``b <-> (is_defined & (total!=5))`` the first is
            ``b <-> (~is_defined | (total!=5))``.


        A clever observation of the implementation below is that for the above 3 expressions, the 'safe'
        domain of a potentially unsafe argument (y or idx) is either one 'trimmed' continuous domain
        (for idx and in case y = [0..n] or [-n..0]), or two 'trimmed' continuous domains (for y=[-n..m]).
        Furthermore, the reformulation for these two cases can be done generically, without needing
        to know the specifics of the partial function being made total.


        :param list_of_expr: list of CPMpy expressions
        :param _toplevel: list of new expressions to put toplevel (used internally)
        :param _nbc: list of new expressions to put in nearest Boolean context (used internally)
        :param safen toplevel: list of expression types that need to be safened, even when toplevel. Used when
                                 a solver does not support unsafe values in it's API (e.g., OR-Tools for `div`).
    """

    if _toplevel is None:
        assert _nbc is None, f"_nbc is an internal argument, should not be filled by caller but got {_nbc}"
        toplevel_call = True
        _toplevel = []
        _nbc = _toplevel # at the toplevel of the contraint model, the neirest Boolean context is just toplevel
    else:
        toplevel_call = False
        assert isinstance(_toplevel, list), f"_toplevel argument must be of type list but got {type(_toplevel)}"
        assert isinstance(_nbc, list), f"_nbc argument must be of type list but got {type(_nbc)}"

    new_lst = []
    for cpm_expr in lst_of_expr:

        if is_num(cpm_expr) or isinstance(cpm_expr, _NumVarImpl):
            new_lst.append(cpm_expr)

        elif isinstance(cpm_expr, (list,tuple)):
            new_lst.append(no_partial_functions(cpm_expr, _toplevel, _nbc, safen_toplevel))

        elif isinstance(cpm_expr, NDVarArray):
            if cpm_expr.has_subexpr():
                # efficiency: flatten into single iterator, then reshape to n-dimensional again
                new_cpm_expr = cpm_array(no_partial_functions(cpm_expr.flat, _toplevel, _nbc, safen_toplevel)).reshape(cpm_expr.shape)
                new_lst.append(new_cpm_expr)
            else:
                new_lst.append(cpm_expr)

        elif isinstance(cpm_expr, DirectConstraint):  # do not recurse into args
            new_lst.append(cpm_expr)

        else:
            assert isinstance(cpm_expr, Expression), f"each `cpm_expr` should be an Expression at this point, not {type(cpm_expr)}"

            args = cpm_expr.args

            if cpm_expr.is_bool() and toplevel_call is False:  # a Boolean context, create a new _nbc
                # if we are currently toplevel, no need to create a new nbc, will append to toplevel anyway
                _nbc = []

            # recurse over the arguments of the expression
            if cpm_expr.has_subexpr():
                new_args = no_partial_functions(args, _toplevel, _nbc, safen_toplevel=safen_toplevel)
                if any((a1 is not a2) for a1,a2 in zip(new_args,args)):  # efficiency (hopefully): only copy if an arg changed
                    cpm_expr = copy(cpm_expr)
                    cpm_expr.update_args(new_args)
                    args = new_args

            if cpm_expr.is_bool() and len(_nbc) != 0 and toplevel_call is False:
                # a nested Boolean context, conjoin my Boolean expression with _nbc
                # in `b <-> (Arr[idx] == 5)`, this would trigger for cpm_expr (Arr[idx] == 5)
                cpm_expr = cpm_all(_nbc) & cpm_expr

            elif isinstance(cpm_expr, GlobalFunction) and cpm_expr.name == "element":
                if _nbc is _toplevel and cpm_expr.name not in safen_toplevel: # no need to safen
                    new_lst.append(cpm_expr)
                    continue

                arr, idx = args
                lb, ub = get_bounds(idx)

                if lb < 0 or ub >= len(arr): # index can be out of bounds
                    guard, output_expr, extra_cons = _safen_range(cpm_expr, safe_range=(0, len(arr)-1), idx_to_safen=1)

                    _nbc.append(guard)  # guard must be added to nearest Boolean context
                    _toplevel += extra_cons  # any additional constraint that must be true
                    cpm_expr = output_expr  # replace partial function by this (total) new output expression

            elif cpm_expr.name == "div" or cpm_expr.name == "mod":
                if _nbc is _toplevel and cpm_expr.name not in safen_toplevel: # no need to safen
                    new_lst.append(cpm_expr)
                    continue

                idx_to_safen = 1
                lb, ub = get_bounds(args[idx_to_safen])

                if lb <= 0 <= ub:
                    if lb == ub == 0:
                        # (unlikely) edge case, neirest Boolean context should propagate to False.
                        # introduce dummy numerical integer expression
                        guard, output_expr, extra_cons = BoolVal(False), intvar(0,1), []
                    elif lb == 0:
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
        assert _nbc is _toplevel # should point to the same list here
        return new_lst + _toplevel
    else:
        return new_lst



def _safen_range(partial_expr, safe_range, idx_to_safen):
    """
        Replace partial function `cpm_expr` that has potentially unsafe argument at `idx_to_safen`,
        by a total function using a safe argument with domain `safe_range`. Also returns
        a Boolean flag indicating whether the original argument's value is in the safe range
        (for use in the nearest Boolean context), and a list of toplevel constraints that help
        define the new total function.

        An example is `Element` where the index should be within the array's range.

        :param partial_expr: The partial function expression to make total
        :param safe_range: The range of safe argument values
        :param idx_to_safen: The index of the potentially unsafe argument in the expression

    """
    safe_lb, safe_ub = safe_range

    orig_arg = partial_expr.args[idx_to_safen]
    new_arg = intvar(safe_lb, safe_ub)  # values for which the partial function is defined

    total_expr = copy(partial_expr)  # the new total function, with the new arg
    total_expr.update_args([new_arg if i == idx_to_safen else a for i,a in enumerate(partial_expr.args)])

    is_defined = boolvar()
    toplevel = [is_defined == ((safe_lb <= orig_arg) & (orig_arg <= safe_ub)),
                is_defined.implies(new_arg == orig_arg),
                # Extra: avoid additional solutions when new var is unconstrained
                # (should always work in reified context because total_expr is newly defined?)
                (~is_defined).implies(new_arg == safe_lb)]

    return is_defined, total_expr, toplevel


def _safen_hole(cpm_expr, exclude, idx_to_safen):
    """
        Safen expression where a single value of an argument can cause undefinedness.
        Examples include `div` where 0 has to be removed from the denominator

        Constructs an expression for each interval of safe values, and
        introduces a new `output_var` variable

        :param cpm_expr: The numerical expression to safen
        :param exclude: The domain value to exclude
        :param idx_to_safen: The index of unsafe argument in the expression
    """
    orig_arg = cpm_expr.args[idx_to_safen]
    orig_lb, orig_ub = get_bounds(orig_arg)

    # expr when arg in [orig_lb..exclude-1]
    new_arg_lower = intvar(orig_lb, exclude-1)
    total_expr_lower = copy(cpm_expr)
    total_expr_lower.update_args([new_arg_lower if i == idx_to_safen else a for i,a in enumerate(cpm_expr.args)])
    # expr when arg in [exclude+1..orig_ub]
    new_arg_upper = intvar(exclude+1, orig_ub)
    total_expr_upper = copy(cpm_expr)
    total_expr_upper.update_args([new_arg_upper if i == idx_to_safen else a for i, a in enumerate(cpm_expr.args)])

    is_defined = boolvar()
    is_defined_lower = boolvar()
    is_defined_upper = boolvar()
    output_var = intvar(*get_bounds(cpm_expr))

    toplevel = [
        is_defined_lower == (orig_arg < exclude),
        is_defined_upper == (orig_arg > exclude),
        is_defined == (is_defined_lower | is_defined_upper),

        is_defined_lower.implies(orig_arg == new_arg_lower),
        is_defined_lower.implies(output_var == total_expr_lower),
        is_defined_upper.implies(orig_arg == new_arg_upper),
        is_defined_upper.implies(output_var == total_expr_upper),

        # Extra: avoid additional solutions when new vars are unconstrained
        # (should always work in reified context because total_expr's are newly defined?)
        (~is_defined_lower).implies(new_arg_lower == exclude-1),
        (~is_defined_upper).implies(new_arg_upper == exclude+1),
        (~is_defined).implies(output_var == output_var.lb)
    ]

    return is_defined, output_var, toplevel


