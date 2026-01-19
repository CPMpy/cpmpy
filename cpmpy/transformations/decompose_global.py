"""

Decompose global constraints and global functions not supported by the solver.

This transformation is necessary for all non-CP solvers, and also used to decompose 
global constraints and global functions not implemented in a CP-solver.

While a solver may natively support a global constraint, it may not support it in a reified context.
In this case, we will als also decompose the global constraint.

For numerical global functions, we will only decompose them if they are not supported in non-reified context.
Even if the solver does not explicitely support them in a subexpression, 
we can rewrite them using func:`cpmpy.transformations.reification.reify_rewrite` to a non-reified version when the function is total.
E.g., bv <-> max(a,b,c) >= 4 can be rewritten as [bv <-> IV0 >= 4, IV0 == max(a,b,c)]

Unsupported gobal constraints and global functions are decomposed in-place
E.g., x + ~AllDifferent(a,b,c) >= 2 is decomposed into x + ~((a) != (b) & (a) != (c) & (b) != (c)) >= 2
This allows to post the decomposed expression tree to the solver if it supports it (e.g., SMT-solvers, MiniZinc, CPO)
"""

import copy
import warnings  # for deprecation warning
from typing import List, Set, Optional, Dict

from .normalize import toplevel_list
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.variables import intvar, cpm_array, NDVarArray
from ..expressions.utils import is_any_list, eval_comparison
from ..expressions.python_builtins import all as cpm_all
from .flatten_model import flatten_constraint, normalized_numexpr


def decompose_in_tree(lst_of_expr: List[Expression], supported: Set[str] = set(), supported_reified: Set[str] = set(), csemap: Optional[Dict[Expression, Expression]] = None, _nested=False) -> List[Expression]:
    """
        Decomposes any global constraint not supported by the solver.
        Accepts a list of CPMpy expressions as input and returns a list of CPMpy expressions.
            
        :param supported: a set of supported global constraints or global functions
        :param supported_reified: a set of supported reified global constraints (globals with Boolean return type only)
        :param csemap: a dictionary of CSE-mapped expressions, used to re-use expressions that have already been decomposed
        :param _nested: whether to treat the root-level as nested, false by default. For internal use only.
        
        Special care is taken for unsupported global constraints in reified (nested) contexts

        Supported numerical global functions remain in the expression tree as is. They can be rewritten using
        :func:`cpmpy.transformations.reification.reify_rewrite`
        The following ``bv -> NumExpr <comp> Var/Const`` can be rewritten as  ``[bv -> IV0 <comp> Var/Const, NumExpr == IV0]``.
        So even if numerical constraints are not supported in reified context, we can rewrite them to non-reified versions if they are total.
    """
    
    toplevel: List[Expression] = [] # list of constraints that should be added toplevel

    def decompose_helper(lst_of_expr, nested):
        """
            Recursively decomposes a list of CPMpy expressions.
            Returns a list of CPMpy expressions.
        """

        newlist = []

        for expr in lst_of_expr:
            if is_any_list(expr):
                assert nested is True, "Cannot have nested lists without passing trough an expression, make sure to run " \
                                       "func:`cpmpy.transformations.normalize.toplevel_list` first. "
                if isinstance(expr, NDVarArray): # NDVarArray is also an expression,
                                                 # so we can call has_subexpr on it for a possible early-exit
                    if expr.has_subexpr():
                        newexpr = decompose_helper(expr, nested=True)
                        newlist.append(cpm_array(newexpr))
                    else:
                        newlist.append(expr)
                else: # a normal list-like (list, tuple, np.ndarray), must be called recursively and check all elements
                    newexpr = decompose_helper(expr, nested=True)
                    newlist.append(newexpr)
                continue

            if isinstance(expr, Expression) and expr.has_subexpr():
                 # a non-leaf expression, recurse into arguments
                expr = copy.copy(expr)
                expr.update_args(decompose_helper(expr.args, nested=True))

            if hasattr(expr, "decompose"): # global function or global constraint
                if expr.is_bool():
                    is_supported = (not nested and expr.name in supported) or (nested and expr.name in supported_reified)
                else:
                    is_supported = expr.name in supported

                if is_supported is False:
                    # unsupported, need to decompose
                    if csemap is not None and expr in csemap:
                        newlist.append(csemap[expr])
                        continue # no need to decompose, re-use the expression we already have
                    
                    val, define = expr.decompose()
                    if isinstance(val, list) and expr.is_bool():
                        val = cpm_all(val)

                    # val may have new global constraints, decompose recursively
                    val = decompose_helper([val], nested=nested)
                    assert len(val) == 1, f"Decomposition should return a single expression\n{val}"
                    val = val[0]

                    if csemap is not None:
                        csemap[expr] = val
                    
                    toplevel.extend(define)
                    newlist.append(val)
                    continue

            # constants, variables, direct constraints are left as is
            newlist.append(expr)

        assert len(newlist) == len(lst_of_expr), f"Decomposition should not change the number of expressions\n{lst_of_expr}\n{newlist}"
        return newlist
        
    newlist = decompose_helper(lst_of_expr, nested=_nested)
    if len(toplevel):
        toplevel = decompose_in_tree(toplevel_list(toplevel), supported, supported_reified, csemap=csemap, _nested=_nested)
    return newlist + toplevel


def decompose_objective(expr, supported=set(), supported_reified=set(), csemap=None):
    if is_any_list(expr):
        raise ValueError(f"Expected a numerical expression as objective but got a list {expr}")

    decomp_expr, *toplevel = decompose_in_tree([expr], supported=supported, supported_reified=supported_reified, csemap=csemap, _nested=True)
    return decomp_expr, toplevel