"""
Transform constraints to a linear form.

This transformation is necessary for Integer Linear Programming (ILP) solvers, and also
for translating to Pseudo-Boolean or CNF formats.

There are a number of components to getting a good linearisation:
- **Decomposing global constraints/functions** in a 'linear friendly' way.
  A common pattern is that constraints that enforce domain consistency on integer variables,
  that these constraints should be decomposed over a Boolean representation of the domain. This is the
  case for :class:`~cpmpy.expressions.globalconstraints.AllDifferent`, :class:`~cpmpy.expressions.globalfunctions.Element`
  and possibly others (for now, only done for AllDifferent in this module).
  Their default decomposition might not do it this way, in which case we want to use different
  decompositions.

- **Linearising multiplication** of variables, e.g. bool*bool, bool*int and int*int.
  Because '*' is a core Operator and not a global constraint, we need to do it with a helper function here.

- **Disequalities** e.g. `sum(X) != 5` should be rewritten as `(sum(X) < 5) | (sum(X) > 5)`
  and further flattened into implications and linearised.

- **Implications** e.g. `B -> sum(X) <= 4` should be linearised with the Big-M method.

In principle, but not actually currently implemented:
- **Domain encodings** e.g. of `A=cp.intvar(0,4)` would create binary variables and constraints
  like `B0 -> A=0`, `B1 -> A=1`, `B2 -> A=2`, etc and `sum(B0..B4) = 1`.
  However each `B0 -> A=0` would require 2 Big-M constraints. Instead we can linearise the entire
  domain encoding with two constraints: `sum(B0..4) = 1` and `A = sum(B0..4 * [0,1,2,3,4])`.
  We could even go as far as eliminate the original integer decision variable.

Module functions
----------------

Main transformations:
- :func:`linearize_constraint`: Transforms a list of constraints to a linear form.

Helper functions:
- :func:`canonical_comparison`: Canonicalizes comparison expressions by moving variables to the
  left-hand side and constants to the right-hand side.
- :func:`linearize_mul_comparison`: Linearizes multiplication comparisons (bool*bool | bool*int | int*int) <cmp> rhs.

Post-linearisation transformations:
- :func:`only_positive_bv`: Transforms constraints so only boolean variables appear positively
  (no :class:`~cpmpy.expressions.variables.NegBoolView`).
- :func:`only_positive_bv_wsum`: Helper function that replaces :class:`~cpmpy.expressions.variables.NegBoolView`
  in var/sum/wsum expressions with equivalent expressions using only :class:`~cpmpy.expressions.variables._BoolVarImpl`.
- :func:`only_positive_bv_wsum_const`: Same as :func:`only_positive_bv_wsum` but returns the constant
  term separately.
- :func:`only_positive_coefficients`: Transforms constraints so only positive coefficients appear
  in linear constraints.
"""

import copy
import numpy as np
import cpmpy as cp
from typing import List, Set, Optional, Dict, Any, Tuple, Union
from cpmpy.transformations.get_variables import get_variables

from .flatten_model import flatten_constraint, get_or_make_var
from .normalize import toplevel_list, simplify_boolean
from ..exceptions import TransformationNotImplementedError

from ..expressions.core import Comparison, Expression, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.utils import is_bool, is_num, eval_comparison, get_bounds, is_true_cst, is_false_cst, is_int

from ..expressions.variables import boolvar, _BoolVarImpl, NegBoolView, _NumVarImpl
from ..transformations.int2bool import _encode_int_var

def linearize_mul_comparison(cpm_expr: Comparison, supported: Set[str] = {}, reified: bool = False, csemap: Optional[Dict[Expression, _NumVarImpl]] = None) -> Tuple[Expression, List[Expression]]:
    """
    Linearizes multiplication comparisons (const*var | bool*bool | bool*int | int*int) <cmp> rhs.

    Arguments:
        cpm_expr (Comparison): Comparison to linearize, e.g. `bool*bool <cmp> rhs` or `bool*int <cmp> rhs` or `int*int <cmp> rhs`.
        supported (Set[str]): Set of constraint and variable type names that are supported by the target
            solver, e.g. `{"sum", "wsum", "->", "and", "or", "alldifferent"}`.
        reified (bool): Whether the constraint is fully reified. When True, nested implications are linearized using Big-M method instead of indicator constraints.
        csemap (Optional[Dict[Any, Any]]): Optional dictionary for common subexpression elimination, mapping expressions to their flattened variable representations. Used to avoid creating duplicate variables for the same subexpression.

    Returns:
        Tuple of (return expression, list of toplevel constraints)
        Warning: the return expression is either a linearized comparison, or an 'and' of linearized comparisons
    """
    assert isinstance(cpm_expr, Comparison), f"linearize_mul_comparison expects a comparison, got {cpm_expr}"
    lhs, rhs = cpm_expr.args
    assert lhs.name == "mul", f"linearize_mul_comparison expects a multiplication comparison, got {lhs}"

    if is_num(lhs.args[0]): # const * iv <comp> rhs
        newlhs = Operator("wsum",[[lhs.args[0]], [lhs.args[1]]])
        return eval_comparison(cpm_expr.name, newlhs, rhs), []
    
    bv_idx = None
    if isinstance(lhs.args[0], _BoolVarImpl):
        bv_idx = 0
    elif isinstance(lhs.args[1], _BoolVarImpl):
        bv_idx = 1

    if bv_idx is not None:
        # bv * iv <comp> rhs, rewrite to (bv -> iv <comp> rhs) & (~bv -> 0 <comp> rhs)
        # (also works for bv1*bv2 <comp> rhs)
        bv, iv = lhs.args[bv_idx], lhs.args[1-bv_idx]
        tmp_supported = supported
        if reified and "->" in supported:
            # when inside a '->', we have to force Big-M instead of a nested '->'
            tmp_supported = supported - {"->"}
        iftrue = linearize_constraint([eval_comparison(cpm_expr.name, iv, rhs)], supported=tmp_supported, implication_literal=bv, csemap=csemap)
        iffalse = linearize_constraint([eval_comparison(cpm_expr.name, 0, rhs)], supported=tmp_supported, implication_literal=~bv, csemap=csemap)
        # all of this will be cleaner when Mul is a global function...
        # lets put some checks for not having toplevel constraints in iftrue/iffalse
        if cpm_expr.name == '!=' or cpm_expr.name == '==':
            assert len(iftrue) <= 2 and len(iffalse) <= 2, f"Expected at most 2 constraints for != in bv*iv, got {iftrue} and {iffalse}"
        else:
            assert len(iftrue) <= 1 and len(iffalse) <= 1, f"Expected at most 1 constraint in bv*iv, got {iftrue} and {iffalse}"
        # this is the case that returns an 'and'... at least if both are non-trivial expressions
        simplified = simplify_boolean([cp.all(iftrue + iffalse)])[0]
        return simplified, []
    else:
        # iv1 * iv2 <comp> rhs, do Boolean expansion of smallest integer
        # resulting in sum([v*b*i2 for v,b in int2bool(i1)]) <comp> rhs
        # Note that this is a straightforward implementation, one could do more by knowing <comp> rhs and deriving bounds on i1/i2 from that...

        # choose smallest integer as i1 (to minimize the number of Boolean variables)
        i1,i2 = lhs.args
        leni1 = (i1.ub - i1.lb) + 1
        leni2 = (i2.ub - i2.lb) + 1
        if leni2 < leni1:
            # swap, call i1 the smallest integer
            i1,i2 = i2,i1

        # encode i1 with Booleans
        # (with temproary ivarmap: no sharing of integer encodings... where would we store ivarmap?)
        # (TODO: int2bool should support csemap... then it could still reuse the encoding Bools)
        encoding = "direct"
        # XXX I have not figured out how to get it right for the binary encoding... esp if i1 has negative and positive values
        #if min(leni1,leni2) >= 8:  # arbitrary heuristic
        #    encoding = "binary"  # results in fewer Bools
        i1_enc, cons = _encode_int_var({}, i1, encoding)  # {}: no ivarmap used

        # channel i1 to the Bools
        (encpairs, offset) = i1_enc.encode_term()
        # i1 = sum(ws * bs) + offset :: 1*i1 - sum(ws*bs) == offset :: i1 + sum(-ws*bs) == offset
        ws,bs = zip(*encpairs)
        ws = [1] + [-1*w for w in ws]
        bs = [i1] + list(bs)
        cons += [Operator("wsum", [ws, bs]) == offset]

        # Build the sum: aux = sum(v * (b_v * i2) for v,bv in encoding)
        terms = []
        for v, b_v in encpairs:
            mymul = b_v * i2
            if csemap is not None and mymul in csemap:
                myaux = csemap[mymul]
            else:
                myaux = cp.intvar(*mymul.get_bounds())
                if csemap is not None:
                    csemap[mymul] = myaux

                # the definition of myaux, to linearize
                res, subcons = linearize_mul_comparison(mymul == myaux, supported=supported, reified=reified, csemap=csemap)
                if res.name == "and":
                    # to be exepected, will return 2 constraints, both are linear
                    cons += res.args
                else:  # just a single linear constraint
                    cons.append(res)
                cons += subcons
            # the term for in the multiplication
            terms.append((offset + v) * myaux)
        assert len(terms) > 0, f"Expected at least one term, got {terms} for {cpm_expr} with encoding {encoding}"

        # I think everything is linear here...
        return eval_comparison(cpm_expr.name, cp.sum(terms), rhs), cons

def linearize_constraint(lst_of_expr: List[Expression], supported: Set[str] = {}, csemap: Optional[Dict[Expression, _NumVarImpl]] = None, implication_literal: Optional[Union[_BoolVarImpl, NegBoolView]] = None, reified: bool = False) -> List[Expression]:
    """
    Transforms all constraints to a linear form.
    This function assumes all constraints are in 'flat normal form' with only boolean variables on the lhs of an implication.
    Only apply after :func:'cpmpy.transformations.flatten_model.flatten_constraint()' and :func:'cpmpy.transformations.reification.only_implies()'.

    After linearize, the following constraints remain:

    **Trivial:**
        - ``BoolVal``

    **Linear comparisons:**
        Equality and non-strict inequality comparison where the left-hand side is
        a linear expression and the right-hand side is a constant:

        - ``LinExpr == Constant``
        - ``LinExpr >= Constant``
        - ``LinExpr <= Constant``

        The linear expression (LinExpr) can be:
        - A numeric variable (``NumVar``)
        - A sum expression (``sum([...])``)
        - A weighted sum expression (``wsum([weights], [vars])``)

    **General comparisons or expressions, if their name is in `supported`:**

        * Boolean expressions:
          - ``GenExpr`` (when :func:`~cpmpy.expressions.core.Expression.is_bool()`)

        * Numeric comparisons:
          - ``GenExpr == Var/Constant`` (when GenExpr is numeric)
          - ``GenExpr <= Var/Constant`` (when GenExpr is numeric)
          - ``GenExpr >= Var/Constant`` (when GenExpr is numeric)

    **Indicator constraints, if '->' is in 'supported':**
        The left-hand side is always a Boolean variable or its negation, and
        the right-hand side can be a linear comparison or a general expression.

        * Linear comparisons:
          - ``BoolVar -> LinExpr == Constant``
          - ``BoolVar -> LinExpr >= Constant``
          - ``BoolVar -> LinExpr <= Constant``

        * General expressions (when the expression name is in `supported`):
          - ``BoolVar -> GenExpr`` (when GenExpr is boolean)
          - ``BoolVar -> GenExpr == Var/Constant`` (when GenExpr is numeric)
          - ``BoolVar -> GenExpr >= Var/Constant`` (when GenExpr is numeric)
          - ``BoolVar -> GenExpr <= Var/Constant`` (when GenExpr is numeric)

    Arguments:
        lst_of_expr: List of CPMpy expressions to linearize. Must be in 'flat normal form'
            with only boolean variables on the left-hand side of implications.
        supported: Set of *primitive* constraint that should not be linearized (e.g. '->', 'and', 'or', 'mul', 'alldifferent').
            '->' in case the solver supports implication constraints
            'and', 'or' in case of e.g. a SAT solver
            'mul' if the solver supports multiplication of two variables (otherwise it is linearized)
            'alldifferent' if the solver supports it natively (otherwise this function uses a linear-friendly decomposition)
        csemap: Optional dictionary for common subexpression elimination, mapping expressions
            to their flattened variable representations. Used to avoid creating duplicate
            variables for the same subexpression.
        implication_literal: when set to, e.g. 'b' this means we should linearize `b -> lst_of_expr`.
            This is typically done in a recursive call, e.g. this function will call it when encountering a 'b -> expr' if '->' is not in supported.
            When an implication literal is given, a '->' inside it will always be linearized using Big-M even if '->' is in 'supported'.
        reified: DEPRECATED, should always be False (use 'implication_literal' instead)

    Returns:
        List of linearized CPMpy expressions. The constraints are in one of the forms
        described above (linear comparisons, general expressions, or indicator constraints).
    """
    if reified:
        raise DeprecationWarning("linearize_constraint: 'reified' argument is deprecated, use 'implication_literal' instead")
    if csemap is None:
        csemap = dict()

    toplevel = []  # constraints that define auxiliaries
                   # not part of the 'implication_literal -> lst_of_expr' implication
                   # should be linear
    newlist = []
    for cpm_expr in lst_of_expr:

        # pre-process non-comparisons
        if not isinstance(cpm_expr, Comparison):
            if is_bool(cpm_expr):  # constant true/false
                if is_true_cst(cpm_expr):
                    continue  # trivially true, skip
                elif is_false_cst(cpm_expr):
                    if implication_literal is not None:
                        # needs further processing below until we accept a BoolVar as response
                        toplevel.extend(linearize_constraint([~implication_literal], supported=supported, csemap=csemap))
                        continue
                    else:
                        return [BoolVal(False)]
                else:
                    raise ValueError(f"Unexpected numeric expression {cpm_expr} in {lst_of_expr}")

            if isinstance(cpm_expr, Operator):
                # pre-process Boolean operators: 'and', 'or', '->'
                op_name = cpm_expr.name

                if op_name == "->":
                    cond, sub_expr = cpm_expr.args
                    # verify cond is a Boolean literal, always when `only_implies` has been called
                    assert isinstance(cond, _BoolVarImpl), f"Linearize of {cpm_expr} expects a Boolean literal left of the implication"

                    # even if -> is supported, the sub_expr must get the recursive call
                    newlist.extend(linearize_constraint([sub_expr], supported=supported, implication_literal=cond, csemap=csemap))
                    continue
                elif op_name in supported:
                    # operator is supported, do nothing
                    newlist.append(cpm_expr)
                    continue
                elif op_name == "or":
                    # rewrite to linear constraint and process that one further
                    cpm_expr = Operator("sum", cpm_expr.args) >= 1
                elif op_name == "and":
                    # rewrite to linear constraint and process that one further
                    cpm_expr = Operator("sum", cpm_expr.args) >= len(cpm_expr.args)
                else:
                    raise ValueError(f"Unexpected operator {op_name} in {cpm_expr}")

            elif cpm_expr.name in supported or isinstance(cpm_expr, (DirectConstraint, BoolVal)):
                newlist.append(cpm_expr)
                continue

            # pre-process alldifferent in linear-friendly way (ideally done before flattening)
            elif cpm_expr.name == "alldifferent": 
                """
                    Flow-based decomposition. Introduces n^2 new boolean variables.
                    Decomposes through bi-partite matching

                    More efficient implementations possible?
                    http://yetanothermathprogrammingconsultant.blogspot.com/2016/05/all-different-and-mixed-integer.html
                """
                # In some future:
                #value, cons = cpm_expr.decompose_linear()
                #assert value.name == "and", f"Linearize alldiff: expected an 'and' of constraints, got {value}"
                # recurse on arguments, with implication literal, then these are fully handled
                #toplevel.extend(linearize_constraint(value.args, supported=supported, csemap=csemap, implication_literal=implication_literal))
                # recurse on the defining constraints, toplevel so no implication literal
                #toplevel.extend(linearize_constraint(cons, supported=supported, csemap=csemap))
                #continue

                # make a boolvar array for each variable, define the encoding variables
                # lets go through all the manual steps to keep everything nicely linear ourselfs
                args_encoded = []
                for var in cpm_expr.args:
                    if is_num(var):
                        args_encoded.append(var)
                    else:
                        # use int2bool to make a direct encoding (it should really have a csemap...)
                        var_enc, cons = _encode_int_var({}, var, "direct")  # {}: no ivarmap used
                        # channel int to bools:
                        (encpairs, offset) = var_enc.encode_term()
                        # i1 = sum(vs * bs) + offset :: 1*i1 - sum(vs*bs) == offset
                        ws,vs = zip(*encpairs)
                        ws = [1] + [-1*w for w in ws]
                        vs = [var] + list(vs)
                        cons.append(Operator("wsum", [ws, vs]) == offset)

                        # enforce domain and channel constraints:
                        toplevel.extend(linearize_constraint(cons, supported=supported, csemap=csemap))
                        args_encoded.append(var_enc)

                # each value can be taken at most once
                value_cons = []  # replaces alldifferent by the conjunction of the constraint for each value
                lbs, ubs = get_bounds(cpm_expr.args)
                for i in range(min(lbs), max(ubs)+1):
                    var_eq_i = []
                    for var_enc in args_encoded:
                        if is_num(var_enc):
                            if var_enc == i:
                                var_eq_i.append(BoolVal(True))
                            # else: skip, False anyway
                        else:
                            var_eq_i.append(var_enc.eq(i))
                    value_cons.append(cp.sum(var_eq_i) <= 1)
                # Now... we have a list, and we could be reified; so handled recursevly and add to toplevel
                toplevel += linearize_constraint(value_cons, supported=supported, csemap=csemap, implication_literal=implication_literal)
                continue

            # pre-process Boolean literals, handled as trivial linears or unit clauses depending on 'supported'
            # TODO: I think we should accept that it returns BoolVar? easy to support in solver?
            elif isinstance(cpm_expr, _BoolVarImpl):
                if "or" in supported:
                    # for SAT solvers: post a clause explicitly
                    # (don't use cp.any, which will just return the BoolVar)
                    newlist.append(Operator("or", [cpm_expr]))
                    continue
                elif isinstance(cpm_expr, NegBoolView):
                    # negative literal: might as well remove the negation
                    cpm_expr = Operator("sum", [~cpm_expr]) <= 0
                else: # positive literal
                    cpm_expr = Operator("sum", [cpm_expr]) >= 1
            
            else:
                raise ValueError(f"Unexpected non-comparison expression {cpm_expr} in {lst_of_expr}")

            # if you are still here, the expr must be rewritten to a comparison
            assert isinstance(cpm_expr, Comparison), f"Linearize, after pre-process: expected a comparison, got {cpm_expr}"

        # comparisons (including ones created during pre-processing)
        cmp_name = cpm_expr.name
        lhs, rhs = cpm_expr.args
        lhs_name = lhs.name

        if lhs_name in supported:  # TODO, should we discriminate between reified and non-reified?
            # e.g. gurobi's generalized constraints
            # only accepts <=, >= and ==, so transform <, > and !=
            if cmp_name == "<" or cmp_name == ">":
                delta = -1 if cmp_name == "<" else 1
                cmp_name = "<=" if cmp_name == "<" else ">="
                if is_int(rhs):
                    new_rhs = rhs + delta
                else:
                    # need to introduce a new variable for rhs+delta
                    new_rhs, cons = get_or_make_var(rhs + delta, csemap=csemap)
                    toplevel += linearize_constraint(cons, csemap=csemap)

                # the new inequality
                newlist.append(eval_comparison(cmp_name, lhs, new_rhs))
            elif cmp_name == "!=":
                # TODO should we use csemap here? These are implications, not equalities...
                if implication_literal is None:
                    # lhs != rhs :: (lhs < rhs) xor (lhs > rhs)
                    #            :: z -> lhs < rhs, ~z -> lhs > rhs
                    z = boolvar()
                    toplevel += linearize_constraint([lhs < rhs], supported=supported, implication_literal=z, csemap=csemap)
                    toplevel += linearize_constraint([lhs > rhs], supported=supported, implication_literal=~z, csemap=csemap)
                    continue  # this entry fully handled
                else:
                    # b -> lhs != rhs :: b -> z1 + z2 == 1, toplevel: z1 -> lhs < rhs, z2 -> lhs > rhs
                    # we are inside a '->', so we have to force Big-M instead of a nested '->'
                    tmp_supported = supported if '->' not in supported else supported - {"->"}
                    z1 = boolvar()
                    toplevel += linearize_constraint([lhs < rhs], supported=tmp_supported, implication_literal=z1, csemap=csemap)
                    z2 = boolvar()
                    toplevel += linearize_constraint([lhs > rhs], supported=tmp_supported, implication_literal=z2, csemap=csemap)
                    newlist.append(z1 + z2 == 1)
            else:
                # just post the original
                newlist.append(cpm_expr)

        else:
            # lets make the comparison a canonical linear comparison

            # some LHS rewrites before canonicalization
            if lhs_name == "mul":
                # convert x * y to wsum
                reified = implication_literal is not None
                cpm_expr, cons = linearize_mul_comparison(cpm_expr, supported=supported, reified=reified, csemap=csemap)
                toplevel += cons
                if cpm_expr.name == "and":
                    # special case!! our `bv*iv CMP x` is now two constraints...
                    # we need to recurse on cpm_expr.args with our optional implication literal,
                    # and make sure it is not treated further: add to toplevel
                    toplevel += linearize_constraint(cpm_expr.args, supported=supported, csemap=csemap, implication_literal=implication_literal)
                    continue
                elif not isinstance(cpm_expr, Comparison):
                    # for example it can be a b -> linexpr
                    toplevel += linearize_constraint([cpm_expr], supported=supported, csemap=csemap, implication_literal=implication_literal)
                    continue
                # else cpm_expr is a single linear constraint, continue as expected

            # TODO: make this inline?
            [cpm_expr] = canonical_comparison([cpm_expr])  # just transforms the constraint, not introducing new ones
            if is_bool(cpm_expr):
                if is_false_cst(cpm_expr):
                    if implication_literal is None:
                        return [BoolVal(False)]
                    else:
                        toplevel.extend(linearize_constraint([~implication_literal], supported=supported, csemap=csemap))
                        continue
                # else: trivially true, skip
                continue
            # SHOULD INCLUDE THE OLD:
            #if lhs_name == "sub": # convert x - y to wsum
            #    lhs = Operator("wsum", [[1, -1], [lhs.args[0], lhs.args[1]]])
            #    cpm_expr = eval_comparison(cpm_expr.name, lhs, rhs)
            #elif lhs_name == "-": # convert -x to wsum
            #    lhs = Operator("wsum", [[-1], [lhs.args[0]]])
            #    cpm_expr = eval_comparison(cpm_expr.name, lhs, rhs)
            # update the helper variables after canonicalization
            cmp_name = cpm_expr.name
            lhs, rhs = cpm_expr.args
            lhs_name = lhs.name
            assert is_int(rhs), f"Linearize canonical comparison: expected integer rhs, got {rhs} from {cpm_expr}"

            # Should be fixed elsewhere
            #if "or" in supported and lhs_name == "sum" and len(lhs.args) == 1 and isinstance(lhs.args[0], _BoolVarImpl):
            #    # very special case, avoid writing as sum of 1 argument
            #    newlist.append(Operator("or", [lhs.args[0]]))
            #    continue

            # check trivially true/false (not allowed by PySAT Card/PB)
            if cpm_expr.name in ('<', '<=', '>', '>='):
                lb,ub = lhs.get_bounds()
                t_lb = eval_comparison(cpm_expr.name, lb, rhs)
                t_ub = eval_comparison(cpm_expr.name, ub, rhs)
                if t_lb and t_ub:
                    continue
                elif not t_lb and not t_ub:
                    if implication_literal is None:
                        return [BoolVal(False)]
                    else:
                        toplevel.extend(linearize_constraint([~implication_literal], supported=supported, csemap=csemap))
                        continue

            # fix the comparisons if needed
            if cpm_expr.name == "<":
                #cmp_name = "<="
                #rhs = rhs - 1
                newlist.append(lhs <= rhs - 1)
            elif cpm_expr.name == ">":
                #cmp_name = ">="
                #rhs = rhs + 1
                newlist.append(lhs >= rhs + 1)
            elif cpm_expr.name == "!=":
                # Special case: BV != BV
                if isinstance(lhs, _BoolVarImpl) and isinstance(rhs, _BoolVarImpl):
                    newlist.append(lhs + rhs == 1)
                else:
                    # TODO should we use csemap here? These are implications, not equalities...
                    z = boolvar()
                    tmp_supported = supported
                    if implication_literal is not None and '->' in supported:
                        tmp_supported = supported - {"->"}
                    ztrue = linearize_constraint([lhs < rhs], supported=tmp_supported, implication_literal=z, csemap=csemap)
                    zfalse = linearize_constraint([lhs > rhs], supported=tmp_supported, implication_literal=~z, csemap=csemap)
                    toplevel += linearize_constraint(ztrue + zfalse, supported=supported, implication_literal=implication_literal, csemap=csemap)
                    continue  # this entry fully handled
            else:
                newlist.append(cpm_expr)
        
    # we now have canonical comparisons (var|sum|wsum (=|<=|>=) int) in 'newlist'
    # handle the implication part if present
    if implication_literal is not None:
        if '->' in supported:
            # implications natively suppported
            newlist = [implication_literal.implies(cpm_expr) for cpm_expr in newlist]
        else:
            # linearize with Big-M
            # BV -> (C1 and ... and Cn) == (BV -> C1) and ... and (BV -> Cn)
            linlist = []
            # first replace each '==' by two '<=' and '>='
            for lin in newlist:
                if lin.name == '==':
                    linlist.append(lin.args[0] <= lin.args[1])
                    linlist.append(lin.args[0] >= lin.args[1])
                else:
                    linlist.append(lin)
            # new overwrite newlist with the Big-Md constraints
            newlist = []
            for lin in linlist:
                if is_num(lin):
                    if is_false_cst(lin):
                        # cond must be false... (TODO: until we decide to accept boolvar, recurse)
                        return linearize_constraint([~cond], supported=supported, csemap=csemap)
                    # else: trivially true, skip this one
                elif isinstance(lin, _BoolVarImpl):
                    # shortcut for (impl -> expr) :: (~impl | expr) :: (~impl + expr >= 1) :: (1-impl + expr >= 1) :: wsum([-1,1], [impl, expr]) >= 0
                    newlist.append(Operator("wsum", [[-1,1], [implication_literal, lin]]) >= 0)
                else:
                    # need to linearize the implication constraint itself
                    if lin.name == "or":
                        # Argh, stupid special cases... this could happen for other supported constraints too, if '->' is not supported
                        newlist.append(Operator("or", [~implication_literal]+lin.args))
                        continue
                    assert isinstance(lin, Comparison), f"Expected a comparison as rhs of implication constraint, got {lin}"
                    lhs,rhs = lin.args
                    assert lhs.name in frozenset({'sum', 'wsum'}), f"Expected sum or wsum as lhs of implication constraint, but got {lhs}"
                    assert is_num(rhs)  # unnecessary assert
                    lb, ub = get_bounds(lhs)
                    if lin.name == "<=":
                        M = rhs - ub # subtracting M from lhs will always satisfy the implied constraint
                        newlist.append(lhs + M*~implication_literal <= rhs)
                    elif lin.name == ">=":
                        M = rhs - lb # adding M to lhs will always satisfy the implied constraint
                        newlist.append(lhs + M*~implication_literal >= rhs)
                    else:
                        raise ValueError(f"Unexpected linearized rhs of implication {lin} in {cpm_expr}")
    
    if len(newlist) == 0 and len(toplevel) == 0:
        # all constraints are trivially true
        return [BoolVal(True)]
    return newlist + toplevel

def only_positive_bv(lst_of_expr: List[Expression], csemap: Optional[Dict[Any, Any]] = None) -> List[Expression]:
    """
    Replaces :class:`~cpmpy.expressions.comparison.Comparison` containing :class:`~cpmpy.expressions.variables.NegBoolView`
    with equivalent expression using only :class:`~cpmpy.expressions.variables.BoolVar`.

    Comparisons are expected to be linearized. Only apply after applying :func:`linearize_constraint(cpm_expr) <linearize_constraint>`.
    The resulting expression is linear if the original expression was linear.

    Arguments:
        lst_of_expr: List of linearized CPMpy expressions that may contain NegBoolView.
        csemap: Optional dictionary for common subexpression elimination, mapping expressions
            to an equivalent decision variable.

    Returns:
        List of CPMpy expressions where all boolean variables appear positively (no NegBoolView).
    """
    newlist = []
    for cpm_expr in lst_of_expr:

        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            new_lhs = lhs
            new_cons = []

            if isinstance(lhs, _NumVarImpl) or lhs.name in {"sum","wsum"}:
                new_lhs, const = only_positive_bv_wsum_const(lhs)
                rhs -= const
            else:
                # other operators in comparison such as "min", "max"
                nbv_sel = [isinstance(a, NegBoolView) for a in lhs.args]
                if any(nbv_sel):
                    new_args = []
                    for i, nbv in enumerate(nbv_sel):
                        if nbv:
                            aux = cp.boolvar()
                            new_args.append(aux)
                            new_cons += [aux + lhs.args[i]._bv == 1]  # aux == 1 - arg._bv
                        else:
                            new_args.append(lhs.args[i])

                    new_lhs = copy.copy(lhs)
                    new_lhs.update_args(new_args)

            if new_lhs is not lhs:
                newlist.append(eval_comparison(cpm_expr.name, new_lhs, rhs))
                newlist += new_cons  # already linear
            else:
                newlist.append(cpm_expr)

        # reification
        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            cond, subexpr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"{cpm_expr} is not a supported linear expression. Apply " \
                                                   f"`linearize_constraint` before calling `only_positive_bv` "
            # BV -> Expr
            subexpr = only_positive_bv([subexpr], csemap=csemap)
            newlist += [cond.implies(expr) for expr in subexpr]

        elif isinstance(cpm_expr, _BoolVarImpl):
            raise ValueError(f"Unreachable: unexpected Boolean literal (`_BoolVarImpl`) in expression {cpm_expr}, perhaps `linearize_constraint` was not called before this `only_positive_bv `call")
        elif isinstance(cpm_expr, (GlobalConstraint, BoolVal, DirectConstraint)):
            newlist.append(cpm_expr)
        else:
            raise Exception(f"{cpm_expr} is not linear or is not supported. Please report on github")

    return newlist

def only_positive_bv_wsum(expr: Expression) -> Expression:
    """
    Replaces a var/sum/wsum expression containing :class:`~cpmpy.expressions.variables.NegBoolView`
    with an equivalent expression using only :class:`~cpmpy.expressions.variables._BoolVarImpl`.

    It might add a constant term to the expression, e.g. `3*~b + 2*c` is transfromed to `3 + -3*b + 2*c`.
    If you want the constant separately, use :func:`only_positive_bv_wsum_const`.

    Arguments:
        expr: Linear expression (NumVar, sum, or wsum) that may contain NegBoolView.

    Returns:
        Linear expression (NumVar, sum, or wsum) without NegBoolView. The constant term
        (if any) is incorporated into the expression.
    """
    if isinstance(expr, _NumVarImpl) or expr.name in {"sum","wsum"}:
        pos_expr, const = only_positive_bv_wsum_const(expr)
        if const == 0:
            return pos_expr
        else:
            assert isinstance(pos_expr, Operator) and pos_expr.name == "wsum", f"unexpected expression, should be wsum but got {pos_expr}"
            # should we check if it already has a constant term?
            return Operator("wsum", [pos_expr.args[0]+[1], pos_expr.args[1]+[const]])
    else:
        return expr

def only_positive_bv_wsum_const(cpm_expr: Expression) -> Tuple[Expression, int]:
    """
    Replaces a var/sum/wsum expression containing :class:`~cpmpy.expressions.variables.NegBoolView`
    with an equivalent expression using only :class:`~cpmpy.expressions.variables._BoolVarImpl`,
    and returns the constant term separately.

    If you want the expression where the constant term is part of the wsum returned,
    use :func:`only_positive_bv_wsum`.

    Arguments:
        cpm_expr: Linear expression (NumVar, sum, or wsum) that may contain NegBoolView.

    Returns:
        Tuple of:
        - pos_expr: Linear expression (NumVar, sum, or wsum) without NegBoolView.
        - const: The constant term that must be added to pos_expr to make it equivalent
                 to the original expression.
    """
    if isinstance(cpm_expr, _NumVarImpl):
        if isinstance(cpm_expr,NegBoolView):
            return Operator("wsum",[[-1], [cpm_expr._bv]]), 1
        else:
            return cpm_expr, 0

    elif cpm_expr.name == "sum":
        # indicator on arguments being negboolviews
        nbv_sel = [isinstance(a, NegBoolView) for a in cpm_expr.args]
        if any(nbv_sel):
            const = 0
            weights = []
            variables = []
            for i, nbv in enumerate(nbv_sel):
                if nbv:
                    const += 1
                    weights.append(-1)
                    variables.append(cpm_expr.args[i]._bv)
                else:
                    weights.append(1)
                    variables.append(cpm_expr.args[i])
            return Operator("wsum", [weights, variables]), const
        else:
            return cpm_expr, 0

    elif cpm_expr.name == "wsum":
        # indicator on arguments of the wsum variable being negboolviews
        nbv_sel = [isinstance(a, NegBoolView) for a in cpm_expr.args[1]]
        if any(nbv_sel):
            # copy weights and variables
            weights = [w for w in cpm_expr.args[0]]
            variables = [v for v in cpm_expr.args[1]]
            const = 0
            for i, nbv in enumerate(nbv_sel):
                if nbv:
                    const += weights[i]
                    weights[i] = -weights[i]
                    variables[i] = variables[i]._bv
            return Operator("wsum", [weights, variables]), const
        else:
            return cpm_expr, 0

    else:
        raise ValueError(f"unexpected expression, should be sum, wsum or var but got {cpm_expr}")


def canonical_comparison(lst_of_expr: Union[Expression, List[Expression]]) -> List[Expression]:
    """
    Canonicalize comparison expressions.

    Transforms linear expressions, or a reification thereof, into canonical form by:
    - moving all variables to the left-hand side
    - moving constants to the right-hand side

    Expects the input constraints to be flat. Only apply after applying :func:`flatten_constraint`.

    Arguments:
        lst_of_expr: Single expression or list of CPMpy expressions to canonicalize.
                     Can be a single :class:`~cpmpy.expressions.core.Expression` or a list.

    Returns:
        List of canonicalized CPMpy expressions. All variables are on the left-hand side
        and all constants are on the right-hand side of comparisons.
    """

    lst_of_expr = toplevel_list(lst_of_expr) # ensure it is a list

    newlist = []
    for cpm_expr in lst_of_expr:

        if isinstance(cpm_expr, Operator) and cpm_expr.name == '->':    # half reification of comparison
            lhs, rhs = cpm_expr.args
            if isinstance(rhs, Comparison):
                rhs = canonical_comparison(rhs)[0]
                newlist.append(lhs.implies(rhs))
            elif isinstance(lhs, Comparison):
                lhs = canonical_comparison(lhs)[0]
                newlist.append(lhs.implies(rhs))
            else:
                newlist.append(cpm_expr)

        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            if isinstance(lhs, Comparison) and (is_bool(rhs) or isinstance(rhs, Expression) and rhs.is_bool()):
                assert cpm_expr.name == "==", "Expected a reification of a comparison here, but got {}".format(cpm_expr.name)
                lhs = canonical_comparison(lhs)[0]
            elif is_num(lhs) or isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name in {"sum", "wsum", "sub"}):
                if lhs.name == "sub":
                    lhs = Operator("wsum", [[1,-1],lhs.args])
                # bring all vars to lhs
                lhs2 = []
                if isinstance(rhs, _NumVarImpl):
                    lhs2, rhs = [-1 * rhs], 0
                elif isinstance(rhs, BoolVal):
                    lhs2, rhs = [-1] if rhs.value() else [], 0 
                elif isinstance(rhs, Operator) and rhs.name == "-":
                    lhs2, rhs = [rhs.args[0]], 0
                elif isinstance(rhs, Operator) and rhs.name == "sum":
                    lhs2, rhs = [-1 * b if isinstance(b, _NumVarImpl) else -1 * b.args[0] for b in rhs.args
                                 if isinstance(b, _NumVarImpl) or isinstance(b, Operator)], \
                                 sum(b for b in rhs.args if is_num(b))
                elif isinstance(rhs, Operator) and rhs.name == "wsum":
                    lhs2, rhs = [-a * b for a, b in zip(rhs.args[0], rhs.args[1])
                                    if isinstance(b, _NumVarImpl)], \
                                    sum(-a * b for a, b in zip(rhs.args[0], rhs.args[1])
                                    if not isinstance(b, _NumVarImpl))
                
                # 2) add collected variables to lhs
                if isinstance(lhs, Operator) and lhs.name == "sum":
                    lhs = sum([1 * a for a in lhs.args] + lhs2)
                elif isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name == "wsum"):
                    lhs = lhs + lhs2
                else:
                    raise ValueError(f"unexpected expression on lhs of expression, should be sum, wsum or intvar but got {lhs}")

                assert not is_num(lhs), "lhs cannot be an integer at this point!"

                # bring all const to rhs
                if isinstance(lhs, Operator):
                    if lhs.name == "sum":
                        new_args = []
                        for i, arg in enumerate(lhs.args):
                            if is_num(arg):
                                rhs -= int(arg)
                            else:
                                new_args.append(arg)
                        lhs = Operator("sum", new_args)

                    elif lhs.name == "wsum":
                        new_weights, new_args = [], []
                        for i, (w, arg) in enumerate(zip(*lhs.args)):
                            if is_num(arg):
                                rhs -= w * int(arg)
                            else:
                                new_weights.append(w)
                                new_args.append(arg)
                        lhs = Operator("wsum", [new_weights, new_args])
                    else:
                        raise ValueError(f"lhs should be sum or wsum, but got {lhs}")
                else:
                    assert isinstance(lhs, _NumVarImpl), f"Expected variable here, but got {lhs} in expression {cpm_expr}"
                    lhs = Operator("sum", [lhs])

            newlist.append(eval_comparison(cpm_expr.name, lhs, rhs))
        else:   # rest of expressions
            newlist.append(cpm_expr)

    return newlist

def only_positive_coefficients(lst_of_expr: List[Expression]) -> List[Expression]:
    """
    Replaces Boolean terms with negative coefficients in linear constraints with terms
    with positive coefficients by negating the literal.

    This can simplify a `wsum` into `sum` when all coefficients become 1.
    Input expressions are expected to be canonical comparisons. Only apply after
    applying :func:`canonical_comparison(cpm_expr) <canonical_comparison>`.

    Arguments:
        lst_of_expr: List of canonical CPMpy expressions (comparisons) that may contain
                     boolean variables with negative coefficients.

    Returns:
        List of CPMpy expressions where all boolean variables have positive coefficients.
        The resulting expression is linear if the original expression was linear.
    """
    newlist = []
    for cpm_expr in lst_of_expr:
        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args

            #    ... -c*b + ... <= k
            # :: ... -c*(1 - ~b) + ... <= k
            # :: ... -c + c* ~b + ... <= k
            # :: ... + c*~b + ... <= k+c
            if lhs.name == "wsum":
                weights, args = lhs.args
                idxes = {i for i, (w, a) in enumerate(zip(weights, args)) if w < 0 and isinstance(a, _BoolVarImpl)}
                nw, na = zip(*[(-w, ~a) if i in idxes else (w, a) for i, (w, a) in enumerate(zip(weights, args))])
                rhs += sum(-weights[i] for i in idxes)

                # Simplify wsum to sum if all weights are 1
                if all(w == 1 for w in nw):
                    lhs = Operator("sum", [list(na)])
                else:
                    lhs = Operator("wsum", [list(nw), list(na)])

            newlist.append(eval_comparison(cpm_expr.name, lhs, rhs))

        # reification
        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            cond, subexpr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"{cpm_expr} is not a supported linear expression. Apply " \
                                                   f"`linearize_constraint` before calling `only_positive_coefficients` "
            subexpr = only_positive_coefficients([subexpr])
            newlist += [cond.implies(expr) for expr in subexpr]

        else:
            newlist.append(cpm_expr)

    return newlist
