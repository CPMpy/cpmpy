"""
Flattening a model (or individual constraints) into 'flat normal form'.

In flat normal form, constraints belong to one of three families with all arguments
either constants, variables, list of constants or list of variables, and
some binary constraints have a canonical order of variables.

Furthermore, it is 'negated normal' meaning that the ~ (negation operator) only appears
before a Boolean variable (in CPMpy, absorbed in a 'NegBoolView'),
and it is 'negation normal' meaning that the - (negative operator) only appears before
a constant, that is a - b :: a + -1*b :: wsum([1,-1],[a,b])

The three families of possible constraints are:

Base constraints: (no nesting)
------------------------------

    - Boolean variable
    - Boolean operators: and([Var]), or([Var])             (CPMpy class 'Operator', is_bool())
    - Boolean impliciation: Var -> Var                     (CPMpy class 'Operator', is_bool())
    - Boolean equality: Var == Var                         (CPMpy class 'Comparison')
                        Var == Constant                    (CPMpy class 'Comparison')
    - Global constraint (Boolean): global([Var]*)          (CPMpy class 'GlobalConstraint', is_bool())

Comparison constraints: (up to one nesting on one side)
-------------------------------------------------------

    - Numeric equality:  Numexpr == Var                    (CPMpy class 'Comparison')
                         Numexpr == Constant               (CPMpy class 'Comparison')
    - Numeric disequality: Numexpr != Var                  (CPMpy class 'Comparison')
                           Numexpr != Constant             (CPMpy class 'Comparison')
    - Numeric inequality (>=,>,<,<=): Numexpr >=< Var      (CPMpy class 'Comparison')

    Numexpr:

        - Operator (non-Boolean) with all args Var/constant (examples: +,*,/,mod,wsum)
                                                           (CPMpy class 'Operator', not is_bool())
        - Global constraint (non-Boolean) (examples: Max,Min,Element)
                                                           (CPMpy class 'GlobalConstraint', not is_bool()))

    wsum: wsum([Const],[Var]) represents sum([Const]*[Var]) # TODO: not implemented yet

Reify/imply constraint: (up to two nestings on one side)
--------------------------------------------------------

    - Reification (double implication): Boolexpr == Var    (CPMpy class 'Comparison')
    - Implication: Boolexpr -> Var                         (CPMpy class 'Operator', is_bool())
                   Var -> Boolexpr                         (CPMpy class 'Operator', is_bool())

    Boolexpr:

        - Boolean operators: and([Var]), or([Var])             (CPMpy class 'Operator', is_bool())
        - Boolean equality: Var == Var                         (CPMpy class 'Comparison')
        - Global constraint (Boolean): global([Var]*)          (CPMpy class 'GlobalConstraint', is_bool())
        - Comparison constraint (see above)                    (CPMpy class 'Comparison')
    
    Reification of a comparison is the most complex case as it can allow up to 3 levels of nesting in total, e.g.:

        - (wsum([1,2,3],[IV1,IV2,IV3]) > 5) == BV
        - (IV1 == IV2) == BV
        - (BV1 == BV2) == BV3

Objective: (up to one nesting)
------------------------------

    - Satisfaction problem: None
    - Decision variable: Var
    - Linear: sum([Var])                                   (CPMpy class 'Operator', name 'sum')
              wsum([Const],[Var])                          (CPMpy class 'Operator', name 'wsum')

The output after calling flatten_model() or flatten_constraint() will ONLY contain expressions
of the form specified above.

The flattening does not promise to do common subexpression elimination or to automatically group
commutative expressions (and, or, sum, wsum, ...) but such optimisations should be added later.

TODO: update behind_the_scenes.rst doc with the new 'flat normal form'
TODO: small optimisations, e.g. and/or chaining (potentially after negation), see test_flatten
"""
import copy
import math
import builtins
import numpy as np

from .normalize import toplevel_list, simplify_boolean
from ..expressions.core import *
from ..expressions.core import _wsum_should, _wsum_make
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView
from ..expressions.utils import is_num, is_any_list, is_boolexpr
from .negation import recurse_negation, push_down_negation


def flatten_model(orig_model):
    """
        Receives model, returns new model where every constraint is in 'flat normal form'
    """
    from ..model import Model  # otherwise circular dependency...

    # the top-level constraints
    basecons = flatten_constraint(orig_model.constraints)

    # the objective
    if orig_model.objective_ is None:
        return Model(*basecons)  # no objective, satisfaction problem
    else:
        (newobj, newcons) = flatten_objective(orig_model.objective_)
        basecons += newcons
        if orig_model.objective_is_min:
            return Model(*basecons, minimize=newobj)
        else:
            return Model(*basecons, maximize=newobj)


def flatten_constraint(expr):
    """
        input is any expression; except is_num(), pure _NumVarImpl,
        or Operator/GlobalConstraint with not is_bool()
        
        output: see definition of 'flat normal form' above.

        it will return 'Exception' if something is not supported
        TODO, what built-in python error is best?
        RE TODO: we now have custom NotImpl/NotSupported
    """
    from ..expressions.globalconstraints import GlobalConstraint  # avoid circular import

    newlist = []
    # for backwards compatibility reasons, we now consider it a meta-
    # transformation, that calls (preceding) transformations itself
    # e.g. `toplevel_list()` ensures it is a list
    lst_of_expr = toplevel_list(expr)               # ensure it is a list
    lst_of_expr = push_down_negation(lst_of_expr)   # push negation into the arguments to simplify expressions
    lst_of_expr = simplify_boolean(lst_of_expr)     # simplify boolean expressions, and ensure types are correct
    for expr in lst_of_expr:

        if isinstance(expr, _BoolVarImpl):
            newlist.append(expr)

        elif isinstance(expr, Operator):
            """
            - Base Boolean operators: and([Var]), or([Var])        (CPMpy class 'Operator', is_bool())
            - Base Boolean impliciation: Var -> Var                (CPMpy class 'Operator', is_bool())
            - Implication: Boolexpr -> Var                         (CPMpy class 'Operator', is_bool())
                           Var -> Boolexpr                         (CPMpy class 'Operator', is_bool())
            """
            # does not type-check that arguments are bool... Could do now with expr.is_bool()!
            if all(__is_flat_var(arg) for arg in expr.args):
                newlist.append(expr)
                continue
            elif expr.name == 'or':
                # rewrites that avoid auxiliary var creation, should go to normalize?
                # in case of an implication in a disjunction, merge in
                if builtins.any(isinstance(a, Operator) and a.name == '->' for a in expr.args):
                    newargs = list(expr.args)  # take copy
                    for i,a in enumerate(newargs):
                        if isinstance(a, Operator) and a.name == '->':
                            newargs[i:i+1] = [~a.args[0],a.args[1]]
                    # there could be nested implications
                    newlist.extend(flatten_constraint(Operator('or', newargs)))
                    continue
                # conjunctions in disjunctions could be split out by applying distributivity,
                # but this would explode the number of constraints in favour of having less auxiliary variables.
                # Testing has proven that this is not worth it.
            elif expr.name == '->':
                # some rewrite rules that avoid creating auxiliary variables
                # 1) if rhs is 'and', split into individual implications a0->and([a11..a1n]) :: a0->a11,...,a0->a1n
                if expr.args[1].name == 'and':
                    a1s = expr.args[1].args
                    a0 = expr.args[0]
                    newlist.extend(flatten_constraint([a0.implies(a1) for a1 in a1s]))
                    continue
                # 2) if lhs is 'or' then or([a01..a0n])->a1 :: ~a1->and([~a01..~a0n] and split
                elif expr.args[0].name == 'or':
                    a0s = expr.args[0].args
                    a1 = expr.args[1]
                    newlist.extend(flatten_constraint([(~a1).implies(~a0) for a0 in a0s]))
                    continue
                # 2b) if lhs is ->, like 'or': a01->a02->a1 :: (~a01|a02)->a1 :: ~a1->a01,~a1->~a02
                elif expr.args[0].name == '->':
                    a01,a02 = expr.args[0].args
                    a1 = expr.args[1]
                    newlist.extend(flatten_constraint([(~a1).implies(a01), (~a1).implies(~a02)]))
                    continue

                # ->, allows a boolexpr on one side
                elif isinstance(expr.args[0], _BoolVarImpl):
                    # LHS is var, ensure RHS is normalized 'Boolexpr'
                    lhs,lcons = expr.args[0], ()
                    rhs,rcons = normalized_boolexpr(expr.args[1])
                else:
                    # make LHS normalized 'Boolexpr', RHS must be a var
                    lhs,lcons = normalized_boolexpr(expr.args[0])
                    rhs,rcons = get_or_make_var(expr.args[1])

                newlist.append(Operator(expr.name, (lhs,rhs)))
                newlist.extend(lcons)
                newlist.extend(rcons)
                continue



            # if none of the above cases + continue matched:
            # a normalizable boolexpr
            (con, flatcons) = normalized_boolexpr(expr)
            newlist.append(con)
            newlist.extend(flatcons)

        elif isinstance(expr, Comparison):
            """
    - Base Boolean equality: Var == Var                         (CPMpy class 'Comparison')
                             Var == Constant                    (CPMpy class 'Comparison')
    - Numeric equality:  Numexpr == Var                    (CPMpy class 'Comparison')
                         Numexpr == Constant               (CPMpy class 'Comparison')
    - Numeric disequality: Numexpr != Var                  (CPMpy class 'Comparison')
                           Numexpr != Constant             (CPMpy class 'Comparison')
    - Numeric inequality (>=,>,<,<=,): Numexpr >=< Var     (CPMpy class 'Comparison')
    - Reification (double implication): Boolexpr == Var    (CPMpy class 'Comparison')
            """
            exprname = expr.name  # so it can be modified
            lexpr, rexpr = expr.args
            rewritten = False

            # rewrite 'Var == Expr' to normalzed 'Expr == Var'
            if (expr.name == '==' or expr.name == '!=') \
                    and __is_flat_var(lexpr) and not __is_flat_var(rexpr):
                lexpr, rexpr = rexpr, lexpr
                rewritten = True

            # rewrite 'BoolExpr != BoolExpr' to normalized 'BoolExpr == ~BoolExpr'
            if exprname == '!=' and lexpr.is_bool() and rexpr.is_bool():
                exprname = '=='
                rexpr = ~rexpr
                rewritten = True

            # already flat?
            if all(__is_flat_var(arg) for arg in [lexpr, rexpr]):
                if not rewritten:
                    newlist.append(expr)  # original
                else:
                    newlist.append(Comparison(exprname, lexpr, rexpr))
                continue

            # ensure rhs is var
            (rvar, rcons) = get_or_make_var(rexpr)
            # Reification (double implication): Boolexpr == Var
            # normalize the lhs (does not have to be a var, hence we call normalize instead of get_or_make_var
            if exprname == '==' and lexpr.is_bool():
                if rvar.is_bool():
                    # this is a reification
                    (lhs, lcons) = normalized_boolexpr(lexpr)
                else:
                    # integer comparison
                    (lhs, lcons) = get_or_make_var(lexpr)
            else:
                (lhs, lcons) = normalized_numexpr(lexpr)

            newlist.append(Comparison(exprname, lhs, rvar))
            newlist.extend(lcons)
            newlist.extend(rcons)

        elif isinstance(expr, GlobalConstraint):
            """
    - Global constraint: global([Var]*)          (CPMpy class 'GlobalConstraint')
            """
            (con, flatcons) = normalized_boolexpr(expr)
            newlist.append(con)
            newlist.extend(flatcons)

        else:
            # any other case (e.g. DirectConstraint), pass as is
            newlist.append(expr)

    return newlist


def flatten_objective(expr, supported=frozenset(["sum","wsum"])):
    """
    - Decision variable: Var
    - Linear: sum([Var])                                   (CPMpy class 'Operator', name 'sum')
              wsum([Const],[Var])                          (CPMpy class 'Operator', name 'wsum')
    """
    # lets be very explicit here
    if is_any_list(expr):
        # one source of errors is sum(v) where v is a matrix, use v.sum() instead
        raise Exception(f"Objective expects a single variable/expression, not a list of expressions")

    expr = simplify_boolean([expr])[0]
    (flatexpr, flatcons) = normalized_numexpr(expr)  # might rewrite expr into a (w)sum
    if isinstance(flatexpr, Expression) and flatexpr.name in supported:
        return (flatexpr, flatcons)
    else:
        # any other numeric expression,
        var, cons = get_or_make_var(flatexpr)
        return (var, cons+flatcons)


def __is_flat_var(arg):
    """ True if the variable is a numeric constant, or a _NumVarImpl (incl subclasses)
    """
    return is_num(arg) or isinstance(arg, _NumVarImpl)

def __is_flat_var_or_list(arg):
    """ True if the variable is a numeric constant, or a _NumVarImpl (incl subclasses)
        or a list of __is_flat_var_or_list
    """
    return is_num(arg) or isinstance(arg, _NumVarImpl) or \
           is_any_list(arg) and all(__is_flat_var_or_list(el) for el in arg)


def get_or_make_var(expr):
    """
        Must return a variable, and list of flat normal constraints
        Determines whether this is a Boolean or Integer variable and returns
        the equivalent of: (var, normalize(expr) == var)
    """
    if __is_flat_var(expr):
        return (expr, [])

    if is_any_list(expr):
        raise Exception(f"Expected single variable, not a list for: {expr}")

    if expr.is_bool():
        # normalize expr into a boolexpr LHS, reify LHS == bvar
        (flatexpr, flatcons) = normalized_boolexpr(expr)

        if isinstance(flatexpr,_BoolVarImpl):
            #avoids unnecessary bv == bv or bv == ~bv assignments
            return flatexpr,flatcons
        bvar = _BoolVarImpl()
        return (bvar, [flatexpr == bvar]+flatcons)

    else:
        # normalize expr into a numexpr LHS,
        # then compute bounds and return (newintvar, LHS == newintvar)
        (flatexpr, flatcons) = normalized_numexpr(expr)

        lb, ub = flatexpr.get_bounds()
        ivar = _IntVarImpl(lb, ub)
        return (ivar, [flatexpr == ivar]+flatcons)

def get_or_make_var_or_list(expr):
    """ Like get_or_make_var() but also accepts and recursively transforms lists
        Used to convert arguments of globals
    """
    if __is_flat_var_or_list(expr):
        return (expr,[])
    elif is_any_list(expr):
        flatvars, flatcons = zip(*[get_or_make_var(arg) for arg in expr])
        return (flatvars, [c for con in flatcons for c in con])
    else:
        return get_or_make_var(expr)


def normalized_boolexpr(expr):
    """
        input is any Boolean (is_bool()) expression
        output are all 'flat normal form' Boolean expressions that can be 'reified', meaning that
            - subexpr == BoolVar
            - subexpr -> BoolVar

        are valid output expressions.

        Currently, this is the case for subexpr:
        - Boolean operators: and([Var]), or([Var])             (CPMpy class 'Operator', is_bool())
        - Boolean equality: Var == Var                         (CPMpy class 'Comparison')
        - Global constraint: global([Var]*)                    (CPMpy class 'GlobalConstraint')
        - Comparison constraint (see elsewhere)                (CPMpy class 'Comparison')

        output: (base_expr, base_cons) with:
            base_expr: same as 'expr', but all arguments are variables
            base_cons: list of flat normal constraints
    """
    assert(not __is_flat_var(expr))
    assert(expr.is_bool()) 

    if isinstance(expr, Operator):
        # and, or, ->

        # apply De Morgan's transform for "implies"
        if expr.name == '->':
            # TODO, optimisation if args0 is an 'and'?
            (lhs,lcons) = get_or_make_var(expr.args[0])
            # TODO, optimisation if args1 is an 'or'?
            (rhs,rcons) = get_or_make_var(expr.args[1])
            return ((~lhs | rhs), lcons+rcons)
        if expr.name == 'not':
            flatvar, flatcons = get_or_make_var(expr.args[0])
            return (~flatvar, flatcons)
        if all(__is_flat_var(arg) for arg in expr.args):
            return (expr, [])
        else:
            # one of the arguments is not flat, flatten all
            flatvars, flatcons = zip(*[get_or_make_var(arg) for arg in expr.args])
            newexpr = Operator(expr.name, flatvars)
            return (newexpr, [c for con in flatcons for c in con])

    elif isinstance(expr, Comparison):
        if expr.name != '!=' and all(__is_flat_var(arg) for arg in expr.args):
            return (expr, [])  # shortcut
        else:
            # LHS can be boolexpr, RHS has to be variable

            lexpr, rexpr = expr.args
            exprname = expr.name

            # ==,!=: can swap if lhs is var and rhs is not
            if (exprname == '==' or exprname == '!=') and \
                not __is_flat_var(rexpr) and __is_flat_var(lexpr):
                lexpr, rexpr = rexpr, lexpr

            # ensure rhs is var
            (rvar, rcons) = get_or_make_var(rexpr)

            # LHS: check if Boolexpr == smth:
            if (exprname == '==' or exprname == '!=') and lexpr.is_bool():
                # this is a reified constraint, so lhs must be var too to be in normal form
                (lhs, lcons) = get_or_make_var(lexpr)
                if expr.name == '!=' and rvar.is_bool():
                    # != not needed, negate RHS variable
                    rvar = ~rvar
                    exprname = '=='
            else:
                # other cases: LHS is numexpr
                (lhs, lcons) = normalized_numexpr(lexpr)

            return (Comparison(exprname, lhs, rvar), lcons+rcons)

    else:
        """
        - Global constraint (Boolean): global([Var]*)          (CPMpy class 'GlobalConstraint', is_bool())
        """
        # just recursively flatten args, which can be lists
        if all(__is_flat_var_or_list(arg) for arg in expr.args):
            return (expr, [])
        else:
            # recursively flatten all children
            flatargs, flatcons = zip(*[get_or_make_var_or_list(arg) for arg in expr.args])

            # take copy, replace args
            newexpr = copy.copy(expr) # shallow or deep? currently shallow
            newexpr.args = flatargs
            return (newexpr, [c for con in flatcons for c in con])


def normalized_numexpr(expr):
    """
        all 'flat normal form' numeric expressions...

        Currently, this is the case for:

        - Operator (non-Boolean) with all args Var/constant (examples: +,*,/,mod,wsum)
                                                           (CPMpy class 'Operator', not is_bool())
        - Global constraint (non-Boolean) (examples: Max,Min,Element)
                                                           (CPMpy class 'GlobalConstraint', not is_bool()))

        output: (base_expr, base_cons) with:
            base_expr: same as 'expr', but all arguments are variables
            base_cons: list of flat normal constraints
    """
    # XXX a boolexpr is also a valid numexpr... e.g. 30*(iv > 5) + ... see mario obj.
    if __is_flat_var(expr):
        return (expr, [])

    elif expr.is_bool():
        # unusual case, but its truth-value is a valid numexpr
        # so reify and return the boolvar
        return get_or_make_var(expr)

    elif isinstance(expr, Operator):
        # rewrite -a, const*a and a*const into a weighted sum, so it can be used as objective
        if expr.name == '-' or (expr.name == 'mul' and _wsum_should(expr)):
            return normalized_numexpr(Operator("wsum", _wsum_make(expr)))

        if all(__is_flat_var(arg) for arg in expr.args):
            return (expr, [])

        # pre-process sum, to fold in nested subtractions and const*Exprs, e.g. x - y + 2*(z+r)
        if expr.name == "sum" and \
           all(isinstance(a, Expression) for a in expr.args) and \
           any((a.name == "-" or _wsum_should(a)) for a in expr.args):
            we = [_wsum_make(a) for a in expr.args]
            w = [wi for w,_ in we for wi in w]
            e = [ei for _,e in we for ei in e]
            return normalized_numexpr(Operator("wsum", (w,e)))

        # wsum needs special handling because expr.args is a tuple of which only 2nd one has exprs
        if expr.name == 'wsum':
            weights, sub_exprs  = expr.args
            # while here, avoid creation of auxiliary variables for compatible operators -/sum/wsum
            i = 0
            while(i < len(sub_exprs)): # can dynamically change
                if isinstance(sub_exprs[i], Operator) and \
                    ((sub_exprs[i].name in ['-', 'sum'] and \
                        all(isinstance(a, Expression) for a in sub_exprs[i].args)) or \
                     (sub_exprs[i].name == 'wsum' and \
                        all(isinstance(a, Expression) for a in sub_exprs[i].args[1]))):  # TODO: avoid constants for now...
                    w,e = _wsum_make(sub_exprs[i])
                    # insert in place, and next iteration over same 'i' again
                    weights[i:i+1] = [weights[i]*wj for wj in w]
                    sub_exprs[i:i+1] = e
                else:
                    i = i+1

            # now flatten the resulting subexprs
            flatvars, flatcons = map(list, zip(*[get_or_make_var(arg) for arg in sub_exprs])) # also bool, reified...
            newexpr = Operator(expr.name, (weights, flatvars))
            return (newexpr, [c for con in flatcons for c in con])

        else: # generic operator
            # recursively flatten all children
            flatvars, flatcons = zip(*[get_or_make_var(arg) for arg in expr.args])

            newexpr = Operator(expr.name, flatvars)
            return (newexpr, [c for con in flatcons for c in con])
    else:
        # Globalfunction (examples: Max,Min,Element)

        # just recursively flatten args, which can be lists
        if all(__is_flat_var_or_list(arg) for arg in expr.args):
            return (expr, [])
        else:
            # recursively flatten all children
            flatvars, flatcons = zip(*[get_or_make_var_or_list(arg) for arg in expr.args])

            # take copy, replace args
            newexpr = copy.copy(expr) # shallow or deep? currently shallow
            newexpr.args = flatvars
            return (newexpr, [c for con in flatcons for c in con])

    raise Exception("Operator '{}' not allowed as numexpr".format(expr)) # or bug

