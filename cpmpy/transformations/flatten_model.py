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

=============================  ====================================  ==============
Boolean variable               ``Var``                                                                                                                                                    
Boolean operators              ``and([Var])``, ``or([Var])``         :class:`~cpmpy.expressions.core.Operator`, :func:`~cpmpy.expressions.core.Operator.is_bool()`                        
Boolean implication            ``Var -> Var``                        :class:`~cpmpy.expressions.core.Operator`, :func:`~cpmpy.expressions.core.Operator.is_bool()`                                                                                
Boolean equality               ``Var == Var``, ``Var == Constant``   :class:`~cpmpy.expressions.core.Comparison`                                                                          
Global constraint (Boolean)    ``global([Var]*)``                    :class:`~cpmpy.expressions.globalconstraints.GlobalConstraint`, :func:`~cpmpy.expressions.core.Operator.is_bool()`                                           
=============================  ====================================  ==============

Comparison constraints: (up to one nesting on one side)
-------------------------------------------------------

===============================  ============================================  ==============
Numeric equality                 ``Numexpr == Var``, ``Numexpr == Constant``   :class:`~cpmpy.expressions.core.Comparison`
Numeric disequality              ``Numexpr != Var``, ``Numexpr != Constant``   :class:`~cpmpy.expressions.core.Comparison`
Numeric inequality (>=,>,<,<=)   ``Numexpr >=< Var``                           :class:`~cpmpy.expressions.core.Comparison`
===============================  ============================================  ==============                                                    

**Numexpr:**

==================================================  ======================================  ==============
Operator (non-Boolean) with all args Var/constant   ``+``, ``*``, ``/``, ``mod``, ``wsum``  :class:`~cpmpy.expressions.core.Operator`, not :func:`~cpmpy.expressions.core.Operator.is_bool()`                      
Global constraint (non-Boolean)                     ``Max``, ``Min``, ``Element``           :class:`~cpmpy.expressions.globalconstraints.GlobalConstraint`, not :func:`~cpmpy.expressions.core.Operator.is_bool()` 
==================================================  ======================================  ==============

**wsum:**

.. todo::
    wsum([Const],[Var]) represents sum([Const]*[Var]) # TODO: not implemented yet

Reify/imply constraint: (up to two nestings on one side)
--------------------------------------------------------
=================================  =========================================  ==============
Reification (double implication)   ``Boolexpr == Var``                        :class:`~cpmpy.expressions.core.Comparison`                                                   
Implication                        ``Boolexpr -> Var``, ``Var -> Boolexpr``   :class:`~cpmpy.expressions.core.Operator`, :func:`~cpmpy.expressions.core.Operator.is_bool()` 
=================================  =========================================  ==============

**Boolexpr:**

==================================  =============================  ==============
Boolean operators                   ``and([Var])``, ``or([Var])``  :class:`~cpmpy.expressions.core.Operator`, not :func:`~cpmpy.expressions.core.Operator.is_bool()`                      
Boolean equality                    ``Var == Var``                 :class:`~cpmpy.expressions.core.Comparison`                                                                            
Global constraint (Boolean)         ``global([Var]*)``             :class:`~cpmpy.expressions.globalconstraints.GlobalConstraint`, :func:`~cpmpy.expressions.core.Operator.is_bool()`     
Comparison constraint (see above)                                  :class:`~cpmpy.expressions.core.Comparison`       
==================================  =============================  ==============                                                                     

    
Reification of a comparison is the most complex case as it can allow up to 3 levels of nesting in total, e.g.:

- (wsum([1,2,3],[IV1,IV2,IV3]) > 5) == BV
- (IV1 == IV2) == BV
- (BV1 == BV2) == BV3

Objective: (up to one nesting)
------------------------------
======================  ========================================  ============
Type                    Example                                   Notes                                                  
======================  ========================================  ============
Satisfaction problem    ``None``                                                                                         
Decision variable       ``Var``                                   :class:`~cpmpy.expressions.core.Operator`, name `sum`  
Linear                  ``sum([Var])``, ``wsum([Const],[Var])``   :class:`~cpmpy.expressions.core.Operator`, name `wsum` 
======================  ========================================  ============


The output after calling :func:`flatten_model()` or :func:`flatten_constraint()` will ONLY contain expressions
of the form specified above.

The flattening does not promise to do common subexpression elimination or to automatically group
commutative expressions (``and``, ``or``, ``sum``, ``wsum``, ...) but such optimisations should be added later.

.. todo::
    TODO: update behind_the_scenes.rst doc with the new 'flat normal form'
    TODO: small optimisations, e.g. and/or chaining (potentially after negation), see test_flatten
"""
import math
import builtins
import cpmpy as cp

from .normalize import toplevel_list, simplify_boolean
from ..expressions.core import *
from ..expressions.core import _wsum_should, _wsum_make
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl
from ..expressions.utils import is_num, is_any_list, is_int, is_star
from .negation import push_down_negation


def flatten_model(orig_model, csemap=None):
    """
        Receives model, returns new model where every constraint is in 'flat normal form'
    """

    # the top-level constraints
    basecons = flatten_constraint(orig_model.constraints, csemap=csemap)

    # the objective
    if orig_model.objective_ is None:
        return cp.Model(*basecons)  # no objective, satisfaction problem
    else:
        (newobj, newcons) = flatten_objective(orig_model.objective_, csemap=csemap)
        basecons += newcons
        if orig_model.objective_is_min:
            return cp.Model(*basecons, minimize=newobj)
        else:
            return cp.Model(*basecons, maximize=newobj)


def flatten_constraint(expr, csemap=None):
    """
        input is any expression; except is_num(), pure _NumVarImpl,
        or Operator/GlobalConstraint with not is_bool()
        
        output: see definition of 'flat normal form' above.

        it will return 'Exception' if something is not supported

        .. todo::
            TODO, what built-in python error is best?
            RE TODO: we now have custom NotImpl/NotSupported
    """

    newlist = []
    # for backwards compatibility reasons, we now consider it a meta-
    # transformation, that calls (preceding) transformations itself
    # e.g. `toplevel_list()` ensures it is a list
    lst_of_expr = toplevel_list(expr)               # ensure it is a list
    lst_of_expr = push_down_negation(lst_of_expr)   # push negation into the arguments to simplify expressions
    lst_of_expr = simplify_boolean(lst_of_expr)     # simplify boolean expressions, and ensure types are correct
    for expr in lst_of_expr:

        if not expr.has_subexpr():
            newlist.append(expr)  # no need to do anything
            continue

        elif isinstance(expr, Operator):
            """
            - Base Boolean operators: and([Var]), or([Var])        (CPMpy class 'Operator', is_bool())
            - Base Boolean impliciation: Var -> Var                (CPMpy class 'Operator', is_bool())
            - Implication: Boolexpr -> Var                         (CPMpy class 'Operator', is_bool())
                           Var -> Boolexpr                         (CPMpy class 'Operator', is_bool())
            """
            # does not type-check that arguments are bool... Could do now with expr.is_bool()!
            if expr.name == 'or':
                # rewrites that avoid auxiliary var creation, should go to normalize?
                # in case of an implication in a disjunction, merge in
                if builtins.any(isinstance(a, Operator) and a.name == '->' for a in expr.args):
                    newargs = list(expr.args)  # take copy
                    for i,a in enumerate(newargs):
                        if isinstance(a, Operator) and a.name == '->':
                            newargs[i:i+1] = [~a.args[0],a.args[1]]
                    # there could be nested implications
                    newlist.extend(flatten_constraint(Operator('or', newargs), csemap=csemap))
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
                    newlist.extend(flatten_constraint([a0.implies(a1) for a1 in a1s], csemap=csemap))
                    continue
                # 2) if lhs is 'or' then or([a01..a0n])->a1 :: ~a1->and([~a01..~a0n] and split
                elif expr.args[0].name == 'or':
                    a0s = expr.args[0].args
                    a1 = expr.args[1]
                    newlist.extend(flatten_constraint([(~a1).implies(~a0) for a0 in a0s], csemap=csemap))
                    continue
                # 2b) if lhs is ->, like 'or': a01->a02->a1 :: (~a01|a02)->a1 :: ~a1->a01,~a1->~a02
                elif expr.args[0].name == '->':
                    a01,a02 = expr.args[0].args
                    a1 = expr.args[1]
                    newlist.extend(flatten_constraint([(~a1).implies(a01), (~a1).implies(~a02)], csemap=csemap))
                    continue

                # ->, allows a boolexpr on one side
                elif isinstance(expr.args[0], _BoolVarImpl):
                    # LHS is var, ensure RHS is normalized 'Boolexpr'
                    lhs,lcons = expr.args[0], ()
                    rhs,rcons = normalized_boolexpr(expr.args[1], csemap=csemap)
                else:
                    # make LHS normalized 'Boolexpr', RHS must be a var
                    lhs,lcons = normalized_boolexpr(expr.args[0], csemap=csemap)
                    rhs,rcons = get_or_make_var(expr.args[1], csemap=csemap)

                newlist.append(Operator(expr.name, (lhs,rhs)))
                newlist.extend(lcons)
                newlist.extend(rcons)
                continue



            # if none of the above cases + continue matched:
            # a normalizable boolexpr
            (con, flatcons) = normalized_boolexpr(expr, csemap=csemap)
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

            # rewrite 'Var # Expr' to normalized 'Expr # Var' (where # is any comparator)
            if __is_flat_var(lexpr) and not __is_flat_var(rexpr):
                assert (expr.name in ('==', '!=', '>', '>=', '<', '<='))
                lexpr, rexpr = rexpr, lexpr
                rewritten = True

                # flip comparator in case of inequality
                if exprname == '>':
                    exprname = '<'
                elif exprname == '>=':
                    exprname = '<='
                elif exprname == '<':
                    exprname = '>'
                elif exprname == '<=':
                    exprname = '>='

            # already flat?
            if not expr.has_subexpr():
                if not rewritten:
                    newlist.append(expr)  # original
                else:
                    newlist.append(Comparison(exprname, lexpr, rexpr))
                continue

            # ensure rhs is var
            (rvar, rcons) = get_or_make_var(rexpr, csemap=csemap)
            # Reification (double implication): Boolexpr == Var
            # normalize the lhs (does not have to be a var, hence we call normalize instead of get_or_make_var
            if exprname == '==' and lexpr.is_bool():
                if rvar.is_bool():
                    # this is a reification
                    (lhs, lcons) = normalized_boolexpr(lexpr, csemap=csemap)
                else:
                    # integer comparison
                    (lhs, lcons) = get_or_make_var(lexpr, csemap=csemap)
            else:
                (lhs, lcons) = normalized_numexpr(lexpr, csemap=csemap)

            newlist.append(Comparison(exprname, lhs, rvar))
            newlist.extend(lcons)
            newlist.extend(rcons)

        elif isinstance(expr, cp.expressions.globalconstraints.GlobalConstraint):
            """
    - Global constraint: global([Var]*)          (CPMpy class 'GlobalConstraint')
            """
            (con, flatcons) = normalized_boolexpr(expr, csemap=csemap)
            newlist.append(con)
            newlist.extend(flatcons)

        else:
            # any other case (e.g. DirectConstraint), pass as is
            newlist.append(expr)

    return newlist


def flatten_objective(expr, supported=frozenset(["sum", "wsum"]), csemap=None):
    """
    - Decision variable: Var
    - Linear: 
        ======================                       ========
        sum([Var])                                   (CPMpy class 'Operator', name 'sum')
        wsum([Const],[Var])                          (CPMpy class 'Operator', name 'wsum')
        ======================                       ========
    """
    # lets be very explicit here
    if is_any_list(expr):
        # one source of errors is sum(v) where v is a matrix, use v.sum() instead
        raise Exception(f"Objective expects a single variable/expression, not a list of expressions: {expr}")

    expr = simplify_boolean([expr])[0]
    (flatexpr, flatcons) = normalized_numexpr(expr, csemap=csemap)  # might rewrite expr into a (w)sum
    if isinstance(flatexpr, Expression) and flatexpr.name in supported:
        return (flatexpr, flatcons)
    else:
        # any other numeric expression,
        var, cons = get_or_make_var(flatexpr, csemap=csemap)
        return (var, cons+flatcons)


def __is_flat_var(arg):
    """ True if the variable is a numeric constant, or a _NumVarImpl (incl subclasses)
    """
    return is_num(arg) or isinstance(arg, _NumVarImpl)

def __is_flat_var_or_list(arg):
    """ True if the variable is a numeric constant, or a _NumVarImpl (incl subclasses)
        or a list of __is_flat_var_or_list, or it is a wildcard as used in the ShortTable global constraint
    """
    return is_num(arg) or isinstance(arg, _NumVarImpl) or \
           is_any_list(arg) and all(__is_flat_var_or_list(el) for el in arg) or \
           is_star(arg)

def get_or_make_var(expr, csemap=None):
    """
        Must return a variable, and list of flat normal constraints
        Determines whether this is a Boolean or Integer variable and returns
        the equivalent of: (var, normalize(expr) == var)
    """

    if __is_flat_var(expr):
        return (expr, [])

    if is_any_list(expr):
        raise Exception(f"Expected single variable, not a list for: {expr}")

    if csemap is not None and expr in csemap:
        return csemap[expr], []

    if expr.is_bool():
        # normalize expr into a boolexpr LHS, reify LHS == bvar
        (flatexpr, flatcons) = normalized_boolexpr(expr, csemap=csemap)

        if isinstance(flatexpr,_BoolVarImpl):
            # avoids unnecessary bv == bv or bv == ~bv assignments
            return flatexpr,flatcons
        bvar = _BoolVarImpl()

        # save expr in dict
        if csemap is not None:
            csemap[expr] = bvar
        return bvar, [flatexpr == bvar] + flatcons

    else:
        # normalize expr into a numexpr LHS,
        # then compute bounds and return (newintvar, LHS == newintvar)
        (flatexpr, flatcons) = normalized_numexpr(expr, csemap=csemap)

        lb, ub = flatexpr.get_bounds()
        if not is_int(lb) or not is_int(ub):
            warnings.warn(f"CPMpy only uses integer variables, but found expression ({expr}) with domain {lb}({type(lb)}"
                          f" - {ub}({type(ub)}. CPMpy will rewrite this constriants with integer bounds instead.")
            lb, ub = math.floor(lb), math.ceil(ub)
        ivar = _IntVarImpl(lb, ub)

        # save expr in dict
        if csemap is not None:
            csemap[expr] = ivar
        return ivar, [flatexpr == ivar] + flatcons

def get_or_make_var_or_list(expr, csemap=None):
    """ Like get_or_make_var() but also accepts and recursively transforms lists
        Used to convert arguments of globals
    """

    if __is_flat_var_or_list(expr):
        return (expr,[])
    elif is_any_list(expr):
        flatvars, flatcons = zip(*[get_or_make_var(arg, csemap=csemap) for arg in expr])
        return (flatvars, [c for con in flatcons for c in con])
    else:
        return get_or_make_var(expr, csemap=csemap)


def normalized_boolexpr(expr, csemap=None):
    """
        input is any Boolean (is_bool()) expression
        output are all 'flat normal form' Boolean expressions that can be 'reified', meaning that

        .. code-block:: text

            subexpr == BoolVar
            subexpr -> BoolVar

        are valid output expressions.

        Currently, this is the case for subexpr:

        =====================================  =============================  ==============
        Boolean operators                      ``and([Var])``, ``or([Var])``  :class:`~cpmpy.expressions.core.Operator`, not :func:`~cpmpy.expressions.core.Operator.is_bool()`                      
        Boolean equality                       ``Var == Var``                 :class:`~cpmpy.expressions.core.Comparison`                                                                            
        Global constraint (Boolean)            ``global([Var]*)``             :class:`~cpmpy.expressions.globalconstraints.GlobalConstraint`, :func:`~cpmpy.expressions.core.Operator.is_bool()`     
        Comparison constraint (see elsewhere)                                 :class:`~cpmpy.expressions.core.Comparison`       
        =====================================  =============================  ==============                                                                     

        Result:
            (base_expr, base_cons) with:
            
            - base_expr: same as 'expr', but all arguments are variables
            - base_cons: list of flat normal constraints
    """
    assert(not __is_flat_var(expr))
    assert(expr.is_bool()) 

    if isinstance(expr, Operator):
        # and, or, ->

        # apply De Morgan's transform for "implies"
        if expr.name == '->':
            # TODO, optimisation if args0 is an 'and'?
            (lhs,lcons) = get_or_make_var(expr.args[0], csemap=csemap)
            # TODO, optimisation if args1 is an 'or'?
            (rhs,rcons) = get_or_make_var(expr.args[1], csemap=csemap)
            return ((~lhs | rhs), lcons+rcons)
        if expr.name == 'not':
            flatvar, flatcons = get_or_make_var(expr.args[0], csemap=csemap)
            return (~flatvar, flatcons)
        if not expr.has_subexpr():
            return (expr, [])
        else:
            # one of the arguments is not flat, flatten all
            flatvars, flatcons = zip(*[get_or_make_var(arg, csemap=csemap) for arg in expr.args])
            newexpr = Operator(expr.name, flatvars)
            return (newexpr, [c for con in flatcons for c in con])

    elif isinstance(expr, Comparison):
        if (expr.name != '!=') and (not expr.has_subexpr()):
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
            (rvar, rcons) = get_or_make_var(rexpr, csemap=csemap)

            # LHS: check if Boolexpr == smth:
            if (exprname == '==' or exprname == '!=') and lexpr.is_bool():
                # this is a reified constraint, so lhs must be var too to be in normal form
                (lhs, lcons) = get_or_make_var(lexpr, csemap=csemap)
                if expr.name == '!=' and rvar.is_bool():
                    # != not needed, negate RHS variable
                    rvar = ~rvar
                    exprname = '=='
            else:
                # other cases: LHS is numexpr
                (lhs, lcons) = normalized_numexpr(lexpr, csemap=csemap)

            return (Comparison(exprname, lhs, rvar), lcons+rcons)

    else:
        """
        - Global constraint (Boolean): global([Var]*)          (CPMpy class 'GlobalConstraint', is_bool())
        """
        # just recursively flatten args, which can be lists
        if not expr.has_subexpr():
            return (expr, [])
        else:
            # recursively flatten all children
            flatargs, flatcons = zip(*[get_or_make_var_or_list(arg, csemap=csemap) for arg in expr.args])

            # take copy, replace args
            newexpr = copy.copy(expr) # shallow or deep? currently shallow
            newexpr.update_args(flatargs)
            return (newexpr, [c for con in flatcons for c in con])


def normalized_numexpr(expr, csemap=None):
    """
        all 'flat normal form' numeric expressions...

        Currently, this is the case for:

        - Operator (non-Boolean) with all args Var/constant (examples: +,*,/,mod,wsum)
                                                           (CPMpy class 'Operator', not is_bool())
        - Global constraint (non-Boolean) (examples: Max,Min,Element)
                                                           (CPMpy class 'GlobalConstraint', not is_bool()))

        Result:
            (base_expr, base_cons) with:
            
            - base_expr: same as 'expr', but all arguments are variables
            - base_cons: list of flat normal constraints
    """
    # XXX a boolexpr is also a valid numexpr... e.g. 30*(iv > 5) + ... see mario obj.
    if __is_flat_var(expr):
        return (expr, [])

    elif expr.is_bool():
        # unusual case, but its truth-value is a valid numexpr
        # so reify and return the boolvar
        return get_or_make_var(expr, csemap=csemap)

    elif isinstance(expr, Operator):
        # rewrite -a, const*a and a*const into a weighted sum, so it can be used as objective
        if expr.name == '-' or (expr.name == 'mul' and _wsum_should(expr)):
            return normalized_numexpr(Operator("wsum", _wsum_make(expr)), csemap=csemap)

        if not expr.has_subexpr():
            return (expr, [])

        # pre-process sum, to fold in nested subtractions and const*Exprs, e.g. x - y + 2*(z+r)
        if expr.name == "sum" and \
           all(isinstance(a, Expression) for a in expr.args) and \
           any((a.name == "-" or _wsum_should(a)) for a in expr.args):
            we = [_wsum_make(a) for a in expr.args]
            w = [wi for w,_ in we for wi in w]
            e = [ei for _,e in we for ei in e]
            return normalized_numexpr(Operator("wsum", (w,e)), csemap=csemap)

        # wsum needs special handling because expr.args is a tuple of which only 2nd one has exprs
        if expr.name == 'wsum':
            weights, sub_exprs  = expr.args
            # while here, avoid creation of auxiliary variables for compatible operators -/sum/wsum
            i = 0
            while(i < len(sub_exprs)): # can dynamically change
                if isinstance(sub_exprs[i], Operator) and \
                    ((sub_exprs[i].name in ['-', 'sum'] and
                      all(isinstance(a, Expression) for a in sub_exprs[i].args)) or
                     (sub_exprs[i].name == 'wsum' and
                      all(isinstance(a, Expression) for a in sub_exprs[i].args[1]))):  # TODO: avoid constants for now...
                    w,e = _wsum_make(sub_exprs[i])
                    # insert in place, and next iteration over same 'i' again
                    weights[i:i+1] = [weights[i]*wj for wj in w]
                    sub_exprs[i:i+1] = e
                else:
                    i = i+1

            # now flatten the resulting subexprs
            flatvars, flatcons = map(list, zip(*[get_or_make_var(arg, csemap=csemap) for arg in sub_exprs])) # also bool, reified...
            newexpr = Operator(expr.name, (weights, flatvars))
            return (newexpr, [c for con in flatcons for c in con])

        else: # generic operator
            # recursively flatten all children
            flatvars, flatcons = zip(*[get_or_make_var(arg, csemap=csemap) for arg in expr.args])

            newexpr = Operator(expr.name, flatvars)
            return (newexpr, [c for con in flatcons for c in con])
    else:
        # Globalfunction (examples: Max,Min,Element)

        # just recursively flatten args, which can be lists
        if not expr.has_subexpr():
            return (expr, [])
        else:
            # recursively flatten all children
            flatvars, flatcons = zip(*[get_or_make_var_or_list(arg, csemap=csemap) for arg in expr.args])

            # take copy, replace args
            newexpr = copy.copy(expr) # shallow or deep? currently shallow
            newexpr.update_args(flatvars)
            return (newexpr, [c for con in flatcons for c in con])
