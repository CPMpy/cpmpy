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
    - Boolean operators: and([Var]), or([Var]), xor([Var]) (CPMpy class 'Operator', is_bool())
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

        - Boolean operators: and([Var]), or([Var]), xor([Var]) (CPMpy class 'Operator', is_bool())
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

TODO: remove zipcycle (no longer needed)
TODO: use normalized_boolexpr when possible in the flatten_cons operator case.
TODO: update behind_the_scenes.rst doc with the new 'flat normal form'
TODO: small optimisations, e.g. and/or chaining (potentially after negation), see test_flatten
"""
import copy
import math
from ..expressions.core import *
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView
from ..expressions.utils import is_num, is_any_list

from itertools import cycle
def __zipcycle(vars1, vars2):
    v1 = [vars1] if not is_any_list(vars1) else vars1
    v2 = [vars2] if not is_any_list(vars2) else vars2
    return zip(v1, cycle(v2)) if len(v2) < len(v1) else zip(cycle(v1), v2)

def flatten_model(orig_model):
    """
        Receives model, returns new model where every constraint is in 'flat normal form'
    """
    from ..model import Model # otherwise circular dependency...

    # the top-level constraints
    basecons = []
    for con in orig_model.constraints:
        basecons += flatten_constraint(con)

    # the objective
    if orig_model.objective is None:
        return Model(*basecons) # no objective, satisfaction problem
    else:
        (newobj, newcons) = flatten_objective(orig_model.objective)
        basecons += newcons
        if orig_model.objective_max:
            return Model(*basecons, maximize=newobj)
        else:
            return Model(*basecons, minimize=newobj)


def flatten_constraint(expr):
    """
        input is any expression; except is_num(), pure _NumVarImpl,
        or Operator/GlobalConstraint with not is_bool()
        
        output: see definition of 'flat normal form' above.

        it will return 'Exception' if something is not supported
        TODO, what built-in python error is best?
    """
    # base cases
    if isinstance(expr, _BoolVarImpl) or isinstance(expr, bool):
        return [expr]
    elif is_num(expr) or isinstance(expr, _NumVarImpl):
        raise Exception("Numeric constants or numeric variables not allowed as base constraint")

    # recursively flatten list of constraints
    if is_any_list(expr):
        flatcons = [flatten_constraint(e) for e in expr]
        return [c for con in flatcons for c in con]
    # recursively flatten top-level 'and'
    if isinstance(expr, Operator) and expr.name == 'and':
        flatcons = [flatten_constraint(e) for e in expr.args]
        return [c for con in flatcons for c in con]

    assert expr.is_bool(), f"Boolean expressions only in flatten_constraint, `{expr}` not allowed."

    if isinstance(expr, Operator):
        """
            - Base Boolean operators: and([Var]), or([Var]), xor([Var]) (CPMpy class 'Operator', is_bool())
            - Base Boolean impliciation: Var -> Var                     (CPMpy class 'Operator', is_bool())
            - Implication: Boolexpr -> Var                         (CPMpy class 'Operator', is_bool())
                           Var -> Boolexpr                         (CPMpy class 'Operator', is_bool())
        """
        # does not type-check that arguments are bool... Could do now with expr.is_bool()!
        if all(__is_flat_var(arg) for arg in expr.args):
            return [expr]
        elif not expr.name == '->':
            # and, or, xor
            # recursively flatten all children
            flatvars, flatcons = zip(*[get_or_make_var(arg) for arg in expr.args])

            newexpr = Operator(expr.name, flatvars)
            return [newexpr]+[c for con in flatcons for c in con]
        else:
            # ->, allows a boolexpr on one side
            if isinstance(expr.args[0], _BoolVarImpl):
                # LHS is var, ensure RHS is normalized 'Boolexpr'
                lhs = expr.args[0]
                (rhs,flatcons) = normalized_boolexpr(expr.args[1])
            else:
                # make LHS normalized 'Boolexpr', RHS must be a var
                (lhs,lcons) = normalized_boolexpr(expr.args[0])
                (rhs,rcons) = get_or_make_var(expr.args[1])
                flatcons = lcons+rcons

            newexpr = Operator(expr.name, (lhs,rhs))
            return [newexpr]+[c for c in flatcons]

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

        flatcons = []
        # zipcycle: unfolds 'arr1 == arr2' pairwise
        # XXX: zipcycle no longer needed, vectorized now handled at creation level!
        for lexpr, rexpr in __zipcycle(expr.args[0], expr.args[1]):
            if __is_flat_var(lexpr) and __is_flat_var(rexpr):
                flatcons += [Comparison(expr.name, lexpr, rexpr)]
            else:
                # RHS must be var (or const)
                lexpr,rexpr = expr.args
                exprname = expr.name
                # ==,!=: can swap if lhs is var and rhs is not
                # TODO: this is very similar to (part of) normalize_boolexpr??
                # XXX indeed, every case that is not 'reification' can be
                # delegated to normalize_boolexpr... TODO
                if (exprname == '==' or exprname == '!=') and \
                    not __is_flat_var(rexpr) and __is_flat_var(lexpr):
                    (lexpr,rexpr) = (rexpr,lexpr)
                # ensure rhs is var
                (rvar, rcons) = get_or_make_var(rexpr)

                # LHS: check if Boolexpr == smth:
                if (exprname == '==' or exprname == '!=') and lexpr.is_bool():
                    if is_num(rexpr):
                        # BoolExpr == 0|False
                        # special case, handled in normalized_boolexpr()
                        (con, subs) = normalized_boolexpr(expr)
                        flatcons += [con] + subs
                        continue # ready with this one

                    # Reification (double implication): Boolexpr == Var
                    if __is_flat_var(lexpr):
                        (lhs, lcons) = (lexpr, [])
                    else:
                        (lhs, lcons) = normalized_boolexpr(lexpr)
                    if expr.name == '!=':
                        # != not needed, negate RHS variable
                        rhs = ~rvar
                        exprname = '=='
                else:
                    # other cases: LHS is numexpr
                    (lhs, lcons) = normalized_numexpr(lexpr)

                flatcons += [Comparison(exprname, lhs, rvar)]+lcons+rcons

        return flatcons

    else:
        """
    - Global constraint (Boolean): global([Var]*)          (CPMpy class 'GlobalConstraint', is_bool())
        """
        # just recursively flatten args, which can be lists
        if all(__is_flat_var_or_list(arg) for arg in expr.args):
            return [expr]
        else:
            # recursively flatten all children
            flatvars, flatcons = zip(*[get_or_make_var_or_list(arg) for arg in expr.args])

            # take copy, replace args
            newexpr = copy.copy(expr) # shallow or deep? currently shallow
            newexpr.args = flatvars
            return [newexpr]+[c for con in flatcons for c in con]


def flatten_objective(expr):
    """
    - Decision variable: Var
    - Linear: sum([Var])                                   (CPMpy class 'Operator', name 'sum')
              wsum([Const],[Var])                          (CPMpy class 'Operator', name 'wsum')
    """
    if __is_flat_var(expr):
        return (expr, [])

    # lets be very explicit here
    if is_any_list(expr):
        # one source of errors is sum(v) where v is a matrix, use v.sum() instead
        raise Exception(f"Objective expects a single variable/expression, not a list of expressions")

    if isinstance(expr, Operator) and (expr.name == 'sum' or expr.name == 'wsum'):
        if expr.name == 'sum':
            if all(__is_flat_var(arg) for arg in expr.args):
                return (expr, [])
            else:
                # one of the arguments is not flat, flatten all
                flatvars, flatcons = zip(*[get_or_make_var(arg) for arg in expr.args])
                newexpr = Operator(expr.name, flatvars)
                return (newexpr, [c for con in flatcons for c in con])
        elif expr.name == 'wsum':
            raise NotImplementedError(expr) # TODO, wsum
    
    # any other numeric expression
    return get_or_make_var(expr)


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
        (newexpr, flatcons) = normalized_boolexpr(expr)

        bvar = _BoolVarImpl()
        return (bvar, [newexpr == bvar]+flatcons)

    #else:
    # normalize expr into a numexpr LHS, return LHS == intvar
    # includes estimating appropriate bounds for intvar...

    # special case, -var... 
    # XXX until we do weighted sum, turn into -1*args[0]
    if isinstance(expr, Operator) and expr.name == '-': # unary
        return get_or_make_var(-1*expr.args[0])

    if isinstance(expr, Operator):
        # TODO: more like above, call normalized_numexpr() on expr, then equate...
        flatvars, flatcons = zip(*[get_or_make_var(arg) for arg in expr.args]) # also bool, reified...
        lbs = [var.lb if isinstance(var, _NumVarImpl) else var for var in flatvars]
        ubs = [var.ub if isinstance(var, _NumVarImpl) else var for var in flatvars]

        if expr.name == 'abs': # unary
            if lbs[0] < 0 and ubs[0] > 0:
                lb = 0 # from neg to pos, so includes 0
            else:
                lb = min(abs(lbs[0]), abs(ubs[0])) # same sign, take smallest
            ub = max(abs(lbs[0]), abs(ubs[0])) # largest abs value
            ivar = _IntVarImpl(lb, ub)
        elif expr.name == 'mul': # binary
            v0 = [lbs[0], ubs[0]]
            v1 = [lbs[1], ubs[1]]
            bnds = [v0[i]*v1[j] for i in [0,1] for j in [0,1]]
            ivar = _IntVarImpl(min(bnds),max(bnds)) 
        elif expr.name == 'div': # binary
            num = [lbs[0], ubs[0]]
            denom = [lbs[1], ubs[1]]
            bnds = [num[i]/denom[j] for i in [0,1] for j in [0,1]]
            # the above can give fractional values, tighten bounds to integer
            ivar = _IntVarImpl(math.ceil(min(bnds)), math.floor(max(bnds))) 
        elif expr.name == 'mod': # binary 
            # broadest possible assumptions
            # (negative possible if divisor is negative)
            ivar = _IntVarImpl(lbs[0], ubs[0])
        elif expr.name == 'pow': # binary
            base = [lbs[0], ubs[0]]
            exp = [lbs[1], ubs[1]]
            if exp[0] < 0:
                raise NotImplementedError("Power operator: For integer values, exponent must be non-negative")
            bnds = [base[i]**exp[j] for i in [0,1] for j in [0,1]]
            if exp[1] > 0: # even/uneven behave differently when base is negative
                bnds += [base[0]**(exp[1]-1), base[1]**(exp[1]-1)]
            ivar = _IntVarImpl(min(bnds), max(bnds))
        elif expr.name == 'sum': # n-ary
            ivar = _IntVarImpl(sum(lbs), sum(ubs)) 
        # TODO: weighted sum
        elif expr.is_bool(): # Boolean
            ivar = _BoolVarImpl() # TODO: we can support Bool? check, update docs
        else:
            raise Exception("Operator '{}' not known in get_or_make_var".format(expr.name)) # or bug

        newexpr = (Operator(expr.name, flatvars) == ivar)
        return (ivar, [newexpr]+[c for con in flatcons for c in con])

    else:
        """
        - Global constraint (non-Boolean) (examples: Max,Min,Element)
        """
        # just recursively flatten args, which can be lists
        if all(__is_flat_var_or_list(arg) for arg in expr.args):
            newexpr = expr
            flatcons = []
        else:
            flatvars, flatcons = zip(*[get_or_make_var_or_list(arg) for arg in expr.args]) # also bool, reified...
            #idx, icons = flatten_numexpr(idx)
            #arr, acons = zip(*[flatten_numexpr(e) for e in arr])
            #basecons = icons+[c for con in acons for c in con]

            # take copy, replace args
            newexpr = copy.copy(expr) # shallow or deep? currently shallow
            newexpr.args = flatvars

        # XXX Also, how to get the bounds on the new variable? have the solver handle it?
        # XXX Add to GlobalCons as function? e.g. (lb,ub) = expr.get_bounds()? would also work for Operator...
        ivar = _IntVarImpl(-2147483648, 2147483647) # TODO, this can breaks solvers

        return (ivar, [newexpr == ivar]+[c for con in flatcons for c in con])
    

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
        all 'flat normal form' Boolean expressions that can be 'reified', meaning that

            - expr == BoolVar
            - expr != BoolVar
            - expr -> BoolVar

        are valid expressions.

        Currently, this is the case for:
        - Boolean operators: and([Var]), or([Var]), xor([Var]) (CPMpy class 'Operator', is_bool())
        - Boolean equality: Var == Var                         (CPMpy class 'Comparison')
        - Global constraint (Boolean): global([Var]*)          (CPMpy class 'GlobalConstraint', is_bool())
        - Comparison constraint (see elsewhere)                (CPMpy class 'Comparison')

        output: (base_expr, base_cons) with:
            base_expr: same as 'expr', but all arguments are variables
            base_cons: list of flat normal constraints
    """
    assert(not __is_flat_var(expr))
    assert(expr.is_bool()) 

    if isinstance(expr, Operator):
        # and, or, xor, ->

        # apply De Morgan's transform for "implies"
        if expr.name == '->':
            # TODO, optimisation if args0 is an 'and'?
            (lhs,lcons) = get_or_make_var(expr.args[0])
            # TODO, optimisation if args1 is an 'or'?
            (rhs,rcons) = get_or_make_var(expr.args[1])
            return ((~lhs | rhs), lcons+rcons)

        if all(__is_flat_var(arg) for arg in expr.args):
            return (expr, [])
        else:
            # one of the arguments is not flat, flatten all
            flatvars, flatcons = zip(*[get_or_make_var(arg) for arg in expr.args])
            newexpr = Operator(expr.name, flatvars)
            return (newexpr, [c for con in flatcons for c in con])

    elif isinstance(expr, Comparison):
        if all(__is_flat_var(arg) for arg in expr.args):
            return (expr, [])
        else:
            # LHS can be numexpr, RHS has to be variable

            # TODO: optimisations that swap directions instead when it can avoid to create vars
            """
            if expr.name == '==' or expr.name == '!=':
                # RHS has to be variable, LHS can be more
                if __is_flat_var(lexpr) and not __is_flat_var(rexpr):
                    # LHS is var and RHS not, swap for new expression
                    lexpr, rexpr = rexpr, lexpr

                if __is_numexpr(lexpr) and __is_numexpr(rexpr):
                    # numeric case
                    (lrich, lcons) = flatten_objective(lexpr)
                    (rvar, rcons) = flatten_numexpr(rexpr)
                else:
                    # Boolean case
                    # make LHS reify_ready, RHS var
                    (lrich, lcons) = reify_ready_boolexpr(lexpr)
                    (rvar, rcons) = flatten_boolexpr(rexpr)
                flatcons += [Comparison(expr.name, lrich, rvar)]+lcons+rcons

            else: # inequalities '<=', '<', '>=', '>'
                newname = expr.name
                # LHS can be linexpr, RHS a var
                if __is_flat_var(lexpr) and not __is_flat_var(rexpr):
                    # LHS is var and RHS not, swap for new expression (incl. operator name)
                    lexpr, rexpr = rexpr, lexpr
                    if   expr.name == "<=": newname = ">="
                    elif expr.name == "<":  newname = ">"
                    elif expr.name == ">=": newname = "<="
                    elif expr.name == ">":  newname = "<"

                # make LHS like objective, RHS var
                (lrich, lcons) = flatten_objective(lexpr)
                (rvar, rcons) = flatten_numexpr(rexpr)
                flatcons += [Comparison(newname, lrich, rvar)]+lcons+rcons
            """

            # XXX This is duplicate code from flatten_constraint!!! (except return)
            # -> move into shared function??? or call this one there?
            # But should support one 'less' level of nesting?

            # RHS must be var (or const)
            lexpr,rexpr = expr.args
            exprname = expr.name
            # ==,!=: can swap if lhs is var and rhs is not
            if (exprname == '==' or exprname == '!=') and \
                not __is_flat_var(rexpr) and __is_flat_var(lexpr):
                (lexpr,rexpr) = (rexpr,lexpr)
            # ensure rhs is var
            (rvar, rcons) = get_or_make_var(rexpr)

            # LHS: check if Boolexpr == smth:
            if (exprname == '==' or exprname == '!=') and lexpr.is_bool():
                if is_num(rexpr):
                    # BoolExpr == 0|False
                    assert(not rexpr) # 'true' is preprocessed away

                    nnexpr = negated_normal(lexpr)
                    return normalized_boolexpr(nnexpr)

                # Reification (double implication): Boolexpr == Var
                (lhs, lcons) = normalized_boolexpr(lexpr)
                if expr.name == '!=':
                    # != not needed, negate RHS variable
                    rhs = ~rvar
                    exprname = '=='
            else:
                # other cases: LHS is numexpr
                (lhs, lcons) = normalized_numexpr(lexpr)

            return (Comparison(exprname, lhs, rvar), lcons+rcons)

    else:
        """
        - Global constraint (Boolean): global([Var]*)          (CPMpy class 'GlobalConstraint', is_bool())
        """
        # XXX literal copy from flatten_cons... (except return)
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

    # special case, -var... 
    # XXX until we do weighted sum, turn into -1*args[0]
    if isinstance(expr, Operator) and expr.name == '-': # unary
        return normalized_numexpr(-1*expr.args[0])

    if isinstance(expr, Operator):
        if all(__is_flat_var(arg) for arg in expr.args):
            return (expr, [])
        
        # recursively flatten all children
        flatvars, flatcons = zip(*[get_or_make_var(arg) for arg in expr.args])

        newexpr = Operator(expr.name, flatvars)
        return (newexpr, [c for con in flatcons for c in con])

    else:
        """
        - Global constraint (non-Boolean) (examples: Max,Min,Element)
        """
        # XXX literal copy from flatten_cons... (except return)
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

def negated_normal(expr):
    """
        WORK IN PROGRESS
        Negate 'expr' by pushing the negation down into it and its args

        Comparison: swap comparison sign
        Operator.is_bool(): apply DeMorgan
        Global: should call decompose and negate that?

        This function only ensures 'negated normal' for the top-level
        constraint (negating arguments recursively as needed),
        it does not ensure flatness (except if the input is flat)
    """

    if __is_flat_var(expr):
        return ~expr

    elif isinstance(expr, Comparison):
        newexpr = copy.copy(expr)
        if   expr.name == '==': newexpr.name = '!='
        elif expr.name == '!=': newexpr.name = '=='
        elif expr.name == '<=': newexpr.name = '>'
        elif expr.name == '<':  newexpr.name = '>='
        elif expr.name == '>=': newexpr.name = '<'
        elif expr.name == '>':  newexpr.name = '<='
        return newexpr

    elif isinstance(expr, Operator):
        assert(expr.is_bool())

        if expr.name == 'and':
            return Operator('or', [negated_normal(arg) for arg in expr.args])
        elif expr.name == 'or':
            return Operator('and', [negated_normal(arg) for arg in expr.args])
        elif expr.name == '->':
            return expr.args[0] & negated_normal(expr.args[1])
        elif expr.name == 'xor':
            assert (len(expr.args) == 2)
            # not xor: must be equal to each other
            return (expr.args[0] == expr.args[1])
            # one could also stay in xor-space:
            # doesn't matter which one is negated
            #return expr.args[0] ^ negated_normal(expr.args[1])
        else:
            #raise NotImplementedError("negate_normal {}".format(expr))
            return expr == 0 # can't do better than this...

    else:
        # global...
        #raise NotImplementedError("negate_normal {}".format(expr))
        return expr == 0 # can't do better than this...

