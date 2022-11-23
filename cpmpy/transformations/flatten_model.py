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

TODO: update behind_the_scenes.rst doc with the new 'flat normal form'
TODO: small optimisations, e.g. and/or chaining (potentially after negation), see test_flatten
"""
import copy
import math
import numpy as np
from ..expressions.core import *
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView
from ..expressions.utils import is_num, is_any_list

def flatten_model(orig_model):
    """
        Receives model, returns new model where every constraint is in 'flat normal form'
    """
    from ..model import Model  # otherwise circular dependency...

    # the top-level constraints
    basecons = []
    for con in orig_model.constraints:
        basecons += flatten_constraint(con)

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
        input is any expression; except bool, is_num(), pure _NumVarImpl,
        or Operator/GlobalConstraint with not is_bool()
        
        output: see definition of 'flat normal form' above.

        it will return 'Exception' if something is not supported
        TODO, what built-in python error is best?
    """
    # base cases
    if isinstance(expr, bool):
        if expr:
            return []
        else:
            return [expr]  # not sure about this one... means False is a valid FNF expression
    elif isinstance(expr, _BoolVarImpl):
        return [expr]
    elif is_num(expr) or isinstance(expr, _NumVarImpl):
        raise Exception("Numeric constants or numeric variables not allowed as base constraint")

    # recursively flatten list of constraints
    if is_any_list(expr):
        flatcons = []
        for e in expr:
            flatcons += flatten_constraint(e)  # add all at end
        return flatcons
    # recursively flatten top-level 'and'
    if isinstance(expr, Operator) and expr.name == 'and':
        flatcons = []
        for e in expr.args:
            flatcons += flatten_constraint(e)  # add all at end
        return flatcons

    assert expr.is_bool(), f"Boolean expressions only in flatten_constraint, `{expr}` not allowed."

    if isinstance(expr, Operator):
        """
            - Base Boolean operators: and([Var]), or([Var])        (CPMpy class 'Operator', is_bool())
            - Base Boolean impliciation: Var -> Var                (CPMpy class 'Operator', is_bool())
            - Implication: Boolexpr -> Var                         (CPMpy class 'Operator', is_bool())
                           Var -> Boolexpr                         (CPMpy class 'Operator', is_bool())
        """
        # does not type-check that arguments are bool... Could do now with expr.is_bool()!
        if all(__is_flat_var(arg) for arg in expr.args):
            return [expr]

        if expr.name == '->':
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
            return [newexpr]+flatcons
        else:
            # a normalizable boolexpr
            (con, flatcons) = normalized_boolexpr(expr)
            return [con]+flatcons


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
        if all(__is_flat_var(arg) for arg in expr.args):
            return [expr]

        # swap 'Var == Expr' to normal 'Expr == Var'
        lexpr, rexpr = expr.args
        if (expr.name == '==' or expr.name == '!=') \
                and __is_flat_var(lexpr) and not __is_flat_var(rexpr):
            lexpr, rexpr = rexpr, lexpr

        # ensure rhs is var
        (rvar, rcons) = get_or_make_var(rexpr)

        exprname = expr.name  # so it can be modified
        # 'BoolExpr != Rvar' to normal 'BoolExpr == ~Rvar'
        if exprname == '!=' and lexpr.is_bool():  # negate rvar
            exprname = '=='
            rvar = ~rvar

        # Reification (double implication): Boolexpr == Var
        if exprname == '==' and lexpr.is_bool():
            if is_num(rexpr):
                # shortcut, full original one is normalizable BoolExpr
                # such as And(v1,v2,v3) == 0
                (con, flatcons) = normalized_boolexpr(expr)
                return [con]+flatcons
            else:
                (lhs, lcons) = normalized_boolexpr(lexpr)
        else:
            # other cases: LHS is numexpr
            (lhs, lcons) = normalized_numexpr(lexpr)

        return [Comparison(exprname, lhs, rvar)]+lcons+rcons

    else:
        """
    - Global constraint (Boolean): global([Var]*)          (CPMpy class 'GlobalConstraint', is_bool())
        """
        (con, flatcons) = normalized_boolexpr(expr)
        return [con] + flatcons


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

    if isinstance(expr, Expression) and expr.name in supported:
        return normalized_numexpr(expr)
    else:
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
        (flatexpr, flatcons) = normalized_boolexpr(expr)

        bvar = _BoolVarImpl()
        return (bvar, [flatexpr == bvar]+flatcons)

    else:
        # normalize expr into a numexpr LHS,
        # then compute bounds and return (newintvar, LHS == newintvar)
        (flatexpr, flatcons) = normalized_numexpr(expr)

        if isinstance(flatexpr, Operator) and expr.name == "wsum":
            # more complex args, and weights can be negative, so more complex lbs/ubs
            weights, flatvars  = flatexpr.args
            bounds = np.array([[w * fvar.lb for w, fvar in zip(weights, flatvars)],
                               [w * fvar.ub for w, fvar in zip(weights, flatvars)]])
            lb, ub = bounds.min(axis=0).sum(), bounds.max(axis=0).sum() # for every column is axis=0...
            ivar = _IntVarImpl(lb, ub)
            return (ivar, [flatexpr == ivar]+flatcons)

        elif isinstance(flatexpr, Operator):
            lbs = [var.lb if isinstance(var, _NumVarImpl) else var for var in flatexpr.args]
            ubs = [var.ub if isinstance(var, _NumVarImpl) else var for var in flatexpr.args]

            if flatexpr.name == 'abs': # unary
                if lbs[0] < 0 and ubs[0] > 0:
                    lb = 0 # from neg to pos, so includes 0
                else:
                    lb = min(abs(lbs[0]), abs(ubs[0])) # same sign, take smallest
                ub = max(abs(lbs[0]), abs(ubs[0])) # largest abs value
                ivar = _IntVarImpl(lb, ub)
            elif flatexpr.name == "sub": # binary
                lb = lbs[0] - ubs[1]
                ub = ubs[0] - lbs[1]
                ivar = _IntVarImpl(lb,ub)
            elif flatexpr.name == 'mul': # binary
                v0 = [lbs[0], ubs[0]]
                v1 = [lbs[1], ubs[1]]
                bnds = [v0[i]*v1[j] for i in [0,1] for j in [0,1]]
                ivar = _IntVarImpl(min(bnds),max(bnds))
            elif flatexpr.name == 'div': # binary
                num = [lbs[0], ubs[0]]
                denom = [lbs[1], ubs[1]]
                bnds = [num[i]/denom[j] for i in [0,1] for j in [0,1]]
                # the above can give fractional values, tighten bounds to integer
                ivar = _IntVarImpl(math.ceil(min(bnds)), math.floor(max(bnds)))
            elif flatexpr.name == 'mod': # binary
                l = np.arange(lbs[0], ubs[0]+1)
                r = np.arange(lbs[1], ubs[1]+1)
                # check all possibilities
                remainders = np.mod(l[:,None],r)
                lb, ub = np.min(remainders), np.max(remainders)
                ivar = _IntVarImpl(lb,ub)
            elif flatexpr.name == 'pow': # binary
                base = [lbs[0], ubs[0]]
                exp = [lbs[1], ubs[1]]
                if exp[0] < 0:
                    raise NotImplementedError("Power operator: For integer values, exponent must be non-negative")
                bnds = [base[i]**exp[j] for i in [0,1] for j in [0,1]]
                if exp[1] > 0: # even/uneven behave differently when base is negative
                    bnds += [base[0]**(exp[1]-1), base[1]**(exp[1]-1)]
                ivar = _IntVarImpl(min(bnds), max(bnds))
            elif flatexpr.name == 'sum': # n-ary
                ivar = _IntVarImpl(sum(lbs), sum(ubs))
            elif flatexpr.is_bool(): # Boolean
                ivar = _BoolVarImpl() # TODO: we can support Bool? check, update docs
            else:
                raise Exception("Operator '{}' not known in get_or_make_var".format(expr.name)) # or bug

            return (ivar, [flatexpr == ivar]+flatcons)

        else:
            """
            - Global constraint (non-Boolean) (examples: Max,Min,Element)
            """
            # we don't currently have a generic way to get bounds from non-Boolean globals...
            # XXX Add to GlobalCons as function? e.g. (lb,ub) = expr.get_bounds()? would also work for Operator...
            ivar = _IntVarImpl(-2147483648, 2147483647) # TODO, this can breaks solvers

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
        all 'flat normal form' Boolean expressions that can be 'reified', meaning that

            - expr == BoolVar
            - expr != BoolVar
            - expr -> BoolVar

        are valid expressions.

        Currently, this is the case for:
        - Boolean operators: and([Var]), or([Var])             (CPMpy class 'Operator', is_bool())
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
        # and, or, ->

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

            # RHS must be var (or const)
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
                if is_num(rexpr):
                    # BoolExpr == 0|False
                    assert (not rexpr), f"should be false: {rexpr}" # 'true' is preprocessed away
                    if exprname == '==':
                        nnexpr = negated_normal(lexpr)
                        return normalized_boolexpr(nnexpr)
                    else: # !=, should only be possible in dubble negation
                        return normalized_boolexpr(lexpr)

                # this is a reified constraint, so lhs must be var too to be in normal form
                (lhs, lcons) = get_or_make_var(lexpr)
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
        # special case, -var, turn into -1*args[0]
        if expr.name == '-': # unary
            return normalized_numexpr(-1*expr.args[0])

        if all(__is_flat_var(arg) for arg in expr.args):
            return (expr, [])

        elif expr.name == 'wsum': # unary
            weights, sub_exprs  = expr.args
            flatvars, flatcons = map(list, zip(*[get_or_make_var(arg) for arg in sub_exprs])) # also bool, reified...
            newexpr = Operator(expr.name, (weights, flatvars))
            return (newexpr, [c for con in flatcons for c in con])

        else: # generic operator
            # recursively flatten all children
            flatvars, flatcons = zip(*[get_or_make_var(arg) for arg in expr.args])

            newexpr = Operator(expr.name, flatvars)
            return (newexpr, [c for con in flatcons for c in con])
    else:
        # Global constraint (non-Boolean) (examples: Max,Min,Element)

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
        Global: decompose and negate that

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
        else:
            #raise NotImplementedError("negate_normal {}".format(expr))
            return expr == 0 # can't do better than this...

    elif expr.name == 'xor':
        # avoid circular import
        from ..expressions.globalconstraints import Xor
        # stay in xor space
        # only negated last element
        return Xor(expr.args[:-1]) ^ negated_normal(expr.args[-1])

    else: # circular if I import GlobalConstraint here...
        if hasattr(expr, "decompose"):
            # global... decompose and negate that
            return negated_normal(Operator('and', expr.decompose()))
        else:
            raise NotImplementedError("negate_normal {}".format(expr))
