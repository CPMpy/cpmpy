import copy
from ..expressions import *
from ..variables import *

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
-----------------
    - Boolean operators: and([Var]), or([Var]), xor([Var]) (CPMpy class 'Operator', is_bool())
    - Boolean impliciation: Var -> Var                     (CPMpy class 'Operator', is_bool())
    - Boolean equality: Var == Var                         (CPMpy class 'Comparison')
                        Var == Constant                    (CPMpy class 'Comparison')
    - Global constraint (Boolean): global([Var]*)          (CPMpy class 'GlobalConstraint', is_bool())

Comparison constraints: (up to one nesting on one side)
-----------------------
    - Numeric equality:  Var == Numexpr                    (CPMpy class 'Comparison')
                         Numexpr == Constant               (CPMpy class 'Comparison')
    - Numeric disequality: Var != Numexpr                  (CPMpy class 'Comparison')
                           Numexpr != Constant             (CPMpy class 'Comparison')
    - Numeric inequality (>=,>,<,<=,): Numexpr >=< Var     (CPMpy class 'Comparison')

    Numexpr:
        - Operator (non-Boolean) with all args Var/constant (examples: +,*,/,mod,wsum)
                                                           (CPMpy class 'Operator', not is_bool())
        - Global constraint (non-Boolean) (examples: Max,Min,Element)
                                                           (CPMpy class 'GlobalConstraint', not is_bool()))
    wsum: wsum([Const],[Var]) represents sum([Const]*[Var]) # TODO: not implemented yet

Reify/imply constraint: (up to two nestings on one side)
-----------------------
    - Reification (double implication): Var == Boolexpr    (CPMpy class 'Comparison')
    - Implication: Var -> Boolexpr                         (CPMpy class 'Operator', is_bool())
                   Boolexpr -> Var                         (CPMpy class 'Operator', is_bool())

    Boolexpr:
        - Boolean operators: and([Var]), or([Var]), xor([Var]) (CPMpy class 'Operator', is_bool())
        - Boolean equality: Var == Var                         (CPMpy class 'Comparison')
        - Global constraint (Boolean): global([Var]*)          (CPMpy class 'GlobalConstraint', is_bool())
        - Comparison constraint (see above)                    (CPMpy class 'Comparison')
    
    Reification of a comparison is the most complex case as it can allow up to 3 levels of nesting in total, e.g.:
        - BV == (wsum([1,2,3],[IV1,IV2,IV3]) > 5)
        - BV == (IV1 == IV2)
        - BV1 == (BV2 == BV3)

The output after calling flatten_model() or flatten_constraint() will ONLY contain expressions
of the form specified above.

The flattening does not promise to do common subexpression elimination or to automatically group
commutative expressions (and, or, sum, wsum, ...) but such optimisations should be added later.

TODO: not entirely implemented yet : )
TODO: clean up behind_the_scenes.rst which sketches the previous normal form
"""

from itertools import cycle
def __zipcycle(vars1, vars2):
    v1 = [vars1] if not is_any_list(vars1) else vars1
    v2 = [vars2] if not is_any_list(vars2) else vars2
    return zip(v1, cycle(v2)) if len(v2) < len(v1) else zip(cycle(v1), v2)

def flatten_model(orig_model):
    """
        Receives model, returns new model where every constraint is a 'base' constraint
    """
    from ..model import Model # otherwise circular dependency...

    # the top-level constraints
    basecons = []
    for con in orig_model.constraints:
        basecons += flatten_constraint(con)

    # the objective
    if orig_model.objective is None:
        return Model(basecons) # no objective, satisfaction problem
    else:
        (newobj, newcons) = flatten_objective(orig_model.objective)
        basecons += newcons
        if orig_model.objective_max:
            return Model(basecons, maximize=newobj)
        else:
            return Model(basecons, minimize=newobj)

    return new_model


def flatten_constraint(expr):
    """
        input is any expression; except is_num(), pure NumVarImpl, Operator with is_type_num() and Element with len(args)=2
        output is a list of base constraints, each base constraint is one of:
            * BoolVar
            * Operator with is_bool(), all args are: BoolVar
            * Operator '->' with args [boolexpr,BoolVar]
            * Comparison, all args are is_num() or NumVar 
            * Comparison '==' with args [boolexpr,BoolVar]
            * Comparison '!=' with args [boolexpr,BoolVar]
            * Comparison '==' with args [comparisonexpr,BoolVar]
            * Comparison '!=' with args [comparisonexpr,BoolVar]
            * Comparison '==' with args [numexpr,NumVar]
            * Comparison '!=' with args [numexpr,NumVar]
            * Element with len(args)==3 and args = ([NumVar], NumVar, NumVar)
            * Global, all args are: NumVar

        will return 'Error' if something is not supported
        TODO, what built-in python error is best?
    """
    # base cases
    if isinstance(expr, BoolVarImpl) or isinstance(expr, bool):
        return [expr]
    elif is_num(expr) or isinstance(expr, NumVarImpl):
        raise Exception("Numeric constants or numeric variables not allowed as base constraint")

    # recursively flatten list of constraints
    if is_any_list(expr):
        flatcons = [flatten_constraint(e) for e in expr]
        return [c for con in flatcons for c in con]


    if isinstance(expr, Operator):
        assert(expr.is_bool()) # and, or, xor, ->

        # does not type-check that arguments are bool...
        if all(__is_flat_var(arg) for arg in expr.args):
            return [expr]
        else:
            # recursively flatten all children, which are boolexpr
            flatvars, flatcons = zip(*[flatten_boolexpr(arg) for arg in expr.args])

            newexpr = Operator(expr.name, flatvars)
            return [newexpr]+[c for con in flatcons for c in con]


    elif isinstance(expr, Comparison):
        #allowed = {'==', '!=', '<=', '<', '>=', '>'}
        flatcons = []
        # zipcycle: unfolds 'arr1 == arr2' pairwise
        for lexpr, rexpr in __zipcycle(expr.args[0], expr.args[1]):
            # XXX somehow, this code feels like it should be in reify_ready_boolexpr?
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

        return flatcons

    else:
        # everything else (Element, globals)
        # just recursively flatten args, which can be lists
        if all(__is_flat_var_or_list(arg) for arg in expr.args):
            return [expr]
        else:
            # recursively flatten all children
            flatvars, flatcons = zip(*[flatten_subexpr(arg) for arg in expr.args])

            # take copy, replace args
            newexpr = copy.copy(expr) # shallow or deep? currently shallow
            newexpr.args = flatvars
            return [newexpr]+[c for con in flatcons for c in con]


def __is_flat_var(arg):
    """ True if the variable is a numeric constant, or a NumVarImpl (incl subclasses)
    """
    return is_num(arg) or isinstance(arg, NumVarImpl)

def __is_flat_var_or_list(arg):
    """ True if the variable is a numeric constant, or a NumVarImpl (incl subclasses)
        or a list of __is_flat_var
    """
    return is_num(arg) or isinstance(arg, NumVarImpl) or \
           is_any_list(arg) and all(__is_flat_var(el) for el in arg)


def flatten_objective(expr):
    """
        input: expression of type:
            * is_num()
            * NumVarImpl
            * Operator with not is_bool()
            * Element with len(args)==2
        output: tuple (base_expr, base_cons) with:
            base_expr one of:
                * is_num()
                * NumVarImpl
                * Operator with not is_bool(), all args are: is_num() or NumVarImpl
                * Operator 'sum', all args are: is_num() or NumVarImpl or Operator '*'[is_num(), NumVarImpl] # TODO
            base_cons: list of additional flattened constraints
    """
    if __is_flat_var(expr):
        return (expr, [])

    if isinstance(expr, Operator):
        assert(not expr.is_bool()) # only non-logic operators allowed

        if all(__is_flat_var(arg) for arg in expr.args):
            return (expr, [])
        else:
            # one of the arguments is not flat, flatten all
            flatvars, flatcons = zip(*[flatten_numexpr(arg) for arg in expr.args])
            newexpr = Operator(expr.name, flatvars)
            return (newexpr, [c for con in flatcons for c in con])

    elif isinstance(expr, Element):
        assert(len(expr.args) == 2) # Arr[Idx]
        return flatten_numexpr(expr)

    raise Exception("Expression '{}' not allowed in objective".format(expr)) # or bug

        
def flatten_numexpr(expr):
    """
        input: expression of type:
            * is_num()
            * NumVarImpl
            * Operator with not is_bool()
            * Element with len(args)==2
        output: tuple (base_expr, base_cons) with:
            base_expr one of:
                * is_num()
                * NumVarImpl
            base_cons: list of flattened constraints (with flatten_constraint(con))
    """
    # XXX a boolexpr is also a valid numexpr? e.g. 30*(iv > 5) + ... see mario obj.
    if __is_flat_var(expr):
        return (expr, [])

    # special case, -var... 
    # XXX until we do weighted sum, turn into -1*args[0]
    if isinstance(expr, Operator) and expr.name == '-': # unary
        return flatten_numexpr(-1*expr.args[0])

    if isinstance(expr, Operator):
        assert(not expr.is_bool()) # only non-logic operators allowed

        flatvars, flatcons = zip(*[flatten_subexpr(arg) for arg in expr.args]) # also bool, reified...
        lbs = [var.lb if isinstance(var, NumVarImpl) else var for var in flatvars]
        ubs = [var.ub if isinstance(var, NumVarImpl) else var for var in flatvars]

        # TODO: weighted sum
        if expr.name == 'abs': # unary
            ivar = IntVarImpl(0, ubs[0])
        elif expr.name == 'mul': # binary
            lb = lbs[0] * lbs[1]
            ub = ubs[0] * ubs[1]
            if lb > ub: # a negative nr
                lb,ub = ub,lb
            ivar = IntVarImpl(lb, ub) 
        elif expr.name == 'div': # binary
            lb = lbs[0] // lbs[1]
            ub = ubs[0] // ubs[1]
            if lb > ub: # a negative nr
                lb,ub = ub,lb
            ivar = IntVarImpl(lb, ub) 
        elif expr.name == 'mod': # binary 
            ivar = IntVarImpl(0, ubs[0])
        elif expr.name == 'pow': # binary
            ivar = IntVarImpl(lbs[0] ** lbs[1], ubs[0] ** ubs[1])
        elif expr.name == 'sum': # n-ary
            ivar = IntVarImpl(sum(lbs), sum(ubs)) 
        else:
            raise Exception("Operator '{}' not known in numexpr".format(expr.name)) # or bug

        newexpr = (Operator(expr.name, flatvars) == ivar)
        return (ivar, [newexpr]+[c for con in flatcons for c in con])

    elif isinstance(expr, Element):
        assert(len(expr.args) == 2) # Arr[Idx]

        arr,idx = expr.args
        basecons = []
        if not __is_flat_var(idx) or \
           any(not __is_flat_var(ai) for ai in arr):
            # one of the arguments is not flat, flatten all
            idx, icons = flatten_numexpr(idx)
            arr, acons = zip(*[flatten_numexpr(e) for e in arr])
            basecons = icons+[c for con in acons for c in con]
            
        lb = min([var.lb if isinstance(var, NumVarImpl) else var for var in arr]) 
        ub = max([var.ub if isinstance(var, NumVarImpl) else var for var in arr])
        ivar = IntVarImpl(lb, ub)

        newexpr = Element([arr, idx, ivar])
        return (ivar, [newexpr]+basecons)

    raise Exception("Operator '{}' not allowed as numexpr".format(expr)) # or bug


def flatten_boolexpr(expr):
    """
        input: expression of type:
            * True/False
            * BoolVar
            * Operator with is_bool()
            * Comparison
        output: tuple (base_expr, base_cons) with:
            base_expr one of:
                * True/False
                * BoolVar
            base_cons: list of flattened constraints (with flatten_constraint(con))

        these are all the expression that can be 'reified'
    """
    if __is_flat_var(expr):
        return (expr, [])

    # flatten expr into a LHS, return LHS == bvar
    bvar = BoolVarImpl()
    (newexpr, flatcons) = reify_ready_boolexpr(expr)
    return (bvar, [newexpr == bvar]+flatcons)

def reify_ready_boolexpr(expr):
    """
        alle expressions that can be 'reified', meaning that

            - expr == BoolVar
            - expr != BoolVar
            - expr -> BoolVar

        are valid expressions.

        Currently, this is the case for:
            * Operator with is_bool()
            * Comparison

        output: (base_expr, base_cons) with:
            base_expr: same as 'expr', but all arguments are variables
            base_cons: list of flattened constraints (with flatten_constraint(con))
    """
    assert(not __is_flat_var(expr))

    if isinstance(expr, Operator):
        assert(expr.is_bool()) # and, or, xor, ->

        # apply De Morgan's transform for "implies"
        if expr.name is '->':
            return reify_ready_boolexpr(~args[0] | args[1])

        if all(__is_flat_var(arg) for arg in expr.args):
            return (expr, [])
        else:
            # recursively flatten all children, which are boolexpr
            flatvars, flatcons = zip(*[flatten_boolexpr(arg) for arg in expr.args])
            return (Operator(expr.name, flatvars), [c for con in flatcons for c in con])

    if isinstance(expr, Comparison):
        if all(__is_flat_var(arg) for arg in expr.args):
            return (expr, [])
        else:
            # TODO: special case of <reify_ready_expr> == 0, e.g. (a > 10) == 0 :: (a <= 10)
            #bvar = args[0] if __is_flat_var(args[0]) else args[1]
            #base_cons = [ __check_or_flip_base_const(subexpr)]

            # recursively flatten children, which may be boolexpr or numexpr
            # TODO: actually one side can be a linexpr, then sufficient to flatten_objective...
            # e.g. flatten_boolexpr(a + b > 2) does not need splitting a+b away
            (var0, bco0) = flatten_subexpr(expr.args[0])
            (var1, bco1) = flatten_subexpr(expr.args[1])
            return (Comparison(expr.name, var0, var1), bco0+bco1)


def flatten_subexpr(expr):
    """
        can be both nested boolexpr or linexpr
    """
    if __is_numexpr(expr):
        return flatten_numexpr(expr)
    else:
        return flatten_boolexpr(expr)

def __is_numexpr(expr):
    """
        True if:
            * is_num()
            * NumVarImpl
            * Operator with not is_bool()
            * Element with len(args)==2
    """
    if __is_flat_var(expr):
        return True
    if isinstance(expr, Operator) and not expr.is_bool():
        return True
    if isinstance(expr, Element) and len(expr.args) == 2:
        return True
    return False

    

######## Unused code, detects linexp and swap Comparison so Operator is on left side...

def __is_nested_linexpr(subexpr):
    if __is_flat_var(subexpr):
        return False
    if isinstance(subexpr, Operator):
        # extend definition to consider Arithmetic operator expression as valid
        return subexpr.name is 'sum' and not all([__is_flat_var(arg) for arg in subexpr.args])
    return not all([__is_flat_var(arg) for arg in subexpr.args])

def __check_lincons(subexpr):
    LHS, RHS = subexpr.args
    # LHS or RHS is a Var (IntVar, BoolVar, Num)
    if not (__is_flat_var(LHS) or __is_flat_var(RHS)):
        return False 
    
    # LHS is a Var :: RHS does not have nested expr
    elif __is_flat_var(LHS):
        if __is_flat_var(RHS):
            return True 
        # RHS is an LinExp and that's ok
        # as long as it does not have more nested expression 
        if any([__is_nested_linexpr(arg) for arg in RHS.args]):
            return False
        return True

    # Same logic if RHS is a Var
    #TODO: Refactor to avoid code dupcliation
    elif __is_flat_var(RHS):
        if __is_flat_var(LHS):
            return True 
        if any([__is_nested_linexpr(arg) for arg in LHS.args]):
            return False
        return True

    return False 

def __check_or_flip_base_const(subexpr):
    assert __check_lincons(subexpr)
    if __is_flat_var(subexpr.args[1]):
        return subexpr
    # flip the base constraint to have 
    # BoolExp == boolvar format only 
    return Comparison(subexpr.name, subexpr.args[1], subexpr.args[0])
    
