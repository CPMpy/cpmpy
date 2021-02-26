from ..expressions import *
from ..variables import *

"""
Flattening a model (or individual constraints) into a normal form. See docs/behind_the_scenes.srt

- flatten_model(model): flattens constraints and objective, returns new model
- flatten_constraint(constraint): flattens the constraint, returns a list of base constraints

THIS IS ONLY A POTENTIAL STRUCTURE, not tested or run...
"""

def flatten_model(orig_model):
    from ..model import Model # otherwise circular dependency...

    """
        Receives model, returns new model where every constraint is a 'base' constraint
    """
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


def flatten_constraint(con):
    """
        input is any expression; except is_num(), pure NumVarImpl, Operator with is_type_num() and Element with len(args)=2
        output is a list of base constraints, each base constraint is one of:
            * BoolVar
            * Operator with is_type_bool(), all args are: BoolVar
            * Operator '->' with args [boolexpr,BoolVar]
            * Operator '->' with args [Comparison,BoolVar]
            * Comparison, all args are is_num() or NumVar 
            * Comparison '==' with args [boolexpr,BoolVar]
            * Comparison '!=' with args [boolexpr,BoolVar]
            * Comparison '==' with args [numexpr,NumVar]
            * Comparison '!=' with args [numexpr,NumVar]
            * Element with len(args)==3 and args = ([NumVar], NumVar, NumVar)
            * Global, all args are: NumVar

        will return 'Error' if something is not supported
        TODO, what built-in python error is best?
    """
    # base cases
    if isinstance(con, BoolVarImpl) or isinstance(con, bool):
        return [con]
    elif is_num(con) or isinstance(con, NumVarImpl):
        raise Exception("Numeric constants or numeric variables not allowed as base constraint")

    basecons = []

    # recursively flatten list of constraints
    if is_any_list(con):
        for con_x in con:
            basecons += flatten_constraint(con_x)
        return basecons

    if isinstance(con, Operator):
        # only Boolean operators allowed as top-level constraint
        # bool: 'and'/n, 'or'/n, 'xor'/n, '->'/2
        if not con.is_bool():
            raise Exception("Operator '{}' not allowed as base constraint".format(expr.name))

        return [con] # TODO

    elif isinstance(con, Comparison):
        #allowed = {'==', '!=', '<=', '<', '>=', '>'}
        return [con] # TODO
        for lvar, rvar in zipcycle(args[0], args[1]):
            if expr.name == '==' or expr.name == '!=':
                # special case... allows some nesting of LHS
                # and a variable on RHS
                # check whether needs swap on LHS...
                return [con] # TODO
            else: # inequalities '<=', '<', '>=', '>'
                # special case... allows some nesting of LHS
                return [con] # TODO

    elif isinstance(con, Element):
        return [con] # TODO

    # rest: global constraints
    else:
        return [con] # TODO


def __is_flat_var(arg):
    """ True if the variable is a numeric constant, or a NumVarImpl (incl subclasses)
    """
    return is_num(arg) or isinstance(arg, NumVarImpl)


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
    if __is_flat_var(expr):
        return (expr, [])

    if isinstance(expr, Operator):
        assert(not expr.is_bool()) # only non-logic operators allowed

        flatvars, flatcons = zip(*[flatten_numexpr(arg) for arg in expr.args])
        lbs = [var.lb if isinstance(var, NumVarImpl) else var for var in flatvars]
        ubs = [var.ub if isinstance(var, NumVarImpl) else var for var in flatvars]

        if expr.name == 'abs': # unary
            ivar = IntVarImpl(0, ubs[0])
        elif expr.name == '-': # unary
            ivar = IntVarImpl(-ubs[0], -lbs[0])
        elif expr.name == 'mul': # binary
            ivar = IntVarImpl(lbs[0] * lbs[1], ubs[0] * ubs[1]) 
        elif expr.name == 'div': # binary
            ivar = IntVarImpl(lbs[0] // ubs[1], ubs[0] // lbs[1] )
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

    raise Exception("Operator '{}' not allowed as numexpr".format(expr.name)) # or bug


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
    """
    if __is_flat_var(expr):
        return (expr, [])

    if isinstance(expr, Operator):
        assert(expr.is_bool()) # and, or, xor, ->

        # apply De Morgan's transform for "implies"
        if expr.name is '->':
            return flatten_boolexpr(~args[0] | args[1])

        bvar = BoolVarImpl()
        if all(__is_flat_var(arg) for arg in expr.args):
            return (bvar, [expr == bvar])
        else:
            # recursively flatten all children, which are boolexpr
            flatvars, flatcons = zip(*[flatten_boolexpr(arg) for arg in expr.args])

            newexpr = (Operator(expr.name, flatvars) == bvar)
            return (bvar, [newexpr]+[c for con in flatcons for c in con])

    if isinstance(expr, Comparison):
        bvar = BoolVarImpl()
        if all(__is_flat_var(arg) for arg in expr.args):
            return (bvar, [expr == bvar])
        else:
            # TODO: special case of <expr> == 0
            #bvar = args[0] if __is_flat_var(args[0]) else args[1]
            #base_cons = [ __check_or_flip_base_const(subexpr)]

            # recursively flatten children, which may be boolexpr or numexpr
            # TODO: actually one side can be a linexpr, then sufficient to flatten_objective...
            # e.g. flatten_boolexpr(a + b > c) does not need splitting a+b away
            (var0, bco0) = flatten_subexpr(expr.args[0])
            (var1, bco1) = flatten_subexpr(expr.args[1])

            newexpr = (Comparison(expr.name, var0, var1) == bvar)
            return (bvar, [newexpr]+bco0+bco1)


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
    
