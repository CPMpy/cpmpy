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

def flatten_objective(expr):
    """
        input: expression of type:
            * is_num()
            * NumVarImpl
            * Operator with is_type_num()
            * Element with len(args)==2
        output: tuple (base_expr, base_cons) with:
            base_expr one of:
                * is_num()
                * NumVarImpl
                * Operator with is_type_num(), all args are: is_num() or NumVarImpl
                * Operator 'sum', all args are: is_num() or NumVarImpl or Operator '*'[is_num(), NumVarImpl]
            base_cons: list of flattened constraints (with flatten_constraint(con))
    """
    if is_num(expr) or isinstance(expr, NumVarImpl):
        return (expr, [])

    basecons = []
    if isinstance(expr, Operator):
        # only Numeric operators allowed
        if expr.is_bool():
            raise Exception("Boolean operator '{}' not allowed in objective".format(expr.name)) # or bug

        flat_args = [flatten_numexpr(e) for e in expr.args]
        if all(arg is flatarg[0] for (arg,flatarg) in zip(expr.args, flat_args)):
            return (expr, [])
        else:
            # one of the args was changed
            newargs = [flatarg[0] for flatarg in flat_args]
            new_expr = Operator(expr.name, newargs)
            return (new_expr, [c for flatarg in flat_args for c in flatarg[1]])

    elif isinstance(expr, Element):
        if len(expr.args) != 2:
            raise Exception("Only Element expr of type Arr[Var] allowed in objective".format(expr.name))
        return flatten_numexpr(expr)

    raise Exception("Expression '{}' not allowed in objective".format(expr)) # or bug
        
def flatten_numexpr(expr):
    """
        input: expression of type:
            * is_num()
            * NumVarImpl
            * Operator with is_type_num()
            * Element with len(args)==2
        output: tuple (base_expr, base_cons) with:
            base_expr one of:
                * is_num()
                * NumVarImpl
                * Operator with is_type_num(), all args are: is_num() or NumVarImpl
                * Operator 'sum', all args are: is_num() or NumVarImpl or Operator '*'[is_num(), NumVarImpl]
            base_cons: list of flattened constraints (with flatten_constraint(con))
    """
    if is_num(expr) or isinstance(expr, NumVarImpl):
        return (expr, [])
    args = expr.args
    if isinstance(expr, Operator):
        # only Numeric operators allowed
        # unary int: '-', 'abs'
        # binary int: 'sub', 'mul', 'div', 'mod', 'pow'
        # nary int: 'sum'
        if expr.is_bool():
            raise Exception("Operator '{}' not allowed as numexpr".format(expr.name))

        # TODO: actually need to flatten THIS expression (and recursively the arguments)
        
        arity = Operator.allowed[expr.name][0]
        # unary int op
        if arity == 1:
            var, bcon = flatten_numexpr(args[0])
            basecons = [bcon]
            if expr.name == 'abs':
                ivar = IntVarImpl(0, var.ub)
            else: # '-'
                ivar = IntVarImpl(-var.ub, -var.lb)

            basecons = [Operator(expr.name, [var]) == ivar]
            return ivar, basecons

        # binary int op 
        elif arity == 2:
            var1, bcon1 = flatten_numexpr(args[0])
            var2, bcon2 = flatten_numexpr(args[1])
            basecons = bcon1 + bcon2
            lb = ([var.lb if isinstance(var, NumVarImpl) else var for var in [var1,var2]]) 
            ub = ([var.ub if isinstance(var, NumVarImpl) else var for var in [var1,var2]])
            if expr.name == 'mul': 
                ivar = IntVarImpl(lb[0] * lb[1], ub[0] * ub[1]) 
            elif expr.name == 'div':
                ivar = IntVarImpl(lb[0] // ub[1], ub[0] // lb[1] )
            elif expr.name == 'mod': 
                ivar = IntVarImpl(0, ub[0])
            elif expr.name == 'pow':
                ivar = IntVarImpl(lb[0] ** lb[1], ub[0] ** lb[1])
            
            basecons += [Operator(expr.name, [var1, var2]) == ivar]
            return ivar, basecons

        else: # arity > 2 (sum)
            varrs, bcons = zip(*[flatten_numexpr(arg) for arg in args])
            basecons = [c for con in bcons for c in con] # flatten
            lb = sum([var.lb if isinstance(var, NumVarImpl) else var for var in varrs]) 
            ub = sum([var.ub if isinstance(var, NumVarImpl) else var for var in varrs])
            ivar = IntVarImpl(lb, ub) 

            return (ivar, [Operator('sum', varrs) == ivar]+basecons)

    elif isinstance(expr, Element):
        if len(expr.args) != 2:
            raise Exception("Only Element expr of type Arr[Var] allowed in objective".format(expr.name))
        arr = expr.args[0]
        idx = expr.args[1]
        if __is_flatten_var(idx) and \
           all(__is_flatten_var(ai) for ai in arr):
            # expression is flat
            lb = min([var.lb if isinstance(var, NumVarImpl) else var for var in arr]) 
            ub = max([var.ub if isinstance(var, NumVarImpl) else var for var in arr])
            ivar = IntVarImpl(lb, ub)
            return (ivar, [Element([arr, idx, ivar])]) # need to make new one as 'expr == ivar' would overwrite expr
        else:
            # one of the args was changed
            new_idx, icons = flatten_numexpr(idx)
            new_arr, acons = zip(*[flatten_numexpr(e) for e in arr])

            lb = min([var.lb if isinstance(var, NumVarImpl) else var for var in new_arr]) 
            ub = max([var.ub if isinstance(var, NumVarImpl) else var for var in new_arr])
            ivar = IntVarImpl(lb, ub)
            return (ivar, [Element([new_arr, new_idx, ivar])]+icons+[c for con in acons for c in con])

    raise Exception("Operator '{}' not allowed as numexpr".format(expr.name)) # or bug


def __is_flatten_var(arg):
    return is_num(arg) or isinstance(arg, NumVarImpl)

def __is_nested_linexpr(subexpr):
    if __is_flatten_var(subexpr):
        return False
    if isinstance(subexpr, Operator):
        # extend definition to consider Arithmetic operator expression as valid
        return subexpr.name is 'sum' and not all([__is_flatten_var(arg) for arg in subexpr.args])
    return not all([__is_flatten_var(arg) for arg in subexpr.args])

def __check_lincons(subexpr):
    LHS, RHS = subexpr.args
    # LHS or RHS is a Var (IntVar, BoolVar, Num)
    if not (__is_flatten_var(LHS) or __is_flatten_var(RHS)):
        return False 
    
    # LHS is a Var :: RHS does not have nested expr
    elif __is_flatten_var(LHS):
        if __is_flatten_var(RHS):
            return True 
        # RHS is an LinExp and that's ok
        # as long as it does not have more nested expression 
        if any([__is_nested_linexpr(arg) for arg in RHS.args]):
            return False
        return True

    # Same logic if RHS is a Var
    #TODO: Refactor to avoid code dupcliation
    elif __is_flatten_var(RHS):
        if __is_flatten_var(LHS):
            return True 
        if any([__is_nested_linexpr(arg) for arg in LHS.args]):
            return False
        return True

    return False 

def __check_or_flip_base_const(subexpr):
    assert __check_lincons(subexpr)
    if __is_flatten_var(subexpr.args[1]):
        return subexpr
    # flip the base constraint to have 
    # BoolExp == boolvar format only 
    return Comparison(subexpr.name, subexpr.args[1], subexpr.args[0])

def flatten_boolexpr(subexpr):
    """
        input: expression of type:
            * True/False
            * BoolVar
            * Operator with is_type_bool()
            * Comparison
        output: tuple (base_expr, base_cons) with:
            base_expr one of:
                * True/False
                * BoolVar
            base_cons: list of flattened constraints (with flatten_constraint(con))
    """

    if isinstance(subexpr, BoolVarImpl) or isinstance(subexpr, bool):
        # base case: single boolVar
        return (subexpr, [])

    if isinstance(subexpr, Comparison):
        args = subexpr.args
        if __is_flatten_var(args[0]) and __is_flatten_var(args[1]):
            # base constraint, no need to flatten children
            bvar = BoolVarImpl()
            return (bvar, [subexpr == bvar])
        else:
            # TODO: special case of <expr> == 0
            #bvar = args[0] if __is_flatten_var(args[0]) else args[1]
            #base_cons = [ __check_or_flip_base_const(subexpr)]

            # recursively flatten children, which may be boolexpr or numexpr
            (var0, bco0) = check_or_flatten_subexpr(args[0])
            (var1, bco1) = check_or_flatten_subexpr(args[1])
            newcomp = Comparison(subexpr.name, var0, var1)
            bvar = BoolVarImpl()
            return (bvar, [newcomp == bvar]+bco0+bco1)

    elif isinstance(subexpr, Operator):
        assert(subexpr.is_bool()) # and, or, xor, ->

        # apply De Morgan's transform for "implies"
        if subexpr.name is '->':
            return flatten_boolexpr(~args[0] | args[1])

        args = subexpr.args
        if isinstance(args[0], BoolVarImpl) and isinstance(args[1], BoolVarImpl):
            bvar = BoolVarImpl()
            return (bvar, [subexpr == bvar])
        else:
            # recursively flatten all children, which are boolexpr
            flatres = [flatten_boolexpr(arg) for arg in args]
            flat_vars = [v for (v,_) in flatres]
            flat_cons = [c for (_,cons) in flatres for c in cons] # flatten
            newop = Operator(subexpr.name, flat_vars)
            bvar = BoolVarImpl()
            return (bvar, [newop == bvar]+flat_cons)

def check_or_flatten_subexpr(subexpr):
    """
        can be both nested boolexpr or linexpr
    """
    if isinstance(subexpr, BoolVarImpl) or isinstance(subexpr, bool):
        # base case: single boolVar
        return (subexpr, [])

    if __is_numexpr(subexpr):
        return flatten_numexpr(subexpr)
    else:
        return flatten_boolexpr(subexpr)

def __is_numexpr(expr):
    if isinstance(expr, Operator) and not expr.is_bool():
        return True
    if isinstance(expr, Element) and len(expr.args) == 2:
        return True
    return False

    

    
