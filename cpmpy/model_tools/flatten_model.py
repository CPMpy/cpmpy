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
                * Element with len(args)==2 and args = ([NumVar], NumVar)
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
            return (new_expr, [flatarg[1] for flatarg in flat_args])

    elif isinstance(expr, Element):
        if len(expr.args) != 2:
            raise Exception("Only Element expr of type Arr[Var] allowed in objective".format(expr.name))
        flat_idx = flatten_numexpr(expr.args[1])
        flat_arr = [flatten_numexpr(e) for e in expr.args[0]]
        if flat_idx[0] is expr.args[1] and \
           all(arri is flati[0] for (arri, flati) in zip(expr.args[0],flat_arr)):
            return (expr, [])
        else:
            # one of the args was changed
            new_arr = [flati[0] for flati in flat_arr]
            new_expr = Element([new_arr, flat_idx[0]])
            new_cons = flat_idx[1] + [flati[1] for flati in flat_arr]
            return(new_expr, new_cons)

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
                * Element with len(args)==2 and args = ([NumVar], NumVar)
            base_cons: list of flattened constraints (with flatten_constraint(con))
    """
    if is_num(expr) or isinstance(expr, NumVarImpl):
        return (expr, [])

    basecons = []
    if isinstance(expr, Operator):
        # only Numeric operators allowed
        # unary int: '-', 'abs'
        # binary int: 'sub', 'mul', 'div', 'mod', 'pow'
        # nary int: 'sum'
        if expr.is_bool():
            raise Exception("Operator '{}' not allowed as numexpr".format(expr.name))

        # TODO: actually need to flatten THIS expression (and recursively the arguments)
        return (expr, []) # TODO
        newargs = [flatten_numexpr(e) for e in expr.args]


    elif isinstance(expr, Comparison):
        #allowed = {'==', '!=', '<=', '<', '>=', '>'}
        return (expr, []) # TODO

    elif isinstance(expr, Element):
        # A0[A1] == A2
        return (expr, []) # TODO

    # rest: global constraints
    else:
        return (expr, []) # TODO


def is_flatten_var(arg):
    #TODO: extend definition to consider Arithmetic operator expression as valid,
    # e.g. (2a+c-b)/d 
    return is_num(arg) or isinstance(arg, NumVarImpl)

def is_nested_expr(subexpr):
    if is_flatten_var(subexpr):
        return False
    return not all([is_flatten_var(arg) for arg in subexpr.args])

def check_lincons(subexpr, opname):
    LHS, RHS = subexpr.args
    # LHS or RHS is a Var (IntVar, BoolVar, Num)
    if not (is_flatten_var(LHS) or is_flatten_var(RHS)):
        return False 
    
    # LHS is a Var :: RHS does not have nested expr
    elif is_flatten_var(LHS):
        if is_flatten_var(RHS):
            return True 
        # RHS is an LinExp and that's ok
        # as long as it does not have more nested expression 
        if any([is_nested_expr(arg) for arg in RHS.args]):
            return False
        return True

    # Same logic if RHS is a Var
    #TODO: Refactor to avoid code dupcliation
    elif is_flatten_var(RHS):
        if is_flatten_var(LHS):
            return True 
        if any([is_nested_expr(arg) for arg in LHS.args]):
            return False
        return True

    return False 


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
                * Operator with is_type_bool() EXCEPT '->', all args are: BoolVar
            base_cons: list of flattened constraints (with flatten_constraint(con))
    """

    if isinstance(subexpr, BoolVarImpl) or isinstance(subexpr, bool):
        # base case: single boolVar
        return (subexpr, [])

    args = subexpr.args
    base_cons = []

    if isinstance(subexpr, Comparison):
        allowed = {'>','<','<=','>=','==','!='}
        if any([check_lincons(subexpr, opname) for opname in allowed]):
            # Base case: already in base constraint form
            if is_num(args[0]) or is_num(args[1]):
                bvar = BoolVarImpl()
                base_cons += [subexpr == bvar]
            else:
                bvar = args[0] if is_flatten_var(args[0]) else args[1]
                base_cons += [subexpr]
            return (bvar, base_cons)

        # recursive calls to LHS, RHS
        (var1, bco1) = flatten_boolexpr(args[0])
        (var2, bco2) = flatten_boolexpr(args[1])
        bvar = BoolVarImpl()
        base_cons += [bco1, bco2]
        base_cons += [Comparison(subexpr.name, var1, var2) == bvar]
        return (bvar, base_cons)

    elif isinstance(subexpr, Operator):
        # apply De Morgan's transform for "implies"
        if subexpr.name is '->':
            return flatten_boolexpr(Operator('or', -args[0], args[1]))

        if isinstance(subexpr.args[0], BoolVarImpl) and isinstance(subexpr.args[1], BoolVarImpl):
            bvar = BoolVarImpl()
            base_cons += [subexpr == bvar]
            return (bvar, base_cons)

        # nested AND, OR
        # recurisve function call to LHS and RHS
        # XXX: merge nested AND at top level
        (var1, bco1) = flatten_boolexpr(args[0])
        (var2, bco2) = flatten_boolexpr(args[1])
        bvar = BoolVarImpl()
        base_cons += [bco1, bco2]
        base_cons += [Operator(subexpr.name, [var1, var2]) == bvar]
        return (bvar, base_cons)

    

    
