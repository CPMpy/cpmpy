from ..model import *
from ..expressions import *
from ..variables import *

"""
Flattening a model (or individual constraints) into a normal form. See docs/behind_the_scenes.srt

- flatten_model(model): flattens constraints and objective, returns new model
- flatten_constraint(constraint): flattens the constraint, returns a list of base constraints

THIS IS ONLY A POTENTIAL STRUCTURE, not tested or run...
"""

def flatten_model(orig_model):
    """
        Receives model, returns new model where every constraint is a 'base' constraint
    """
    # TODO: New cppy model
    new_model = None
    basecons = []

    # the top-level constraints
    for con in orig_model.constraints:
        basecons += flatten_constraint(con)

    # the objective
    if cppy_model.objective is None:
        pass # no objective, satisfaction problem
    else:
        (newobj, newcons) = flatten_numexpr(orig_model.objective)
        basecons += newcons
        if orig_model.objective_max:
            new_model.Maximize(obj)
        else:
            new_model.Minimize(obj)

    new_model.constraints = basecons
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
    if isinstance(con, BoolVarImpl) or con == True or con == False:
        # TODO: also NegBoolView, to create...
        return [con]
    elif is_num(expr) or isinstance(expr, NumVarImpl):
        raise Exception("Numeric constants or numeric variables not allowed as base constraint")

    basecons = []

    # recursively flatten list of constraints
    if is_any_list(con):
        for con_x in con:
            basecons += flatten_constraint(con_x)
        return basecons

    if isinstance(expr, Operator):
        # only Boolean operators allowed as top-level constraint
        # bool: 'and'/n, 'or'/n, 'xor'/n, '->'/2
        allowed = ['and', 'or', 'xor', '->']
        if expr.name not in allowed:
            raise Exception("Operator '{}' not allowed as base constraint".format(expr.name))

        newargs = [flatten_boolexpr(e) for e in expr.args]
        if any(x for (x,_,_) in newargs):
            # one of the args was changed
            raise NotImplementedError()
            new_expr = None # ...
            for i,arg in enumerate(expr.args):
                (changed,newvar,newcons) = newargs[i]
                if not changed:
                    new_expr.args.append(arg)
                else:
                    new_expr.args.append(newvar)
                    basecons += newcons

    elif isinstance(expr, Comparison):
        #allowed = {'==', '!=', '<=', '<', '>=', '>'}
        for lvar, rvar in zipcycle(args[0], args[1]):
            if expr.name == '==' or expr.name == '!=':
                # special case... allows some nesting of LHS
                # and a variable on RHS
                # check whether needs swap on LHS...
                raise NotImplementedError()
            else: # inequalities '<=', '<', '>=', '>'
                # special case... allows some nesting of LHS
                raise NotImplementedError()


        if isinstance(expr, Element):
            # A0[A1] == A2
            raise NotImplementedError()

        # rest: global constraints
        else:
            raise NotImplementedError()
        
# Should probably be typed, see 'flatten_numexpr' and 'flatten_boolexpr' below
def check_or_make_variable(subexpr):
    """
        input: expression
        output: tuple (is_new, new_expr, new_cons) with:
            is_new: False or True
            new_var: None or NumVar
            new_cons: None or list of flattened constraints (with flatten_constraint(con))
    """
    raise NotImplementedError()
    if False: # is_num...
        return False, None, None
    if False: # numvar...
        return False, None, None

    # handle other cases... (incl. reifying expressions)

def flatten_numexpr(subexpr):
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
    #raise NotImplementedError()


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
    #raise NotImplementedError()
    

    if isinstance(subexpr, BoolVarImpl) or isinstance(subexpr, bool):
        # base case: single boolVar
        return (subexpr, [subexpr])

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
            return bvar, base_cons

        # recursive calls to LHS, RHS
        (var1, bco1) = flatten_boolexpr(args[0])
        (var2, bco2) = flatten_boolexpr(args[1])
        bvar = BoolVarImpl()
        base_cons += [bco1, bco2]
        base_cons += [Comparison(subexpr.name, var1, var2) == bvar]
        return bvar, base_cons

    elif isinstance(subexpr, Operator):
        # apply De Morgan's transform for "implies"
        if subexpr.name is '->':
            return flatten_boolexpr(Operator('or', -args[0], args[1]))

        if isinstance(subexpr.args[0], BoolVarImpl) and isinstance(subexpr.args[1], BoolVarImpl):
            bvar = BoolVarImpl()
            base_cons += [subexpr == bvar]
            return bvar, base_cons

        # nested AND, OR
        # recurisve function call to LHS and RHS
        # XXX: merge nested AND at top level
        (var1, bco1) = flatten_boolexpr(args[0])
        (var2, bco2) = flatten_boolexpr(args[1])
        bvar = BoolVarImpl()
        base_cons += [bco1, bco2]
        base_cons += [Operator(subexpr.name, [var1, var2]) == bvar]
        return bvar, base_cons

    

    