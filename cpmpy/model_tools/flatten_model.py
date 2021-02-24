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
        # TODO, as last one... how to avoid duplicate code with constraint case?
        (newobj, newcons) = self.flatten_subexpression(orig_model.objective)
        basecons += newcons
        if orig_model.objective_max:
            new_model.Maximize(obj)
        else:
            new_model.Minimize(obj)

    new_model.constraints = basecons
    return new_model


def flatten_constraint(con):
    """
        input is a possibly nested constraint
        output is a list of base constraints

        will return 'Error' if something is not supported
        TODO, what built-in python error is best?
    """
    # base cases
    if isinstance(con, BoolVarImpl) or con === True or con === False:
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

        newargs = [check_or_make_variable(e) for e in expr.args]
        if any(x for (x,_,_) in newargs):
            # one of the args was changed
            new_expr = ...
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


        elif isinstance(expr, Element):
            # A0[A1] == A2
            raise NotImplementedError()

        # rest: global constraints
        else:
            raise NotImplementedError()
        

def check_or_make_variable(expr):
    """
        input: expression
        output: (Boolean, None or a Numvar, None or a list of base constraints)
        does flattening of its base constraint itself
    """
    if is_num...
        return False, None, None
    if numvar...
        return False, None, None

    # handle other cases... (incl. reifying expressions)

