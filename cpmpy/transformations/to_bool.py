
from cpmpy.expressions.core import Comparison
from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl, NDVarArray, boolvar, intvar
from cpmpy.expressions.python_builtins import any
from itertools import combinations
import numpy as np

def exactly_one(lst):
    # return sum(lst) == 1
    # (l1|l2|..|ln) & (-l1|-l2) & ...
    allpairs = [(~a|~b) for a, b in combinations(lst, 2)]
    return [any(lst)] + allpairs

def reified_intvar_to_boolvar(int_vars):

        iv_mapping = {}
        bv_mapping = {}
        constraints = []

        # reification var
        bv = boolvar()

        reification_vars = [bv]

        if isinstance(int_vars, (list, NDVarArray)):
            for ivar in int_vars:
                sub_iv_mapping, sub_bv_mapping, int_cons = intvar_to_boolvar(ivar)
                iv_mapping.update(sub_iv_mapping)
                bv_mapping.update(sub_bv_mapping)
                constraints += [bv.implies(cons) for cons in int_cons]

        else:
            iv_mapping, bv_mapping, constraints = intvar_to_boolvar(int_vars)
            constraints = [bv.implies(con) for con in constraints]

        return iv_mapping, bv_mapping, constraints, reification_vars

def intvar_to_boolvar(int_var):

    constraints = []
    intvar_to_boolvar_mapping = {}
    boolvar_to_intvar_mapping = {}

    if isinstance(int_var, _BoolVarImpl):
        intvar_to_boolvar_mapping = {int_var:int_var}
        boolvar_to_intvar_mapping = {int_var:int_var}

    elif isinstance(int_var, (list, NDVarArray)):

        for ivar in int_var:
            sub_iv_mapping, sub_bv_mapping, int_cons = intvar_to_boolvar(ivar)
            constraints += int_cons
            intvar_to_boolvar_mapping.update(sub_iv_mapping)
            boolvar_to_intvar_mapping.update(sub_bv_mapping)

    elif not isinstance(int_var, _IntVarImpl):
        raise Exception("Only intvars!")

    elif int_var.lb == 0 and int_var.ub == 1:
        bv = boolvar(name="bv-"+int_var.name)
        intvar_to_boolvar_mapping = {int_var: bv}
        boolvar_to_intvar_mapping = {bv: int_var}

    else:
        bv_array = boolvar(name=int_var.name, shape=(int_var.ub-int_var.lb+1))

        intvar_to_boolvar_mapping = {
            int_var: {i: bv_array[id] for id, i in enumerate(range(int_var.lb, int_var.ub+1))}
        }

        boolvar_to_intvar_mapping = {
            bv_array[id]: (int_var, i) for id, i in enumerate(range(int_var.lb, int_var.ub+1))
        }

        constraints = exactly_one(bv_array)

    return intvar_to_boolvar_mapping, boolvar_to_intvar_mapping, constraints

def translate_unit_comparison(constraint, mapping):
    bool_constraints=[]

    # assignment constraint
    left, right = constraint.args

    operator = constraint.name

    if operator in [">", ">="]:
        # exchange 2 constraints arguments since x > 5 is the same as 5 < x
        left, right = right, left

    if operator == '==':
        ## 1 variables equal to a value
        if all(True if isinstance(arg, (int, np.int64)) else False for arg in constraint.args):
            if constraint.args[0] == constraint.args[1]:
                return []
            # 2 different numbers are equal ??
            else:
                return False

        elif any(True if isinstance(arg, (int, np.int64)) else False for arg in constraint.args):
            value, var = (left, right) if isinstance(left, (int, np.int64)) else (right, left)
            assert var.lb <= value and value <= var.ub, "Value must be between bounds!"

            for bv_value, bv_var in mapping[var].items():
                if bv_value == value:
                    bool_constraints.append(bv_var)
                else:
                    bool_constraints.append(~bv_var)

            return bool_constraints

        if left.ub < right.lb or right.ub < left.lb:
            raise Exception("No intersection between bounds", [left.lb, left.ub], [right.lb, right.ub])

        ## 2 variables equal to each other 
        # example[ 1, 7] [2, 5]
        smallest_lb_var, largest_lb_var = (left, right) if left.lb < right.lb else (right, left)
        smallest_ub_var, largest_ub_var = (left, right) if left.ub < right.ub else (right, left)

        # Before 2
        for i in range(smallest_lb_var.lb, largest_lb_var.lb):
            bool_constraints.append(~mapping[smallest_lb_var][i])

        # After 5
        for i in range(smallest_ub_var.ub+1, largest_ub_var.ub+1):
            bool_constraints.append(~mapping[largest_ub_var][i])

        # TODO: check this is correct!
        # between intersection: between 2 and 5
        for i in range(largest_lb_var.lb, smallest_ub_var.ub + 1):
            for j in range(largest_lb_var.lb, smallest_ub_var.ub + 1):
                if i != j:
                    bool_constraints += [(~mapping[left][i] | ~mapping[right][j])]

        return bool_constraints

    # different constraint
    elif operator == "!=":
        if any(True if isinstance(arg, int) else False for arg in constraint.args):
            value, var = (left, right) if isinstance(left, (int, np.int64)) else (right, left)

            for bv_value, bv_var in mapping[var].items():
                # Can only do this assumption!
                if bv_value == value:
                    bool_constraints.append(~bv_var)

            return bool_constraints

        ## 2 variables equal to each other
        smallest_lb_var, largest_lb_var = (left, right) if left.lb < right.lb else (right, left)
        smallest_ub_var, largest_ub_var = (left, right) if left.ub < right.ub else (right, left)

        # only take care of common interval
        # TODO: check this is correct!
        for i in range(largest_lb_var.lb, smallest_ub_var.ub + 1):
            bool_constraints += [(~mapping[left][i] | ~mapping[right][i])]

        return bool_constraints

    elif operator in ["<", ">"]:
        ## case1: var < value => values larger cannot be true
        if isinstance(left, _IntVarImpl) and isinstance(right, (int, np.int64)):
            ## add constraint on values that cannot be true 
            for i in range(right, left.ub+1):
                bool_constraints += [~mapping[left][i]]
        ## case2: value < var
        elif isinstance(left, (int, np.int64)) and isinstance(right, _IntVarImpl):
            ## add constraint on values that cannot be true 
            for i in range(right.lb, left):
                bool_constraints += [~mapping[right][i]]
        ## case3: var < var
        elif isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
            ## can be faster 
            # left  = [0 , 1, 2, 3]
            # right =     [1, 2]
            # left < right
            # TODO: check this is correct!
            for i in range(left.lb, left.ub+1):
                if i >= right.ub:
                    bool_constraints += [(~mapping[left][i])]
                    bool_constraints += [~mapping[right][j] for j in range(right.lb, right.ub+1)]
                else:
                    for j in range(right.lb, right.ub+1):
                        if j <= i:
                            # combination cannot be true
                            bool_constraints += [(~mapping[left][i] & ~mapping[right][j])]
        else:
            raise Exception("Val <= val?? or constraint < constraint ??")

        return bool_constraints

    elif operator in ["<=", ">="]:
        ## case1: var <= value
        if isinstance(left, _IntVarImpl) and isinstance(right, (int, np.int64)):
            # left <= 5
            for i in range(right+1, left.ub+1):
                bool_constraints += [~mapping[left][i]]
        ## case2: value <= var
        elif isinstance(left, (int, np.int64)) and isinstance(right, _IntVarImpl):
            # 5 <= right
            for i in range(left +1, right.ub+1):
                bool_constraints += [~mapping[right][i]]
        ## case3: left <= right
        ### left [0, 1, 2, 3]
        ### right   [1, 2]
        # TODO: check this is correct!
        elif isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
            for i in range(left.lb, left.ub+1):
                # when left.ub >= right.ub
                if i > right.ub:
                    bool_constraints += [(~mapping[left][i])]
                    bool_constraints += [~mapping[right][j] for j in range(right.lb, right.ub+1)]
                else:
                    for j in range(right.lb, right.ub+1):
                        if j < i:
                            bool_constraints += [(~mapping[left][i] & ~mapping[right][j])]
        else:
            raise Exception("Val <= val?? or constraint < constraint ??")


        return bool_constraints

    else:
        raise Exception("COnstraint not handled yet!")

def reify_translate_constraint(constraint, mapping):
    bool_constraints = []
    reification_vars = []
    if isinstance(constraint, (list, NDVarArray)):
        for con in constraint:
            sub_bool_constraints, sub_reification_vars = reify_translate_constraint(con, mapping)

            bool_constraints += sub_bool_constraints
            reification_vars += sub_reification_vars

    # SPECIAL CASE: ASSIGNMENT: boolvars are considered reification vars
    elif isinstance(constraint, Comparison) and constraint.name == "==" and any(isinstance(arg, (int, np.int64)) for arg in constraint.args):
        sub_bool_constraints = translate_unit_comparison(constraint, mapping)
        assert all(isinstance(sub_con, (_BoolVarImpl)) for sub_con in sub_bool_constraints), "Transalted assignment should be boolvars"

        reification_vars += sub_bool_constraints

    # assignment: boolvars are considered reification vars
    elif isinstance(constraint, Comparison):
        sub_bool_constraints = translate_unit_comparison(constraint, mapping)
        bv = boolvar()
        reification_vars.append(bv)

        for sub_con in sub_bool_constraints:
            # boolean variable are added to reification_vars
            bool_constraints.append(bv.implies(sub_con))

    # global constraints
    elif constraint.name == "alldifferent":
        bv = boolvar()
        reification_vars.append(bv)
        for con in constraint.decompose():
            sub_bool_constraints = translate_unit_comparison(con, mapping)
            bool_constraints.append(bv.implies(sub_bool_constraints))
    else:
        raise Exception("COnstraint not handled yet!", type(constraint), constraint)

    return bool_constraints, reification_vars

def translate_constraint(constraint, mapping):
    ## composition of constraints
    bool_constraints = []
    print(type(constraint),":\t", constraint)
    if isinstance(constraint, (list, NDVarArray)):

        for con in constraint:
            bool_constraints += translate_constraint(con, mapping)

    # base constraints
    elif isinstance(constraint, Comparison):

        bool_constraints += translate_unit_comparison(constraint, mapping)

    # global constraints
    elif constraint.name == "alldifferent":

        for con in constraint.decompose():
            bool_constraints += translate_unit_comparison(con, mapping)
    else:
        raise Exception("COnstraint not handled yet!", type(constraint), constraint)

    return bool_constraints
