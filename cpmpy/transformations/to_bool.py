
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

def intvar_to_boolvar(int_var):

    if isinstance(int_var, _BoolVarImpl):
        return {int_var:int_var}, {int_var:int_var}, []

    if isinstance(int_var, (list, NDVarArray)):
        constraints = []
        boolvar_to_intvar_mapping = {}
        intvar_to_boolvar_mapping = {}

        for ivar in int_var:
            int_mapping, bool_mapping, sub_constraints = intvar_to_boolvar(ivar)
            intvar_to_boolvar_mapping.update(int_mapping)
            boolvar_to_intvar_mapping.update(bool_mapping)
            constraints += sub_constraints

        return intvar_to_boolvar_mapping, boolvar_to_intvar_mapping, constraints

    if not isinstance(int_var, _IntVarImpl):
        raise Exception("Only intvars!")

    if int_var.lb == 0 and int_var.ub == 1:
        bv = boolvar(name="bv-"+int_var.name)
        return {int_var: bv}, {bv: int_var}, []

    bv_array = boolvar(name="bv-"+int_var.name, shape=(int_var.ub-int_var.lb+1))

    intvar_to_boolvar_mapping = {int_var: {i: bv_array[id] for id, i in enumerate(range(int_var.lb, int_var.ub+1))}}
    boolvar_to_intvar_mapping = {bv_array[id]: (int_var, i) for id, i in enumerate(range(int_var.lb, int_var.ub+1))}

    constraints = exactly_one(bv_array)

    return intvar_to_boolvar_mapping, boolvar_to_intvar_mapping, constraints

def translate_unit_comparison(constraint, mapping):
    bool_constraints=[]
    # assignment constraint
    left, right = constraint.args
    if constraint.name in [">", ">="]:
        # exchange 2 constraints arguments since x > 5 is the same as 5 < x
        left, right = right, left

    if constraint.name == '==':
        ## 1 variables equal to a value
        if any(True if isinstance(arg, (int, np.int64)) else False for arg in constraint.args):
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
    elif constraint.name == "!=":
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

    elif constraint.name in ["<", ">"]:
        ## case1: var < value => values larger cannot be true
        if isinstance(left, _IntVarImpl) and isinstance(right, (int, np.int64)):
            for i in range(right, left.ub+1):
                bool_constraints += [~mapping[left][i]]
        ## case2: value < var
        elif isinstance(left, (int, np.int64)) and isinstance(right, _IntVarImpl):
            for i in range(left.lb, right):
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

    elif constraint.name in ["<=", ">="]:
        ## case1: var <= value
        if isinstance(left, _IntVarImpl) and isinstance(right, (int, np.int64)):
            # left <= 5
            for i in range(right+1, left.ub+1):
                bool_constraints += [~mapping[left][i]]
        ## case2: value <= var
        elif isinstance(left, (int, np.int64)) and isinstance(right, _IntVarImpl):
            # 5 <= right
            for i in range(left.lb, left):
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

def translate_constraint(constraint, mapping):
    ## composition of constraints
    if isinstance(constraint, NDVarArray):
        bool_constraints = []
        for con in constraint:
            bool_constraints += translate_unit_comparison(con, mapping)
        return bool_constraints
    elif isinstance(constraint, list):
        bool_constraints = []
        for con in constraint:
            bool_constraints += translate_constraint(con, mapping)
        return bool_constraints
    # base constraints
    elif isinstance(constraint, Comparison):
        return translate_unit_comparison(constraint, mapping)
    # global constraints
    elif constraint.name == "alldifferent":
        bool_constraints = []
        for con in constraint.decompose():
            bool_constraints += translate_unit_comparison(con, mapping)
        return bool_constraints

    else:
        raise Exception("COnstraint not handled yet!", type(constraint), constraint)