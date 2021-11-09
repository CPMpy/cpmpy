import itertools
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.expressions.globalconstraints import AllDifferent, AllEqual, Circuit, Table
from cpmpy.expressions.utils import is_int
from cpmpy.transformations.get_variables import get_all_elems, get_variables
from ..expressions.variables import _BoolVarImpl, _IntVarImpl, NDVarArray, boolvar
# from ..expressions.python_builtins import any

import numpy as np


def intvar_to_boolvar(int_var):
    '''
        Basic encoding of integer domain variables to boolean variables.
        We introduce a boolean variable for every value the integer variable
        can take and ensure only 1 value can be selected.

        `intvar_to_boolvar` returns
        (1) the mapping of every intvar to corresponding bool variables 
            as a dictionary

            ```
            dom(x1) = [4, 8]

                {
                    x1: {
                        4: bv4,
                        5: bv5, 
                        ...,
                        8: bv8
                    },
                    ...,
                    xn: {...}
                }

        (2) Exactly one constraint on the boolean variables

        Example:

            ```python
            x1 = intvar(lb=4, ub=8) # dom(x1) = [4, 8]

            # Introduce a boolean variable "linked" to an int value
            bv4, bv5, bv6, bv7, bv8 = boolvar(shape=(ub-lb+1))

            #Ensure only 1 value can be selected;
            exactlyone_val_constraint = sum([bv4, bv5, bv6, bv7, bv8]) == 1

            # The following encoding with the exactlyone constraint is then 
            # equivalent to specifying x1 as an integer variable.
            x1 = 4 * bv4 + 5 * bv5 + 6 * bv6 + 7 * bv7 + 8 * bv8
    '''
    constraints = []
    ivarmap = {}
    # Bool
    if isinstance(int_var, _BoolVarImpl):
        ivarmap[int_var] = int_var

    # takes care of empty list!
    elif isinstance(int_var, list):
        for ivar in int_var:
            sub_iv_mapping, int_cons = intvar_to_boolvar(ivar)
            constraints += int_cons
            ivarmap.update(sub_iv_mapping)

    elif isinstance(int_var, NDVarArray):
        lb, ub = int_var.flat[0].lb ,int_var.flat[0].ub
        # reusing numpy notation if possible
        bvs = boolvar(shape=int_var.shape + (ub - lb + 1,))

        for i, ivar in np.ndenumerate(int_var):
            ivarmap[ivar] = {ivar_val: bv for bv, ivar_val in zip(bvs[i], range(lb, ub+1))}
            constraints.append(sum(bvs[i]) == 1)
    else:
        lb, ub = int_var.lb ,int_var.ub
        bvs = boolvar(shape=(ub - lb +1))
        ivarmap[int_var] = {ivar_val: bv for bv, ivar_val in zip(bvs, range(lb, ub+1))}

        constraints.append(sum(bvs) == 1)

    return ivarmap, constraints

def linear_to_bool_constraint(constraint, ivarmap):
    print("\n")
    print(constraint, type(constraint))
    print("\t name:", constraint.name)
    print("\t args:", constraint.args, "\n")
    bool_constraints = []
    if isinstance(constraint, Comparison) and isinstance(constraint.args[0], Operator):
        left, right = constraint.args
        print("\t\t left.name:", left.name)
        print("\t\t left.args:", left.args, "\n")
        exprs_, vars_ = get_all_elems(left.args)
        # base sum
        if left.name == "sum" and all(isinstance(var, _IntVarImpl) for var in ):
            bool_constraints.append(sum([val * bv for var in get_variables(left) for val, bv in ivarmap[var].items()]))
            print("\t\t", left)
            print("\t\t", bool_constraints)
        
        else:
            raise NotImplementedError(f"Constraint {constraint} not supported...")    
    else:
        # TODO: Handle linear constraints with appropriate transformations
        raise NotImplementedError(f"Constraint {constraint} not supported...")
    return bool_constraints

def to_bool_constraint(constraint, ivarmap):
    '''
        Decomposition of integer constraint using the provided mapping
        of integer variables to boolvariables

    :param constraint: Input flattened expression encoded in given boolean variable
    encoding.
    :type Expression:

    :param ivarmap: Encoding of intvar to boolvar
    :type dict

    :return: bool: the computed output:
        - True      if a solution is found (not necessarily optimal, e.g. could be after timeout)
        - False     if no solution is found
    '''
    assert all(
        True if iv in ivarmap else False for iv in get_variables(constraint)
    ),f"""int var(s):
        {[iv for iv in get_variables(constraint) if iv not in ivarmap]}
    has not been mapped to a boolvar."""

    bool_constraints = []

    # CASE 1: True/False
    if isinstance(constraint, bool):
        return constraint

    # CASE 2: Only bool vars in constraint
    elif all(True if isinstance(arg, _BoolVarImpl) else False for arg in constraint.args):
        return constraint

    # CASE 3: Decompose list of constraints and handle individually
    elif isinstance(constraint, (list, NDVarArray)):
        for con in constraint:
            bool_constraints += to_bool_constraint(con, ivarmap)

    # CASE 4: base comparison constraints + ensure only handling what it can
    elif isinstance(constraint, Comparison) and all(is_int(arg) or isinstance(arg, (_IntVarImpl, _BoolVarImpl)) for arg in constraint.args) :
        bool_constraints += to_unit_comparison(constraint, ivarmap)

    # CASE 5: global constraints
    elif isinstance(constraint, (AllDifferent, AllEqual, Circuit, Table)):
        for con in constraint.decompose():
            bool_constraints += to_unit_comparison(con, ivarmap)

    # CASE 6: Linear constraints & others (ex: Global consraints Min/Max/...)
    else:
        bool_constraints += linear_to_bool_constraint(constraint, ivarmap)


    return bool_constraints

def to_unit_comparison(con, ivarmap):
    """Encoding of comparison constraint with input int-to-bool variable encoding.
    The comparison constraint is encoded using a negative version instead of keeping 
    only positives. 

    :param constraint: Input flattened expression encoded in given boolean variable
    encoding.
    :type Comparison

    :param ivarmap: Encoding of intvar to boolvar
    :type dict

    :return: Expression: encoding of int variable-based comparison constraint
    with given int-to-bool variable encoding.

    For example:
        x1 = intvar(lb=4, ub=8)
        x2 = intvar(lb=4, ub=8)
        # Introduce a boolean variable "linked" to an int value
        bv4, bv5, bv6, bv7, bv8 = boolvar(shape=(ub-lb+1))

        #Ensure only 1 value can be selected;
        exactlyone_val_constraint = sum([bv4, bv5, bv6, bv7, bv8]) == 1

        # The following encoding with the exactlyone constraint is then 
        # equivalent to specifying x1 as an integer variable.
        x1 = 4 * bv4 + 5 * bv5 + 6 * bv6 + 7 * bv7 + 8 * bv8

    Encoded using their negation:

        1. x1 > 5 ---> [[~bv4], [~bv5]]
        2. x1 < 5 ---> [[~bv5], [~bv6], [~bv7], [~bv8]]
        3. x1 < x2 --> [[~(x1i & x2i)] for x1i in x1 for x2i in x2 if x1i >= x2i]

    """
    bool_constraints=[]

    left, right = con.args
    operator = con.name

    if operator in [">", ">="]:
        right, left = left, right
        operator = operator.replace('>', '<')

    if operator == "==" and isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
        # x1 ==  x2
        if left.ub < right.lb or right.ub < left.lb:
            return bool_constraints

        # example x1 = [ 1, 7] x2 = [2, 5]
        smallest_lb, largest_lb = (left, right) if left.lb < right.lb else (right, left)
        # small : exclude [1, 2[
        for i in range(smallest_lb.lb, largest_lb.lb):
            bool_constraints.append(~ivarmap[smallest_lb][i])

        # large: exclude [6, 7]
        smallest_ub, largest_ub = (left, right) if left.ub < right.ub else (right, left)
        for i in range(smallest_ub.ub + 1, largest_ub.ub+1):
            bool_constraints.append(~ivarmap[largest_ub][i])

        # exclude tuples that have different values
        # [2, 6[
        for i in range(largest_lb.lb, smallest_ub.ub+1):
            # [2, 6[
            for j in range(largest_lb.lb, smallest_ub.ub+1):
                if i != j:
                    bool_constraints.append( ~(ivarmap[left][i] & ivarmap[right][j]))

        return bool_constraints

    # Value Assignment x == 5
    elif operator == "==" and any(True if is_int(arg) else False for arg in con.args):
        value, var = (left, right) if is_int(left) else (right, left)
        if var.lb <= value and value <= var.ub:
            bool_constraints.append(ivarmap[var][value])

    # x1  != x2
    elif operator == "!=" and isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
        for i in range(left.lb, left.ub+1):
            for j in range(right.lb, right.ub+1):
                if i == j:
                    bool_constraints.append(~(ivarmap[left][i] & ivarmap[right][j]))
    # x1  != 3
    elif operator == "!=" and any(True if isinstance(arg, (int, np.integer)) else False for arg in con.args):
        value, var = (left, right) if is_int(left) else  (right, left)
        if var.lb <= value and value <= var.ub:
            bool_constraints.append(~ivarmap[var][value])

    # x1  < x2
    elif operator == '<' and isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
        for i in range(left.lb, left.ub+1):
            for j in range(right.lb, right.ub+1):
                if i >= j:
                    bool_constraints.append(~(ivarmap[left][i] & ivarmap[right][j]))

    # 5 < x1 ------> x1 != 5, x1!=4, ...
    elif operator == '<' and is_int(left) and isinstance(right, _IntVarImpl):
        for i in range(right.lb, right.ub+1):
            if i <= left:
                bool_constraints.append(~ivarmap[right][i])

    # x1 < 5 ----> x1 != 5, x1 != 6, x1 != 7, ...
    elif operator == '<' and isinstance(left, _IntVarImpl) and is_int(right):
        for i in range(left.lb, left.ub+1):
            if i >= right:
                bool_constraints.append(~ivarmap[left][i])

    elif operator == '<=' and isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
        # x1  <= x2
        for i in range(left.lb, left.ub+1):
            for j in range(right.lb, right.ub+1):
                if i > j:
                    bool_constraints.append(~(ivarmap[left][i] & ivarmap[right][j]))
        # 5 <= x1
    elif operator == '<=' and is_int(left) and isinstance(right, _IntVarImpl):
        for i in range(right.lb, right.ub+1):
            if i < left:
                bool_constraints.append(~ivarmap[right][i])
        # x1 <= 5
    elif operator == '<=' and isinstance(left, _IntVarImpl) and is_int(right):
        for i in range(left.lb, left.ub+1):
            if i > right:
                bool_constraints.append(~ivarmap[left][i])
    else:
        raise NotImplementedError(f"Constraint {con} not supported...")

    return bool_constraints

def extract_boolvar(ivarmap):
    print(ivarmap)
    all_boolvars = []
    for varmap in ivarmap.values():
        all_boolvars += [bv for bv in varmap.values()]
    return all_boolvars
