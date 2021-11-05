from cpmpy.expressions.core import Comparison, Operator
from cpmpy.expressions.globalconstraints import AllDifferent, AllEqual, Circuit, GlobalConstraint, Table
from ..expressions.variables import _BoolVarImpl, _IntVarImpl, NDVarArray, boolvar, intvar
# from ..expressions.python_builtins import any

import numpy as np

def to_unit_comparison(con, ivarmap):
    bool_constraints=[]

    # assignment constraint
    left, right = con.args
    operator = con.name
    if operator in [">", ">="]:
        right, left = left, right
        operator = operator.replace('>', '<')

    if operator == "==":
        if  isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
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
        elif any(True if isinstance(arg, (int, np.int64)) else False for arg in con.args):
            value, var = (left, right) if isinstance(left, (int, np.int64)) else (right, left)
            if var.lb <= value and value <= var.ub:
                bool_constraints.append(ivarmap[var][value])
            else:
                raise NotImplementedError(f"Constraint {con} not supported...")
        else:
            raise NotImplementedError(f"Constraint {con} not supported...")
    elif operator == "!=":
        # x1  != x2
        if  isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
            for i in range(left.lb, left.ub+1):
                for j in range(right.lb, right.ub+1):
                    if i == j:
                        bool_constraints.append(~(ivarmap[left][i] & ivarmap[right][j]))
        # x1  != 3
        elif any(True if isinstance(arg, (int, np.int64)) else False for arg in con.args):
            value, var = (left, right) if isinstance(left, (int, np.int64)) else  (right, left)
            if var.lb <= value and value <= var.ub:
                bool_constraints.append(~ivarmap[var][value])
        else:
            raise NotImplementedError(f"Constraint {con} not supported...")
    elif operator == '<':
        if  isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
            # x1  < x2
            for i in range(left.lb, left.ub+1):
                for j in range(right.lb, right.ub+1):
                    if i >= j:
                        bool_constraints.append(~(ivarmap[left][i] & ivarmap[right][j]))

        # 5 < x1 ------> x1 != 5, x1!=4, ...
        elif isinstance(left, (int, np.int64)) and isinstance(right, _IntVarImpl):
            for i in range(right.lb, right.ub+1):
                if i <= left:
                    bool_constraints.append(~ivarmap[right][i])
        # x1 < 5
        elif isinstance(left, _IntVarImpl) and isinstance(right, (int, np.int64)):
            for i in range(left.lb, left.ub+1):
                if i >= right:
                    bool_constraints.append(~ivarmap[left][i])
        else:
            raise NotImplementedError(f"Constraint {con} not supported...")
    elif operator == '<=':
        if  isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
            # x1  <= x2
            for i in range(left.lb, left.ub+1):
                for j in range(right.lb, right.ub+1):
                    if i > j:
                        bool_constraints.append(~(ivarmap[left][i] & ivarmap[right][j]))
        # 5 <= x1
        elif isinstance(left, (int, np.int64)) and isinstance(right, _IntVarImpl):
            for i in range(right.lb, right.ub+1):
                if i < left:
                    bool_constraints.append(~ivarmap[right][i])
        # x1 <= 5
        elif isinstance(left, _IntVarImpl) and isinstance(right, (int, np.int64)):
            for i in range(left.lb, left.ub+1):
                if i > right:
                    bool_constraints.append(~ivarmap[left][i])
        else:
            raise NotImplementedError(f"Constraint {con} not supported...")
    return bool_constraints

def to_bool_constraint(constraint, ivarmap):
    ## composition of constraints
    print(constraint, constraint.name, type(constraint))
    bool_constraints = []

    if isinstance(constraint, bool):
        return constraint

    elif isinstance(constraint, (list, NDVarArray)):
        for con in constraint:
            bool_constraints += to_bool_constraint(con, ivarmap)

    elif all(True if isinstance(arg, _BoolVarImpl) else False for arg in constraint.args):
        return constraint

    # base comparison constraints
    elif isinstance(constraint, Comparison):
        bool_constraints += to_unit_comparison(constraint, ivarmap)

    # global constraints
    elif isinstance(constraint, (AllDifferent, AllEqual, Circuit, Table)):
        for con in constraint.decompose():
            bool_constraints += to_unit_comparison(con, ivarmap)
    else:
        raise NotImplementedError(f"Constraint {constraint} not supported...")

    return bool_constraints

def extract_boolvar(ivarmap):
    all_boolvars = []
    for varmap in ivarmap.values():
        all_boolvars += [bv for bv in varmap.values()]
    return all_boolvars

def intvar_to_boolvar(int_var):

    constraints = []
    ivarmap = {}

    if isinstance(int_var, _BoolVarImpl):
        ivarmap[int_var] = int_var

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