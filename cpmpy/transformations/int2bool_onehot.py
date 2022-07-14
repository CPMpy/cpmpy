from ..expressions.core import Comparison, Operator
from ..expressions.globalconstraints import AllDifferent, AllEqual, Circuit, GlobalConstraint, Table
from ..expressions.utils import is_any_list, is_int
from ..transformations.to_cnf import to_cnf
from ..transformations.get_variables import get_variables, get_variables_model
from ..expressions.variables import  _BoolVarImpl, _IntVarImpl, boolvar
import numpy as np

def is_boolvar_constraint(constraint):
    return all(var.is_bool() for var in get_variables(constraint))

def is_bool_model(model):
    return all(var.is_bool() for var in get_variables_model(model))

def int2bool_model(model):
    '''
    Flatten model to ensure flat int variable-based constraints can be 
    encoded to a boolean version.

    :return: (dict, Model):
        - dict: mapping of int variable values to boolean variables
        - model: new boolean encoding of int model
    '''
    from ..model import Model

    assert isinstance(model, Model), f"Input expected Cpmpy.Model got ({type(model)})"

    return int2bool_constraints(model.constraints)

def int2bool_constraints(constraints, ivarmap=None):
    '''
    Encode the int variables-based constraints into their boolean encoding
    and keep track of int->bool variable mapping.

    :return: (dict, Model):
        - dict: mapping of int variable values to boolean variables
        - model: new boolean encoding of int model
    '''
    # already bool variables no transformation to apply
    if ivarmap is None:
        ivarmap = {}

    if all(var.is_bool() for var in get_variables(constraints)):
        return (ivarmap, constraints)

    bool_constraints = []
    ## added extra loop for debugging purposes during to_cnf
    for constraint in constraints:
        cpm_cons = to_cnf(constraint)

        for cpm_con in cpm_cons:
            con_vars = get_variables(cpm_con)
            if not is_boolvar_constraint(cpm_con):
                ### encoding int vars
                iv_not_mapped = [iv for iv in con_vars if iv not in ivarmap and not iv.is_bool()]
                new_ivarmap, iv_bool_constraints = intvar_to_boolvar(iv_not_mapped)
                ivarmap.update(new_ivarmap)
                bool_constraints += iv_bool_constraints

                ### encoding constraints int -> bool
                new_bool_cons = to_bool_constraint(cpm_con, ivarmap)
                bool_constraints += new_bool_cons
            else:
                bool_constraints.append(constraint)

    # raise NotImplementedError(f"Function not finished")
    return (ivarmap, bool_constraints)

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

        (2) an 'exactly one' constraint per integer variable on its corresponding
            Boolean variables.

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
    ivarmap = {}
    constraints = []

    # takes care of empty list!
    if is_any_list(int_var):
        for ivar in int_var:
            sub_ivarmap, sub_cons = intvar_to_boolvar(ivar)
            ivarmap.update(sub_ivarmap)
            constraints += sub_cons
    elif int_var.is_bool():
        return ivarmap, constraints
    else:
        lb, ub = int_var.lb ,int_var.ub
        d = dict()
        for v in range(lb,ub+1):
            # use debug-friendly naming scheme
            d[v] = boolvar(name=f"i2b_{int_var.name}={v}")

        ivarmap[int_var] = d
        # Only 1 Boolvar should be set to True
        constraints.append(sum(d.values()) == 1) 

    return ivarmap, constraints


def to_bool_constraint(constraint, ivarmap=dict()):
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
    bool_constraints = []

    # CASE 1: Decompose list of constraints and handle individually
    if is_any_list(constraint):
        for con in constraint:
            bool_constraints += to_bool_constraint(con, ivarmap)

    # CASE 2: base comparison constraints + ensure only handling what it can
    elif isinstance(constraint, Comparison) and all(is_int(arg) or isinstance(arg, (_IntVarImpl)) for arg in constraint.args):
        bool_constraints += to_unit_comparison(constraint, ivarmap)
    # CASE 3: Linear WEIGHTED SUM
    elif isinstance(constraint, Comparison) and isinstance(constraint.args[0], Operator) and is_int(constraint.args[1]):
        bool_constraints += encode_linear_constraint(constraint, ivarmap)
    elif isinstance(constraint, Comparison) and isinstance(constraint.args[1], _IntVarImpl):
        bool_constraints += encode_var_comparison(constraint, ivarmap)
        # raise NotImplementedError(f"Comparison Constraint {constraint} not supported...")
    # CASE 4: global constraints
    elif isinstance(constraint, (AllDifferent, AllEqual, Circuit, Table)):
        for con in constraint.decompose():
            bool_constraints += to_unit_comparison(con, ivarmap)
            # print("decomposed - bool_constraints", to_unit_comparison(con, ivarmap))
    elif isinstance(constraint, GlobalConstraint):
        raise NotImplementedError(f"Global Constraint {constraint} not supported...")

    # assertion to be removed
    else:
        assert all(isinstance(var, bool) or var.is_bool() for var in get_variables(constraint)) or isinstance(constraint, bool), f"Operation not handled {constraint} yet"

    return bool_constraints

def encode_var_comparison(constraint, ivarmap):
    # print("Enconding as var comparison!")
    # print(f"\t{constraint=}")
    bool_constraints = []
    left, right = constraint.args
    if isinstance(left, Operator) and isinstance(right, _IntVarImpl):
        cons = to_bool_constraint(
            Comparison(name=constraint.name, left=left-right, right=0),
            ivarmap
        )
        bool_constraints += cons
    # (italy == red) == BV
    elif isinstance(left, Comparison) and isinstance(right, _BoolVarImpl):
        from cpmpy.expressions.python_builtins import any
        any_left = any(to_unit_comparison_pos(left, ivarmap))
        bool_constraints += [any_left.implies(right)]
        bool_constraints += [right.implies(any_left)]
    else:
        raise NotImplementedError(f"Intvar Comparison {constraint} not supported...")

    return bool_constraints

def encode_linear_constraint(con, ivarmap):
    """Encode the linear sum integer variable constraint with input int-to-bool 
    variable encoding.

    Args:
        con (cpmpy.Expression.Comparison): Comparison operator with 
        ivarmap ([type]): [description]
    """
    op, val = con.args[0].name, con.args[1]
    w, x = [], []
    # SUM CASE
    # TODO: CASES need to be handled by weighted sum !
    if op == "sum":
        op_args = con.args[0].args

        for expr in op_args:
            # unweighted sum
            if isinstance(expr, _IntVarImpl):
                for wi, bv in ivarmap[expr].items():
                    w.append(wi)
                    x.append(bv)
            # Weighted sum
            elif isinstance(expr, Operator) and expr.name == "mul":
                coeff = expr.args[0]
                var = expr.args[1]
                for wi, bv in ivarmap[var].items():
                    w.append(wi * coeff)
                    x.append(bv)
            # Other functions
            elif isinstance(expr, Operator) and expr.name == "-":
                sub_expr = expr.args[0]
                if isinstance(sub_expr, _IntVarImpl):
                    for wi, bv in ivarmap[sub_expr].items():
                        w.append(-wi)
                        x.append(bv)
                elif isinstance(sub_expr, Operator) and sub_expr.name == "mul" and isinstance(sub_expr.args[1], _IntVarImpl):
                    coeff = sub_expr.args[0]
                    var = sub_expr.args[1]
                    for wi, bv in ivarmap[var].items():
                        w.append(wi * coeff)
                        x.append(bv)
                else:
                    raise NotImplementedError(f"Other sum expressions {expr=} not supported yet...")
            else:
                raise NotImplementedError(f"Other sum expressions {expr=} not supported yet...")
        if val < 0:
            w = [-wi for wi in w]
        return [Comparison(con.name, Operator("sum", [wi *xi for wi, xi in zip(w, x)]), abs(val))]
        # return [Comparison(con.name, Operator("wsum", (w, x)), val)]
    elif op == "mul" and is_int(con.args[0].args[0]) and isinstance(con.args[0].args[1], _IntVarImpl):
        coeff = con.args[0].args[0]
        var = con.args[0].args[1]
        for wi, bv in ivarmap[var].items():
            w.append(wi * coeff)
            x.append(bv)
        return [Comparison(con.name, Operator("sum", [wi *xi for wi, xi in zip(w, x)]), abs(val))]
    elif op == "-":
        if val < 0:
            return [-con]
        return [con]
    elif op == "abs" and isinstance(con.args[0].args[0], _IntVarImpl):
        var = con.args[0].args[0]
        bool_cons = []# # print(f"\t abs: {var=} {var.lb=} {var.ub=}")
        for wi, bv in ivarmap[var].items():
            if abs(wi) != val:
                bool_cons.append([~bv])
        return bool_cons
    else:
        raise NotImplementedError(f"Comparison {con=} {op=} {con.args[0].args=} not supported yet...")

def to_unit_comparison_pos(con, ivarmap):
    """Encoding of comparison constraint with input int-to-bool variable encoding.
    The comparison constraint is encoded using the positives version instead of keeping
    only the negations. 

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
    bool_constraints =[]

    left, right = con.args
    operator = con.name

    if operator == "==" and isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
        # x1 ==  x2
        for i in range(left.lb, left.ub+1):
            for j in range(right.lb, right.ub+1):
                if i == j:
                    bool_constraints.append(ivarmap[left][i] & ivarmap[right][j])

    elif operator == "<"  and isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
        for i in range(left.lb, left.ub+1):
            for j in range(right.lb, right.ub+1):
                if i < j:
                    bool_constraints.append(ivarmap[left][i] & ivarmap[right][j])

    elif operator == "<="  and isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
        for i in range(left.lb, left.ub+1):
            for j in range(right.lb, right.ub+1):
                if i <= j:
                    bool_constraints.append(ivarmap[left][i] & ivarmap[right][j])

    elif operator == ">"  and isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
        for i in range(left.lb, left.ub+1):
            for j in range(right.lb, right.ub+1):
                if i > j:
                    bool_constraints.append(ivarmap[left][i] & ivarmap[right][j])

    elif operator == ">="  and isinstance(left, _IntVarImpl) and isinstance(right, _IntVarImpl):
        for i in range(left.lb, left.ub+1):
            for j in range(right.lb, right.ub+1):
                if i >= j:
                    bool_constraints.append(ivarmap[left][i] & ivarmap[right][j])
    else:
        raise NotImplementedError(f"Constraint {con} not supported...")

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
                    bool_constraints.append( (~ivarmap[left][i] | ~ivarmap[right][j]))

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
                    bool_constraints.append((~ivarmap[left][i] | ~ivarmap[right][j]))
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
                    bool_constraints.append((~ivarmap[left][i] | ~ivarmap[right][j]))

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
                    bool_constraints.append((~ivarmap[left][i] | ~ivarmap[right][j]))
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
