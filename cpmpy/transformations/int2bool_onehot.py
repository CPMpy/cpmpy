
from ..expressions.core import Comparison, Operator
from ..expressions.globalconstraints import AllDifferent, AllEqual, Circuit, GlobalConstraint, Table
from ..expressions.utils import is_any_list, is_int
from ..transformations.get_variables import get_variables, get_variables_model
from ..transformations.flatten_model import flatten_constraint
from ..expressions.variables import  _IntVarImpl, boolvar, intvar
import numpy as np

def int2bool_onehot(model):
    '''
    Flatten model to ensure flat int variable-based constraints can be 
    encoded to a boolean version.

    :return: (dict, Model):
        - dict: mapping of int variable values to boolean variables
        - model: new boolean encoding of int model
    '''
    from ..model import Model

    assert isinstance(model, Model), f"Input expected Cpmpy.Model got ({type(model)})"

    return int2bool(model.constraints)

def int2bool(constraints, ivarmap=None):
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

    flattened_constraints = flatten_constraint(constraints)

    for constraint in flattened_constraints:

        if not is_boolvar_constraint(constraint):
            new_bool_cons, new_ivarmap = to_bool_constraint(constraint, ivarmap)
            ivarmap.update(new_ivarmap)
            bool_constraints += new_bool_cons
        else:
            bool_constraints.append(constraint)

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

    elif intvar.is_bool():
        return ivarmap, constraints
    else:
        lb, ub = int_var.lb ,int_var.ub
        d = dict()
        for v in range(lb,ub+1):
            # use debug-friendly naming scheme
            d[v] = boolvar(name=f"i2b_{int_var.name}={v}")

        ivarmap[int_var] = d
        constraints.append(sum(d.values()) == 1) # the created Boolean vars

    return ivarmap, constraints

def is_boolvar_constraint(constraint):
    return all(var.is_bool() for var in get_variables(constraint))

def is_bool_model(model):
    return all(var.is_bool() for var in get_variables_model(model))

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
    user_vars = get_variables(constraint)
    iv_not_mapped = [iv for iv in user_vars if iv not in ivarmap and not iv.is_bool()]

    if len(iv_not_mapped) > 0:
        new_ivarmap, new_bool_constraints = intvar_to_boolvar(iv_not_mapped)
        ivarmap.update(new_ivarmap)
        bool_constraints += new_bool_constraints

    # CASE 1: Decompose list of constraints and handle individually
    if is_any_list(constraint):
        for con in constraint:
            new_bool_constraints, new_ivarmap = to_bool_constraint(con, ivarmap)
            ivarmap.update(new_ivarmap)
            bool_constraints += new_bool_constraints

    # CASE 2: base comparison constraints + ensure only handling what it can
    elif isinstance(constraint, Comparison) and all(is_int(arg) or isinstance(arg, (_IntVarImpl)) for arg in constraint.args):
        bool_constraints += to_unit_comparison(constraint, ivarmap)
    elif isinstance(constraint, Comparison) and isinstance(constraint.args[0], Operator) and is_int(constraint.args[1]):
        bool_constraints += encode_linear_constraint(constraint, ivarmap)

    # CASE 3: global constraints
    elif isinstance(constraint, (AllDifferent, AllEqual, Circuit, Table)):
        for con in constraint.decompose():
            bool_constraints += to_unit_comparison(con, ivarmap)
    elif isinstance(constraint, GlobalConstraint):
        raise NotImplementedError(f"Global Constraint {constraint} not supported...")

    # assertion to be removed
    else:
        assert all(isinstance(var, bool) or var.is_bool() for var in user_vars) or isinstance(constraint, bool), f"Operation not handled {constraint} yet"

    return bool_constraints, ivarmap

def encode_linear_constraint(con, ivarmap):
    """Encode the linear sum integer variable constraint with input int-to-bool 
    variable encoding.

    Args:
        con (cpmpy.Expression.Comparison): Comparison operator with 
        ivarmap ([type]): [description]
    """
    op, val = con.args[0].name, con.args[1]
    # SUM CASE
    if op == "sum":
        op_args = con.args[0].args
        w, x = [], []

        for var in op_args:
            for wi, bv in ivarmap[var].items():
                w.append(wi)
                x.append(bv)

        return Operator(con.name, [Operator("wsum", (w, x)), val])
    # WEIGHTED SUM
    elif op == "wsum":
        w_in, x_in = con.args[0].args
        w_out, x_out = [], []

        for wi, xi in w_in, x_in:
            for wj, bv in ivarmap[xi].items():
                w_out.append(wi * wj)
                x_out.append(bv)
        return Operator(con.name, [Operator("wsum", (w_out, x_out)), val])
    # TODO other comparison ??
    else:
        raise NotImplementedError(f"Comparison {con} not supported yet...")

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
