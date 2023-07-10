from cpmpy.expressions.core import Comparison, Operator
from cpmpy.expressions.globalconstraints import AllDifferent, AllEqual, Circuit, GlobalConstraint, Table, DirectConstraint
from cpmpy.expressions.utils import is_any_list, is_int, is_bool
from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.get_variables import get_variables, get_variables_model
from cpmpy.expressions.variables import  _BoolVarImpl, _IntVarImpl, boolvar
from cpmpy.solvers.solver_interface import SolverInterface
from cpmpy import SolverLookup
import numpy as np
import builtins


class CPM_int2bool_direct(SolverInterface):
    """
    Meta-interface: converts integer variables and constraints to
    Boolean variables (direct/one-hot encoding) and constraints

    Keeps mapping of intvars to boolvars, and updates the intvars after solve

    All other functions are passed straight-through to the underlying solver

    Creates the following attributes (see parent constructor for more):
    - subsolver: SolverInterface, the solver used to solve the Boolean model
    - ivarmap: map from IntVars to their Boolean Variables
    """

    @staticmethod
    def supported():
        return True  # well, depends on the subsolver, will be checked in constructor

    @staticmethod
    def solvernames():
        """
            Returns subsolvers supported by PySAT on your system
        """
        from cpmpy import SolverLookup
        return [s for s,_ in SolverLookup.base_solvers() if s != "int2bool"]


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: str, name of a subsolver (required!)
        """
        assert(subsolver is not None), "CPM_Meta_int2bool: you must supply the name of a CPMpy solver"

        from .utils import SolverLookup
        # init the subsolver with an empty model
        self.subsolver = SolverLookup.get(subsolver)
        self.ivarmap = dict()

        if not self.subsolver.supported():
            raise Exception(f"CPM_Meta_int2bool: subsolver {subsolver} is not supported it seems")

        # initialise everything else and post the constraints/objective
        super().__init__(name="int2bool:"+subsolver, cpm_model=cpm_model)


    def solve(self, time_limit=None, **kwargs):
        """
            Call the subsolver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            <Please document key solver arguments that the user might wish to change
             for example: log_output=True, var_ordering=3, num_cores=8, ...>
            <Add link to documentation of all solver parameters>
        """
        ret = self.subsolver.solve(time_limit, **kwargs)

        # remapping the solution values (of user specified intvars only)
        int2bool_direct_decode(self.ivarmap)

        self.cpm_status = self.subsolver.cpm_status
        return ret


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created

            I am not sure what should happen if you pass an intvar here...
            For now, it will only return on what the subsolver knows
        """
        self.subsolver.solver_var(cpm_var)


    def transform(self, cpm_expr):
        """
            Transform constraints with integer variables to equivalent ones on the Boolean variables corresponding to the Integer ones

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        return int2bool_direct_constraints(cpm_expr, ivarmap=self.ivarmap)


    def __add__(self, cpm_expr):
        """
            Transform and post supported CPMpy constraints to the subsolver

            What 'supported' means depends on the solver capabilities, and in effect on what transformations
            are applied in `transform()`.

            Solvers can raise 'NotImplementedError' for any constraint not supported after transformation
        """
        self.subsolver += self.transform(cpm_expr)
        return self


    # Other functions from SolverInterface that you can overwrite:
    # solveAll, solution_hint, get_core
    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        return self.subsolver.solveAll(display, time_limit, solution_limit, call_from_model, **kwargs)
    def solution_hint(self, cpm_vars, vals):
        return self.subsolver.solution_hint(cpm_vars, vals)
    def get_core(self):
        return self.subsolver.get_core()



def int2bool_direct_encode(ivar, ivarmap):
    '''
        Basic 'one hot' encoding of integer domain variables to boolean variables.
        We introduce a boolean variable for every value the integer variable
        can take and ensure only 1 value can be selected.

        `int2bool_direct_encode` will ensure that ivarmap is:
        the mapping of every intvar to corresponding bool variables 
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

        and return an 'exactly one' constraint on the corresponding Boolean variables.

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
    assert isinstance(ivar, _IntVarImpl) and not ivar.is_bool(), f"ivar {ivar} must be an intvar"
    assert isinstance(ivarmap, dict), f"ivarmap {ivarmap} must be a (potentially empty) dict"
    assert (not ivar in ivarmap), f"ivar {ivar} is already defined in the ivarmap"

    # create the boolvars and mapping to values
    lb, ub = ivar.lb, ivar.ub
    d = dict()  # TODO: since it is always a range, it could be more efficient to use (offset, arr)
    for v in range(lb,ub+1):
        # use debug-friendly naming scheme
        d[v] = boolvar(name=f"i2b_{ivar.name}={v}")

    # add to ivarmap
    ivarmap[ivar] = d

    # return 'exactly one' constraint on the Bools
    return [sum(d.values()) == 1]


def int2bool_direct_decode(ivarmap):
    """
        Decode and set the value of all intvars in intvarmap,
        based on the assignment of their Boolean mapped variables

        To be used after calling `solve()` on the Boolean model

        Args:
        - ivarmap: a dictionary of intvar: value_dict (see `int2bool_direct_encode()`)
    """
    for iv, value_dict in ivarmap.items():
        iv._value = None  # reset in case of False solve

        # check if (at most) one is assigned True, set its corresponding value
        noneyet = True
        for iv_val, bv in value_dict.items():
            if bv.value():
                assert noneyet, f"{iv} should have at most one of its Booleans true: {[bv.value() for bv in value_dict.values()]}"
                iv._value = iv_val
                noneyet = False


def int2bool_direct_constraints(constraints, ivarmap):
    '''
    Encode the int variables-based constraints into their direct boolean encoding
    and keep track of int->bool variable mapping.

    :return: list of Expressions that only contain boolvars
    Args:
    - constraints: list of Expressions (possibly containing intvars)
    - ivarmap: dictionary (required, but can be empty, e.g. {})
               will by filled by `int2bool_direct_encode()`
    '''
    assert isinstance(ivarmap, dict), f"ivarmap {ivarmap} must be a (potentially empty) dict"

    # already bool variables no transformation to apply
    if builtins.all(var.is_bool() for var in get_variables(constraints)):
        return constraints

    bool_constraints = []
    for cpm_con in flatten_constraint(constraints):  # no `only_bv_implies()`, it would break up equalities

        con_vars = get_variables(cpm_con)
        if builtins.all(var.is_bool() for var in con_vars):
            bool_constraints.append(constraint)

        else:
            ### preload encoding of int vars in ivarmap
            for var in con_vars:
                if not var.is_bool() and not var in ivarmap:
                    bool_constraints += int2bool_direct_encode(var, ivarmap)

            ### encoding constraints int -> bool
            bool_constraints += int2bool_direct_flatconstraint(cpm_con, ivarmap)

    return bool_constraints


def int2bool_direct_flatconstraint(constraint, ivarmap):
    '''
        Decomposition of flat constraint using the provided mapping
        of integer variables to boolvariables

    :param constraint: Input flattened expression, with all intvars encoded in given boolean variable
    encoding.
    :type Expression:

    :param ivarmap: Direct encoding of intvar to boolvar
    :type dict

    :return: list of Expression: only containing Boolean variables
    '''
    # Flat expressions that can contain an intvar
    if isinstance(constraint, Comparison):
        lhs, rhs = constraint.args

        if isinstance(lhs, _IntVarImpl) and not is_bool(rhs):  # lhs may be bool (used as int)
            # Comparison constraints: (no nesting, e.g. only NumVar >=< NumVar/Const)
            return int2bool_direct_comparison(constraint, ivarmap)
        elif lhs.name == 'sum' or lhs.name == 'wsum':
            # (weighted) sum, e.g. (w)sum >=< IVar/Const
            return int2bool_direct_linear(constraint, ivarmap)
        elif hasattr(lhs, 'decompose_numeric'):
            # TODO: numeric global, decompose it...
            raise NotImplementedError(f"Numeric global {constraint} not supported...")
        elif lhs.is_bool() and is_bool(rhs):
            # - Reification (double implication): Boolexpr == BVar    (CPMpy class 'Comparison')
            # (italy == red) == BV
            if isinstance(lhs, Comparison) and isinstance(rhs, _BoolVarImpl):
                from cpmpy.expressions.python_builtins import any
                subboolcons = any(int2bool_direct_comparison_pos(lhs, ivarmap))
                return [any_left == right]
            else:
                # TODO, see implication below, more generic possible?
                raise NotImplementedError(f"Reified Comparison {constraint} not supported...")
        else:
            raise NotImplementedError(f"Comparison Constraint {constraint} not supported...")

    # - Implication: Boolexpr -> BVar or BVar -> BoolExpr                 (CPMpy class 'Operator', is_bool())
    elif constraint.name == "->":
        # one of the two is guaranteed to be a BVar, the other a Boolexpr (or Bvar)
        lhsboolvar = isinstance(lhs, _BoolVarImpl)
        rhsboolvar = isinstance(rhs, _BoolVarImpl)
        if lhsboolvar and rhsboolvar:
            return [constraint]
        elif lhsboolvar:
            bexpr = rhs
        else:  # rhsboolvar
            bexpr = lhs

        # convert the Boolexpr (containing NumVars) to a list of constraints over Booleans
        from cpmpy.expressions.python_builtins import all
        subboolcons = all(int2bool_direct_flatconstraint(bexpr, ivarmap))
        if lhsboolvar:
            return [lhs.implies(subboolcons)]
        else:
            return [subboolcons.implies(rhs)]

    # Decomposable global constraints
    elif hasattr(constraint, 'decompose'):
        return int2bool_direct_constraints(constraint.decompose(), ivarmap)  # may potentially create other intvars?

    elif isinstance(constraint, DirectConstraint):
        return [constraint]

    else:
        raise f"to bool: Constraint {constraint=} of type {type(constraint)} not handled yet"


def int2bool_direct_linear(con, ivarmap):
    """Encode the linear sum integer variable constraint with direct
    variable encoding.

    Args:
        con (cpmpy.Expression.Comparison): Comparison operator with 
        ivarmap ([type]): [description]
    """
    left, right = constraint.args
    if isinstance(left, Operator) and isinstance(right, _IntVarImpl):
        # TODO
        raise "TODO: check that this case is properly coved"
        cons = int2bool_direct_flatconstraint(
            Comparison(name=constraint.name, left=left-right, right=0),
            ivarmap
        )

    op, val = con.args[0].name, con.args[1]
    w, x = [], []
    # SUM CASE
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


def int2bool_direct_comparison(con, ivarmap):
    """Encoding of comparison constraint with direct variable encoding.
    The comparison constraint is encoded using a negative version instead of keeping 
    only positives. 

    :param constraint: Input flattened comparison with intvars
    :type Comparison

    :param ivarmap: Direct encoding of intvar to boolvar
    :type dict

    :return: Expression: encoding of int variable-based comparison constraint
    with direct variable encoding.

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


def int2bool_direct_comparison_pos(con, ivarmap):
    """Encoding of comparison constraint with direct variable encoding.
    The comparison constraint is encoded using the positives version instead of keeping
    only the negations. 

    :param constraint: Input flattened comparison with intvar(s)
    :type Comparison

    :param ivarmap: Direct encoding of intvar to boolvar
    :type dict

    :return: Expression: encoding of int variable-based comparison constraint
    with direct variable encoding.

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

