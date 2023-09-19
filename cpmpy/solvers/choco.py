#!/usr/bin/env python
## choco.py
##
"""
    Interface to Choco solver's Python API

    ...

    Documentation of the solver's own Python API:
    https://pypi.org/project/pychoco/
    https://pychoco.readthedocs.io/en/latest/

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_choco
"""
import time

import numpy as np

from cpmpy.exceptions import NotSupportedError

import cpmpy
from ..transformations.normalize import toplevel_list
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.utils import is_num
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.comparison import only_numexpr_equality
from ..transformations.linearize import canonical_comparison
from ..transformations.reification import only_bv_reifies, reify_rewrite


class CPM_choco(SolverInterface):
    """
    Interface to the Choco solver python API

    Requires that the 'pychoco' python package is installed:
    $ pip install pychoco

    See detailed installation instructions at:
    https://pypi.org/project/pychoco/
    https://pychoco.readthedocs.io/en/latest/

    Creates the following attributes (see parent constructor for more):
    chc_model: the pychoco.Model() created by _model()
    chc_solver: the choco Model().get_solver() instance used in solve()

    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pychoco as chc
            return True
        except ImportError:
            return False

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        choco model and solver object (chc_model and chc_solver)

        chc_model and chc_solver can both be modified externally before
        calling solve(), a prime way to use more advanced solver features

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: None
        """
        if not self.supported():
            raise Exception("Install the python 'pychoco' package to use this solver interface")

        import pychoco as chc

        assert (subsolver is None), "Choco does not support any subsolver"

        # initialise the native solver objects
        self.chc_model = chc.Model()
        self.chc_solver = chc.Model().get_solver()
        self.helper_var = self.chc_model.intvar(0, 0)

        # for the objective
        self.has_obj = False
        self.obj = None
        self.maximize_obj = None

        # for solving with assumption variables, TO-CHECK

        # initialise everything else and post the constraints/objective
        super().__init__(name="choco", cpm_model=cpm_model)

    def solve(self, time_limit=None, **kwargs):
        """
            Call the Choco solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

        """

        if time_limit is not None:
            raise NotSupportedError("Pychoco time_limit is not working properly. Not implemented in CPMpy")

        # call the solver, with parameters
        self.chc_solver = self.chc_model.get_solver()
        start = time.time()
        if self.has_objective():
            sol = self.chc_solver.find_optimal_solution(maximize=self.maximize_obj,
                                                                    objective=self.solver_var(self.obj))
        else:
            sol = self.chc_solver.find_solution()
        end = time.time()

        # new status, get runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = end - start

        # translate exit status
        if sol is not None:
            if self.has_objective() and (time_limit is None or self.cpm_status.runtime < time_limit):
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif (sol is None) and time_limit is not None and self.cpm_status.runtime >= time_limit:
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            # can happen when timeout is reached...

        # True/False depending on self.chc_status
        has_sol = sol is not None

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                cpm_var._value = sol.get_int_val(self.solver_var(cpm_var))

            # translate objective
            if self.has_objective():
                self.objective_value_ = sol.get_int_val(self.solver_var(self.obj))

        return has_sol

    def solveAll(self, display=None, time_limit=None, solution_limit=None, **kwargs):
        """
            Compute all (optimal) solutions, map them to CPMpy and optionally display the solutions.

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - solution_limit: stop after this many solutions (default: None)
                - time_limit:  maximum solve time in seconds (float, default: None)

            Returns: number of solutions found
        """

        if time_limit is not None:
            raise Exception("Pychoco time_limit is not working properly. Not implemented in CPMpy")

        self.chc_solver = self.chc_model.get_solver()
        start = time.time()
        if self.has_objective():
            raise NotSupportedError("Pychoco does not support finding all optimal solutions currently.")
            # Normally the following, but currently have a bug
#            sols = self.chc_solver.find_all_optimal_solutions(maximize=self.maximize_obj,
#                                                                         solution_limit=solution_limit,
#                                                                         objective=self.solver_var(self.obj))
        else:
            sols = self.chc_solver.find_all_solutions(solution_limit=solution_limit)
        end = time.time()

        # new status, get runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = end - start

        # display if needed
        if display is not None:
            for sol in sols:
                # map the solution to user vars
                for cpm_var in self.user_vars:
                    cpm_var._value = sol.get_int_val(self.solver_var(cpm_var))
                # print the desired display
                if isinstance(display, Expression):
                    print(display.value())
                elif isinstance(display, list):
                    print([v.value() for v in display])
                else:
                    display()  # callback

        return len(sols)

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var):  # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return self.chc_model.bool_not_view(self.solver_var(cpm_var._bv))

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.chc_model.boolvar(name=str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.chc_model.intvar(cpm_var.lb, cpm_var.ub, name=str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        return self._varmap[cpm_var]

    def objective(self, expr, minimize):
        """
            Post the given expression to the solver as objective to minimize/maximize

            - expr: Expression, the CPMpy expression that represents the objective function
            - minimize: Bool, whether it is a minimization problem (True) or maximization problem (False)

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: constraints created during conversion of the objective
            are premanently posted to the solver. Choco accepts variables to maximize or minimize
            so it is needed to post constraints and create auxiliary variables)
        """

        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr)
        self += flat_cons  # add potentially created constraints
        get_variables(flat_obj, collect=self.user_vars)  # add objvars to vars
        lb, ub= flat_obj.get_bounds()
        obj = cpmpy.intvar(lb, ub)
        obj_con = flat_obj == obj

        # add constraint for objective variable
        self += obj_con

        self.has_obj = True
        self.obj = obj
        self.maximize_obj = not minimize  # Choco has as default to maximize

    def has_objective(self):
        return self.has_obj

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function

            Accepted by Choco:
            - Decision variable: Var
            - Linear: sum([Var])                                   (CPMpy class 'Operator', name 'sum')
                      wsum([Const],[Var])                          (CPMpy class 'Operator', name 'wsum')
        """

        lhs = cpm_expr.args[0]
        rhs = cpm_expr.args[1]
        op = cpm_expr.name
        if op == "==": op = "="  # choco uses "=" for equality
        if not isinstance(rhs,Expression): rhs = int(rhs)

        # decision variables, check in varmap
        if isinstance(lhs, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.chc_model.arithm(self.solver_var(lhs), op, self.solver_var(rhs))

        # sum or weighted sum
        if isinstance(lhs, Operator):
            if lhs.name == 'sum':
                return self.chc_model.sum(self.solver_vars(lhs.args), op, self.solver_var(rhs))
            elif lhs.name == "sub":
                a, b = self.solver_vars(lhs.args)
                return self.chc_model.arithm(a, "-", b, op, self.solver_var(rhs))
            elif lhs.name == 'wsum':
                w = [int(wght) for wght in lhs.args[0]]
                x = self.solver_vars(lhs.args[1])
                return self.chc_model.scalar(x, w, op, self.solver_var(rhs))

        raise NotImplementedError("Choco: Not a known supported numexpr {}".format(cpm_expr))

    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the 'Adding a new solver' docs on readthedocs for more information.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """

        cpm_cons = toplevel_list(cpm_expr)
        supported = {"min", "max", "abs", "count", "element", "alldifferent", "alldifferent_except0", "allequal",
                     "table", "InDomain", "cumulative", "circuit", "gcc", "inverse"}
        supported_reified = {"alldifferent", "alldifferent_except0", "allequal",
                     "table", "InDomain", "cumulative", "circuit", "gcc", "inverse"}
        cpm_cons = decompose_in_tree(cpm_cons, supported, supported_reified)
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # support >, <, !=
        cpm_cons = canonical_comparison(cpm_cons)
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(["sum", "wsum", "alldifferent", "alldifferent_except0", "allequal",
                     "table", "InDomain", "cumulative", "circuit", "gcc", "inverse"]))  # constraints that support reification
        cpm_cons = only_bv_reifies(cpm_cons)

        return cpm_cons

    def __add__(self, cpm_expr):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: self
        """
        # add new user vars to the set
        get_variables(cpm_expr, collect=self.user_vars)

        # transform and post the constraints
        for con in self.transform(cpm_expr):
            c = self._get_constraint(con)
            if c is not None:
                c.post()

        return self

    def _get_constraint(self, cpm_expr):
        """
        Get a solver's constraint by a supported CPMpy constraint

        :param cpm_expr: CPMpy expression
        :type cpm_expr: Expression

        """
        import pychoco as chc
        from pychoco.variables.intvar import IntVar

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        if isinstance(cpm_expr, Operator):
            # 'and'/n, 'or'/n, '->'/2
            if cpm_expr.name == 'and':
                return self.chc_model.and_(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == 'or':
                return self.chc_model.or_(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == '->':
                lhs = self.solver_var(cpm_expr.args[0]) # should always be boolvar due to only_bv_reifies
                # right hand side
                if isinstance(cpm_expr.args[1], _BoolVarImpl):
                    # bv -> bv
                    # PyChoco does not have "implies" constraint
                    return self.chc_model.or_([~lhs, self.solver_var(cpm_expr.args[1])])
                else:
                    # bv -> boolexpr
                    # the `reify_rewrite()` transformation ensures that only reifiable rhs remain here
                    if cpm_expr.args[1].name == 'not':
                        bv = self._get_constraint(cpm_expr.args[1].args[0]).reify()
                        return self.chc_model.or_([~lhs, ~bv])
                    else:
                        bv = self._get_constraint(cpm_expr.args[1]).reify()
                        return self.chc_model.or_([~lhs, bv])
            else:
                raise NotImplementedError("Not a known supported Choco Operator '{}' {}".format(
                    cpm_expr.name, cpm_expr))

        # Comparisons: both numeric and boolean ones
        # numexpr `comp` bvar|const
        # Choco accepts only int32, not int64
        elif isinstance(cpm_expr, Comparison):
            for i in range(len(cpm_expr.args)):
                if isinstance(cpm_expr.args[i], np.integer):
                    cpm_expr.args[i] = int(cpm_expr.args[i])
            lhs = cpm_expr.args[0]
            rhs = cpm_expr.args[1]

            if lhs.is_bool() and rhs.is_bool(): #boolean equality -- Reification
                if isinstance(rhs, _NumVarImpl):
                    return self.chc_model.all_equal(self.solver_vars([lhs, rhs]))
                else:
                    bv = self._get_constraint(rhs).reify()
                    return self.chc_model.all_equal([self.solver_var(lhs), bv])
            elif isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and (
                    lhs.name == 'sum' or lhs.name == 'wsum' or lhs.name == "sub")):
                # a BoundedLinearExpression LHS, special case, like in objective
                chc_numexpr = self._make_numexpr(cpm_expr)
                return chc_numexpr
            elif cpm_expr.name == '==':
                chcrhs = self.solver_var(rhs)
                # NumExpr == IV, supported by Choco (thanks to `only_numexpr_equality()` transformation)
                if lhs.name == 'min':
                    if isinstance(rhs, int):     # Choco does not accept an int in rhs
                        chcrhs = self.chc_model.intvar(rhs, rhs)  # convert to "variable"
                    elif not isinstance(rhs, _NumVarImpl):
                        raise Exception(f"Choco cannot accept min operation equal to: {rhs}")
                    return self.chc_model.min(chcrhs, self.solver_vars(lhs.args))
                elif lhs.name == 'max':
                    if isinstance(rhs, int):     # Choco does not accept an int in rhs
                        chcrhs = self.chc_model.intvar(rhs, rhs)  # convert to "variable"
                    elif not isinstance(rhs, _NumVarImpl):
                        raise Exception(f"Choco cannot accept max operation equal to: {rhs}")
                    return self.chc_model.max(chcrhs, self.solver_vars(lhs.args))
                elif lhs.name == 'abs':
                    if isinstance(rhs, int):     # Choco does not accept an int in rhs
                        chcrhs = self.chc_model.intvar(rhs, rhs)  # convert to "variable"
                    elif not isinstance(rhs, _NumVarImpl):
                        raise Exception(f"Choco cannot accept absolute operation equal to: {rhs}")
                    return self.chc_model.absolute(chcrhs, self.solver_var(lhs.args[0]))
                elif lhs.name == 'count':
                    arr, val = self.solver_vars(lhs)
                    return self.chc_model.count(val, arr, chcrhs)
                elif lhs.name == 'mul':
                    return self.chc_model.times(self.solver_vars(lhs.args[0]), self.solver_vars(lhs.args[1]),
                                                chcrhs)
                elif lhs.name == 'div':
                    # Choco needs divisor to be a variable
                    if isinstance(lhs.args[1], int):
                        divisor = self.chc_model.intvar(lhs.args[1], lhs.args[1])  # convert to "variable"
                    elif isinstance(lhs.args[1], _NumVarImpl):
                        divisor = self.solver_var(lhs.args[1])  # use variable
                    else:
                        raise Exception(f"Cannot accept division with the divisor being: {lhs.args[1]}")
                    # Choco needs result to be a variable
                    if isinstance(rhs, int):
                        result = self.chc_model.intvar(rhs, rhs)  # convert to "variable"
                    elif isinstance(rhs, _NumVarImpl):
                        result = self.solver_var(rhs)  # use variable
                    else:
                        raise Exception(f"Cannot accept division with the result being: {rhs}")
                    return self.chc_model.div(self.solver_var(lhs.args[0]), divisor, result)
                elif lhs.name == 'element':
                    if isinstance(rhs, int):
                        result = self.chc_model.intvar(rhs, rhs)  # convert to "variable"
                    elif isinstance(rhs, _NumVarImpl):
                        result = self.solver_var(rhs)  # use variable
                    else:
                        raise Exception(f"Cannot accept the right hand side of the element constraint being: {rhs}")
                    return self.chc_model.element(result, self.solver_vars(lhs.args[0]),
                                                  self.solver_var(lhs.args[1]))
                elif lhs.name == 'mod':
                    divisor = lhs.args[1]
                    if not isinstance(rhs, _NumVarImpl): # if divisor is variable then result must be variable too
                        rhs = self.chc_model.intvar(rhs, rhs)  # convert to "variable"
                    else:
                        rhs = self.solver_vars(rhs)  # get choco variable
                    return self.chc_model.mod(self.solver_var(lhs.args[0]), self.solver_var(divisor), rhs)
                elif lhs.name == 'pow':
                    if isinstance(rhs, int):     # Choco does not accept an int in rhs
                        chcrhs = self.chc_model.intvar(rhs, rhs)  # convert to "variable"
                    elif not isinstance(rhs, _NumVarImpl):
                        raise Exception(f"Choco cannot accept power operation equal to: {rhs}")
                    return self.chc_model.pow(self.solver_vars(lhs.args[0]), self.solver_vars(lhs.args[1]),
                                              chcrhs)
            raise NotImplementedError(
                "Not a known supported Choco left-hand-side '{}' {}".format(lhs.name, cpm_expr))

        # base (Boolean) global constraints
        elif isinstance(cpm_expr, GlobalConstraint):

            if cpm_expr.name == 'alldifferent':
                vars = self.solver_vars(cpm_expr.args)
                for i in range(len(vars)):
                    if isinstance(vars[i], int):
                        vars[i] = self.chc_model.intvar(vars[i], vars[i])  # convert to "variable"
                    elif not isinstance(vars[i], IntVar):
                        raise Exception(f"Choco cannot accept alldifferent with: {vars[i]}")
                return self.chc_model.all_different(vars)
            elif cpm_expr.name == 'alldifferent_except0':
                vars = self.solver_vars(cpm_expr.args)
                for i in range(len(vars)):
                    if isinstance(vars[i], int):
                        vars[i] = self.chc_model.intvar(vars[i], vars[i])  # convert to "variable"
                    elif not isinstance(vars[i], IntVar):
                        raise Exception(f"Choco cannot accept alldifferent_except0 with: {vars[i]}")
                return self.chc_model.all_different_except_0(vars)
            elif cpm_expr.name == 'allequal':
                vars = self.solver_vars(cpm_expr.args)
                for i in range(len(vars)):
                    if isinstance(vars[i], int):
                        vars[i] = self.chc_model.intvar(vars[i], vars[i])  # convert to "variable"
                    elif not isinstance(vars[i], IntVar):
                        raise Exception(f"Choco cannot accept allequal with: {vars[i]}")
                return self.chc_model.all_equal(vars)
            elif cpm_expr.name == 'table':
                assert (len(cpm_expr.args) == 2)  # args = [array, table]
                array, table = self.solver_vars(cpm_expr.args)
                return self.chc_model.table(array, table)
            elif cpm_expr.name == 'InDomain':
                assert (len(cpm_expr.args) == 2)  # args = [array, table]
                expr, table = self.solver_vars(cpm_expr.args)
                return self.chc_model.member(expr, table)
            elif cpm_expr.name == "cumulative":
                start, dur, end, demand, cap = self.solver_vars(cpm_expr.args)
                # Everything given to cumulative in Choco needs to be a variable.
                # Convert demands to variables
                if is_num(demand):  # Create list for demand per task
                    demand = [demand] * len(start)
                if isinstance(demand, _NumVarImpl):
                    demand = self.solver_vars(demand)
                else:
                    demand = [self.chc_model.intvar(d, d) for d in demand]  # Create variables for demand
                # Create task variables. Choco can create them only one by one
                tasks = [self.chc_model.task(s, d, e) for s, d, e in zip(start, dur, end)]
                # Convert capacity to variable
                # Choco needs result to be a variable
                if isinstance(cap, int):
                    capacity = self.chc_model.intvar(cap, cap)  # convert to "variable"
                elif isinstance(cap, IntVar):
                    capacity = self.solver_var(cap)  # use variable
                else:
                    raise Exception(f"Choco cannot accept cumulative with the capacity being: {cap}")
                return self.chc_model.cumulative(tasks, demand, capacity)
            elif cpm_expr.name == "circuit":
                return self.chc_model.circuit(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == "gcc":
                vars, vals, occ = self.solver_vars(cpm_expr.args)
                for i in range(len(occ)):
                    if isinstance(occ[i], int):
                        occ[i] = self.chc_model.intvar(occ[i], occ[i])  # convert to "variable"
                    elif isinstance(occ[i], IntVar):
                        occ[i] = occ[i]  # use variable
                    else:
                        raise Exception(f"Choco cannot accept gcc including the following in the occurrences: {occ[i]}")
                return self.chc_model.global_cardinality(vars, vals, occ)
            elif cpm_expr.name == 'inverse':
                assert len(cpm_expr.args) == 2, "inverse() expects two args: fwd, rev"
                fwd, rev = self.solver_vars(cpm_expr.args)
                return self.chc_model.inverse_channeling(fwd, rev)
            else:
                raise NotImplementedError(
                    f"Unknown global constraint {cpm_expr}, should be decomposed! If you reach this, please report on github.")

        # unlikely base case: Boolean variable
        elif isinstance(cpm_expr, _BoolVarImpl):
            return self.chc_model.and_([self.solver_var(cpm_expr)])

        # unlikely base case: True or False
        elif isinstance(cpm_expr, BoolVal):
            # Choco does not allow to post True or False. Post "certainly True or False" constraints instead
            if cpm_expr.args[0] is True:
                return self.chc_model.arithm(self.helper_var, ">=", 0)
            else:
                return self.chc_model.arithm(self.helper_var, "<", 0)

        # a direct constraint, pass to solver
        elif isinstance(cpm_expr, DirectConstraint):
            c = cpm_expr.callSolver(self, self.chc_model)
            return c

        # else
        raise NotImplementedError(cpm_expr)  # if you reach this... please report on github