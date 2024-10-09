#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## gurobi.py
##
"""
    Interface to the python 'gurobi' package

    Requires that the 'gurobipy' python package is installed:

        $ pip install gurobipy
    
    
    In contrast to other solvers in this package, Gurobi is not free to use and requires an active licence
    You can read more about available licences at https://www.gurobi.com/downloads/

    Documentation of the solver's own Python API:
    https://www.gurobi.com/documentation/current/refman/py_python_api_details.html

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_gurobi

    ==============
    Module details
    ==============
"""

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import *
from ..expressions.utils import argvals
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.globalconstraints import DirectConstraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, only_positive_bv
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_implies, reify_rewrite, only_bv_reifies

try:
    import gurobipy as gp

    GRB_ENV = None
except ImportError:
    pass


class CPM_gurobi(SolverInterface):
    """
    Interface to Gurobi's API

    Requires that the 'gurobipy' python package is installed:
    $ pip install gurobipy

    See detailed installation instructions at:
    https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-

    Creates the following attributes (see parent constructor for more):
        - grb_model: object, TEMPLATE's model object

    The `DirectConstraint`, when used, calls a function on the `grb_model` object.
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import gurobipy as gp
            global GRB_ENV
            if GRB_ENV is None:
                # initialise the native gurobi model object
                GRB_ENV = gp.Env()
                GRB_ENV.setParam("OutputFlag", 0)
                GRB_ENV.start()
            return True
        except:
            return False

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: a CPMpy Model()
        - subsolver: None, not used
        """
        if not self.supported():
            raise Exception(
                "CPM_gurobi: Install the python package 'gurobipy' and make sure your licence is activated!")
        import gurobipy as gp

        # TODO: subsolver could be a GRB_ENV if a user would want to hand one over
        self.grb_model = gp.Model(env=GRB_ENV)
        self.cons = []

        # initialise everything else and post the constraints/objective
        # it is sufficient to implement __add__() and minimize/maximize() below
        super().__init__(name="gurobi", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.grb_model

    def solve(self, time_limit=None, solution_callback=None, **kwargs):
        """
            Call the gurobi solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            Examples of gurobi supported arguments include:
                - Threads : int
                - MIPFocus: int
                - ImproveStartTime : bool
                - FlowCoverCuts: int

            For a full list of gurobi parameters, please visit https://www.gurobi.com/documentation/9.5/refman/parameters.html#sec:Parameters
        """
        from gurobipy import GRB

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        if time_limit is not None:
            self.grb_model.setParam("TimeLimit", time_limit)

        # call the solver, with parameters
        for param, val in kwargs.items():
            self.grb_model.setParam(param, val)

        _ = self.grb_model.optimize(callback=solution_callback)
        grb_objective = self.grb_model.getObjective()

        grb_status = self.grb_model.Status

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.grb_model.runtime

        # translate exit status
        if grb_status == GRB.OPTIMAL and not self.has_objective():
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif grb_status == GRB.OPTIMAL and self.has_objective():
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        elif grb_status == GRB.INFEASIBLE:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif grb_status == GRB.TIME_LIMIT:
            if self.grb_model.SolCount == 0:
                # can be sat or unsat
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        else:  # another?
            raise NotImplementedError(
                f"Translation of gurobi status {grb_status} to CPMpy status not implemented")  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                solver_val = self.solver_var(cpm_var).X
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = int(solver_val)
            # set _objective_value
            if self.has_objective():
                self.objective_value_ = grb_objective.getValue()

        return has_sol

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
            raise Exception(
                "Negative literals should not be part of any equation. See /transformations/linearize for more details")

        # create if it does not exit
        if cpm_var not in self._varmap:
            from gurobipy import GRB
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.grb_model.addVar(vtype=GRB.BINARY, name=cpm_var.name)
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.grb_model.addVar(cpm_var.lb, cpm_var.ub, vtype=GRB.INTEGER, name=str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        # return from cache
        return self._varmap[cpm_var]

    def lex_solve(self, objectives, minimize=True):
        """
            Post the given set of objectives to the solver as objective to lex minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective
                are premanently posted to the solver)
        """
        from gurobipy import GRB
        if minimize:
            self.grb_model.ModelSense = GRB.MINIMIZE
        else:
            self.grb_model.ModelSense = GRB.MAXIMIZE
        for index in range(len(objectives)):
            # make objective function non-nested
            (flat_obj, flat_cons) = (flatten_objective(objectives[index]))
            self += flat_cons
            get_variables(flat_obj, collect=self.user_vars)  # add potentially created constraints
            # make objective function or variable and post
            obj = self._make_numexpr(flat_obj)
            if minimize:
                self.grb_model.setObjectiveN(obj,index,len(objectives)-index-1)
            else:
                self.grb_model.setObjective(obj,index,len(objectives)-index-1)

    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective
                are premanently posted to the solver)
        """
        from gurobipy import GRB

        # make objective function non-nested
        (flat_obj, flat_cons) = (flatten_objective(expr))
        self += flat_cons
        get_variables(flat_obj, collect=self.user_vars)  # add potentially created constraints

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        if minimize:
            self.grb_model.setObjective(obj, sense=GRB.MINIMIZE)
        else:
            self.grb_model.setObjective(obj, sense=GRB.MAXIMIZE)

    def has_objective(self):
        return self.grb_model.getObjective().size() != 0  # TODO: check if better way to do this...

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function
        """
        import gurobipy as gp

        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # sum
        if cpm_expr.name == "sum":
            return gp.quicksum(self.solver_vars(cpm_expr.args))
        if cpm_expr.name == "sub":
            a, b = self.solver_vars(cpm_expr.args)
            return a - b
        # wsum
        if cpm_expr.name == "wsum":
            return gp.quicksum(w * self.solver_var(var) for w, var in zip(*cpm_expr.args))

        raise NotImplementedError("gurobi: Not a known supported numexpr {}".format(cpm_expr))

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
        # apply transformations, then post internally
        # expressions have to be linearized to fit in MIP model. See /transformations/linearize
        cpm_cons = toplevel_list(cpm_expr)
        supported = {"min", "max", "abs", "alldifferent"}  # alldiff has a specialized MIP decomp in linearize
        cpm_cons = decompose_in_tree(cpm_cons, supported)
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_implies(cpm_cons)  # anything that can create full reif should go above...
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset(
            {"sum", "wsum", "sub", "min", "max", "mul", "abs", "pow", "div"}))  # the core of the MIP-linearization
        cpm_cons = only_positive_bv(cpm_cons)  # after linearization, rewrite ~bv into 1-bv
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
            self._post_constraint(con)

        return self

    def add_temp(self,cpm_expr):

        for con in self.transform(cpm_expr):
            self._post_constraint(con,temp=True)


    def remove_temp(self):
        [self.grb_model.remove(c) for c in self.cons]
        self.cons = []
        return self

    def _post_constraint(self, cpm_expr,temp=False):
        """
          Post a supported CPMpy constraint directly to the underlying solver's API

          What 'supported' means depends on the solver capabilities, and in effect on what transformations
          are applied in `transform()`.

      :param cpm_expr: CPMpy expression
      :type cpm_expr: Expression
      """
        from gurobipy import GRB

        # Comparisons: only numeric ones as 'only_implies()' has removed the '==' reification for Boolean expressions
        # numexpr `comp` bvar|const
        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            grbrhs = self.solver_var(rhs)

            # Thanks to `only_numexpr_equality()` only supported comparisons should remain
            if cpm_expr.name == '<=':
                grblhs = self._make_numexpr(lhs)
                c = self.grb_model.addLConstr(grblhs, GRB.LESS_EQUAL, grbrhs)
                if temp:
                    self.cons.append(c)
            elif cpm_expr.name == '>=':
                grblhs = self._make_numexpr(lhs)
                c = self.grb_model.addLConstr(grblhs, GRB.GREATER_EQUAL, grbrhs)
                if temp:
                    self.cons.append(c)
            elif cpm_expr.name == '==':
                if isinstance(lhs, _NumVarImpl) \
                        or (
                        isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum' or lhs.name == "sub")):
                    # a BoundedLinearExpression LHS, special case, like in objective
                    grblhs = self._make_numexpr(lhs)
                    c = self.grb_model.addLConstr(grblhs, GRB.EQUAL, grbrhs)
                    if temp:
                        self.cons.append(c)
                elif lhs.name == 'mul':
                    assert len(lhs.args) == 2, "Gurobi only supports multiplication with 2 variables"
                    a, b = self.solver_vars(lhs.args)
                    self.grb_model.setParam("NonConvex", 2)
                    c = self.grb_model.addConstr(a * b == grbrhs)
                    if temp:
                        self.cons.append(c)
                elif lhs.name == 'div':
                    assert is_num(lhs.args[1]), "Gurobi only supports division by constants"
                    a, b = self.solver_vars(lhs.args)
                    c = self.grb_model.addLConstr(a / b, GRB.EQUAL, grbrhs)
                    if temp:
                        self.cons.append(c)
                else:
                    # General constraints
                    # grbrhs should be a variable for gurobi in the subsequent, fake it
                    if is_num(grbrhs):
                        grbrhs = self.solver_var(intvar(lb=grbrhs, ub=grbrhs))
                    if lhs.name == 'min':
                        c = self.grb_model.addGenConstrMin(grbrhs, self.solver_vars(lhs.args))
                        if temp:
                            self.cons.append(c)
                    elif lhs.name == 'max':
                        c = self.grb_model.addGenConstrMax(grbrhs, self.solver_vars(lhs.args))
                        if temp:
                            self.cons.append(c)
                    elif lhs.name == 'abs':
                        c = self.grb_model.addGenConstrAbs(grbrhs, self.solver_var(lhs.args[0]))
                        if temp:
                            self.cons.append(c)
                    elif lhs.name == 'pow':
                        x, a = self.solver_vars(lhs.args)
                        assert a == 2, "Gurobi: 'pow', only support quadratic constraints (x**2)"
                        c = self.grb_model.addGenConstrPow(x, grbrhs, a)
                        if temp:
                            self.cons.append(c)
                    else:
                        raise NotImplementedError(
                            "Not a known supported gurobi comparison '{}' {}".format(lhs.name, original_expr))
            else:
                raise NotImplementedError(
                    "Not a known supported gurobi comparison '{}' {}".format(lhs.name, original_expr))

        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            # Indicator constraints
            # Take form bvar -> sum(x,y,z) >= rvar
            cond, sub_expr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
            assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"
            if isinstance(cond, NegBoolView):
                cond, bool_val = self.solver_var(cond._bv), False
            else:
                cond, bool_val = self.solver_var(cond), True
            lhs, rhs = sub_expr.args
            if isinstance(lhs, _NumVarImpl) or lhs.name == "sum" or lhs.name == "wsum":
                lin_expr = self._make_numexpr(lhs)
            else:
                raise Exception(f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}")
            if sub_expr.name == "<=":
                c = self.grb_model.addGenConstrIndicator(cond, bool_val, lin_expr, GRB.LESS_EQUAL, self.solver_var(rhs))
                if temp:
                    self.cons.append(c)
            elif sub_expr.name == ">=":
                c = self.grb_model.addGenConstrIndicator(cond, bool_val, lin_expr, GRB.GREATER_EQUAL, self.solver_var(rhs))
                if temp:
                    self.cons.append(c)
            elif sub_expr.name == "==":
                c = self.grb_model.addGenConstrIndicator(cond, bool_val, lin_expr, GRB.EQUAL, self.solver_var(rhs))
                if temp:
                    self.cons.append(c)
            else:
                raise Exception(f"Unknown linear expression {sub_expr} name")
        # True or False
        elif isinstance(cpm_expr, BoolVal):
            c = self.grb_model.addConstr(cpm_expr.args[0])
            if temp:
                self.cons.append(c)
        # a direct constraint, pass to solver
        elif isinstance(cpm_expr, DirectConstraint):
            cpm_expr.callSolver(self, self.grb_model)
        else:
            raise NotImplementedError(cpm_expr)  # if you reach this... please report on github
        return self

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            This is the generic implementation, solvers can overwrite this with
            a more efficient native implementation

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - call_from_model: whether the method is called from a CPMpy Model instance or not
                - any other keyword argument

            Returns: number of solutions found
        """

        if time_limit is not None:
            self.grb_model.setParam("TimeLimit", time_limit)

        if solution_limit is None:
            raise Exception(
                "Gurobi does not support searching for all solutions. If you really need all solutions, try setting solution limit to a large number")

        # Force gurobi to keep searching in the tree for optimal solutions
        sa_kwargs = {"PoolSearchMode": 2, "PoolSolutions": solution_limit}

        # solve the model
        self.solve(time_limit=time_limit, **sa_kwargs, **kwargs)

        optimal_val = None
        solution_count = self.grb_model.SolCount
        opt_sol_count = 0

        for i in range(solution_count):
            # Specify which solution to query
            self.grb_model.setParam("SolutionNumber", i)
            sol_obj_val = self.grb_model.PoolObjVal
            if optimal_val is None:
                optimal_val = sol_obj_val
            if optimal_val is not None:
                # sub-optimal solutions
                if sol_obj_val != optimal_val:
                    break
            opt_sol_count += 1

            # Translate solution to variables
            for cpm_var in self.user_vars:
                solver_val = self.solver_var(cpm_var).Xn
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = int(solver_val)

            # Translate objective
            if self.has_objective():
                self.objective_value_ = self.grb_model.PoolObjVal

            if display is not None:
                if isinstance(display, Expression):
                    print(argval(display))
                elif isinstance(display, list):
                    print(argvals(display))
                else:
                    display()  # callback

        # Reset pool search mode to default
        self.grb_model.setParam("PoolSearchMode", 0)

        return opt_sol_count
