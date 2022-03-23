#!/usr/bin/env python
"""
    Interface to the python 'gurobi' package

    Requires that the 'gurobipy' python package is installed:

        $ pip install gurobipy
    
    as well as the Gurobi bundled binary packages, downloadable from:
    https://www.gurobi.com/
    
    In contrast to other solvers in this package, Gurobi is not free to use and requires an active licence
    You can read more about available licences at https://www.gurobi.com/downloads/

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
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..transformations.flatten_model import flatten_constraint, flatten_objective, get_or_make_var
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, only_positive_bv
from ..transformations.reification import only_bv_implies

try:
    import gurobipy as gp
    GRB_env = gp.Env()
    GRB_env.setParam("OutputFlag",0)
    GRB_env.start()
except ImportError as e:
    pass


class CPM_gurobi(SolverInterface):
    """
    Interface to Gurobi's API

    Requires that the 'gurobipy' python package is installed:
    $ pip install gurobipy

    See detailed installation instructions at:
    https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-

    Creates the following attributes:
    user_vars: set(), variables in the original (non-transformed) model,
                    for reverse mapping the values after `solve()`
    cpm_status: SolverStatus(), the CPMpy status after a `solve()`
    tpl_model: object, TEMPLATE's model object
    _varmap: dict(), maps cpmpy variables to native solver variables
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import gurobipy as gp
            from datetime import datetime
            valid_until = gp.Model().LicenseExpiration
            today = datetime.today()
            return today.year * 1e4 + today.month * 1e2 + today.day <= valid_until
        except ImportError as e:
            return False

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: a CPMpy Model()
        """
        if not self.supported():
            raise Exception(
                "CPM_gurobi: Install the python package 'gurobipy' and make sure your licence is activated!")
        import gurobipy as gp

        # initialise the native gurobi model object
        self.grb_model = gp.Model(env=GRB_env)

        # initialise everything else and post the constraints/objective
        # it is sufficient to implement __add__() and minimize/maximize() below
        super().__init__(name="gurobi", cpm_model=cpm_model)

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
        import gurobipy as gp
        from gurobipy import GRB

        if time_limit is not None:
            self.grb_model.setParam("TimeLimit", time_limit)

        # call the solver, with parameters
        for param, val in kwargs.items():
            self.grb_model.setParam(param, val)

        _ = self.grb_model.optimize(callback=solution_callback)
        grb_objective = self.grb_model.getObjective()

        is_optimization_problem = grb_objective.size() != 0 # TODO: check if better way to do this...

        grb_status = self.grb_model.Status

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.grb_model.runtime

        # translate exit status
        if grb_status == GRB.OPTIMAL and not is_optimization_problem:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif grb_status == GRB.OPTIMAL and is_optimization_problem:
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        elif grb_status == GRB.INFEASIBLE:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        else:  # another?
            raise NotImplementedError(
                f"Translation of gurobi status {grb_status} to CPMpy status not implemented")  # a new status type was introduced, please report on github
        # TODO: what about interrupted solves? Gurobi can return sub-optimal values too

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user vars only)
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                solver_val = self.solver_var(cpm_var).X
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = int(solver_val)
            # set _objective_value
            if is_optimization_problem:
                self.objective_value_ = grb_objective.getValue()

        return has_sol

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        from gurobipy import GRB

        if is_num(cpm_var):
            return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            raise Exception("Negative literals should not be part of any equation. See /transformations/linearize for more details")

        # create if it does not exit
        if not cpm_var in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.grb_model.addVar(vtype=GRB.BINARY, name=cpm_var.name)
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.grb_model.addVar(cpm_var.lb, cpm_var.ub, vtype=GRB.INTEGER, name=str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        # return from cache
        return self._varmap[cpm_var]

    def objective(self, expr, minimize=True):
        """
            Post the expression to optimize to the solver.

            'objective()' can be called multiple times, onlu the last one is used.

            (technical side note: any constraints created during conversion of the objective
                are premanently posted to the solver)
        """

        from gurobipy import GRB

        # make objective function non-nested
        (flat_obj, flat_cons) = (flatten_objective(expr))
        self += flat_cons  # add potentially created constraints
        self.user_vars.update(get_variables(flat_obj))

        obj = self._make_numexpr(flat_obj)
        if minimize:
            self.grb_model.setObjective(obj, sense=GRB.MINIMIZE)
        else:
            self.grb_model.setObjective(obj, sense=GRB.MAXIMIZE)

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
        # wsum
        if cpm_expr.name == "wsum":
            return gp.quicksum(w * self.solver_var(var) for w, var in zip(*cpm_expr.args))

        raise NotImplementedError("gurobi: Not a know supported numexpr {}".format(cpm_expr))

    def __add__(self, cpm_con):
        """
        Post a (list of) CPMpy constraints(=expressions) to the solver

        Note that we don't store the constraints in a cpm_model,
        we first transform the constraints into primitive constraints,
        then post those primitive constraints directly to the native solver

        :param cpm_con CPMpy constraint, or list thereof
        :type cpm_con (list of) Expression(s)
        """
        # add new user vars to the set
        self.user_vars.update(get_variables(cpm_con))

        # apply transformations, then post internally
        # expressions have to be linearized to fit in MIP model. See /transformations/linearize

        cpm_cons = flatten_constraint(cpm_con)
        cpm_cons = only_bv_implies(cpm_cons)
        cpm_cons = linearize_constraint(cpm_cons)
        cpm_cons = only_positive_bv(cpm_cons)

        for con in cpm_cons:
            self._post_constraint(con)

        return self

    def _post_constraint(self, cpm_expr):
        """
            Post a primitive CPMpy constraint to the native solver API

            What 'primitive' means depends on the solver capabilities,
            more specifically on the transformations applied in `__add__()`

            Solvers do not need to support all constraints.
        """
        from gurobipy import GRB

        # Comparisons: only numeric ones as 'only_bv_implies()' has removed the '==' reification for Boolean expressions
        # numexpr `comp` bvar|const
        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            rvar = self.solver_var(rhs)


            # TODO: this should become a transformation!!
            if cpm_expr.name != '==' and not is_num(lhs) and \
                    not isinstance(lhs, _NumVarImpl) and \
                    not lhs.name == "sum" and \
                    not lhs.name == "wsum":
                # functional globals only exist for equality in gurobi
                # example: min(x) > 10 :: min(x) == aux, aux > 10
                # create the equality and overwrite lhs with auxiliary (will handle appropriate bounds)
                (lhs, cons) = get_or_make_var(lhs)
                self += cons

            # all but '==' now only have as lhs: const|ivar|sum|wsum
            # translate ivar|sum|wsum so they can be posted directly below
            if isinstance(lhs, _NumVarImpl):
                lhs = self.solver_var(lhs) # Case can be omitted -> handled in _make_num_expr
            elif isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum'):
                # a BoundedLinearExpression LHS, special case, like in objective
                lhs = self._make_numexpr(lhs)
                # assumes that gurobi accepts sum(x) >= y without further simplification

            # post the comparison
            if cpm_expr.name == '<=':
                return self.grb_model.addLConstr(lhs, GRB.LESS_EQUAL, rvar)
            elif cpm_expr.name == '<':
                raise Exception(f"{cpm_expr} should have been linearized, see /transformations/linearize.py")
            elif cpm_expr.name == '>=':
                return self.grb_model.addLConstr(lhs, GRB.GREATER_EQUAL, rvar)
            elif cpm_expr.name == '>':
                raise Exception(f"{cpm_expr} should have been linearized, see /transformations/linearize.py")
            elif cpm_expr.name == '!=':
                raise Exception(f"{cpm_expr} should have been linearized, see /transformations/linearize.py")
            elif cpm_expr.name == '==':
                if not isinstance(lhs, Expression):
                    # base cases: const|ivar|sum|wsum with prepped lhs above
                    return self.grb_model.addLConstr(lhs, GRB.EQUAL, rvar)

                elif lhs.name == 'mul':
                    assert len(lhs.args) == 2, "Gurobi only supports multiplication with 2 variables"
                    a, b = self.solver_vars(lhs.args)
                    self.grb_model.setParam("NonConvex", 2)
                    return self.grb_model.addConstr(a * b == rvar)

                elif lhs.name == 'div':
                    if not isinstance(lhs.args[1], _NumVarImpl):
                        a, b = self.solver_vars(lhs.args)
                        return self.grb_model.addLConstr(a / b, GRB.EQUAL, rvar)
                    raise Exception("Gurobi only supports division by constants")

                # General constraints
                # rvar should be a variable, not a constant
                if not isinstance(rhs, _NumVarImpl):
                    rvar = self.solver_var(intvar(lb=rhs, ub=rhs))

                if lhs.name == "and" or lhs.name == "or":
                    raise Exception(f"{cpm_expr} should have been linearized, see /transformations/linearize.py")
                elif lhs.name == 'min':
                    return self.grb_model.addGenConstrMin(rvar, self.solver_vars(lhs.args))
                elif lhs.name == 'max':
                    return self.grb_model.addGenConstrMax(rvar, self.solver_vars(lhs.args))
                elif lhs.name == 'abs':
                    return self.grb_model.addGenConstrAbs(rvar, self.solver_var(lhs.args[0]))
                elif lhs.name == 'pow':
                    x, a = self.solver_vars(lhs.args)
                    assert a == 2, "Only support quadratic constraints"
                    assert not isinstance(a, _NumVarImpl), f"Gurobi only supports power expressions with positive exponents."
                    return self.grb_model.addGenConstrPow(x, rvar, a)

            raise NotImplementedError(
                "Not a know supported gurobi left-hand-side '{}' {}".format(lhs.name, cpm_expr))

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
                return self.grb_model.addGenConstrIndicator(cond, bool_val, lin_expr, GRB.LESS_EQUAL, self.solver_var(rhs))
            if sub_expr.name == ">=":
                return self.grb_model.addGenConstrIndicator(cond, bool_val, lin_expr, GRB.GREATER_EQUAL, self.solver_var(rhs))
            if sub_expr.name == "==":
                return self.grb_model.addGenConstrIndicator(cond, bool_val, lin_expr, GRB.EQUAL, self.solver_var(rhs))

        # Global constraints
        else:
            self += cpm_expr.decompose()
            return

        raise NotImplementedError(cpm_expr)  # if you reach this... please report on github

    def solveAll(self, display=None, time_limit=None, solution_limit=None, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            This is the generic implementation, solvers can overwrite this with
            a more efficient native implementation

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - any other keyword argument

            Returns: number of solutions found
        """

        if time_limit is not None:
            self.grb_model.setParam("TimeLimit", time_limit)
        if solution_limit is None:
            raise Exception(
                "Gurobi does not support searching for all solutions. If you really need all solutions, try setting solution limit to a large number and set time_limit to be not None.")

        # Force gurobi to keep searching in the tree for optimal solutions
        self.grb_model.setParam("PoolSearchMode", 2)
        self.grb_model.setParam("PoolSolutions", solution_limit)

        for param, val in kwargs.items():
            self.grb_model.setParam(param, val)
        # Solve the model
        self.grb_model.optimize()


        solution_count = self.grb_model.SolCount
        for i in range(solution_count):
            # Specify which solution to query
            self.grb_model.setParam("SolutionNumber", i)
            # Translate solution to variables
            for cpm_var in self.user_vars:
                solver_val = self.solver_var(cpm_var).Xn
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = int(solver_val)
            # Translate objective
            if self.grb_model.getObjective().size() != 0:                # TODO: check if better way to do this...
                self.objective_value_ = self.grb_model.getObjective().getValue()

            if display is not None:
                if isinstance(display, Expression):
                    print(display.value())
                elif isinstance(display, list):
                    print([v.value() for v in display])
                else:
                    display()  # callback

        # Reset pool search mode to default
        self.grb_model.setParam("PoolSearchMode", 0)

        return solution_count
