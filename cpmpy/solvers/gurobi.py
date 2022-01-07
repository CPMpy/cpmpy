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
from ..transformations.linearize import linearize_constraint
from ..transformations.reification import only_bv_implies


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

        # initialise the native solver object
        self.env = gp.Env()
        self.env.setParam("LogToConsole", 0)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

        self.grb_model = gp.Model(env=self.env)

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
            <Please document key solver arguments that the user might wish to change
             for example: log_output=True, var_ordering=3, num_cores=8, ...>
            <Add link to documentation of all solver parameters>
        """
        from gurobipy import GRB

        if time_limit is not None:
            raise NotImplementedError("TEMPLATE: TODO, implement time_limit")

        # call the solver, with parameters
        for param, val in kwargs.items():
            self.grb_model.setParam(param, val)

        _ = self.grb_model.optimize(callback=solution_callback)

        my_status = self.grb_model.Status

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.grb_model.runtime

        # translate exit status
        if my_status == GRB.OPTIMAL and self._objective_value is None:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status == GRB.OPTIMAL and self._objective_value is not None:
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        elif my_status == GRB.INFEASIBLE:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        else:  # another?
            raise NotImplementedError(
                f"Translation of gurobi status {my_status} to CPMpy status not implemented")  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user vars only)
        if has_sol:
            # fill in variable values
            # set _
            for cpm_var in self.user_vars:
                solver_val = self.solver_var(cpm_var).X
                if cpm_var.is_bool():
                    cpm_var._value = bool(solver_val)
                else:
                    cpm_var._value = int(solver_val)

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
            return 1-self.solver_var(cpm_var._bv)

        # create if it does not exit
        if not cpm_var in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.grb_model.addVar(vtype=GRB.BINARY, name=cpm_var.name)
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.grb_model.addVar(cpm_var.lb, cpm_var.ub, vtype=GRB.INTEGER, name=str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar
            # Update model to make new vars visible for constraints
            self.grb_model.update()

        # return from cache
        return self._varmap[cpm_var]

    # if TEMPLATE does not support objective functions, you can delete minimize()/maximize()/_make_numexpr()
    def minimize(self, expr):
        """
            Minimize the given objective function

            `minimize()` can be called multiple times, only the last one is used

            (technical side note: any constraints created during conversion of the objective
            are premanently posted to the solver)
        """
        from gurobipy import GRB

        # make objective function non-nested
        (flat_obj, flat_cons) = (flatten_objective(expr))
        self += linearize_constraint(flat_cons)  # add potentially created constraints

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        self.grb_model.setObjective(obj, sense=GRB.MINIMIZE)

    def maximize(self, expr):
        """
            Maximize the given objective function

            `maximize()` can be called multiple times, only the last one is used

            (technical side note: any constraints created during conversion of the objective
            are premanently posted to the solver)
        """
        from gurobipy import GRB

        # make objective function non-nested
        (flat_obj, flat_cons) = (flatten_objective(expr))
        self += linearize_constraint(flat_cons)  # add potentially created constraints

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        self.grb_model.setObjective(obj, sense=GRB.MAXIMIZE)

    def

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
            return gp.quicksum([self.solver_var(var) for var in cpm_expr.args])

        if cpm_expr.name == "wsum":
            weights, cpm_vars = cpm_expr.args
            gbi_vars = [self.solver_var(var) for var in cpm_vars]
            return gp.LinExpr(weights, gbi_vars)

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
        self.user_vars.update(set(get_variables(cpm_con)))

        # apply transformations, then post internally
        # XXX chose the transformations your solver needs, see cpmpy/transformations/

        cpm_cons = flatten_constraint(cpm_con)
        cpm_cons = linearize_constraint(cpm_cons)

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

        # Base case: Boolean variable
        if isinstance(cpm_expr, _BoolVarImpl):
            return self.grb_model.addLConstr(self.solver_var(cpm_expr), ">", 1)


        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        elif isinstance(cpm_expr, Operator):
            # 'and'/n, 'or'/n, 'xor'/n, '->'/2
            if cpm_expr.name in ["and", "or", "xor", "->"]:
                raise Exception(f"{cpm_expr} should have been linearized, see /transformations/linearize.py")
            else:
                raise NotImplementedError("Not a know supported ORTools Operator '{}' {}".format(
                    cpm_expr.name, cpm_expr))


        # Comparisons: only numeric ones as 'only_bv_implies()' has removed the '==' reification for Boolean expressions
        # numexpr `comp` bvar|const
        elif isinstance(cpm_expr, Comparison):
            lhs = cpm_expr.args[0]
            rvar = self.solver_var(cpm_expr.args[1])

            # TODO: this should become a transformation!!
            if cpm_expr.name != '==' and not is_num(lhs) and not isinstance(lhs, _NumVarImpl):
                # functional globals only exist for equality in ortools
                # example: min(x) > 10 :: min(x) == aux, aux > 10
                # create the equality and overwrite lhs with auxiliary (will handle appropriate bounds)
                (lhs, cons) = get_or_make_var(lhs)
                self += cons

            # all but '==' now only have as lhs: const|ivar|sum|wsum
            # translate ivar|sum|wsum so they can be posted directly below
            if isinstance(lhs, _NumVarImpl):
                lhs = self.solver_var(lhs)
            elif isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum'):
                # a BoundedLinearExpression LHS, special case, like in objective
                lhs = self._make_numexpr(lhs)
                # assumes that gurobi accepts sum(x) >= y without further simplification

            # post the comparison
            if cpm_expr.name == '<=':
                return self.grb_model.addLConstr(lhs, "<", rvar)
            elif cpm_expr.name == '<':
                raise Exception(f"{cpm_expr} should have been linearized, see /transformations/linearize.py")
            elif cpm_expr.name == '>=':
                return self.grb_model.addLConstr(lhs, ">", rvar)
            elif cpm_expr.name == '>':
                raise Exception(f"{cpm_expr} should have been linearized, see /transformations/linearize.py")
            elif cpm_expr.name == '!=':
                raise Exception(f"{cpm_expr} should have been linearized, see /transformations/linearize.py")
            elif cpm_expr.name == '==':
                if not isinstance(lhs, Expression):
                    # base cases: const|ivar|sum|wsum with prepped lhs above
                    return self.grb_model.addLConstr(lhs, "=", rvar)
                elif lhs.name == "and":
                    return self.grb_model.addGenConstrAnd(rvar, self.solver_vars(lhs.args))
                elif lhs.name == "or":
                    return self.grb_model.addGenConstrOr(rvar, self.solver_vars(lhs.args))
                elif lhs.name == 'min':
                    return self.grb_model.addGenConstrMin(rvar, self.solver_vars(lhs.args))
                elif lhs.name == 'max':
                    return self.grb_model.addGenConstrMax(rvar, self.solver_vars(lhs.args))
                elif lhs.name == 'abs':
                    # TODO put this in the correct place
                    if isinstance(cpm_expr.args[1], _NumVarImpl):
                        return self.grb_model.addGenConstrAbs(rvar, self.solver_var(lhs.args[0]))
                    # right side is a constant, not support by gurobi, so add new
                    self += abs(lhs.args[0]) == intvar(rvar,rvar)
                    return


                elif lhs.name == 'mul':
                    assert len(lhs.args) == 2, "Gurobi only supports multiplication with 2 variables"
                    a, b = self.solver_vars(lhs.args)
                    self.grb_model.setParam("NonConvex", 2)
                    return self.grb_model.addConstr(a * b == rvar)

                elif lhs.name == 'div':
                    if not isinstance(lhs.args[1], _NumVarImpl):
                        a, b = self.solver_vars(lhs.args)
                        return self.grb_model.addLConstr(a / b, "=", rvar)
                    raise Exception("Gurobi only supports division by constants")

                elif lhs.name == 'pow':
                    x, a = self.solver_vars(lhs.args)
                    assert not isinstance(a, _NumVarImpl) or a.lb >= 0, f"Gurobi only supports power expressions with positive exponents."
                    return self.grb_model.addGenConstrPow(x, rvar, a)

            raise NotImplementedError(
                        "Not a know supported gurobi left-hand-side '{}' {}".format(lhs.name, cpm_expr))

        # Global constraints
        else:
            self += cpm_expr.decompose()
            return

        raise NotImplementedError(cpm_expr)  # if you reach this... please report on github

