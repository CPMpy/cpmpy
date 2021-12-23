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
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl
from ..expressions.utils import is_any_list
from ..transformations.flatten_model import flatten_constraint
from ..transformations.get_variables import get_variables
from ..transformations.linearize import no_global_constraints, linearize_constraint


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


    def __init__(self, cpm_model=None, solver=None, name="gurobi"):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: a CPMpy Model()
        """
        if not self.supported():
            raise Exception("CPM_gurobi: Install the python package 'gurobipy' and make sure your licence is activated!")
        import gurobipy as gp

        # initialise the native solver object
        self.gbi_model = gp.Model()
        self._objective_value = None

        # initialise everything else and post the constraints/objective
        # it is sufficient to implement __add__() and minimize/maximize() below
        super().__init__(cpm_model, solver, name)

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

        cpm_cons = no_global_constraints(cpm_con)
        cpm_cons = flatten_constraint(cpm_cons)
        cpm_cons = linearize_constraint(cpm_cons)

        for con in cpm_cons:
            self._post_constraint(con)

        return self

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
            self.gbi_model.setParam(param, val)

        _ = self.gbi_model.optimize(callback=solution_callback)
        my_status = self.gbi_model.Status

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.gbi_model.runtime


        # translate exit status
        if my_status == GRB.OPTIMAL and self._objective_value is None:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status == GRB.OPTIMAL and self._objective_value is not None:
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        elif my_status == GRB.INFEASIBLE:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        else:  # another?
            raise NotImplementedError(f"Translation of gurobi status {my_status} to CPMpy status not implemented") # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user vars only)
        if has_sol:
            # fill in variable values
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
            return -self.solver_var(cpm_var._bv)

        # create if it does not exit
        if not cpm_var in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.gbi_model.addVar(vtype=GRB.BINARY, name=cpm_var.name)
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.gbi_model.addVar(cpm_var.lb, cpm_var.ub, vtype=GRB.INTEGER, name=str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar
            # Update model to make new vars visible for constraints
            self.gbi_model.update()

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
        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr)
        self += flat_cons # add potentially created constraints

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        TEMPLATEpy.Minimize(obj)

    def maximize(self, expr):
        """
            Maximize the given objective function

            `maximize()` can be called multiple times, only the last one is used

            (technical side note: any constraints created during conversion of the objective
            are premanently posted to the solver)
        """
        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr)
        self += flat_cons # add potentially created constraints

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        TEMPLATEpy.Maximize(obj)

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function
        """
        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl): # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # if the solver only supports a decision variable as argument to its minimize/maximize then
        # well, smth generic like
        #(obj_var, obj_cons) = get_or_make_var(cpm_expr)
        #self += obj_cons
        #return self.solver_var(obj_var)

        # else if the solver support e.g. a linear expression as objective, built it here
        # something like
        #if isinstance(cpm_expr, Operator):
        #    args = self.solver_vars(cpm_expr.args) # TODO: soon
        #    if cpm_expr.name == 'sum':
        #        return sum(args) # if TEMPLATEpy supports this

        raise NotImplementedError("TEMPLATE: Not a know supported numexpr {}".format(cpm_expr))


    def _post_constraint(self, cpm_expr):
        """
            Post a primitive CPMpy constraint to the native solver API

            What 'primitive' means depends on the solver capabilities,
            more specifically on the transformations applied in `__add__()`

            Solvers do not need to support all constraints.
        """
        from gurobipy import GRB
        import gurobipy as gp

        native_operators = {"sum", "wsum", "sub", "mul", "div", "->"}
        native_comps = {"<=", ">=", "=="}



        #Base case
        if isinstance(cpm_expr, _BoolVarImpl):
            self.gbi_model.addConstr(self.solver_var(cpm_expr) >= 1)


        #Comparisons
        elif isinstance(cpm_expr, Comparison) and cpm_expr.name in native_comps:
            # Native mapping to gurobi API
            lhs, rhs = cpm_expr.args
            sense = cpm_expr.name[0]
            if isinstance(lhs, Operator):
                if lhs.name in native_operators:
                    lvars = [self.solver_var(var) for var in lhs.args]
                    if lhs.name == "sum":
                        self.gbi_model.addLConstr(gp.quicksum(lvars), sense, self.solver_var(rhs))
                    if lhs.name == "wsum":
                        self.gbi_model.addLConstr(gp.LinExpr(lhs.args[0], lhs.args[1]), sense, self.solver_var(rhs))
                    if lhs.name == "sub":
                        self.gbi_model.addLconstr(lvars[0] - lvars[1], sense, self.solver_var(rhs))
                    if lhs.name == "mul":
                        self.gbi_model.addQConstr(np.prod(lvars), sense, self.solver_var(rhs))
                    if lhs.name == "div":
                        if isinstance(lhs.args[1], _NumVarImpl):
                            raise NotImplementedError("Gurobi does not support division by an variable. If you need this, please report on github.")
                        self.gbi_model.addLConstr(lvars[0] / lvars[1], sense, self.solver_var(rhs))
                else:
                    raise NotImplementedError(f"Cannot post constraint {cpm_expr} to gurobi")

            else:
                # Add lhs >=< rhs to model
                self.gbi_model.addLConstr(self.solver_var(lhs), sense, self.solver_var(rhs))



        #Operators
        elif isinstance(cpm_expr, Operator) and cpm_expr.name  in native_operators:
            args = [self.solver_var(var) for var in cpm_expr.args]

            raise NotImplementedError(f"TODO: implement operators, raised by adding constaint {cpm_expr} to the model")

        else:
            raise NotImplementedError(f"Cannot post constraint {cpm_expr} to gurobi optimizer")
