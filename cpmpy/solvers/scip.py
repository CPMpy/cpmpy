#!/usr/bin/env python
"""
    Interface to the python 'scip' package

    Requires that the 'PySCIPOpt' python package is installed:

        $ pip install PySCIPOpt

    Scip is an open source Mixed Integer Solver. Its full documentation can be accessed here: https://www.scipopt.org/doc-8.0.1/html/

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_scip

    ==============
    Module details
    ==============
"""

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import *
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..transformations.comparison import only_numexpr_equality
from ..transformations.flatten_model import flatten_constraint, flatten_objective, get_or_make_var
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, only_positive_bv
from ..transformations.reification import only_bv_implies, reify_rewrite

class CPM_scip(SolverInterface):
    """
    Interface to scip's API

    Requires that the 'PySCIPOpt' python package is installed:
    $ conda install --channel conda-forge pyscipopt

    See detailed installation instructions at:
    https://github.com/scipopt/PySCIPOpt

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
            import pyscipopt as scp
            return True
        except:
            return False

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: a CPMpy Model()
        """
        if not self.supported():
            raise Exception(
                "CPM_scip: Install the python package 'PySCIPOpt'!")
        import pyscipopt as scp

        self.scp_model = scp.Model()
        self.scp_model.hideOutput()

        # initialise everything else and post the constraints/objective
        # it is sufficient to implement __add__() and minimize/maximize() below
        super().__init__(name="scip", cpm_model=cpm_model)

    def solve(self, time_limit=None, solution_callback=None, **kwargs):
        """
            Call the scip solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            Examples of scip supported arguments include:
                TODO: small list of parameters which might be useful

            For a full list of scip parameters, please visit https://www.scipopt.org/doc-8.0.1/html/PARAMETERS.php
        """
        import pyscipopt as scp

        if time_limit is not None:
            self.scp_model.setParam("limits/time", time_limit)

        # call the solver, with parameters
        for param, val in kwargs.items():
            self.scp_model.setParam(param, val)

        _ = self.scp_model.optimize()
        scp_objective = self.scp_model.getObjective()
        is_optimization_problem = len(scp_objective.terms) != 0 # TODO: check if better way to do this...

        scp_status = self.scp_model.getStatus()

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.scp_model.getTotalTime()

        # translate exit status
        if scp_status == 'optimal' and not is_optimization_problem:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif scp_status == 'optimal' and is_optimization_problem:
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        elif scp_status == 'infeasible':
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif scp_status == 'timelimit':
            if self.scp_model.getBestSol() is None or self.scp_model.getStage() == scp.SCIP_STAGE.PRESOLVING:
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        else:  # another?
            raise NotImplementedError(
                f"Translation of scip status {scp_status} to CPMpy status not implemented")  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user vars only)
        if has_sol:
            # fill in variable values
            scp_sol = self.scp_model.getBestSol()
            for cpm_var in self.user_vars:
                solver_val = self.scp_model.getSolVal(sol=scp_sol, expr=self.solver_var(cpm_var))
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = int(solver_val)
            # set _objective_value
            if is_optimization_problem:
                self.objective_value_ = scp_objective.getValue()

        return has_sol

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var):
            return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            raise Exception("Negative literals should not be part of any equation. See /transformations/linearize for more details")

        # create if it does not exit
        if not cpm_var in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.scp_model.addVar(vtype='B', name=str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.scp_model.addVar(lb=cpm_var.lb, ub=cpm_var.ub, vtype='I', name=str(cpm_var))
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

        # make objective function non-nested
        (flat_obj, flat_cons) = (flatten_objective(expr))
        self += flat_cons  # add potentially created constraints
        self.user_vars.update(get_variables(flat_obj))

        obj = self._make_numexpr(flat_obj)
        if minimize:
            self.scp_model.setObjective(obj, sense="minimize")
        else:
            self.scp_model.setObjective(obj, sense="maximize")

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function
        """
        import pyscipopt as scp

        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # sum
        if cpm_expr.name == "sum":
            return scp.quicksum(self.solver_vars(cpm_expr.args))
        # wsum
        if cpm_expr.name == "wsum":
            return scp.quicksum(w * self.solver_var(var) for w, var in zip(*cpm_expr.args))
        # sub
        if cpm_expr.name == "sub":
            a, b = self.solver_vars(cpm_expr.vars)
            return a - b
        # mul
        if cpm_expr.name == "mul":
            assert len(cpm_expr.args) == 2, "scip only supports multiplication with 2 variables"
            a, b = self.solver_vars(cpm_expr.args)
            return a * b
        # div
        if cpm_expr.name == "div":
            a, b = self.solver_vars(cpm_expr.args)
            return a / b
        # abs
        if cpm_expr.name == "abs":
            return abs(self.solver_var(cpm_expr.args[0]))
        # pow
        if cpm_expr.name == "pow":
            base, exp = self.solver_vars(cpm_expr.args)
            return base ** exp

        raise NotImplementedError("scip: Not a know supported numexpr {}".format(cpm_expr))

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
        cpm_cons = reify_rewrite(cpm_cons, supported={"sum", "wsum", "or", "and"})
        cpm_cons = only_bv_implies(cpm_cons, supported_eq={"or", "and"})
        cpm_cons = linearize_constraint(cpm_cons, gen_constr={"or", "and"})
        cpm_cons = only_numexpr_equality(cpm_cons)
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
        import pyscipopt as scp

        # Comparisons: only numeric ones as 'only_bv_implies()' has removed the '==' reification for Boolean expressions
        # numexpr `comp` bvar|const
        supported_num = ["sum", "wsum", "sub", "div", "mul", "abs", "pow"]

        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            scprhs = self.solver_var(rhs)

            # Thanks to `only_numexpr_equality()` only supported comparisons should remain
            if cpm_expr.name == '<=':
                scplhs = self._make_numexpr(lhs)
                return self.scp_model.addCons(scplhs <= scprhs)
            elif cpm_expr.name == '>=':
                scplhs = self._make_numexpr(lhs)
                return self.scp_model.addCons(scplhs >= scprhs)
            elif cpm_expr.name == '==':
                if isinstance(lhs, _NumVarImpl) \
                        or (isinstance(lhs, Operator) and (lhs.name in supported_num)):
                    # a BoundedLinearExpression LHS, special case, like in objective
                    scplhs = self._make_numexpr(lhs)
                    return self.scp_model.addCons(scplhs == scprhs)

                if lhs.name == "and":
                    return self.scp_model.addConsAnd(self.solver_vars(lhs.args), scprhs)
                if lhs.name == "or":
                    return self.scp_model.addConsOr(self.solver_vars(lhs.args), scprhs)

            raise NotImplementedError(
                "Not a know supported scip comparison '{}' {}".format(lhs.name, cpm_expr))

        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            # Indicator constraints
            # Take form bvar -> linexpr >=< rvar
            cond, sub_expr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
            assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"

            if isinstance(cond, NegBoolView):
                cond, bool_val = self.solver_var(cond._bv), False
            else:
                cond, bool_val = self.solver_var(cond), True

            lhs, rhs = sub_expr.args
            if isinstance(lhs, _NumVarImpl) or lhs.name in supported_num:
                lin_expr = self._make_numexpr(lhs)
            else:
                raise Exception(f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}")
            if sub_expr.name == "<=":
                return self.scp_model.addConsIndicator(lin_expr <= self.solver_var(rhs), cond, activeone=bool_val)
            if sub_expr.name == ">=":
                return self.scp_model.addConsIndicator(lin_expr >= self.solver_var(rhs), cond, activeone=bool_val)
            if sub_expr.name == "==":
                leq = lin_expr <= self.solver_var(rhs)
                geq = lin_expr >= self.solver_var(rhs)
                print(f"Adding {leq} and {geq}")
                self.scp_model.addConsIndicator(cons=lin_expr <= self.solver_var(rhs), binvar=cond, activeone=bool_val)
                return self.scp_model.addConsIndicator(cons=lin_expr >= self.solver_var(rhs), binvar=cond, activeone=bool_val)

        # Global constraints
        if cpm_expr.name == "xor":
            return self.scp_model.addConsXor(self.solver_vars(cpm_expr.args), True)
        else:
            self += cpm_expr.decompose()
            return

        raise NotImplementedError(cpm_expr)  # if you reach this... please report on github

    # def solveAll(self, display=None, time_limit=None, solution_limit=None, **kwargs):
    #     """
    #         Compute all solutions and optionally display the solutions.
    #
    #         This is the generic implementation, solvers can overwrite this with
    #         a more efficient native implementation
    #
    #         Arguments:
    #             - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
    #                     default/None: nothing displayed
    #             - time_limit: stop after this many seconds (default: None)
    #             - solution_limit: stop after this many solutions (default: None)
    #             - any other keyword argument
    #
    #         Returns: number of solutions found
    #     """
    #
    #     if time_limit is not None:
    #         self.grb_model.setParam("TimeLimit", time_limit)
    #     if solution_limit is None:
    #         raise Exception(
    #             "Gurobi does not support searching for all solutions. If you really need all solutions, try setting solution limit to a large number and set time_limit to be not None.")
    #
    #     # Force gurobi to keep searching in the tree for optimal solutions
    #     self.grb_model.setParam("PoolSearchMode", 2)
    #     self.grb_model.setParam("PoolSolutions", solution_limit)
    #
    #     for param, val in kwargs.items():
    #         self.grb_model.setParam(param, val)
    #     # Solve the model
    #     self.grb_model.optimize()
    #
    #
    #     solution_count = self.grb_model.SolCount
    #     for i in range(solution_count):
    #         # Specify which solution to query
    #         self.grb_model.setParam("SolutionNumber", i)
    #         # Translate solution to variables
    #         for cpm_var in self.user_vars:
    #             solver_val = self.solver_var(cpm_var).Xn
    #             if cpm_var.is_bool():
    #                 cpm_var._value = solver_val >= 0.5
    #             else:
    #                 cpm_var._value = int(solver_val)
    #         # Translate objective
    #         if self.grb_model.getObjective().size() != 0:                # TODO: check if better way to do this...
    #             self.objective_value_ = self.grb_model.getObjective().getValue()
    #
    #         if display is not None:
    #             if isinstance(display, Expression):
    #                 print(display.value())
    #             elif isinstance(display, list):
    #                 print([v.value() for v in display])
    #             else:
    #                 display()  # callback
    #
    #     # Reset pool search mode to default
    #     self.grb_model.setParam("PoolSearchMode", 0)
    #
    #     return solution_count
