

from typing import Optional

import cuopt

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import *
from ..expressions.utils import argvals, argval
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.globalconstraints import DirectConstraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, only_positive_bv, only_positive_bv_wsum
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from ..transformations.safening import no_partial_functions

from .solver_interface import SolverInterface, SolverStatus, ExitStatus


class CPM_cuopt(SolverInterface):
    
    @staticmethod
    def supported():
        return True

    @staticmethod
    def installed():
        return True

    @staticmethod
    def version() -> Optional[str]:
        return "test"

    def __init__(self, cpm_model=None, subsolver=None):
        import cuopt

        from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
        from cuopt.linear_programming.solver_settings import SolverSettings
        
        self.cuopt_model = Problem("CPMpy model")

        super().__init__(name="cuopt", cpm_model=cpm_model)

    @property
    def native_model(self):
        return self.cuopt_model

    def solve(self, time_limit=None, **kwargs):
        from cuopt.linear_programming.solver_settings import SolverSettings
        from cuopt.linear_programming.solution import Solution
        # from cuopt.linear_programming.solver 


        # ensure all user vars are known to solver
        self.solver_vars(list(self.user_vars))

        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")

        # set additional keyword arguments  
        for (kw, val) in kwargs.items():
            setattr(self.ort_solver.parameters, kw, val)
        
        # Configure solver settings
        settings = SolverSettings()
        if time_limit is not None:
            settings.set_parameter("time_limit", time_limit)

        # Solve the problem
        self.cuopt_model.solve(settings)

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.cuopt_model.SolveTime
        cuopt_status = self.cuopt_model.Status.name

        if cuopt_status == "Optimal":
            if self.has_objective():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif cuopt_status == "FeasibleFound":
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif cuopt_status == "Infeasible":
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif cuopt_status == "NoTermination":
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:
            raise NotImplementedError(
                f"Translation of cpuopt status {cuopt_status} to CPMpy status not implemented")  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                try:
                    cpm_var._value = int(self.solver_var(cpm_var).getValue())
                    if isinstance(cpm_var, _BoolVarImpl):
                        cpm_var._value = bool(cpm_var._value) # ort value is always an int
                except IndexError:
                    raise ValueError(f"Var {cpm_var} is unknown to the cuOpt solver, this is unexpected - "
                                     f"please report on github...")
            # translate objective
            if self.has_objective():
                cuopt_obj_val = self.cuopt_model.ObjValue
                if round(cuopt_obj_val) == cuopt_obj_val: # it is an integer?
                    self.objective_value_ = int(cuopt_obj_val)  # ensure it is an integer
                else: # can happen when using floats as coeff in objective
                    self.objective_value_ = float(cuopt_obj_val)
        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None
        return has_sol

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        pass

    def solver_var(self, cpm_var):
        from cuopt.linear_programming.problem import INTEGER

        if is_num(cpm_var): # shortcut, eases posting constraints
            return cpm_var
        
        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            raise Exception("Negative literals should not be part of any equation. "
                            "See /transformations/linearize for more details")

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, (_BoolVarImpl, _IntVarImpl)):
                revar = self.cuopt_model.addVariable(lb=cpm_var.lb, ub=cpm_var.ub, vtype=INTEGER, name=str(cpm_var))
                # print(revar)
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        return self._varmap[cpm_var]

    def objective(self, expr, minimize):
        from cuopt.linear_programming.problem import MAXIMIZE, MINIMIZE

        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr)
        flat_obj = only_positive_bv_wsum(flat_obj)  # remove negboolviews
        self += flat_cons  # add potentially created constraints
        get_variables(flat_obj, collect=self.user_vars)  # add objvars to vars

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        # Set objective
        self.cuopt_model.setObjective(obj, sense=MINIMIZE if minimize else MAXIMIZE)

        
    def has_objective(self):
        return self.cuopt_model.getObjective() is not None

    def transform(self, cpm_expr):
        # apply transformations, then post internally
        # expressions have to be linearized to fit in MIP model. See /transformations/linearize
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons)  # linearize expects safe exprs
        supported = {} # alldiff has a specialized MIP decomp in linearize
        cpm_cons = decompose_in_tree(cpm_cons, supported, csemap=self._csemap)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset([]), csemap=self._csemap)  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]), csemap=self._csemap)  # supports >, <, !=
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)  # anything that can create full reif should go above...
        # gurobi does not round towards zero, so no 'div' in supported set: https://github.com/CPMpy/cpmpy/pull/593#issuecomment-2786707188
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum","sub", "mul"}), csemap=self._csemap)  # the core of the MIP-linearization
        cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)  # after linearization, rewrite ~bv into 1-bv
        return cpm_cons

    def add(self, cpm_expr):
        

        # add new user vars to the set
        get_variables(cpm_expr, collect=self.user_vars)


        # transform and post the constraints
        for cpm_expr in self.transform(cpm_expr):

            # Comparisons: only numeric ones as 'only_implies()' has removed the '==' reification for Boolean expressions
            # numexpr `comp` bvar|const
            if isinstance(cpm_expr, Comparison):
                lhs, rhs = cpm_expr.args
                cuopt_rhs = self.solver_var(rhs)

                # Thanks to `only_numexpr_equality()` only supported comparisons should remain
                if cpm_expr.name == '<=':
                    self.cuopt_model.addConstraint(self._make_numexpr(lhs) <= rhs)
                elif cpm_expr.name == '>=':
                    print(self._make_numexpr(lhs) >= rhs)
                    self.cuopt_model.addConstraint(self._make_numexpr(lhs) >= rhs)
                elif cpm_expr.name == '==':
                    self.cuopt_model.addConstraint(self._make_numexpr(lhs) == rhs)

                    # if isinstance(lhs, _NumVarImpl) \
                    #         or (isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum' or lhs.name == "sub")):
                    #     # a BoundedLinearExpression LHS, special case, like in objective
                    #     cuopt_lhs = self._make_numexpr(lhs)
                    #     self.cuopt_model.addConstraint()
                    #     self.grb_model.addLConstr(cuopt_lhs == cuopt_rhs)

                    # elif lhs.name == 'mul':
                    #     assert len(lhs.args) == 2, "Gurobi only supports multiplication with 2 variables"
                    #     a, b = self.solver_vars(lhs.args)
                    #     self.grb_model.setParam("NonConvex", 2)
                    #     self.grb_model.addConstr(a * b == grbrhs)

                    # elif lhs.name == 'div':
                    #     if not is_num(lhs.args[1]):
                    #         raise NotSupportedError(f"Gurobi only supports division by constants, but got {lhs.args[1]}")
                    #     a, b = self.solver_vars(lhs.args)
                    #     self.grb_model.addLConstr(a / b, GRB.EQUAL, grbrhs)

                    # else:
                    #     # General constraints
                    #     # grbrhs should be a variable for gurobi in the subsequent, fake it
                    #     if is_num(grbrhs):
                    #         grbrhs = self.solver_var(intvar(lb=grbrhs, ub=grbrhs))

                    #     if lhs.name == 'min':
                    #         self.grb_model.addGenConstrMin(grbrhs, self.solver_vars(lhs.args))
                    #     elif lhs.name == 'max':
                    #         self.grb_model.addGenConstrMax(grbrhs, self.solver_vars(lhs.args))
                    #     elif lhs.name == 'abs':
                    #         self.grb_model.addGenConstrAbs(grbrhs, self.solver_var(lhs.args[0]))
                    #     elif lhs.name == 'pow':
                    #         x, a = self.solver_vars(lhs.args)
                    #         self.grb_model.addGenConstrPow(x, grbrhs, a)
                    #     else:
                    #         raise NotImplementedError(
                    #         "Not a known supported gurobi comparison '{}' {}".format(lhs.name, cpm_expr))
                else:
                    raise NotImplementedError(
                    "Not a known supported gurobi comparison '{}' {}".format(lhs.name, cpm_expr))

            elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":

                
                
                # Indicator constraints
                # Take form bvar -> sum(x,y,z) >= rvar
                cond, sub_expr = cpm_expr.args

                assert isinstance(cond, _BoolVarImpl), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
                assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"
                # if isinstance(cond, NegBoolView):
                #     cond, bool_val = self.solver_var(cond._bv), False
                # else:
                #     cond, bool_val = self.solver_var(cond), True

                lhs, rhs = sub_expr.args
                if isinstance(lhs, _NumVarImpl) or lhs.name == "sum" or lhs.name == "wsum":
                    lin_expr = self._make_numexpr(lhs)
                else:
                    raise Exception(f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}")
                
                if sub_expr.name == "<=":
                    lub, llb = get_bounds(lhs)
                    rub, rlb = get_bounds(rhs)
                    M = (lub - rlb)
                    y = cp.boolvar()
                    # self.cuopt_model.addConstraint(self._make_numexpr(lhs - y*M) <= rhs)
                    # self.cuopt_model.addConstraint(self._make_numexpr((-1*cond._bv if isinstance(cond, NegBoolView) else cond) + y) <= 1)
                    self.cuopt_model.addConstraint(lin_expr + self._make_numexpr(- rhs - M + cond*M) <= 0)
                elif sub_expr.name == ">=":
                    lub, llb = get_bounds(lhs)
                    rub, rlb = get_bounds(rhs)
                    M = (rub - llb)
                    y = cp.boolvar()
                    print(" -----> ", self._make_numexpr(rhs - M + cond*M) - lin_expr)
                    # self.cuopt_model.addConstraint(self._make_numexpr(lhs + y*M) >= rhs)
                    # self.cuopt_model.addConstraint(self._make_numexpr((-1*cond._bv if isinstance(cond, NegBoolView) else cond) + y) <= 1)
                    self.cuopt_model.addConstraint( self._make_numexpr(rhs - M + cond*M) - lin_expr <= 0)
                # elif sub_expr.name == "==":
                #     self.grb_model.addGenConstrIndicator(cond, bool_val, lin_expr, GRB.EQUAL, self.solver_var(rhs))
                else:
                    raise Exception(f"Unknown linear expression {sub_expr} name")

            # True or False
            elif isinstance(cpm_expr, BoolVal):
                self.cuopt_model.addConstraint(self.solver_var(cp.boolvar()) >= 2)

            # a direct constraint, pass to solver
            elif isinstance(cpm_expr, DirectConstraint):
                cpm_expr.callSolver(self, self.cuopt_model)

            else:
                raise NotImplementedError(cpm_expr)  # if you reach this... please report on github

        return self

    __add__ = add

    def _make_numexpr(self, cpm_expr):
    
        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # sum
        if cpm_expr.name == "sum":
            return sum(self.solver_vars(cpm_expr.args))
        if cpm_expr.name == "sub":
            a,b = self.solver_vars(cpm_expr.args)
            return a - b
        # wsum
        if cpm_expr.name == "wsum":
            return sum(-w * self.solver_var(var._bv) if isinstance(var, NegBoolView) else w * self.solver_var(var) for w, var in zip(*cpm_expr.args))

        raise NotImplementedError("cuopt: Not a known supported numexpr {}".format(cpm_expr))

        
