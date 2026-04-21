#!/usr/bin/env python
"""
    Interface to the SCIP's python "PySCIPOpt" package

    First install the ScipOptSuite on your machine, follow:
    https://scipopt.org/index.php#download

    Then install the 'pyscipopt' python package:
        $ pip install pyscipopt
    (more information on https://github.com/scipopt/PySCIPOpt)
    
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
import warnings
from typing import Optional

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import BoolVal, Comparison, Operator
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl
from ..expressions.globalconstraints import DirectConstraint, GlobalConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.utils import is_num, is_true_cst, is_false_cst
from ..transformations.comparison import only_numexpr_equality
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import decompose_linear, decompose_linear_objective, linearize_constraint, linearize_reified_variables, only_positive_bv, only_positive_bv_wsum
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_bv_reifies, only_implies, reify_rewrite
from ..transformations.safening import no_partial_functions, safen_objective


class CPM_scip(SolverInterface):
    """
    Interface to SCIP's API

    Requires that the SCIPOptSuite and 'pyscipopt' python package is installed
    See detailed installation instructions at the top of this file.

    Creates the following attributes (see parent constructor for more):
    - scip_model: object, SCIP's Model object

    Detailed documentation on the Model():
    https://scipopt.github.io/PySCIPOpt/docs/html/classpyscipopt_1_1scip_1_1Model.html

    The `DirectConstraint`, when used, calls a function on the `scip_model` object.
    """

    # Globals we keep and how they are translated in add():
    # - "xor": addConsXor();
    # - "abs": addCons(abs(x) <= k);
    # - "mul": addCons(mul == rhs).
    # No native "div": PySCIPOpt uses real division, which does not match CPMpy integer division
    # (round toward zero); same rationale as Gurobi — decompose via Division.decompose().
    supported_global_constraints = frozenset({"xor", "abs", "mul"})
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pyscipopt
            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e

    @classmethod
    def version(cls) -> Optional[str]:
        """Returns the installed version of the solver's Python API (pyscipopt)."""
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("pyscipopt")
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        if not self.supported():
            raise ModuleNotFoundError(
                "CPM_scip: Install SCIPOptSuite and the python package 'pyscipopt' to use this solver interface.")
        assert subsolver is None, "SCIP does not support subsolvers"
        import pyscipopt as scip

        self.scip_model = scip.Model()
        self.scip_model.setParam("display/verblevel", 0)  # remove solver logs from output
        self.objective_value_ = None
        super().__init__(name="scip", cpm_model=cpm_model)

    @property
    def native_model(self):
        """Returns the solver's underlying native model (SCIP Model) for direct solver access."""
        return self.scip_model

    def solve(self, time_limit=None, solution_callback=None, **kwargs):
        """
            Call the SCIP solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional). Persists across solve() calls until overridden.
            - kwargs:      any keyword argument, sets parameters of solver object. 

            Arguments correspond to solver parameters (passed via `setParams`). Due to naming `/`, you can pass these options as a dict with e.g. `solve(**{"limits/nodes": 1000, "limits/solutions": 1, "parallel/maxnthreads": 4, "display/verblevel": 4, "separating/maxrounds": 0})`. For a full list see https://www.scipopt.org/doc/html/PARAMETERS.php. Note, passing `limits/time` overrides `time_limit`.
        """
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            self.scip_model.setParam("limits/time", float(time_limit))

        if solution_callback is not None:
            raise NotSupportedError("SCIP: solution callback not (yet?) supported by CPMpy")

        # apply any solver parameters (can override time limit if e.g. "limits/time" in kwargs)
        self.scip_model.setParams(kwargs)
        self.scip_model.optimize()

        scip_status = self.scip_model.getStatus()

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.scip_model.getSolvingTime()

        # We have not implemented finding all solutions, so the "bestsollimit" status is unexpected (arguably we should set `cpm_status` to OPTIMAL in this case)
        assert scip_status != "bestsollimit", f"Unexpected status {scip_status}, this SCIP usage was not implemented"

        # translate exit status
        if scip_status == "optimal":
            if self.has_objective():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif scip_status == "infeasible":  # proven unsat
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif self.scip_model.getNSols() > 0:  # decide between unknown and feasible based on number of solutions (easier and more reliable than matching on a status)
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        else:
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        # use best solution object so values are correct (getVal can be wrong after transform)
        if has_sol:
            best_sol = self.scip_model.getBestSol()
            assert best_sol is not None, f"Due to status {scip_status}, we expected a solution from SCIP, but there was none. This is a bug, please report on GitHub."
            for cpm_var in self.user_vars:
                assert cpm_var in self._varmap, f"SCIP: The user variable {cpm_var} was never added to the variable map. This is a bug, please report on GitHub."
                scip_var = self.solver_var(cpm_var)
                solver_val = self.scip_model.getSolVal(best_sol, scip_var)
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = round(solver_val)

            if self.has_objective():
                self.objective_value_ = self.scip_model.getObjVal()
        else:
            for cpm_var in self.user_vars:
                cpm_var._value = None

            self.objective_value_ = None

        # From SCIP's Mark Turner:
        # SCIP transforms the problem that you are actual solving into something it believes
        # is easier to solve (presolve techniques etc). All solutions when found are then 
        # transformed back to the original space, because that is obviously what the user 
        # modelled and can interpret. Allowing the user to change the model during solving 
        # gets messy when balancing these things however, so it's forbidden. 
        # What self.scip_model.freeTransform() does is to remove all information from the 
        # transformed space, and just leave the original model. All solutions are kept, 
        # and the user can now change the model as they will. The downside is that potentially 
        # useful information for speeding up the next optimisation call is thrown out.
        self.scip_model.freeTransform()

        return has_sol


    def solver_var(self, cpm_var):
        if is_num(cpm_var): # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view (not supported as first-class var; use 1-bv in constraints)
        if isinstance(cpm_var, NegBoolView):
            raise NotSupportedError(
                "Negative literals should not be part of any equation. See /transformations/linearize for more details"
            )

        # create if it does not exit
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.scip_model.addVar(vtype='B', name=cpm_var.name)
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.scip_model.addVar(lb=cpm_var.lb, ub=cpm_var.ub, vtype='I', name=cpm_var.name)
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        # return from cache
        return self._varmap[cpm_var]


    def objective(self, expr, minimize=True):
        get_variables(expr, collect=self.user_vars)
        self.solver_vars(list(self.user_vars))

        obj, safe_cons = safen_objective(expr)
        obj, decomp_cons = decompose_linear_objective(
            obj,
            supported=self.supported_global_constraints,
            supported_reified=self.supported_reified_global_constraints,
            csemap=self._csemap,
        )
        obj, flat_cons = flatten_objective(obj, csemap=self._csemap)
        obj = only_positive_bv_wsum(obj)

        # transform and add constraints (via `_add_transformed_constraint` as to not pollute `user_vars`)
        for cpm_expr in self.transform(safe_cons + decomp_cons + flat_cons):
            self._add_transformed_constraint(cpm_expr)

        scip_obj = self._make_numexpr(obj)
        if minimize:
            self.scip_model.setObjective(scip_obj, sense='minimize')
        else:
            self.scip_model.setObjective(scip_obj, sense='maximize')

    def has_objective(self):
        obj = self.scip_model.getObjective()
        return obj is not None and getattr(obj, 'terms', False)  # obj could be `Expr({})`

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific numeric expression
        """
        import pyscipopt as scip

        if is_num(cpm_expr):
            return cpm_expr
        elif isinstance(cpm_expr, NegBoolView):  # negated bool (e.g. in objective): 1 - bv
            raise NotSupportedError("Negative literals should not be left as part of any equation. Please report.")
        elif isinstance(cpm_expr, _NumVarImpl):  # decision variables, check in varmap (_BoolVarImpl is subclass of _NumVarImpl)
            return self.solver_var(cpm_expr)
        elif cpm_expr.name == "sum":
            return scip.quicksum(self.solver_vars(cpm_expr.args))
        elif cpm_expr.name == "wsum":
            return scip.quicksum(w * self.solver_var(var) for w, var in zip(*cpm_expr.args))
        elif isinstance(cpm_expr, GlobalFunction):  # GlobalFunction: abs, mul (PySCIPOpt supports these in constraints/objective)
            if cpm_expr.name == "abs":
                return abs(self._make_numexpr(cpm_expr.args[0]))
            elif cpm_expr.name == "mul":
                a, b = self._make_numexpr(cpm_expr.args[0]), self._make_numexpr(cpm_expr.args[1])
                return a * b
            else:
                raise NotImplementedError("scip: Not a known supported GlobalFunction {}".format(cpm_expr))

        raise NotImplementedError("scip: Not a known supported numexpr {}".format(cpm_expr))


    def transform(self, cpm_expr):
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div", "element"})
        cpm_cons = decompose_linear(cpm_cons, supported=self.supported_global_constraints, supported_reified=self.supported_reified_global_constraints, csemap=self._csemap)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(["sum", "wsum"]), csemap=self._csemap)
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum"]), csemap=self._csemap)
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap)
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "abs", "->"}) | self.supported_global_constraints, csemap=self._csemap)
        cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)
        return cpm_cons

    def add(self, cpm_expr):
        get_variables(cpm_expr, collect=self.user_vars)
        # Ensure every user var has a solver variable (so we get values after solve even if the constraint was simplified away and the var never appears in transformed constraints)
        self.solver_vars(list(self.user_vars))

        for con in self.transform(cpm_expr):
            self._add_transformed_constraint(con)

        return self

    __add__ = add

    def _add_transformed_constraint(self, cpm_expr):
        """Add already transformed CPMpy constraints to the solver. Some constraints are further transformed in this file, such as reified linear equality constraints `b -> ... == k` into `b -> ... >= k /\ b -> ... <= k`. In this case, we recursively call this function instead of `self.add`, which avoids both the full transformation pipeline overhead and also does not pollute `user_vars` with `b`."""
        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            lhs_is_operator = isinstance(lhs, Operator)
            sciprhs = self.solver_var(rhs)

            if cpm_expr.name == '<=':
                if (lhs_is_operator and lhs.name == "sum" and all(a.is_bool() and not isinstance(a, NegBoolView) for a in lhs.args)):
                    if rhs == 1:
                        self.scip_model.addConsSOS1(self.solver_vars(lhs.args))
                    else:
                        self.scip_model.addConsCardinality(self.solver_vars(lhs.args), int(rhs))
                else:
                    sciplhs = self._make_numexpr(lhs)
                    self.scip_model.addCons(sciplhs <= sciprhs)
            elif cpm_expr.name == '>=':
                sciplhs = self._make_numexpr(lhs)
                self.scip_model.addCons(sciplhs >= sciprhs)
            elif cpm_expr.name == '==':
                sciplhs = self._make_numexpr(lhs)
                self.scip_model.addCons(sciplhs == sciprhs)
            else:
                raise NotImplementedError(
                    "Not a known supported scip comparison '{}' {}".format(cpm_expr.name, cpm_expr))

        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            cond, sub_expr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
            assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"

            lhs, rhs = sub_expr.args
            assert is_num(rhs), f"linearize should only leave constants on rhs of comparison but got {rhs}"
            assert isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name in ("sum", "wsum")), f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}"

            if sub_expr.name in ("<=", ">="):
                lin_expr = self._make_numexpr(lhs)
                if sub_expr.name == "<=":
                    scip_cons = lin_expr <= rhs
                else:
                    scip_cons = lin_expr >= rhs
                if isinstance(cond, NegBoolView):
                    self.scip_model.addConsIndicator(scip_cons, binvar=self.solver_var(cond._bv), activeone=False)
                else:
                    self.scip_model.addConsIndicator(scip_cons, binvar=self.solver_var(cond), activeone=True)
            elif sub_expr.name == "==":
                self._add_transformed_constraint(cond.implies(lhs <= rhs))
                self._add_transformed_constraint(cond.implies(lhs >= rhs))
            else:
                raise Exception(f"Unknown linear expression {sub_expr} name")

        elif isinstance(cpm_expr, BoolVal):
            if cpm_expr.args[0] is False:
                self.scip_model.addConsXor([], True)  # easiest way to post False to SCIP (e.g. 0 <= -1 is not allowed, bv <= -1 requires adding a dummy variables, ...)

        elif isinstance(cpm_expr, DirectConstraint):
            cpm_expr.callSolver(self, self.scip_model)

        elif isinstance(cpm_expr, GlobalConstraint):
            if cpm_expr.name == "xor":
                # Convert to SCIP arguments, handling constants, post `xor(args) == rhsvar` to SCIP
                scip_args = []
                rhsvar = True
                for arg in cpm_expr.args:
                    if is_false_cst(arg):
                        continue
                    elif is_true_cst(arg):
                        # note: `xor` is "parity" (i.e. it enforces an odd number of true arguments)
                        # every time we see True, we can just flip the RHS
                        rhsvar = not rhsvar
                    else:
                        scip_args.append(self.solver_var(arg))

                # post constraint (note: `addConsXor` is tested to work for empty lists)
                self.scip_model.addConsXor(scip_args, rhsvar)
            else:
                raise NotImplementedError(
                    f"SCIP does not translate global constraint '{cpm_expr.name}' natively; "
                    f"supported globals: {sorted(self.supported_global_constraints)}. "
                    "It should have been decomposed by transform(); please report if you see this."
                )

        else:
            raise NotImplementedError(cpm_expr)

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        warnings.warn("Solution enumeration is not implemented in PyScipOPT, defaulting to CPMpy's naive implementation")
        # Issues to track for future reference:
        # - https://github.com/scipopt/PySCIPOpt/issues/549 and
        # - https://github.com/scipopt/PySCIPOpt/issues/248
        return super().solveAll(display, time_limit, solution_limit, call_from_model, **kwargs)

