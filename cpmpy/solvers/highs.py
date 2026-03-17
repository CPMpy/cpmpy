#!/usr/bin/env python
"""
    Interface to the HiGHS `highspy` Python API

    HiGHS is a high-performance open-source linear/mixed-integer
    programming solver.

    ============
    Installation
    ============

    Install the Python bindings from PyPI:

        $ pip install highspy

    Always use :func:`cp.SolverLookup.get("highs") <cpmpy.solvers.utils.SolverLookup.get>`
    to instantiate the solver object.

    Documentation of the solver's own Python API:
    - https://ergo-code.github.io/HiGHS/dev/interfaces/python/
"""

from typing import Optional

import time
import warnings

import numpy as np

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import *
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.globalconstraints import DirectConstraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import decompose_linear, decompose_linear_objective, linearize_constraint, linearize_reified_variables, only_positive_bv, only_positive_bv_wsum
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_bv_reifies, only_implies, reify_rewrite
from ..transformations.safening import no_partial_functions, safen_objective


class CPM_highs(SolverInterface):
    """
    Interface to HiGHS' Python API (`highspy`).

    This backend treats HiGHS as a pure LP/MIP solver without native
    indicator, SOS, or global-constraint support. Such features are
    handled by CPMpy's transformation/linearization pipeline, similar
    to the Pindakaas backend.

    Creates the following attributes (see parent constructor for more):

    - highs: object, HiGHS `Highs` instance
    """

    # HiGHS does not have native global constraints in the CPMpy sense
    supported_global_constraints = frozenset()
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        try:
            import highspy  # noqa: F401

            return True
        except Exception:
            return False

    @classmethod
    def version(cls) -> Optional[str]:
        """
        Returns the installed version of the solver's Python API (highspy).
        """
        from importlib.metadata import version, PackageNotFoundError

        try:
            return version("highspy")
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: a CPMpy Model()
        """
        if not self.supported():
            raise ModuleNotFoundError("CPM_highs: Install the python package 'highspy' to use this solver interface.")
        if subsolver is not None:
            raise NotSupportedError("HiGHS does not support subsolvers")

        import highspy

        self._inf = highspy.kHighsInf
        self.highs = highspy.Highs()
        # by default, keep HiGHS quiet
        try:
            self.highs.setOptionValue("log_to_console", False)
        except Exception:
            pass

        # track whether an objective was posted
        self._has_obj = False

        # initialise everything else and post the constraints/objective
        super().__init__(name="highs", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
        Returns the solver's underlying native model (HiGHS Highs instance).
        """
        return self.highs

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var):  # shortcut, eases posting constraints
            return cpm_var

        # negative bool views should have been eliminated by transformations
        if isinstance(cpm_var, NegBoolView):
            raise NotSupportedError(
                "Negative literals should not be part of any equation. "
                "They should have been removed by only_positive_bv()/only_positive_bv_wsum. "
                "See /transformations/linearize for more details."
            )

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                hvar = self.highs.addBinary()
            elif isinstance(cpm_var, _IntVarImpl):
                hvar = self.highs.addIntegral(lb=cpm_var.lb, ub=cpm_var.ub)
            else:
                raise NotSupportedError(f"Not a known HiGHS variable type: {cpm_var}")

            self._varmap[cpm_var] = hvar.index

        return self._varmap[cpm_var]

    def _row_from_linexpr(self, lhs):
        """
        Convert a flat linear numeric expression (var/sum/wsum)
        into HiGHS row data: (indices, values, constant).

        The constant captures any numeric terms that are not column
        variables and must be subtracted from the row bounds by the caller.
        """
        if is_num(lhs):
            return [], [], float(lhs)

        if isinstance(lhs, _NumVarImpl):
            col = self.solver_var(lhs)
            return [col], [1.0], 0.0

        if isinstance(lhs, Operator) and lhs.name in ("sum", "wsum"):
            if lhs.name == "sum":
                pairs = [(1.0, v) for v in lhs.args]
            else:
                weights, vars_ = lhs.args
                pairs = list(zip(weights, vars_))

            acc = {}
            const = 0.0
            for w, v in pairs:
                if is_num(v):
                    const += float(w) * float(v)
                else:
                    col = self.solver_var(v)
                    acc[col] = acc.get(col, 0.0) + float(w)
            idxs = [i for i in sorted(acc.keys()) if acc[i] != 0.0]
            return idxs, [acc[i] for i in idxs], const

        raise NotImplementedError(f"HiGHS: unsupported linear expression {lhs}")

    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports.

            Follows the ILP-style pipeline used by SCIP/Gurobi/CPLEX, but
            without native indicator/SOS handling; those are linearized.
        """
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div", "element"})  # linearize and decompose expects safe exprs
        cpm_cons = decompose_linear(cpm_cons, supported=self.supported_global_constraints, supported_reified=self.supported_reified_global_constraints, csemap=self._csemap)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset({"sum", "wsum", "sub"}), csemap=self._csemap)  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset({"sum", "wsum", "sub"}), csemap=self._csemap)  # supports >, <, !=
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap)
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)  # anything that can create full reif should go above...
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum"}), csemap=self._csemap)  # the core of the MIP-linearization; rewrites sub to wsum
        cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)  # after linearisation, rewrite ~bv into 1-bv
        return cpm_cons

    def add(self, cpm_expr_orig):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.
        """
        # track user vars and ensure newly seen ones have solver columns
        get_variables(cpm_expr_orig, collect=self.user_vars)

        for cpm_expr in self.transform(cpm_expr_orig):
            if isinstance(cpm_expr, Comparison):
                lhs, rhs = cpm_expr.args
                assert is_num(rhs), f"RHS of comparison should be numeric after transformations, got {rhs}"

                indices, values, const = self._row_from_linexpr(lhs)
                # effective rhs: lhs_vars + const <op> rhs  =>  lhs_vars <op> rhs - const
                bound = float(rhs) - const

                if cpm_expr.name == "<=":
                    lower = -self._inf
                    upper = bound
                elif cpm_expr.name == ">=":
                    lower = bound
                    upper = self._inf
                elif cpm_expr.name == "==":
                    lower = bound
                    upper = bound
                else:
                    raise NotSupportedError(
                        f"HiGHS: unexpected comparison operator after linearization: {cpm_expr.name}"
                    )

                indices_arr = np.array(indices, dtype=np.int32)
                values_arr = np.array(values, dtype=np.double)
                self.highs.addRow(lower, upper, len(indices), indices_arr, values_arr)

            elif isinstance(cpm_expr, BoolVal):
                if cpm_expr.args[0] is False:
                    # add an always-infeasible row: 1 <= 0
                    indices_arr = np.array([], dtype=np.int32)
                    values_arr = np.array([], dtype=np.double)
                    self.highs.addRow(1.0, 0.0, 0, indices_arr, values_arr)

            elif isinstance(cpm_expr, DirectConstraint):
                # delegate to user-provided function with native model
                cpm_expr.callSolver(self, self.highs)

            else:
                raise NotImplementedError(f"HiGHS: unsupported transformed constraint {cpm_expr}")

        return self

    __add__ = add

    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize.
            Any constraints created during conversion are permanently posted.
        """
        import highspy

        get_variables(expr, collect=self.user_vars)

        obj, safe_cons = safen_objective(expr)
        obj, decomp_cons = decompose_linear_objective(
            obj,
            supported=self.supported_global_constraints,
            supported_reified=self.supported_reified_global_constraints,
            csemap=self._csemap,
        )
        obj, flat_cons = flatten_objective(obj, csemap=self._csemap)
        obj = only_positive_bv_wsum(obj)

        self.add(safe_cons + decomp_cons + flat_cons)

        indices, coeffs, const = self._row_from_linexpr(obj)

        # reset all costs to 0 and then apply objective coefficients
        ncol = self.highs.getNumCol()
        if ncol > 0:
            all_indices = np.arange(ncol, dtype=np.int32)
            costs = np.zeros(ncol, dtype=np.double)
            for idx, c in zip(indices, coeffs):
                costs[idx] = c
            self.highs.changeColsCost(ncol, all_indices, costs)

        self.highs.changeObjectiveOffset(const)

        sense = highspy.ObjSense.kMinimize if minimize else highspy.ObjSense.kMaximize
        self.highs.changeObjectiveSense(sense)

        self._has_obj = True

    def has_objective(self):
        return self._has_obj

    def solve(self, time_limit=None, **kwargs):
        """
            Call the HiGHS solver.

            Arguments:
            - time_limit: maximum solve time in seconds (float, optional)
            - kwargs:     any keyword argument, mapped to HiGHS options via setOptionValue
        """
        import highspy

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)

        # set time limit, if any
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            self.highs.setOptionValue("time_limit", float(time_limit))

        # map additional kwargs to HiGHS options
        for key, val in kwargs.items():
            try:
                self.highs.setOptionValue(key, val)
            except Exception as e:
                warnings.warn(f"HiGHS: failed to set option '{key}' = {val!r}: {e}")

        start = time.time()
        status = self.highs.run()
        end = time.time()

        info = self.highs.getInfo()
        model_status = self.highs.getModelStatus()

        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = end - start

        # map HiGHS model status to CPMpy ExitStatus
        _has_feasible_sol = (info.primal_solution_status
                             == highspy.SolutionStatus.kSolutionStatusFeasible)

        if model_status == highspy.HighsModelStatus.kOptimal:
            if self.has_objective():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif model_status == highspy.HighsModelStatus.kInfeasible:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif model_status in (highspy.HighsModelStatus.kUnbounded,
                              highspy.HighsModelStatus.kUnboundedOrInfeasible):
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        elif model_status in (highspy.HighsModelStatus.kTimeLimit,
                              highspy.HighsModelStatus.kIterationLimit,
                              highspy.HighsModelStatus.kSolutionLimit,
                              highspy.HighsModelStatus.kObjectiveBound,
                              highspy.HighsModelStatus.kObjectiveTarget):
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE if _has_feasible_sol else ExitStatus.UNKNOWN
        elif model_status == highspy.HighsModelStatus.kSolveError:
            self.cpm_status.exitstatus = ExitStatus.ERROR
        else:
            # kUnknown, kModelEmpty, kModelError, kNotset, ...
            if _has_feasible_sol:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            elif status == highspy.HighsStatus.kError:
                self.cpm_status.exitstatus = ExitStatus.ERROR
            else:
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN

        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            solution = self.highs.getSolution()
            col_values = list(solution.col_value)
            for cpm_var in self.user_vars:
                if cpm_var not in self._varmap:
                    continue
                col_idx = self._varmap[cpm_var]
                val = col_values[col_idx]
                if cpm_var.is_bool():
                    cpm_var._value = val >= 0.5
                else:
                    cpm_var._value = int(round(val))

            if self.has_objective():
                self.objective_value_ = info.objective_function_value
        else:
            for cpm_var in self.user_vars:
                cpm_var._value = None

        return has_sol

