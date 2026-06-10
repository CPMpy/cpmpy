#!/usr/bin/env python
"""
    Interface to the HiGHS `highspy` Python API

    HiGHS is a high-performance open-source linear/mixed-integer
    programming solver.

    Always use :func:`cp.SolverLookup.get("highs") <cpmpy.solvers.utils.SolverLookup.get>`
    to instantiate the solver object.

    ============
    Installation
    ============

    Install the Python bindings from PyPI:

    .. code-block:: console

        $ pip install highspy
    
    Detailed installation instructions available at:
    https://ergo-code.github.io/HiGHS/dev/interfaces/python/

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_highs

    ==============
    Module details
    ==============
"""

from typing import Optional

import time
import warnings

import cpmpy as cp
import numpy as np
import numpy.typing as npt

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import BoolVal, Comparison, Operator
from ..expressions.utils import is_num, is_int
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.globalconstraints import DirectConstraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import decompose_linear, decompose_linear_objective, linearize_constraint, linearize_reified_variables, only_positive_bv, only_positive_bv_wsum_const
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_bv_reifies, only_implies, reify_rewrite
from ..transformations.safening import no_partial_functions, safen_objective


class CPM_highs(SolverInterface):
    """
    Interface to HiGHS' Python API (`highspy`).

    Creates the following attributes (see parent constructor for more):

    - highs: object, HiGHS `Highs` instance
    - _inf: numeric, HiGHS' infinity constant (`highspy.kHighsInf`)
    - _obj_cols: Optional[npt.NDArray[np.int32]], columns with nonzero cost in the previous objective; ``None`` means no objective posted yet

    Documentation of the solver's own Python API:
    https://ergo-code.github.io/HiGHS/dev/interfaces/python/model-py/
    """

    # HiGHS does not have native global constraints in the CPMpy sense
    supported_global_constraints = frozenset()
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported() -> bool:
        # try to import the package
        try:
            import highspy  # noqa: F401
            return True
        except ModuleNotFoundError:
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
        - subsolver: None (not supported)
        """
        if not self.supported():
            raise ModuleNotFoundError("CPM_highs: Install the python package 'highspy' to use this solver interface.")
        assert subsolver is None, "HiGHS does not support subsolvers"

        import highspy

        self.highs = highspy.Highs()
        self.highs.setOptionValue("log_to_console", False)

        self._inf = highspy.kHighsInf
        self._obj_cols = None

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
            or returns a constant if the variable is a constant
        """
        if isinstance(cpm_var, _NumVarImpl):
            name = cpm_var.name
            revar = self._varmap.get(name)
            if revar is not None:
                return revar

            # not yet created, make a new solver var
            if cpm_var.is_bool():
                # negative bool views should have been eliminated by transformations
                if isinstance(cpm_var, NegBoolView):
                    raise NotSupportedError(
                        "Negative literals should not be part of any equation. "
                        "They should have been removed by only_positive_bv()/only_positive_bv_wsum. "
                        "See /transformations/linearize for more details."
                    )
                revar = self.highs.addBinary().index
            else:
                revar = self.highs.addIntegral(lb=cpm_var.lb, ub=cpm_var.ub).index

            self._varmap[name] = revar
            return revar

        if is_int(cpm_var):  # shortcut, eases posting constraints
            return cpm_var

        raise NotImplementedError("Not a known var {}".format(cpm_var))

    def _row_from_linexpr(self, linexpr) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64], int|float]:
        """
        Convert a flat linear numeric expression (var/sum/wsum)
        into HiGHS row data: (indices, values, constant).

        The constant captures any numeric terms that are not column
        variables and must be subtracted from the row bounds by the caller.

        Returns:
            tuple[arr[int], arr[float], float]: (col indices, col weights, constant)
        """
        if is_num(linexpr):
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64), linexpr

        elif isinstance(linexpr, _NumVarImpl):
            col = self.solver_var(linexpr)
            return np.array([col], dtype=np.int32), np.array([1.0], dtype=np.float64), 0.0

        elif isinstance(linexpr, Operator):  # sum or wsum (vars only in operand lists)
            coeffs: npt.NDArray[np.float64]
            if linexpr.name == "sum":
                solvars = self.solver_vars(linexpr.args)
                idx = np.array(solvars, dtype=np.int32)
                coeffs = np.ones(idx.size, dtype=np.float64)
            elif linexpr.name == "wsum":
                ws, vs = linexpr.args
                solvars = self.solver_vars(vs)
                idx = np.array(solvars, dtype=np.int32)
                coeffs = np.asarray(ws, dtype=np.float64)
            else:
                raise NotImplementedError(f"HiGHS: unexpected operator {linexpr}, please report on our issue tracker.")

            # if columns are unique, just post
            if np.unique(idx).size == idx.size:
                return idx, coeffs, 0.0

            # merge duplicate columns
            new_idx, group = np.unique(idx, return_inverse=True)
            new_coeffs = np.bincount(group, weights=coeffs).astype(np.float64)
            sel = new_coeffs != 0  # keep only the non-zero vars
            return new_idx[sel], new_coeffs[sel], 0.0

        raise NotImplementedError(f"HiGHS: unexpected linear expression {linexpr}, please report on our issue tracker.")

    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports.

            Follows the ILP-style pipeline with linearize-friendly decompositions and treatment of reified variables.
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

    def _add_transformed(self, con):
        """
        Post a single already-transformed constraint to the HiGHS model.

        Returns the HiGHS row indices created by the constraint. This is also
        used by ``mus_native`` to map HiGHS IIS rows back to CPMpy constraints.
        """
        if isinstance(con, Comparison):
            lhs, rhs = con.args
            if not is_num(rhs):
                raise AssertionError(f"HiGHS: unexpected non-numeric RHS in comparison {con}, please report on our issue tracker.")

            indices, values, const = self._row_from_linexpr(lhs)
            # effective rhs: lhs_vars + const <op> rhs  =>  lhs_vars <op> rhs - const
            bound = rhs - const

            if con.name == "<=":
                lower = -self._inf
                upper = bound
            elif con.name == ">=":
                lower = bound
                upper = self._inf
            elif con.name == "==":
                lower = bound
                upper = bound
            else:
                raise NotSupportedError(f"HiGHS: unexpected comparison operator after linearization: {con.name}, please report on our issue tracker.")

            row_idx = self.highs.getNumRow()
            self.highs.addRow(lower, upper, len(indices), indices, values)
            return [row_idx]

        elif isinstance(con, BoolVal):
            if con.args[0] is False:
                # add an always-infeasible row: 1 <= 0
                indices = np.empty(0, dtype=np.int32)
                values = np.empty(0, dtype=np.float64)
                row_idx = self.highs.getNumRow()
                self.highs.addRow(1, 0, 0, indices, values)
                return [row_idx]
            # BoolVal(True) is a tautology; nothing to post
            return []

        elif isinstance(con, DirectConstraint):
            # delegate to user-provided function with native model
            con.callSolver(self, self.highs)
            return []

        else:
            raise NotImplementedError(f"HiGHS: unexpected transformed constraint {con}, please report on our issue tracker.")

    def add(self, cpm_expr):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.
        """
        # track user vars and ensure newly seen ones have solver columns
        get_variables(cpm_expr, collect=self.user_vars)

        for con in self.transform(cpm_expr):
            self._add_transformed(con)

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
        # only_positive_bv_wsum_const keeps the constant separate so it never ends up
        # as a numeric element in the wsum vars list (which _row_from_linexpr cannot handle)
        obj, obj_const = only_positive_bv_wsum_const(obj)

        self.add(safe_cons + decomp_cons + flat_cons)

        indices, values, const = self._row_from_linexpr(obj)
        const += obj_const

        # reset only columns that carried cost in the previous objective, then set new ones
        if self._obj_cols is not None and len(self._obj_cols):
            zeros = np.zeros(len(self._obj_cols), dtype=np.float64)
            self.highs.changeColsCost(len(self._obj_cols), self._obj_cols, zeros)
        if len(indices):
            self.highs.changeColsCost(len(indices), indices, values)
        self._obj_cols = indices

        self.highs.changeObjectiveOffset(const)

        sense = highspy.ObjSense.kMinimize if minimize else highspy.ObjSense.kMaximize
        self.highs.changeObjectiveSense(sense)

    def has_objective(self):
        return self._obj_cols is not None

    def solve(self, time_limit=None, **kwargs):
        """
            Call the HiGHS solver.

            Arguments:
            - time_limit: maximum solve time in seconds (float, optional)
            - kwargs:     any keyword argument, mapped to HiGHS options via ``setOptionValue``.
                          Unknown/invalid options are ignored with a warning.

            Notable HiGHS options:

        - ``threads`` (int): number of threads; default ``0`` means automatic (parallelism enabled
          according to HiGHS defaults).

        HiGHS option reference: https://ergo-code.github.io/HiGHS/dev/options/definitions/

        Solution callbacks are not connected yet; see HiGHS callback documentation for future reference:
        https://ergo-code.github.io/HiGHS/stable/callbacks/
        """
        import highspy

        # ensure all vars are known to solver
        self.solver_vars(self.user_vars)

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)

        # time limit: omitting clears it so it does not carry over between solve() calls
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            self.highs.setOptionValue("time_limit", time_limit)
        else:
            # (re)set to no limit (for HiGHS: infinity)
            self.highs.setOptionValue("time_limit", highspy.kHighsInf)

        # map additional kwargs to HiGHS options
        for key, val in kwargs.items():
            try:
                self.highs.setOptionValue(key, val)
            except Exception as e:
                warnings.warn(f"HiGHS: failed to set option '{key}' = {val!r}: {e}")

        status = self.highs.run()
        info = self.highs.getInfo()
        model_status = self.highs.getModelStatus()

        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.highs.getRunTime()

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
            col_values = solution.col_value
            for cpm_var in self.user_vars:
                if cpm_var.name not in self._varmap:
                    continue
                col_idx = self._varmap[cpm_var.name]
                val = col_values[col_idx]
                if cpm_var.is_bool():
                    cpm_var._value = val >= 0.5
                else:
                    cpm_var._value = round(val)

            if self.has_objective():
                self.objective_value_ = info.objective_function_value
        else:
            for cpm_var in self.user_vars:
                cpm_var._value = None

        return has_sol

    @classmethod
    def mus_native(cls, soft, hard=[]):
        """
        Compute a MUS using HiGHS' native IIS row extractor.

        A CPMpy soft constraint can transform to multiple rows, but this is not supported.
        Also HiGHS' IIS support is currently LP-only and works at the level of
        native rows. Even when there is a bijection of CPMpy constraints to HiGHS constraints
        the returned IIS may be invalid and hence raise an
        """
        import highspy

        soft_cons = toplevel_list(soft, merge_and=False)
        hard_cons = toplevel_list(hard, merge_and=False)

        s = cls()
        for cpm_con in s.transform(hard_cons):
            raise NotSupportedError("HiGHS does not support hard constraints for MUS extraction.")
            s._add_transformed(cpm_con)

        soft_rows = []
        for soft_con in soft_cons:
            soft_con_tf = s.transform(soft_con)

            if len(soft_con_tf) == 0:
                soft_con_rep = cp.BoolVal(True)
            elif len(soft_con_tf) == 1:
                soft_con_rep = soft_con_tf[0]
            else:
                raise NotSupportedError("HiGHS only supports MUS extraction for linear constraints.")
                assumption = cp.boolvar()
                additional_hard_constraint = assumption.implies(cp.all(soft_con_tf))
                for tf_con in s.transform(additional_hard_constraint):
                    s._add_transformed(tf_con)
                soft_con_rep = assumption >= 1

            soft_rows.append(s._add_transformed(soft_con_rep))

        if s.solve() is not False:
            raise AssertionError("MUS: model must be UNSAT")

        s.highs.setOptionValue("iis_strategy", 15)  # whole LP + IIS reduction


        status, iis = s.highs.getIis()
        if status == highspy.HighsStatus.kError:
            raise NotSupportedError("HiGHS: native MUS extraction failed.")
        if not getattr(iis, "valid_", False):
            raise NotSupportedError("HiGHS: native MUS extraction did not return a valid MUS. The infeasibility may rely on integrality.")

        iis_rows = frozenset(iis.row_index_)
        native_core = [
            soft_con
            for soft_con, rows in zip(soft_cons, soft_rows)
            if any(row in iis_rows for row in rows)
        ]

        return native_core
