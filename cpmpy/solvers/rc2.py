#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## pysat.py
##
"""
    Interface to PySAT's RC2 MaxSAT solver API

    PySAT is a Python (2.7, 3.4+) toolkit, which aims at providing a simple and unified
    interface to a number of state-of-art Boolean satisfiability (SAT) solvers. It also
    includes the RC2 MaxSAT solver.
    (see https://pysathq.github.io/)

    .. warning::
        It does not support satisfaction, only optimization.
    
    Always use :func:`cp.SolverLookup.get("rc2") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'python-sat' package is installed:

    .. code-block:: console

        $ pip install pysat

    If you want to also solve pseudo-Boolean constraints, you should also install its optional dependency 'pypblib', as follows:

    .. code-block:: console

        $ pip install pypblib

    See detailed installation instructions at:
    https://pysathq.github.io/installation

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_rc2

    ==============
    Module details
    ==============
"""
import os

from threading import Timer
from .solver_interface import SolverStatus, ExitStatus
from .pysat import CPM_pysat
from ..transformations.decompose_global import decompose_objective
from ..transformations.safening import safen_objective
from ..exceptions import NotSupportedError
from ..expressions.variables import _IntVarImpl, NegBoolView
from ..transformations.linearize import only_positive_coefficients_
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_objective
from ..transformations.int2bool import replace_int_user_vars, _encode_lin_expr


class CPM_rc2(CPM_pysat):
    """
    Interface to PySAT's RC2 MaxSAT solver API.

    Creates the following attributes (see parent constructor for more):

    - ``pysat_vpool``: a pysat.formula.IDPool for the variable mapping
    - ``pysat_solver``: a pysat.examples.rc2.RC2() (or .RC2Stratified())
    - ``ivarmap``: a mapping from integer variables to their encoding for `int2bool`
    - ``encoding``: the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary"). Set to "auto" but can be changed in the solver object.

    The :class:`~cpmpy.expressions.globalconstraints.DirectConstraint`, when used, calls a function on the ``pysat_solver`` object.

    Documentation of the solver's own Python API:
    https://pysathq.github.io/docs/html/api/examples/rc2.html

    .. note::
        CPMpy uses 'model' to refer to a constraint specification,
        the PySAT docs use 'model' to refer to a solution.

    """

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        PySAT clauses and solver object

        Only supports optimisation problems (MaxSAT)

        Arguments:
            cpm_model (Model(), optional): a CPMpy Model()
            subsolver (None): ignored
        
        """
        if not self.supported():
            raise ModuleNotFoundError("CPM_rc2: Install the python package 'cpmpy[rc2]' (recommended), or 'python-sat' if you do not require 'pblib' to encode (weighted) sums.")

        from pysat.formula import IDPool, WCNF

        # In order for all features to work (e.g. `process` option), RC2 should be bootstrapped from a formula
        self.pysat_solver = WCNF()

        self.pysat_solver.add_clause = lambda c, weight=None: self.pysat_solver.append(c, weight=weight)
        self.pysat_solver.append_formula = lambda c, weights=None: self.pysat_solver.extend(c, weights=weights)
        self.pysat_solver.supports_atmost = lambda: False  # native atmost support disabled for RC2 to reduce complexity

        # objective value related
        self.objective_ = None

        # initialise the native solver object
        self.pysat_vpool = IDPool()
        self.ivarmap = dict()  # for the integer to boolean encoders
        self.encoding = "auto"

        # initialise everything else and post the constraints/objective (skip PySAT itself)
        super(CPM_pysat, self).__init__(name="rc2", cpm_model=cpm_model)

    def has_objective(self):
        return self.objective_ is not None

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.pysat_solver

    def solve(self, time_limit=None, **kwargs):
        """
            Call the RC2 MaxSAT solver

            Arguments:
                time_limit (float, optional):   Maximum solve time in seconds. Auto-interrups in case the
                                                runtime exceeds given time_limit.
                                                
                                                .. warning::
                                                    Warning: the time_limit is not very accurate at subsecond level

            The following `**kwargs` are supported for RC2:

                stratified (bool, optional): use the stratified solver for weighted maxsat (default: True)
                adapt (bool, optional): detect and adapt intrinsic AtMost1 constraint (default: True)
                exhaust (bool, optional): do core exhaustion (default: True)
                minz (bool, optional): do heuristic core reduction (default: True)

            If no `**kwargs` are given, the default values are used as recommended by the PySAT authors, based on their MaxSAT Evaluation 2018 submission, i.e.: `{"solver": "glucose3", "adapt": True, "exhaust": True, "minz": True}`.
            If `**kwargs` are given, these are passed to RC2.
            Note that currently, no args are passed to the underlying oracle.
        """
        from pysat.examples import rc2

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # the user vars should have all and only Booleans (e.g. to ensure solveAll behaves consistently)
        self.user_vars = replace_int_user_vars(self.user_vars, self.ivarmap)

        if not self.has_objective():
            raise NotSupportedError("CPM_rc2: RC2 does not support solving decision problems. Add an objective to your problem.")

        # determine subsolver
        default_kwargs = {"solver": "glucose3", "adapt": True, "exhaust": True, "minz": True}
        stratified = kwargs.get("stratified", True)
        slv_kwargs = kwargs if kwargs else default_kwargs

        # instantiate and configure RC2
        solver = rc2.RC2Stratified(self.pysat_solver, **slv_kwargs) if stratified else rc2.RC2(self.pysat_solver, **slv_kwargs)

        # set time limit
        if time_limit is None:
            solution = solver.compute()
        else:
            if time_limit <= 0:
                raise ValueError("CPM_rc2: Time limit must be positive")
            timer = Timer(time_limit, lambda: solver.interrupt())
            timer.start()
            solution = solver.compute(expect_interrupt=True)
            # ensure timer is stopped
            timer.cancel()
            # this part cannot be added to timer otherwise it "interrupts" the timeout timer too soon
            solver.clear_interrupt()

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = solver.oracle_time()

        # translate exit status
        if solution is None:
            # `None` for either unsat or unknown!
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN if solver.interrupted else ExitStatus.UNSATISFIABLE
        else:
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL

        return self._process_solution(solution)

    def _process_solution(self, sol):
        """Process solution `sol` by using PySAT's `_process_solution()`, and setting the objective value."""
        has_sol = super()._process_solution(sol)
        if self.has_objective():
            self.objective_value_ = self.objective_.value()
        return has_sol
        
    def transform_objective(self, expr):
        """
            Transform the objective to a list of (w,x) and a constant
        """
        # add new user vars to the set
        get_variables(expr, collect=self.user_vars)

        # transform objective
        obj, safe_cons = safen_objective(expr)
        obj, decomp_cons = decompose_objective(
            obj,
            supported=self.supported_global_constraints,
            supported_reified=self.supported_reified_global_constraints,
            csemap=self._csemap
        )
        obj, flat_cons = flatten_objective(obj, csemap=self._csemap)
        self.add(safe_cons + decomp_cons + flat_cons)

        weights, xs, const = [], [], 0
        # we assume obj is a var, a sum or a wsum (over int and bool vars)
        if isinstance(obj, _IntVarImpl) or isinstance(obj, NegBoolView):  # includes _BoolVarImpl
            weights = [1]
            xs = [obj]
        elif obj.name == "sum":
            xs = obj.args
            weights = [1] * len(xs)
        elif obj.name == "wsum":
            weights, xs = obj.args
        else:
            raise NotImplementedError(f"CPM_rc2: Non supported objective {obj} (yet?)")

        terms, cons, k = _encode_lin_expr(self.ivarmap, xs, weights, self.encoding)

        self += cons
        const += k

        # remove terms with coefficient 0 (`only_positive_coefficients_` may return them and RC2 does not accept them)
        terms = [(w, x) for w,x in terms if w != 0]  
        ws, xs = zip(*terms)  # unzip
        new_weights, new_xs, k = only_positive_coefficients_(ws, xs)
        const += k

        return list(new_weights), list(new_xs), const


    def objective(self, expr, minimize):
        """
            Post the given expression to the solver as objective to minimize/maximize.

            Arguments:
                expr: Expression, the CPMpy expression that represents the objective function
                minimize: Bool, whether it is a minimization problem (True) or maximization problem (False)

        """
        # XXX RC2 is incremental, except for its objective is non-incremental; while (soft) clauses can be added, they cannot be removed, so we cannot replace the objective. Adding soft clauses is now also not supported.
        if self.objective_ is not None:
            raise NotSupportedError("CPM_rc2: objective can only be set once")
        self.objective_ = expr

        # maxsat by default maximizes
        if minimize:
            expr = -expr

        # transform the objective to a list of (w,x) and a constant
        weights, xs, const = self.transform_objective(expr)

        assert len(weights) == len(xs), f"CPM_rc2 objective: expected equal nr weights and vars, got {weights, xs}"
        assert isinstance(const, int), f"CPM_rc2 objective: expected constant to be an integer, got {const}"
        assert all(w > 0 for w in weights), f"CPM_rc2 objective: strictly positive weights only, got {weights}"

        # post each weighted literal as a soft clause
        # we don't need to keep the constant, we will recompute the objective value
        self.pysat_solver.append_formula([[x] for x in self.solver_vars(xs)], weights=weights)
