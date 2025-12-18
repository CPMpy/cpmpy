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
from .solver_interface import SolverStatus, ExitStatus
from .pysat import CPM_pysat
from ..exceptions import NotSupportedError
from ..expressions.variables import _BoolVarImpl, _IntVarImpl, NegBoolView
from ..transformations.linearize import only_positive_coefficients_
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_objective
from ..transformations.int2bool import get_user_vars, _encode_lin_expr


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
            raise ImportError("PySAT is not installed. The recommended way to install PySAT is with `pip install cpmpy[pysat]`, or `pip install python-sat` if you do not require `pblib` to encode (weighted) sums.")

        from pysat.formula import IDPool, WCNFPlus, CNFPlus

        # In order for all features to work (e.g. `process` option), RC2 should be bootstrapped from a formula
        self.pysat_solver = WCNFPlus()

        def append_formula(cnf, weight=None):
            """Some RC2 oracles/solvers support AtMostK constraints in the form of CNFPlus objects, which can be passed through this function in addition to regular clauses."""
            clauses = []
            atmosts = []
            if isinstance(cnf, list):
                clauses = cnf
            elif isinstance(cnf, tuple):
                atmosts = cnf
            else:  # CNFPlus
                assert isinstance(cnf, CNFPlus)
                clauses = cnf.clauses
                atmosts = cnf.atmosts

            for c in clauses:
                self.pysat_solver.append(c, weight=weight)
            for c in atmosts:
                self.pysat_solver.append(c, weight=weight, is_atmost=True)

        self.pysat_solver.append_formula = append_formula
        self.pysat_solver.add_clause = lambda c, weight=None: self.pysat_solver.append(c, weight=weight)
        # Note: in case it becomes neccessary in the future to support CNFPlus objects for `add_clause`, this function should be replaced by: `lambda c, weight=None: append_formula([c], weight=weight)`

        # We collect all AtMostK constraints in the WCNFPlus object, then encode them if the solver (set at `solve`) does not accept them
        self.pysat_solver.supports_atmost = lambda: True

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
        from pysat.solvers import Solver
        from pysat.formula import WCNF

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # the user vars are only the Booleans (e.g. to ensure solveAll behaves consistently)
        self.user_vars = get_user_vars(self.user_vars, self.ivarmap)

        # TODO: set time limit (awaiting upstream PR https://github.com/pysathq/pysat/pull/211)
        if time_limit is not None:
            raise NotImplementedError("CPM_rc2: time limit not yet supported")

        # hack to support decision problems
        if not self.has_objective():
            self.pysat_solver.add_clause([self.pysat_solver.nv + 1], weight=1)

        # determine subsolver
        default_kwargs = {"solver": "glucose3", "adapt": True, "exhaust": True, "minz": True}
        stratified = kwargs.get("stratified", True)
        slv_kwargs = kwargs if kwargs else default_kwargs

        # get subsolver kwarg or default, then check if it supports AtMostK constrains
        sub_solver = slv_kwargs.get("solver", default_kwargs["solver"])
        if sub_solver and not Solver(name=sub_solver).supports_atmost():
            # encode AtMostK constraints since they are unsupported by sub-solver oracle
            atmosts = self.pysat_solver.atms
            self.pysat_solver.__class__ = WCNF
            for atmost in atmosts:
                lits, k = atmost
                self.pysat_solver.extend(self._card.CardEnc.atmost(lits=lits, bound=k, vpool=self.pysat_vpool))

        # instantiate and configure RC2
        slv = rc2.RC2Stratified(self.pysat_solver, **slv_kwargs) if stratified else rc2.RC2(self.pysat_solver, **slv_kwargs)

        sol = slv.compute()  # return one solution

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = slv.oracle_time()

        # translate exit status
        if sol is None:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif self.has_objective():
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        else:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE

        return self._process_solution(sol)
        
    def transform_objective(self, expr):
        """
            Transform the objective to a list of (w,x) and a constant
        """
        # add new user vars to the set
        get_variables(expr, collect=self.user_vars)

        # try to flatten the objective
        (flat_obj, flat_cons) = flatten_objective(expr, csemap=self._csemap)
        self.add(flat_cons)

        weights, xs, const = [], [], 0
        # we assume flat_obj is a var, a sum or a wsum (over int and bool vars)
        if isinstance(flat_obj, _IntVarImpl) or isinstance(flat_obj, NegBoolView):  # includes _BoolVarImpl
            weights = [1]
            xs = [flat_obj]
        elif flat_obj.name == "sum":
            xs = flat_obj.args
            weights = [1] * len(xs)
        elif flat_obj.name == "wsum":
            weights, xs = flat_obj.args
        else:
            raise NotImplementedError(f"CPM_rc2: Non supported objective {flat_obj} (yet?)")

        terms, cons, k = _encode_lin_expr(self.ivarmap, xs, weights, self.encoding)

        self += cons
        const += k

        terms = [(w, x) for w,x in terms if w != 0]  # positive coefficients only
        ws, xs = zip(*terms)  # unzip
        new_weights, new_xs, k = only_positive_coefficients_(ws, xs) # this is actually only_non_negative_coefficients
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
        # we don't need to keep the constant, we will recompute the objective value

        # post weighted literals
        for wi,vi in zip(weights, xs):
            assert wi > 0, f"CPM_rc2 objective: strictly positive weights only, got {wi,vi}"
            self.pysat_solver.add_clause([self.solver_var(vi)], weight=wi)


    def objective_value(self):
        """
            Get the objective value of the last optimisation problem
        """
        return self.objective_.value()
