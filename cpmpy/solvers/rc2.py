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
from threading import Timer
from typing import Optional
import warnings
import pkg_resources

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from .pysat import CPM_pysat
from ..exceptions import NotSupportedError
from ..expressions.core import Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, _IntVarImpl, NegBoolView
from ..expressions.globalconstraints import DirectConstraint
from ..transformations.linearize import canonical_comparison, only_positive_coefficients
from ..expressions.utils import is_int, flatlist
from ..transformations.comparison import only_numexpr_equality
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.linearize import linearize_constraint
from ..transformations.normalize import toplevel_list, simplify_boolean
from ..transformations.reification import only_implies, only_bv_reifies, reify_rewrite
from ..transformations.int2bool import int2bool, _encode_int_var, _decide_encoding, IntVarEncDirect


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

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pysat
            # there is actually a non-related 'pysat' package
            # while we need the 'python-sat' package, some more checks:
            from pysat.formula import IDPool
            from pysat.solvers import Solver
            from pysat.examples import rc2

            from pysat import card
            CPM_rc2._card = card  # native

            # try to import pypblib and avoid ever re-import by setting `_pb`
            if not hasattr(CPM_rc2, ("_pb")):
                try:
                    from pysat import pb  # require pypblib
                    """The `pysat.pb` module if its dependency `pypblib` installed, `None` if we have not checked it yet, or `False` if we checked and it is *not* installed"""
                    CPM_rc2._pb = pb
                except (ModuleNotFoundError, NameError):  # pysat returns the wrong error type (latter i/o former)
                    CPM_rc2._pb = None  # not installed, avoid reimporting

            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e

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
        if cpm_model and cpm_model.objective_ is None:
            raise NotSupportedError("CPM_rc2: only optimisation, does not support satisfaction")

        from pysat.formula import IDPool, WCNF

        self.pysat_solver = WCNF()  # not actually the solver...
        # fix an inconsistent API
        self.pysat_solver.add_clause = self.pysat_solver.append
        self.pysat_solver.append_formula = self.pysat_solver.extend
        self.pysat_solver.supports_atmost = lambda: False
        # TODO: accepts native cardinality constraints, not sure how to make clear...

        # objective value related
        self.objective_ = None  # pysat returns the 'cost' of unsatisfied soft clauses, we want the value of the satisfied ones

        # initialise the native solver object
        self.pysat_vpool = IDPool()
        self.ivarmap = dict()  # for the integer to boolean encoders
        self.encoding = "auto"

        # initialise everything else and post the constraints/objective (skip PySAT itself)
        super(CPM_pysat, self).__init__(name="rc2", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.pysat_solver

    def solve(self, time_limit=None, stratified=True, adapt=True, exhaust=True, minz=True, **kwargs):
        """
            Call the RC2 MaxSAT solver

            Arguments:
                time_limit (float, optional):   Maximum solve time in seconds. Auto-interrups in case the
                                                runtime exceeds given time_limit.
                                                
                                                .. warning::
                                                    Warning: the time_limit is not very accurate at subsecond level
                stratified (bool, optional): use the stratified solver for weighted maxsat (default: True)
                adapt (bool, optional): detect and adapt intrinsic AtMost1 constraint (default: True)
                exhaust (bool, optional): do core exhaustion (default: True)
                minz (bool, optional): do heuristic core reduction (default: True)

            The last 4 parameters default values were recommended by the PySAT authors, based on their MaxSAT Evaluation 2018 submission.
        """
        from pysat.examples import rc2

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # the user vars are only the Booleans (e.g. to ensure solveAll behaves consistently)
        user_vars = set()
        for x in self.user_vars:
            if isinstance(x, _BoolVarImpl):
                user_vars.add(x)
            else:
                # extends set with encoding variables of `x`
                user_vars.update(self.ivarmap[x.name].vars())
        self.user_vars = user_vars

        # TODO: set time limit
        if time_limit is not None:
            # rc2 does not support it, also not interrupts like pysat does
            # we will have to manage it externally, e.g in a subprocess or so
            raise NotImplementedError("CPM_rc2: time limit not yet supported")

        # determine subsolver
        if stratified:
            slv = rc2.RC2Stratified(self.pysat_solver, adapt=adapt, exhaust=exhaust, minz=minz, **kwargs)
        else:
            slv = rc2.RC2(self.pysat_solver, adapt=adapt, exhaust=exhaust, minz=minz, **kwargs)

        sol = slv.compute()  # return one solution
        
        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = slv.oracle_time()

        # translate exit status
        if sol is None:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        else:
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL

        # translate solution values (of user specified variables only)
        if sol is not None:
            # fill in variable values
            for cpm_var in self.user_vars:
                if isinstance(cpm_var, _BoolVarImpl):
                    lit = self.solver_var(cpm_var)
                    if lit in sol:
                        cpm_var._value = True
                    else:  # -lit in sol (=False) or not specified (=False)
                        cpm_var._value = False
                elif isinstance(cpm_var, _IntVarImpl):
                    raise TypeError("user_vars should only contain Booleans")
                else:
                    raise NotImplementedError(f"CPM_rc2: variable {cpm_var} not supported")

            # Now assign the user integer variables using their encodings
            # `ivarmap` also contains auxiliary variable, but they will be assigned 'None' as their encoding variables are assigned `None`
            for enc in self.ivarmap.values():
                enc._x._value = enc.decode()

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None
            for enc in self.ivarmap.values():
                enc._x._value = None

        return sol is not None


    def transform_objective(self, expr):
        """
            Transform the objective to a list of (w,x) and a constant
        """
        # add new user vars to the set
        get_variables(expr, collect=self.user_vars)

        # try to flatten the objective
        (flat_obj, flat_cons) = flatten_objective(expr)
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
        
        # transform weighted integers to weighted sum of Booleans
        new_weights, new_xs = [], []
        for w, x in zip(weights, xs):
            if isinstance(x, _BoolVarImpl):
                if w != 0:
                    new_weights.append(w)
                    new_xs.append(x)
            elif isinstance(x, _IntVarImpl):
                # replace the intvar with its linear encoding
                # ensure encoding is created
                self.solver_var(x)
                enc = self.ivarmap[x.name]
                tlst, tconst = enc.encode_term(w)
                const += tconst
                for encw, encx in tlst:
                    if encw != 0:
                        new_weights.append(encw)
                        new_xs.append(encx)
            elif isinstance(x, int):
                const += w*x
            else:
                raise NotImplementedError(f"CPM_rc2: Non supported term {w,x} in objective {flat_obj} (yet?)")

        # positive weights only, flip negative
        for i in range(len(new_weights)):  # inline replace
            assert new_weights[i] != 0, f"CPM_rc2: positive weights only, got {new_weights[i],new_xs[i]}"
            if new_weights[i] < 0:  # negative weight
                # wi*vi == wi*(1-(~vi)) == wi + -wi*~vi  # where wi is negative
                const += new_weights[i]
                new_weights[i] = -new_weights[i]
                new_xs[i] = ~new_xs[i]
        
        return new_weights, new_xs, const


    def objective(self, expr, minimize):
        """
            Post the given expression to the solver as objective to minimize/maximize.

            Arguments:
                expr: Expression, the CPMpy expression that represents the objective function
                minimize: Bool, whether it is a minimization problem (True) or maximization problem (False)

        """
        # XXX WARNING, not incremental! Can NOT overwrite the objective.... only append to it!
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
            self.pysat_solver.append([self.solver_var(vi)], weight=wi)


    def objective_value(self):
        """
            Get the objective value of the last optimisation problem
        """
        return self.objective_.value()
