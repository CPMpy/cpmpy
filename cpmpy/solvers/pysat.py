#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## pysat.py
##
"""
    Interface to PySAT's API

    PySAT is a Python (2.7, 3.4+) toolkit, which aims at providing a simple and unified
    interface to a number of state-of-art Boolean satisfiability (SAT) solvers as well as
    to a variety of cardinality and pseudo-Boolean encodings.
    (see https://pysathq.github.io/)

    .. warning::
        This solver can only be used if the model only uses Boolean variables.
        It does not support optimization.
    
    Always use :func:`cp.SolverLookup.get("pysat") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'python-sat' package is installed:

    .. code-block:: console

        $ pip install python-sat

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

        CPM_pysat

    ==============
    Module details
    ==============
"""
from threading import Timer
from typing import Optional
import warnings
import pkg_resources

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, _IntVarImpl, NegBoolView
from ..expressions.globalconstraints import DirectConstraint
from ..transformations.linearize import only_positive_coefficients
from ..expressions.utils import flatlist
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint
from ..transformations.linearize import linearize_constraint
from ..transformations.normalize import toplevel_list, simplify_boolean
from ..transformations.reification import only_implies, only_bv_reifies
from ..transformations.int2bool import int2bool, _encode_int_var, _decide_encoding


class CPM_pysat(SolverInterface):
    """
    Interface to PySAT's API.

    Creates the following attributes (see parent constructor for more):

    - ``pysat_vpool``: a pysat.formula.IDPool for the variable mapping
    - ``pysat_solver``: a pysat.solver.Solver() (default: glucose4)
    - ``ivarmap``: a mapping from integer variables to their encoding for `int2bool`
    - ``encoding``: the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary"). Set to "auto" but can be changed in the solver object.

    The :class:`~cpmpy.expressions.globalconstraints.DirectConstraint`, when used, calls a function on the ``pysat_solver`` object.

    Documentation of the solver's own Python API:
    https://pysathq.github.io/docs/html/api/solvers.html

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

            from pysat import card
            CPM_pysat._card = card  # native

            # try to import pypblib and avoid ever re-import by setting `_pb`
            if not hasattr(CPM_pysat, ("_pb")):
                try:
                    from pysat import pb  # require pypblib
                    """The `pysat.pb` module if its dependency `pypblib` installed, `None` if we have not checked it yet, or `False` if we checked and it is *not* installed"""
                    CPM_pysat._pb = pb
                except (ModuleNotFoundError, NameError):  # pysat returns the wrong error type (latter i/o former)
                    CPM_pysat._pb = None  # not installed, avoid reimporting

            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e

    @staticmethod
    def solvernames(**kwargs):
        """
            Returns solvers supported by PySAT on your system
        """
        if CPM_pysat.supported():
            from pysat.solvers import SolverNames
            names = []
            for name, attr in vars(SolverNames).items():
                # issue with cryptosat, so we don't include it in our https://github.com/msoos/cryptominisat/issues/765
                if not name.startswith('__') and isinstance(attr, tuple) and name != 'cryptosat':
                    if name not in attr:
                        name = attr[-1]
                    names.append(name)  
            return names
        else:
            warnings.warn("PySAT is not installed or not supported on this system.")
            return []
        
    @staticmethod
    def solverversion(subsolver:str) -> Optional[str]:
        """
        Returns the version of the requested subsolver.

        Arguments:
            subsolver (str): name of the subsolver

        Returns:
            Version number of the subsolver if installed, else None 
    
        Pysat currently does not provide accessible subsolver version numbers.
        """
        # Could try to extract them from solver name, but even then the minor revision numbers are missing
        return None
    
    @staticmethod
    def version() -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        try:
            return pkg_resources.get_distribution('python-sat').version
        except pkg_resources.DistributionNotFound:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        PySAT clauses and solver object

        Only supports satisfaction problems (no objective)

        Arguments:
            cpm_model (Model(), a CPMpy Model(), optional):
            subsolver (str, name of the pysat solver, e.g. glucose4):  see .solvernames() to get the list of available solver(names)
        """
        if not self.supported():
            raise ImportError("PySAT is not installed. The recommended way to install PySAT is with `pip install cpmpy[pysat]`, or `pip install python-sat` if you do not require `pblib` to encode (weighted) sums.")
        if cpm_model and cpm_model.objective_ is not None:
            raise NotSupportedError("CPM_pysat: only satisfaction, does not support an objective function")

        from pysat.formula import IDPool
        from pysat.solvers import Solver

        # determine subsolver
        if subsolver is None or subsolver == 'pysat':
            # default solver
            subsolver = "glucose4" # something recent...
        elif subsolver.startswith('pysat:'):
            subsolver = subsolver[6:] # strip 'pysat:'

        # initialise the native solver object
        self.pysat_vpool = IDPool()
        self.pysat_solver = Solver(use_timer=True, name=subsolver)
        self.ivarmap = dict()  # for the integer to boolean encoders
        self.encoding = "auto"

        # initialise everything else and post the constraints/objective
        super().__init__(name="pysat:"+subsolver, cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.pysat_solver


    def solve(self, time_limit=None, assumptions=None):
        """
            Call the PySAT solver

            Arguments:
                time_limit (float, optional):   Maximum solve time in seconds. Auto-interrups in case the
                                                runtime exceeds given time_limit.
                                                
                                                .. warning::
                                                    Warning: the time_limit is not very accurate at subsecond level
                assumptions: list of CPMpy Boolean variables that are assumed to be true.
                            For use with :func:`s.get_core() <get_core()>`: if the model is UNSAT, get_core() returns a small subset of assumption variables that are unsat together.
                            Note: the PySAT interface is statefull, so you can incrementally call solve() with assumptions and it will reuse learned clauses
        """

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

        if assumptions is None:
            pysat_assum_vars = [] # default if no assumptions
        else:
            pysat_assum_vars = self.solver_vars(assumptions)
            self.assumption_vars = assumptions

        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            
            t = Timer(time_limit, lambda s: s.interrupt(), [self.pysat_solver])
            t.start()
            my_status = self.pysat_solver.solve_limited(assumptions=pysat_assum_vars, expect_interrupt=True)
            # ensure timer is stopped if early stopping
            t.cancel()
            ## this part cannot be added to timer otherwhise it "interrups" the timeout timer too soon
            self.pysat_solver.clear_interrupt()
        else:
            my_status = self.pysat_solver.solve(assumptions=pysat_assum_vars)

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.pysat_solver.time()

        # translate exit status
        if my_status is True:
            # COP
            if self.has_objective():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            # CSP
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status is False:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status is None:
            # can happen when timeout is reached...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:  # another?
            raise NotImplementedError(my_status)  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        if has_sol:
            sol = frozenset(self.pysat_solver.get_model())  # to speed up lookup
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
                    raise NotImplementedError(f"CPM_pysat: variable {cpm_var} not supported")

            # Now assign the user integer variables using their encodings
            # `ivarmap` also contains auxiliary variable, but they will be assigned 'None' as their encoding variables are assigned `None`
            for enc in self.ivarmap.values():
                enc._x._value = enc.decode()

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None


        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created.

            Transforms cpm_var into CNF literal using ``self.pysat_vpool``
            (positive or negative integer).

            So vpool is the varmap (we don't use _varmap here).
        """

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, BoolVal):
            return cpm_var
        elif isinstance(cpm_var, NegBoolView):
            # just a view, get actual var identifier, return -id
            return -self.pysat_vpool.id(cpm_var._bv.name)
        elif isinstance(cpm_var, _BoolVarImpl):
            return self.pysat_vpool.id(cpm_var.name)
        elif isinstance(cpm_var, _IntVarImpl):  # intvar
            if cpm_var.name not in self.ivarmap:
                enc, cons = _encode_int_var(self.ivarmap, cpm_var, _decide_encoding(cpm_var, None, encoding=self.encoding))
                self += cons
            else:
                enc = self.ivarmap[cpm_var.name]
            return self.solver_vars(enc.vars())
        else:
            raise NotImplementedError(f"CPM_pysat: variable {cpm_var} not supported")

    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the :ref:`Adding a new solver` docs on readthedocs for more information.


            In the case of PySAT, the supported constraints are over Boolean variables:

            - Boolean clauses
            - Cardinality constraint (`sum`)
            - Pseudo-Boolean constraints (`wsum`)

            :param cpm_expr: CPMpy expression, or list thereof
            :type cpm_expr: Expression or list of Expression

            :return: list of Expression
        """
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = decompose_in_tree(cpm_cons, supported=frozenset({"alldifferent"}), csemap=self._csemap)
        cpm_cons = simplify_boolean(cpm_cons)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum","wsum", "and", "or"}), csemap=self._csemap)  # the core of the MIP-linearization
        cpm_cons = int2bool(cpm_cons, self.ivarmap, encoding=self.encoding)
        cpm_cons = only_positive_coefficients(cpm_cons)
        return cpm_cons

    def add(self, cpm_expr_orig):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

            What 'supported' means depends on the solver capabilities, and in effect on what transformations
            are applied in `transform()`.
        """
        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        for cpm_expr in self.transform(cpm_expr_orig):
            self._post_constraint(cpm_expr)

        return self

    def _post_constraint(self, cpm_expr):
        """ Add expression to solver _without_ transforming."""
        if cpm_expr.name == 'or':
            self.pysat_solver.add_clause(self.solver_vars(cpm_expr.args))

        elif cpm_expr.name == '->':  # BV -> BE only thanks to only_bv_reifies
            a0,a1 = cpm_expr.args

            if isinstance(a1, _BoolVarImpl):
                # BoolVar() -> BoolVar()
                args = [~a0, a1]
                self.pysat_solver.add_clause(self.solver_vars(args))
            elif isinstance(a1, Operator) and a1.name == 'or':
                # BoolVar() -> or(...)
                args = [~a0]+a1.args
                self.pysat_solver.add_clause(self.solver_vars(args))
            elif isinstance(a1, Comparison) and a1.args[0].name == "sum":  # implied sum comparison (a0->sum(bvs)<>val)
                # implied sum comparison (a0->sum(bvs)<>val)
                # convert sum to cnf
                cnf = self._pysat_cardinality(a1, reified=True)
                # implication of conjunction is conjunction of individual implications
                antecedent = [self.solver_var(~a0)]
                cnf = [antecedent+c for c in cnf]
                self.pysat_solver.append_formula(cnf)
            elif isinstance(a1, Comparison) and a1.args[0].name == "wsum":  # implied pseudo-boolean comparison (a0->wsum(ws,bvs)<>val)
                # implied sum comparison (a0->wsum([w,bvs])<>val or a0->(w*bv<>val))
                cnf = self._pysat_pseudoboolean(a1)
                # implication of conjunction is conjunction of individual implications
                antecedent = [self.solver_var(~a0)]
                self.pysat_solver.append_formula([antecedent+c for c in cnf])
            else:
                raise NotSupportedError(f"Implication: {cpm_expr} not supported by CPM_pysat")

        elif isinstance(cpm_expr, Comparison): # root-level comparisons have been linearized
            if isinstance(cpm_expr.args[0], Operator) and cpm_expr.args[0].name == "sum":
                self.pysat_solver.append_formula(self._pysat_cardinality(cpm_expr))
            elif isinstance(cpm_expr.args[0], Operator) and cpm_expr.args[0].name == "wsum":
                self.pysat_solver.append_formula(self._pysat_pseudoboolean(cpm_expr))
            else:
                raise NotSupportedError(f"Implication: {cpm_expr} not supported by CPM_pysat")

        elif isinstance(cpm_expr, Comparison):
            # comparisons between Booleans will have been transformed out
            # check if comparison of cardinality/pseudo-boolean constraint
            if isinstance(cpm_expr.args[0], Operator):
                if cpm_expr.args[0].name == "sum":
                    # convert to clauses and post
                    clauses = self._pysat_cardinality(cpm_expr)
                    self.pysat_solver.append_formula(clauses)
                elif cpm_expr.args[0].name == "wsum":
                    # convert to clauses and post
                    clauses = self._pysat_pseudoboolean(cpm_expr)
                    self.pysat_solver.append_formula(clauses)
                else:
                    raise NotImplementedError(f"Operator constraint {cpm_expr} not supported by CPM_pysat")
            else:
                raise NotImplementedError(f"Non-operator constraint {cpm_expr} not supported by CPM_pysat")

        elif isinstance(cpm_expr, BoolVal):
            # base case: Boolean value
            if cpm_expr.args[0] is False:
                self.pysat_solver.add_clause([])

        elif isinstance(cpm_expr, _BoolVarImpl):
            # base case, just var or ~var
            self.pysat_solver.add_clause([self.solver_var(cpm_expr)])

        # a direct constraint, pass to solver
        elif isinstance(cpm_expr, DirectConstraint):
            cpm_expr.callSolver(self, self.pysat_solver)

        else:
            raise NotImplementedError(f"CPM_pysat: Non supported constraint {cpm_expr}")


    __add__ = add  # avoid redirect in superclass

    def solution_hint(self, cpm_vars, vals):
        """
        PySAT supports warmstarting the solver with a feasible solution

        In PySAT, this is called setting the 'phases' or the 'polarities' of literals

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """

        cpm_vars = flatlist(cpm_vars)
        vals = flatlist(vals)
        assert (len(cpm_vars) == len(vals)), "Variables and values must have the same size for hinting"

        literals = []
        for (cpm_var, val) in zip(cpm_vars, vals):
            lit = self.solver_var(cpm_var)
            if val:
                # true, so positive literal
                literals.append(lit)
            else:
                # false, so negative literal
                literals.append(-lit)
        self.pysat_solver.set_phases(literals)


    def get_core(self):
        """
            For use with :func:`s.solve(assumptions=[...]) <solve()>`. Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

            CPMpy will return only those assumptions which are False (in the UNSAT core)

            Note that there is no guarantee that the core is minimal.
            More advanced Minimal Unsatisfiable Subset are available in the 'examples' folder on GitHub

        """
        assert hasattr(self, 'assumption_vars'), "get_core(): requires a list of assumption variables, e.g. s.solve(assumptions=[...])"
        assert (self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE), "get_core(): solver must return UNSAT"

        assum_idx = frozenset(self.pysat_solver.get_core()) # to speed up lookup

        return [v for v in self.assumption_vars if self.solver_var(v) in assum_idx]


    def _pysat_cardinality(self, cpm_expr, reified=False):
        """ Convert CPMpy comparison of `sum` (over Boolean variables) into PySAT list of clauses """

        # unpack and transform to PySAT argument
        lhs, rhs = cpm_expr.args
        if lhs.name != "sum":
            raise NotSupportedError(
                f"PySAT: Expect {cpm_expr} to be a 'sum'"
            )

        lits = self.solver_vars(lhs.args)
        pysat_args = { "lits": lits, "bound": rhs, "vpool": self.pysat_vpool }

        # Some subsolvers (e.g. MiniCard) support native root context cardinality constraints
        if not reified and self.pysat_solver.supports_atmost():
            pysat_args["encoding"] = self._card.EncType.native

        if cpm_expr.name == "<=":
            return self._card.CardEnc.atmost(**pysat_args)
        elif cpm_expr.name == ">=":
            return self._card.CardEnc.atleast(**pysat_args)
        elif cpm_expr.name == "==":
            return self._card.CardEnc.equals(**pysat_args)
        else:
            raise ValueError(f"PySAT: Expected Comparison to be either <=, ==, or >=, but was {cpm_expr.name}")

    def _pysat_pseudoboolean(self, cpm_expr):
        """Convert CPMpy comparison of `wsum` (over Boolean variables) into PySAT list of clauses."""
        if self._pb is None:
            raise ImportError("The model contains a PB constraint, for which PySAT needs an additional dependency (PBLib). To install it, run `pip install pypblib`.")

        if cpm_expr.args[0].name != "wsum":
            raise NotSupportedError(
                f"PySAT: Expect {cpm_expr} to be a 'wsum'"
            )

        # unpack and transform to PySAT arguments
        lhs, rhs = cpm_expr.args
        lits = self.solver_vars(lhs.args[1])
        pysat_args = {"weights": lhs.args[0], "lits": lits, "bound": rhs, "vpool":self.pysat_vpool }


        if cpm_expr.name == "<=":
            return self._pb.PBEnc.atmost(**pysat_args).clauses
        elif cpm_expr.name == ">=":
            return self._pb.PBEnc.atleast(**pysat_args).clauses
        elif cpm_expr.name == "==":
            return self._pb.PBEnc.equals(**pysat_args).clauses
        else:
            raise ValueError(f"PySAT: Expected Comparison to be either <=, ==, or >=, but was {cpm_expr.name}")
