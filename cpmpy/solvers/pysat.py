#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## pysat.py
##
"""
    Interface to PySAT's API

    Requires that the 'python-sat' python package is installed:

        $ pip install python-sat[aiger,approxmc,cryptosat,pblib]

    PySAT is a Python (2.7, 3.4+) toolkit, which aims at providing a simple and unified
    interface to a number of state-of-art Boolean satisfiability (SAT) solvers as well as
    to a variety of cardinality and pseudo-Boolean encodings.
    https://pysathq.github.io/

    This solver can be used if the model only has Boolean variables,
    and only logical constraints (and,or,implies,==,!=) or cardinality constraints.

    Documentation of the solver's own Python API:
    https://pysathq.github.io/docs/html/api/solvers.html

    WARNING: CPMpy uses 'model' to refer to a constraint specification,
    the PySAT docs use 'model' to refer to a solution.

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
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, NegBoolView, boolvar
from ..expressions.globalconstraints import DirectConstraint
from ..transformations.linearize import canonical_comparison, only_positive_coefficients
from ..expressions.utils import is_int, flatlist
from ..transformations.comparison import only_numexpr_equality
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint
from ..transformations.linearize import linearize_constraint
from ..transformations.normalize import toplevel_list, simplify_boolean
from ..transformations.reification import only_implies, only_bv_reifies, reify_rewrite


class CPM_pysat(SolverInterface):
    """
    Interface to PySAT's API

    Requires that the 'python-sat' python package is installed:
    $ pip install python-sat

    See detailed installation instructions at:
    https://pysathq.github.io/installation

    Creates the following attributes (see parent constructor for more):
        - pysat_vpool: a pysat.formula.IDPool for the variable mapping
        - pysat_solver: a pysat.solver.Solver() (default: glucose4)

    The `DirectConstraint`, when used, calls a function on the `pysat_solver` object.
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
            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e

    @staticmethod
    def pb_supported():
        try:
            from pypblib import pblib
            from pysat.pb import PBEnc
            import pysat
            return True
        except ImportError as e:
            return False

    @staticmethod
    def solvernames():
        """
            Returns solvers supported by PySAT on your system
        """
        from pysat.solvers import SolverNames
        names = []
        for name, attr in vars(SolverNames).items():
            # issue with cryptosat, so we don't include it in our https://github.com/msoos/cryptominisat/issues/765
            if not name.startswith('__') and isinstance(attr, tuple) and not name == 'cryptosat':
                if name not in attr:
                    name = attr[-1]
                names.append(name)
        return names


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        PySAT clauses and solver object

        Only supports satisfaction problems (no objective)

        Arguments:
        - cpm_model: Model(), a CPMpy Model(), optional
        - subsolver: str, name of the pysat solver, e.g. glucose4
            see .solvernames() to get the list of available solver(names)
        """
        if not self.supported():
            raise Exception("CPM_pysat: Install the python package 'python-sat' to use this solver interface "
                            "(NOT the 'pysat' package!)")
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
            - time_limit:  maximum solve time in seconds (float, optional). Auto-interrups in case the
                           runtime exceeds given time_limit.
                           Warning: the time_limit is not very accurate at subsecond level
            - assumptions: list of CPMpy Boolean variables that are assumed to be true.
                           For use with s.get_core(): if the model is UNSAT, get_core() returns a small subset of assumption variables that are unsat together.
                           Note: the PySAT interface is statefull, so you can incrementally call solve() with assumptions and it will reuse learned clauses
        """

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        if assumptions is None:
            pysat_assum_vars = [] # default if no assumptions
        else:
            pysat_assum_vars = self.solver_vars(assumptions)
            self.assumption_vars = assumptions

        import time
        # set time limit?
        if time_limit is not None:
            from threading import Timer
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
                lit = self.solver_var(cpm_var)
                if lit in sol:
                    cpm_var._value = True
                elif -lit in sol:
                    cpm_var._value = False
                else: # not specified, dummy val
                    cpm_var._value = True

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None


        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created

            Transforms cpm_var into CNF literal using self.pysat_vpool
            (positive or negative integer)

            so vpool is the varmap (we don't use _varmap here)
        """

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            # just a view, get actual var identifier, return -id
            return -self.pysat_vpool.id(cpm_var._bv.name)
        elif isinstance(cpm_var, _BoolVarImpl):
            return self.pysat_vpool.id(cpm_var.name)
        else:
            raise NotImplementedError(f"CPM_pysat: variable {cpm_var} not supported")

    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the 'Adding a new solver' docs on readthedocs for more information.

            In the case of PySAT, the supported constraints are over Boolean variables: Boolean clauses, cardinality constraint (`sum`) and pseudo-Boolean constraints (`wsum`).

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = decompose_in_tree(cpm_cons, supported=frozenset({"alldifferent"}))
        cpm_cons = simplify_boolean(cpm_cons)
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_implies(cpm_cons)
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum","wsum", "and", "or", "bv"}))  # the core of the MIP-linearization
        cpm_cons = only_positive_coefficients(cpm_cons)
        return cpm_cons

    def __add__(self, cpm_expr_orig):
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
            elif isinstance(a1, Comparison) and a1.args[0].name == "sum":
                # implied sum comparison (a0->sum(bvs)<>val)
                # convert sum to clauses
                sum_clauses = self._pysat_cardinality(a1)
                # implication of conjunction is conjunction of individual implications
                nimplvar = [self.solver_var(~a0)]
                clauses = [nimplvar+c for c in sum_clauses]
                self.pysat_solver.append_formula(clauses)

            elif isinstance(a1, Comparison) and a1.args[0].name == "wsum":  # implied pseudo-boolean comparison (a0->wsum(ws,bvs)<>val)
                # implied sum comparison (a0->wsum([w,bvs])<>val or a0->(w*bv<>val))
                # convert wsum to clauses
                wsum_clauses = self._pysat_pseudoboolean(a1)
                # implication of conjunction is conjunction of individual implications
                nimplvar = [self.solver_var(~a0)]
                clauses = [nimplvar+c for c in wsum_clauses]
                self.pysat_solver.append_formula(clauses)
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

      return self

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
            For use with s.solve(assumptions=[...]). Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

            CPMpy will return only those assumptions which are False (in the UNSAT core)

            Note that there is no guarantee that the core is minimal.
            More advanced Minimal Unsatisfiable Subset are available in the 'examples' folder on GitHub

        """
        assert hasattr(self, 'assumption_vars'), "get_core(): requires a list of assumption variables, e.g. s.solve(assumptions=[...])"
        assert (self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE), "get_core(): solver must return UNSAT"

        assum_idx = frozenset(self.pysat_solver.get_core()) # to speed up lookup

        return [v for v in self.assumption_vars if self.solver_var(v) in assum_idx]


    def _pysat_cardinality(self, cpm_expr):
        """ Convert CPMpy comparison of `sum` (over Boolean variables) into PySAT list of clauses """
        assert_normalized_bool_lin(cpm_expr)
        if not CPM_pysat.pb_supported():
            raise ImportError("Please install PyPBLib: pip install pypblib")

        from pysat.card import CardEnc

        # unpack and transform to PySAT argument
        lhs, rhs = cpm_expr.args
        if lhs.name != "sum":
            raise NotSupportedError(
                f"PySAT: Expect {cpm_expr} to be a 'sum'"
            )

        lits = self.solver_vars(lhs.args)
        pysat_args = { "lits": lits, "bound": rhs, "vpool": self.pysat_vpool }

        if cpm_expr.name == "<=":
            return CardEnc.atmost(**pysat_args).clauses
        elif cpm_expr.name == ">=":
            return CardEnc.atleast(**pysat_args).clauses
        elif cpm_expr.name == "==":
            return CardEnc.equals(**pysat_args).clauses

    def _pysat_pseudoboolean(self, cpm_expr):
        """ Convert CPMpy comparison of `wsum` (over Boolean variables) into PySAT list of clauses """
        assert_normalized_bool_lin(cpm_expr)
        if cpm_expr.args[0].name != "wsum":
            raise NotSupportedError(
                f"PySAT: Expect {cpm_expr} to be a 'wsum'"
            )

        if not CPM_pysat.pb_supported():
            raise ImportError("Please install PyPBLib: pip install pypblib")

        from pysat.pb import PBEnc

        # unpack and transform to PySAT arguments
        lhs, rhs = cpm_expr.args
        lits = self.solver_vars(lhs.args[1])
        pysat_args = {"weights": lhs.args[0], "lits": lits, "bound": rhs, "vpool":self.pysat_vpool }

        if cpm_expr.name == "<=":
            return PBEnc.atmost(**pysat_args).clauses
        elif cpm_expr.name == ">=":
            return PBEnc.atleast(**pysat_args).clauses
        elif cpm_expr.name == "==":
            return PBEnc.equals(**pysat_args).clauses

def assert_normalized_bool_lin(cpm_expr):
    # we assume transformations are applied such that the below is true
    ERR = "PySAT: Expected {cpm_expr} to be a normalized linear constraint (`LinExpr <=/==/>= Constant`) over Boolean literals"
    if not isinstance(cpm_expr, Comparison):
        raise NotSupportedError(f"{ERR}, but did not receive a Comparison")
    lhs,rhs = cpm_expr.args
    if not is_int(rhs):
        raise NotSupportedError(
                f"{ERR}, but the RHS was not a Constant"
                )
    if lhs.name == "sum":
        lits = lhs.args
    elif lhs.name == "wsum":
        lits = lhs.args[1]
    else:
        raise NotSupportedError(
                f"{ERR}, but the LHS was not a `sum` or `wsum`"
            )
    for v in lits:
        if not isinstance(v, _BoolVarImpl):
            raise NotSupportedError(
                f"{ERR}, but {v} was not a Boolean literal"
            )

