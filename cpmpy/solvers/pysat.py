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
"""
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, NegBoolView, boolvar
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.utils import is_int, flatlist
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_implies, only_bv_reifies


class CPM_pysat(SolverInterface):
    """
    Interface to PySAT's API

    Requires that the 'python-sat' python package is installed:
    $ pip install python-sat

    See detailed installation instructions at:
    https://pysathq.github.io/installation.html

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
            if not name.startswith('__') and isinstance(attr, tuple):
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
            raise Exception("CPM_pysat: Install the python 'python-sat' package to use this solver interface (NOT the 'pysat' package!)")
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
                else:
                    # not specified...
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

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = decompose_in_tree(cpm_cons)
        cpm_cons = flatten_constraint(cpm_cons)
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_implies(cpm_cons)
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

            # BoolVar() -> BoolVar()
            if isinstance(a1, _BoolVarImpl):
                args = [~a0, a1]
                self.pysat_solver.add_clause(self.solver_vars(args))
            elif isinstance(a1, Operator) and a1.name == 'or':
                args = [~a0]+a1.args
                self.pysat_solver.add_clause(self.solver_vars(args))
            elif hasattr(a1, 'decompose'):  # implied global constraint
                self += a0.implies(cpm_expr.decompose())
            elif isinstance(a1, Comparison) and a1.args[0].name == "sum":  # implied sum comparison (a0->sum(bvs)<>val)
                # convert sum to clauses
                sum_clauses = self._pysat_cardinality(a1)
                # implication of conjunction is conjunction of individual implications
                nimplvar = [self.solver_var(~a0)]
                clauses = [nimplvar+c for c in sum_clauses]
                self.pysat_solver.append_formula(clauses)

        elif isinstance(cpm_expr, Comparison):
            # only handle cardinality encodings (for now)
            if isinstance(cpm_expr.args[0], Operator) and cpm_expr.args[0].name == "sum":
                # convert to clauses and post
                clauses = self._pysat_cardinality(cpm_expr)
                self.pysat_solver.append_formula(clauses)
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


    def _pysat_cardinality(self, cpm_compsum):
        """ convert CPMpy comparison of sum into PySAT list of clauses """
        # we assume transformations are applied such that the below is true
        if not isinstance(cpm_compsum, Comparison):
            raise NotSupportedError(f"PySAT card: input constraint must be Comparison -- {cpm_compsum}")
        if not is_int(cpm_compsum.args[1]):
            raise NotSupportedError(f"PySAT card: sum must have constant at rhs not {cpm_compsum.args[1]} -- {cpm_compsum}")
        if not cpm_compsum.args[0].name == "sum":
            raise NotSupportedError(f"PySAT card: input constraint must be sum, got {cpm_compsum.args[0].name} -- {cpm_compsum}")
        if not (all(isinstance(v, _BoolVarImpl) for v in cpm_compsum.args[0].args)):
            raise NotSupportedError(f"PySAT card: sum must be over Boolvars only -- {cpm_compsum.args[0]}")

        from pysat.card import CardEnc

        lits = self.solver_vars(cpm_compsum.args[0].args)
        bound = cpm_compsum.args[1]

        if cpm_compsum.name == "<":
            return CardEnc.atmost(lits=lits, bound=bound-1, vpool=self.pysat_vpool).clauses
        elif cpm_compsum.name == "<=":
            return CardEnc.atmost(lits=lits, bound=bound, vpool=self.pysat_vpool).clauses
        elif cpm_compsum.name == ">=":
            return CardEnc.atleast(lits=lits, bound=bound, vpool=self.pysat_vpool).clauses
        elif cpm_compsum.name == ">":
            return CardEnc.atleast(lits=lits, bound=bound+1, vpool=self.pysat_vpool).clauses
        elif cpm_compsum.name == "==":
            return CardEnc.equals(lits=lits, bound=bound, vpool=self.pysat_vpool).clauses
        elif cpm_compsum.name == "!=":
            # special cases with bounding 'hardcoded' for clarity
            if bound <= 0:
                return CardEnc.atleast(lits=lits, bound=bound+1, vpool=self.pysat_vpool).clauses
            elif bound >= len(lits):
                return CardEnc.atmost(lits=lits, bound=bound-1, vpool=self.pysat_vpool).clauses
            else:
                ## add implication literals for (strict) atleast/atmost, one must be true
                is_atleast = self.solver_var(boolvar())
                is_atmost = self.solver_var(boolvar())
                clauses = [[is_atleast, is_atmost]]
                clauses += [atl + [-is_atleast] for atl in
                            CardEnc.atleast(lits=lits, bound=bound+1, vpool=self.pysat_vpool).clauses]
                clauses += [atm + [-is_atmost] for atm in
                            CardEnc.atmost(lits=lits, bound=bound-1, vpool=self.pysat_vpool).clauses]
                return clauses

        raise NotImplementedError(f"Non-operator constraint {cpm_compsum} not supported by CPM_pysat")
