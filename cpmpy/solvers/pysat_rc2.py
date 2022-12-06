#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## pysat_rc2.py
##
"""
    Interface to PySAT RC2 MaxSAT API.

    RC2 is an efficient core-guided MaxSAT solver part of the PySAT package
    for solving the (weighted) (partial) Maximum Satisfiability problem.

    Documentation is available at:
    https://pysathq.github.io/docs/html/api/examples/rc2.html

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_RC2
"""
from cpmpy.transformations import get_variables
from ..expressions.core import Operator, Expression
from .pysat import CPM_pysat
from .solver_interface import SolverInterface, SolverStatus, ExitStatus

class CPM_RC2(CPM_pysat):
    """
    Interface to PySAT RC2 MaxSAT solver at:
    https://pysathq.github.io/docs/html/api/examples/rc2.html

    Requires that the 'python-sat' python package is installed:
    $ pip install python-sat

    See detailed installation instructions at:
    https://pysathq.github.io/installation.html

    Creates the following attributes (see parent constructor for more):
    pysat_vpool: a pysat.formula.IDPool for the variable mapping
    rc2_solver: a pysat.examples.rc2 (default: glucose4)
    wcnf: keeping track of the weighted 
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pysat
            # there is actually a non-related 'pysat' package
            # while we need the 'python-sat' package, some more checks:
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
        Constructor of the native RC2 solver object.
        We keep track of the built cnf clauses to reinitialize the solver when assumptions are added to the MaxSAT solver as hard clauses.

        Requires a CPMpy model as input, and will create the corresponding
        PySAT clauses and RC2 solver object.

        Only supports maximal satisfaction problems

        Arguments:
        - cpm_model: Model(), a CPMpy Model(), optional
        """
        if not self.supported():
            raise Exception("CPM_pysat: Install the python 'python-sat' package to use this solver interface (NOT the 'pysat' package!)")

        from pysat.formula import WCNF
        from pysat.examples.rc2 import RC2

        if subsolver is None:
            subsolver = "glucose3"

        assert subsolver in CPM_RC2.solvernames(), f"Wrong solver ({subsolver}) selected from available: ({CPM_RC2.solvernames()})"

        # Keep track of transformed clauses in case the solving requires
        # assumptions
        self.wcnf = WCNF()

        self.rc2_solver = RC2(formula=self.wcnf, solver=subsolver)
        self.pysat_vpool = self.rc2_solver.pool

        # RESTARTS: keep track of subsolver
        self.subsolver = subsolver

        # RESTARTS: Keep track of solving state
        self._solved_assumption = False

        # Efficiency trick: If the solver is called incrementally
        # where assumption variables are the same, then the solver
        # does not need to be restarted.
        self._prev_pysat_assum_vars = set()

        SolverInterface.__init__(self, name="rc2", cpm_model=cpm_model)

    def _post_constraint(self, clause):
        """
        Post the CPMpy constraints transformed into Boolean SAT clauses
        to the RC2 solver.
        """
        self.wcnf.append(clause)

        self.rc2_solver.add_clause(clause)

    def _restart(self, wcnf):
        """
        Restart RC2 solver if assumptions were used during the solving.
        """
        if self.rc2_solver is not None:
            del self.rc2_solver

        self.rc2_solver = RC2(formula=wcnf, solver=self.subsolver)

        self._solved_assumption = False
        self._prev_pysat_assum_vars = set()

    def _add_assumptions(self, assumptions=None):
        """
        Assumptions are added as hard clauses to the RC2 solver.
        To improve efficiency, given assumptions are compared to the
        assumptions (`self._prev_pysat_assum_vars`) used in the previous
        solve call.

        RC2 is called incrementally if `_prev_pysat_assum_vars' is a subset
        of the given assumptions. There is no need to restart RC2 solver.

        Otherwise, restart RC2.
        """
        pysat_vars = set(self.solver_vars(assumptions))

        # No need to restart if solved incrementally
        if not set(self._prev_pysat_assum_vars).issubset(pysat_vars):
            self._restart(self.wcnf)

        for pysat_var in pysat_vars:
            self.rc2_solver.add_clause([pysat_var])

        self._solved_assumption = True

    def solve(self, assumptions=None):
        """
            Call the MaxSAT RC2 solver.
            Assumption are treated as hard constraints.

            1. In case, a new solve call is executed without changing the assumptions
               then the rc2 is not restarted and computes a solution

            2. In case new assumptions are added and they are not a superset of
               previous assumption variables, then the solver is restarted with the
               base constraints of the input model and solved with the current assumptions.

            Arguments:
            - assumptions: list of (negated) CPMpy Boolean variables that are assumed to be true.
                           MaxSAT solver does not support incremental solving and assumptions.
                           CPM_rc2 adds the assumptions as hard clauses to the MaxSAT solver.
        """
        if assumptions is not None:
            self._solved_assumption = True
            self._add_assumptions(assumptions)
        elif self._solved_assumption:
            self._restart(self.wcnf)

        sol = self.rc2_solver.compute()

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.rc2_solver.oracle_time()

        # translate exit status
        if sol is not None:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        else:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        if has_sol:
            sol = frozenset(sol)  # to speed up lookup
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

    def solveAll(self, assumptions=None, block=0, solution_limit=None, display=None):
        """
            A short-hand to (efficiently) compute all solutions, map them to CPMpy and optionally display the solutions.

            Arguments:
                - display: either a list of CPMpy expressions, called with the variables before value-mapping
                        default/None: nothing displayed
                - solution_limit: stop after this many solutions (default: None)
                - block: To block MSSes, one should set the block parameter to 1.
                         To block MCSes, set it to -1.
                         By the default (for blocking MaxSAT models), block is set to 0.

            Returns: number of solutions found
        """
        cb = RC2SolutionPrinter(self, display=display, solution_limit=solution_limit)
        
        ## Previous call was made with assumptions or not
        if assumptions is not None:
            self._solved_assumption = True
            self._add_assumptions(assumptions)
        elif self._solved_assumption:
            self._restart(self.wcnf)

        max_cost = None
        nbsols = 0

        for model in self.rc2_solver.enumerate(block=block):
            cb.on_solution_callback(model)
            cost = self.rc2_solver.cost
            if max_cost is None:
                max_cost = cost
            elif cost > max_cost:
                return nbsols
            elif solution_limit is not None and nbsols >= solution_limit:
                return nbsols

            nbsols +=1

        return nbsols

    def objective(self, expr, minimize=False):
        """
        MaxSAT is usually modelled as a minimization problem where
        the solution minimizes the total cost of the falsified clauses.
        For a traditional constraint optimization problem this corresponds to maximizing
        the objective function.

        Only linear terms are handled by
        """
        if minimize:
            raise NotImplementedError("RC2 does not support minimizing the objective.")

        if isinstance(expr, Operator) and expr.name == "sum":
            pysat_vars = self.solver_vars(expr.args)
            for pysat_var in pysat_vars:
                self.rc2_solver.add_clause([pysat_var], weight=1)
                self.wcnf.append([pysat_var], weight=1)
        elif isinstance(expr, Operator) and expr.name == "wsum":
            weights, vars = expr.args
            pysat_vars = self.solver_vars(vars)
            for weight, pysat_var in zip(weights, pysat_vars):
                self.rc2_solver.add_clause([pysat_var], weight=weight)
                self.wcnf.append([pysat_var], weight=weight)
        else:
            raise NotImplementedError(f"Expression {expr} not handled")

    def get_core(self):
        raise NotImplementedError("RC2 does not support unsat core extraction, check out the PySat solver for this functionality.")


class RC2SolutionPrinter:
    def __init__(self, solver, display=None, verbose=False):
        self.solver = solver
        # identify which variables to populate with their values
        self._cpm_vars = []
        self._display = display

        if isinstance(display, (list,Expression)):
            self._cpm_vars = get_variables(display)
        elif callable(display):
            # might use any, so populate all (user) variables with their values
            self._cpm_vars = solver.user_vars

    def on_solution_callback(self, sol):
        """Called on each new solution."""
        if len(self._cpm_vars):
            # populate values before printing
            for cpm_var in self._cpm_vars:
                # it might be an NDVarArray
                if hasattr(cpm_var, "flat"):
                    for cpm_subvar in cpm_var.flat:
                        lit = self.solver.solver_var(cpm_subvar)
                        if lit in sol:
                            cpm_var._value = True
                        elif -lit in sol:
                            cpm_var._value = False
                        else:
                            # not specified...
                            cpm_var._value = None

                else:
                    lit = self.solver.solver_var(cpm_subvar)
                    if lit in sol:
                        cpm_var._value = True
                    elif -lit in sol:
                        cpm_var._value = False
                    else:
                        # not specified...
                        cpm_var._value = None

            if isinstance(self._display, Expression):
                print(self._display.value())
            elif isinstance(self._display, list):
                # explicit list of expressions to display
                print([v.value() for v in self._display])
            else: # callable
                self._display()

