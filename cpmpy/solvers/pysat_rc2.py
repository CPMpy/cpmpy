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
from ..expressions.core import Operator
from ..expressions.variables import _BoolVarImpl
from ..transformations.to_cnf import to_cnf
from ..transformations.get_variables import get_variables
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

    def __init__(self, cpm_model=None, subsolver="glucose3"):
        """
        Constructor of the native RC2 solver object.
        We keep track of the built cnf clauses to reinitliase the solver when assumptions are added to the maxsat solver as hard clauses.

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

    def _post_clauses(self, clauses):
        self.wcnf.extend(clauses)

        for clause in clauses:
            self.rc2_solver.add_clause(clause)

    def _restart(self, wcnf):
        if self.rc2_solver is not None:
            del self.rc2_solver
        self.rc2_solver = RC2(formula=wcnf, solver=self.subsolver)

        self._solved_assumption = False
        self._prev_pysat_assum_vars = set()

    def _add_assumptions(self, assumptions=None):
        pysat_assum_vars = set(self.solver_vars(assumptions))

        # No need to restart if solved incrementally
        if not set(self._prev_pysat_assum_vars).issubset(pysat_assum_var):
            self._restart(self.wcnf)

        for pysat_assum_var in pysat_assum_vars:
            self.rc2_solver.add_clause([pysat_assum_var])

        self._solved_assumption = True

    def solve(self, assumptions=None):
        """
            Call the MaxSAT RC2 solver.
            1. In case, a new solve call is executed without changing the assumptions
               then the rc2 is not restarted and computes a solution

            2. In case new assumptions are added, then the solver is restarted with the
               base constraints of the input model.

            Arguments:
            - assumptions: list of CPMpy Boolean variables that are assumed to be true.
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
            A shorthand to (efficiently) compute all solutions, map them to CPMpy and optionally display the solutions.

            It is just a wrapper around the use of `OrtSolutionPrinter()` in fact.

            Arguments:
                - display: either a list of CPMpy expressions, called with the variables before value-mapping
                        default/None: nothing displayed
                - solution_limit: stop after this many solutions (default: None)

            Returns: number of solutions found
        """
        if display:
            pysat_rc2_vars = set()
            for cpm_var in display:
                pysat_var = self.solver_var(cpm_var)
                pysat_rc2_vars.add(pysat_var)
                pysat_rc2_vars.add(-pysat_var)

        if assumptions is not None:
            self._solved_assumption = True
            self._add_assumptions(assumptions)
        elif self._solved_assumption:
            self._restart(self.wcnf)

        max_cost = None
        nbsols = 0
        for model in self.rc2_solver.enumerate(block=block):
            cost = self.rc2_solver.cost
            if max_cost is None:
                max_cost = cost
            elif cost > max_cost:
                return nbsols
            elif solution_limit is not None and nbsols >= solution_limit:
                return nbsols

            if display:
                sol = set(model) & pysat_rc2_vars
                print(f'Solution {sol} with cost={cost}')
            nbsols +=1

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
            pysat_assum_vars = self.solver_vars(expr.args)
            for pysat_assum_var in pysat_assum_vars:
                self.rc2_solver.add_clause([pysat_assum_var], weight=1)
                self.wcnf.append([pysat_assum_var], weight=1)
        elif isinstance(expr, Operator) and expr.name == "wsum":
            weights, vars = expr.args
            pysat_assum_vars = self.solver_vars(vars)
            for weight, pysat_assum_var in zip(weights, pysat_assum_vars):
                self.rc2_solver.add_clause([pysat_assum_var], weight=weight)
                self.wcnf.append([pysat_assum_var], weight=weight)
        else:
            raise NotImplementedError(f"Expression {expr} not handled")
        print(self.wcnf.hard)
        print(self.wcnf.soft)
        print(self.wcnf.wght)

    def get_core(self):
        raise NotImplementedError("RC2 does not support unsat core extraction, check out the PySat solver for this functionality.")