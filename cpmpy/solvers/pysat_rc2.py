#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## pysat_rc2.py
##
"""
    Interface to PySAT's API

    PySAT is a Python (2.7, 3.4+) toolkit, which aims at providing a simple and unified
    interface to a number of state-of-art Boolean satisfiability (SAT) solvers as well as
    to a variety of cardinality and pseudo-Boolean encodings.
    https://pysathq.github.io/

    This solver can be used if the model only has Boolean variables,
    and only logical constraints (and,or,xor,implies,==,!=) or cardinality constraints.

    Documentation of the solver's own Python API:
    https://pysathq.github.io/docs/html/api/solvers.html

    WARNING: CPMpy uses 'model' to refer to a constraint specification,
    the PySAT docs use 'model' to refer to a solution.

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

    def __add__(self, cpm_con):
        """
        Post a (list of) CPMpy constraints(=expressions) to the solver

        Note that we store the constraints in a WCNF formula,
        we first transform the constraints into primitive constraints,
        then post those primitive constraints directly to the native solver

        :param cpm_con CPMpy constraint, or list thereof
        :type cpm_con (list of) Expression(s)
        """
        # add new user vars to the set
        self.user_vars.update(get_variables(cpm_con))

        # apply transformations, then post internally
        cpm_cons = to_cnf(cpm_con)
        for con in cpm_cons:
            clauses = self._encode_constraint(con)
            self._post_clauses(clauses)
            ## keep track of the transformed clauses
            self.wcnf.extend(clauses)

        return self

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

    def objective(self, expr, minimize=False):
        """
        Push objective to maxsat solver.
        Only linear sums are supported for the moment.
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
                self.wcnf.append([pysat_assum_var], weight=1)
        else:
            raise NotImplementedError(f"Expression {expr} not handled")

    def get_core(self):
        raise NotImplementedError("RC2 does not support unsat core extraction, check out the PySat solver for this functionality.")