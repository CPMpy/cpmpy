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

        CPM_rc2
"""
from ..expressions.core import Operator
from ..expressions.variables import _BoolVarImpl
from ..transformations.to_cnf import to_cnf
from ..transformations.get_variables import get_variables
from .pysat import CPM_pysat
from .solver_interface import SolverInterface, SolverStatus, ExitStatus

class CPM_RC2(CPM_pysat):
    """
    Interface to PySAT's API

    Requires that the 'python-sat' python package is installed:
    $ pip install python-sat

    See detailed installation instructions at:
    https://pysathq.github.io/installation.html

    Creates the following attributes (see parent constructor for more):
    pysat_vpool: a pysat.formula.IDPool for the variable mapping
    pysat_solver: a pysat.solver.Solver() (default: glucose4)
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

    def __init__(self, cpm_model=None, subsolver="glucose3"):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        PySAT clauses and solver object

        Only supports satisfaction problems (no objective)

        Arguments:
        - cpm_model: Model(), a CPMpy Model(), optional
        """
        if not self.supported():
            raise Exception("CPM_pysat: Install the python 'python-sat' package to use this solver interface (NOT the 'pysat' package!)")

        from pysat.formula import WCNF
        from pysat.examples.rc2 import RC2

        assert subsolver in CPM_RC2.solvernames(), f"Wrong solver ({subsolver}) selected from available: ({CPM_RC2.solvernames()})"

        self.pysat_solver = RC2(formula=WCNF(), solver=subsolver)
        self.pysat_vpool = self.pysat_solver.pool

        SolverInterface.__init__(self, name="rc2", cpm_model=cpm_model)

    def solve(self, assumptions=None):
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
        if assumptions is not None:
            pysat_assum_vars = self.solver_vars(assumptions)
            for pysat_assum_var in pysat_assum_vars:
                self.pysat_solver.add_clause([pysat_assum_var])

        sol = self.pysat_solver.compute()

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.pysat_solver.oracle_time()

        # translate exit status
        if sol is not None:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif sol is None:
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

    def objective(self, expr, minimize):
        if minimize:
            raise NotImplementedError()

        if isinstance(expr, Operator) and expr.name == "sum":
            pysat_assum_vars = self.solver_vars(expr.args)
            for pysat_assum_var in pysat_assum_vars:
                self.pysat_solver.add_clause([pysat_assum_var], weight=1)
        elif isinstance(expr, Operator) and expr.name == "wsum":
            weights, vars = expr.args
            pysat_assum_vars = self.solver_vars(vars)
            for weight, pysat_assum_var in zip(weights, pysat_assum_vars):
                self.pysat_solver.add_clause([pysat_assum_var], weight=weight)
        else:
            raise NotImplementedError()

    def get_core(self):
        raise NotImplementedError("Does not work.")