#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## pysat.py
##
"""
    Interface to PySAT's API

    This solver can be used if the model only has Boolean variables,
    and only logical constraints (and,or,xor,implies,==,!=)

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_pysat
"""
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import *
from ..expressions.variables import _BoolVarImpl, NegBoolView
from ..expressions.utils import is_any_list
from ..transformations.get_variables import get_variables_model
from ..transformations.to_cnf import to_cnf

class CPM_pysat(SolverInterface):
    """
    Interface to PySAT's API

    Requires that the 'python-sat' python package is installed:
    $ pip install python-sat

    See detailed installation instructions at:
    https://pysathq.github.io/installation.html

    Creates the following attributes:
    user_vars: variables in the original (unflattened) model (for reverse mapping the values after solve)
    pysat_vpool: a pysat.formula.IDPool for the variable mapping
    pysat_solver: a pysat.solver.Solver() (default: glucose4)
    cpm_status: the corresponding CPMpy status
    """

    @staticmethod
    def supported():
        try:
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
            if not name.startswith('__') and isinstance(attr, tuple):
                if name not in attr:
                    name = attr[-1]
                names.append(name)
        return names

    def __init__(self, cpm_model, solver=None):
        """
        Constructor of the solver object

        Requires a CPMpy model as input, and will create the corresponding
        PySAT clauses and solver object

        WARNING: CPMpy uses 'model' to refer to a constraint specification,
        the PySAT docs use 'model' to refer to a solution.

        Only supports satisfaction problems (no objective)

        Arguments:
        - cpm_model: a CPMpy Model()
        - solver: name of the pysat solver, e.g. glucose4
            see .solvernames() to get the list of available solver(names)
        """
        if not self.supported():
            raise Exception("CPM_pysat: Install the python 'python-sat' package to use this solver interface")
        if cpm_model.objective is not None:
            raise Exception("CPM_pysat: only satisfaction, does not support an objective function")
        from pysat.formula import IDPool
        from pysat.solvers import Solver

        super().__init__()

        # determine solvername, set cpmpy name
        solvername = solver
        if solver is None:
            # default solver
            solvername = "glucose4" # something recent...
        elif solvername.startswith('pysat:'):
            solvername = solvername[6:] # strip 'pysat:'
        self.name = "pysat:"+solvername

        # store original vars
        self.user_vars = get_variables_model(cpm_model)

        # ID pool of variables
        self.pysat_vpool = IDPool()

        # create constraint model (list of clauses)
        cnf = self.make_cnf(cpm_model)
        # create the solver instance
        self.pysat_solver = Solver(bootstrap_with=cnf.clauses, use_timer=True, name=solvername)


    def pysat_var(self, cpm_var):
        """
            Transforms cpm_var into CNF literal using self.pysat_vpool
            (positive or negative integer)
        """
        if isinstance(cpm_var, NegBoolView):
            # just a view, get actual var identifier, return -id
            return -self.pysat_vpool.id(cpm_var._bv.name)
        elif isinstance(cpm_var, _BoolVarImpl):
            return self.pysat_vpool.id(cpm_var.name)
        else:
            raise NotImplementedError(f"CPM_pysat: variable {cpm_var} not supported")

    def __add__(self, cpm_con):
        """
        Direct solver access constraint addition,
        immediately adds the constraint to PySAT

        Note that we don't store the resulting cpm_model, we translate
        directly to the ort_model

        :param cpm_con CPMpy constraint, or list thereof
        :type cpm_con (list of) Expression(s)
        """
        # base case, just var or ~var
        if isinstance(cpm_con, _BoolVarImpl):
            self.pysat_solver.add_clause([ self.pysat_var(cpm_con) ])
        else:
            cpm_con = to_cnf(cpm_con)
            for con in cpm_con:
                if isinstance(con, Operator) and con.name == 'or':
                    self.pysat_solver.add_clause([ self.pysat_var(var) for var in con.args ])
                else:
                    raise NotImplementedError("PySAT: to_cnf create non-clause constraint",con)
                    
        return self


    def solution_hint(self, cpm_vars, vals):
        """
        PySAT supports warmstarting the solver with a feasible solution

        In PySAT, this is called setting the 'phases' or the 'polarities' of literals

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
        literals = []
        for (cpm_var, val) in zip(cpm_vars, vals):
            lit = self.pysat_var(cpm_var)
            if val:
                # true, so positive literal
                literals.append(lit)
            else:
                # false, so negative literal
                literals.append(-lit)
        self.pysat_solver.set_phases(literals)


    def solve(self, time_limit=None, assumptions=None):
        """
            Call the PySAT solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - assumptions: list of CPMpy Boolean variables that are assumed to be true.
                           For use with s.get_core(): if the model is UNSAT, get_core() returns a small subset of assumption variables that are unsat together.
                           Note: the PySAT interface is statefull, so you can incrementally call solve() with assumptions and it will reuse learned clauses
        """

        # set time limit?
        if time_limit is not None:
            raise NotImplementedError("Didn't get to it yet, see pysat.solver.interrupt() for an example of what to implement")

        if assumptions is None:
            pysat_assum_vars = [] # default if no assumptions
        else:
            pysat_assum_vars = [self.pysat_var(v) for v in assumptions]
            self.assumption_vars = assumptions

        pysat_status = self.pysat_solver.solve(assumptions=pysat_assum_vars)

        return self._after_solve(pysat_status)

    def _after_solve(self, pysat_status):
        """
            To be called immediately after calling pysat solve() or solve_limited()
            Translate pysat status and variable values to CPMpy corresponding things

            - pysat_status: Boolean or None
        """

        # translate exit status
        self.cpm_status = SolverStatus(self.name)
        if pysat_status is True:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif pysat_status is False:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif pysat_status is None:
            # can happen when timeout is reached...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else: # another?
            raise NotImplementedError(pysat_status) # a new status type was introduced, please report on github

        # translate runtime
        self.cpm_status.runtime = self.pysat_solver.time()

        # translate solution values (of original vars only)
        if self.cpm_status.exitstatus == ExitStatus.FEASIBLE:
            sol = frozenset(self.pysat_solver.get_model()) # to speed up lookup
            # fill in variables
            for cpm_var in self.user_vars:
                lit = self.pysat_var(cpm_var)
                if lit in sol:
                    cpm_var._value = True
                elif -lit in sol:
                    cpm_var._value = False
                else:
                    # not specified...
                    cpm_var._value = None
                    pass

        return self._solve_return(self.cpm_status, pysat_status)


    def get_core(self):
        """
            For use with s.solve(assumptions=[...]). Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

            CPMpy will return only those variables that are False (in the UNSAT core)

            Note that there is no guarantee that the core is minimal, though this interface does open up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!
        """
        assert hasattr(self, 'assumption_vars'), "get_core(): requires a list of assumption variables, e.g. s.solve(assumptions=[...])"
        assert (self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE), "get_core(): solver must return UNSAT"

        assum_idx = frozenset(self.pysat_solver.get_core()) # to speed up lookup

        return [v for v in self.assumption_vars if self.pysat_var(v) in assum_idx]

    def make_cnf(self, cpm_model):
        """
            Makes a pysat.formulae CNF out of 
            a CPMpy model (only supports clauses for now)

            Typically only needed for internal use
        """
        from pysat.formula import CNF

        # check only BoolVarImpl (incl. NegBoolView)
        for var in get_variables_model(cpm_model):
            if not isinstance(var, _BoolVarImpl):
                raise NotImplementedError("Non-Boolean variables not (yet) supported. Reach out on github if you want to help implement a translation")

        # CNF object
        cnf = CNF()

        # Post the constraint expressions to the solver
        # only CNF (list of disjunctions) supported for now
        for con in to_cnf(cpm_model.constraints):
            # base case, just var or ~var
            if isinstance(con, _BoolVarImpl):
                cnf.append([ self.pysat_var(con) ])
            elif isinstance(con, Operator):
                if con.name == 'or':
                    cnf.append([ self.pysat_var(var) for var in con.args ])
                else:
                    raise NotImplementedError("Only 'or' operator supported by CPM_pysat for now (more possible with aiger, contact us on github")
                    
            else:
                raise NotImplementedError(f"Non-operator constraint {con} not supported by CPM_pysat")

        return cnf

