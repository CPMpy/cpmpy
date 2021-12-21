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

        CPM_pysat
"""
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import *
from ..expressions.variables import _BoolVarImpl, NegBoolView, boolvar
from ..expressions.utils import is_any_list
from ..transformations.get_variables import get_variables
from ..transformations.to_cnf import to_cnf

class CPM_pysat(SolverInterface):
    """
    Interface to PySAT's API

    Requires that the 'python-sat' python package is installed:
    $ pip install python-sat

    See detailed installation instructions at:
    https://pysathq.github.io/installation.html

    Creates the following attributes:
    pysat_vpool: a pysat.formula.IDPool for the variable mapping
    pysat_solver: a pysat.solver.Solver() (default: glucose4)
    And in the constructor of the superclass:
    user_vars: set(), variables in the original (non-transformed) model,
                    for reverse mapping the values after `solve()`
    cpm_status: SolverStatus(), the CPMpy status after a `solve()`
    tpl_model: object, TEMPLATE's model object
    _varmap: dict(), maps cpmpy variables to native solver variables
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


    def __init__(self, cpm_model=None, solver=None):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        PySAT clauses and solver object

        Only supports satisfaction problems (no objective)

        Arguments:
        - cpm_model: Model(), a CPMpy Model()
        - solver: str, name of the pysat solver, e.g. glucose4
            see .solvernames() to get the list of available solver(names)
        """
        if not self.supported():
            raise Exception("CPM_pysat: Install the python 'python-sat' package to use this solver interface (NOT the 'pysat' package!)")
        if cpm_model and cpm_model.objective is not None:
            raise Exception("CPM_pysat: only satisfaction, does not support an objective function")

        from pysat.formula import IDPool
        from pysat.solvers import Solver

        # determine solvername, set cpmpy name
        solvername = solver
        if solver is None or solvername == 'pysat':
            # default solver
            solvername = "glucose4" # something recent...
        elif solvername.startswith('pysat:'):
            solvername = solvername[6:] # strip 'pysat:'

        # Create solver specific objects
        self.pysat_vpool = IDPool()
        self.pysat_solver = Solver(use_timer=True, name=solvername)

        super().__init__(cpm_model, solver=solver, name=solvername)

    def __add__(self, cpm_con):
        """
        Direct solver access constraint addition,
        immediately adds the constraint to PySAT

        Note that we don't store the resulting cpm_model, we translate
        directly to the pysat model

        :param cpm_con CPMpy constraint, or list thereof
        :type cpm_con (list of) Expression(s)
        """
        self.user_vars.update(get_variables(cpm_con))

        # base case, just var or ~var
        if isinstance(cpm_con, _BoolVarImpl):
            self.pysat_solver.add_clause([self.solver_var(cpm_con)])
            return self

        # Post the constraint expressions to the solver
        # only CNF (list of disjunctions) supported for now
        cpm_cons = to_cnf(cpm_con)
        for con in cpm_cons:
            self._post_constraint(con)

        return self

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
            pysat_assum_vars = [self.solver_var(v) for v in assumptions]
            self.assumption_vars = assumptions

        # Run pysat solver
        pysat_status = self.pysat_solver.solve(assumptions=pysat_assum_vars)
        self.cpm_status.runtime = self.pysat_solver.time()

        # translate exit status
        self.cpm_status = SolverStatus(self.name)
        if pysat_status is True:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif pysat_status is False:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif pysat_status is None:
            # can happen when timeout is reached...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:  # another?
            raise NotImplementedError(pysat_status)  # a new status type was introduced, please report on github

        has_sol = self._solve_return(self.cpm_status)
        # translate solution values (of original vars only)
        if has_sol:
            sol = frozenset(self.pysat_solver.get_model())  # to speed up lookup
            # fill in variables
            for cpm_var in self.user_vars:
                lit = self.solver_var(cpm_var)
                if lit in sol:
                    cpm_var._value = True
                elif -lit in sol:
                    cpm_var._value = False
                else:
                    # not specified...
                    cpm_var._value = None
                    pass

        return has_sol

    def solver_var(self, cpm_var):
        """
            Transforms cpm_var into CNF literal using self.pysat_vpool
            (positive or negative integer)
        """
        if isinstance(cpm_var, NegBoolView):
            # just a view, get actual var identifier, return -id
            pysat_var = -self.pysat_vpool.id(cpm_var._bv.name)
        elif isinstance(cpm_var, _BoolVarImpl):
            pysat_var = self.pysat_vpool.id(cpm_var.name)
        else:
            raise NotImplementedError(f"CPM_pysat: variable {cpm_var} not supported")

        # Add to varmap
        self._varmap[cpm_var] = pysat_var
        return pysat_var

    def _post_constraint(self, cpm_expr):
        """
            Post a primitive CPMpy constraint to the pysat solver API
        """
        from pysat.card import CardEnc

        # base case, just var or ~var
        if isinstance(cpm_expr, _BoolVarImpl):
            self.pysat_solver.add_clause([self.solver_var(cpm_expr)])
        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == 'or':
                self.pysat_solver.add_clause([self.solver_var(var) for var in cpm_expr.args])
            else:
                raise NotImplementedError(
                    f"Automatic conversion of Operator {cpm_expr} to CNF not supported, please report on github.")
        elif isinstance(cpm_expr, Comparison):
            # only handle cardinality encodings
            if isinstance(cpm_expr.args[0], Operator) and cpm_expr.args[0].name == "sum" and all(
                    isinstance(v, _BoolVarImpl) for v in cpm_expr.args[0].args):
                lits = [self.solver_var(var) for var in cpm_expr.args[0].args]
                bound = cpm_expr.args[1]
                # TODO: Edge case where sum(x) < 0: Raises error
                if cpm_expr.name == "<":
                    atmost = CardEnc.atmost(lits=lits, bound=bound - 1, vpool=self.pysat_vpool)
                    self.pysat_solver.append_formula(atmost.clauses)
                elif cpm_expr.name == "<=":
                    atmost = CardEnc.atmost(lits=lits, bound=bound, vpool=self.pysat_vpool)
                    self.pysat_solver.append_formula(atmost.clauses)
                elif cpm_expr.name == ">=":
                    atleast = CardEnc.atleast(lits=lits, bound=bound, vpool=self.pysat_vpool)
                    self.pysat_solver.append_formula(atleast.clauses)
                elif cpm_expr.name == ">":
                    atleast = CardEnc.atleast(lits=lits, bound=bound + 1, vpool=self.pysat_vpool)
                    # self.pysat_solver.add_clause(atleast.clauses)
                    self.pysat_solver.append_formula(atleast)
                elif cpm_expr.name == "==":
                    equals = CardEnc.equals(lits=lits, bound=bound, vpool=self.pysat_vpool)
                    self.pysat_solver.append_formula(equals.clauses)
                # special cases with bounding 'hardcoded' for clarity
                elif cpm_expr.name == "!=" and bound <= 0:
                    atleast = CardEnc.atleast(lits=lits, bound=bound + 1, vpool=self.pysat_vpool)
                    self.pysat_solver.append_formula(atleast.clauses)
                elif cpm_expr.name == "!=" and bound >= len(lits):
                    atmost = CardEnc.atmost(lits=lits, bound=bound - 1, vpool=self.pysat_vpool)
                    self.pysat_solver.append_formula(atmost.clauses)
                elif cpm_expr.name == "!=":
                    ## add implication literal
                    is_atleast = self.solver_var(boolvar())
                    atleast = [cl + [-is_atleast] for cl in
                               CardEnc.atleast(lits=lits, bound=bound + 1, vpool=self.pysat_vpool).clauses]
                    self.pysat_solver.append_formula(atleast)

                    is_atmost = self.solver_var(boolvar())
                    atmost = [cl + [-is_atmost] for cl in
                              CardEnc.atmost(lits=lits, bound=bound - 1, vpool=self.pysat_vpool).clauses]
                    self.pysat_solver.append_formula(atmost)

                    ## add is_atleast or is_atmost
                    self.pysat_solver.add_clause([is_atleast, is_atmost])
                else:
                    raise NotImplementedError(f"Non-operator constraint {cpm_expr} not supported by CPM_pysat")
            else:
                raise NotImplementedError(f"Non-operator constraint {cpm_expr} not supported by CPM_pysat")

        else:
            raise NotImplementedError(f"Non-operator constraint {cpm_expr} not supported by CPM_pysat")

    def solution_hint(self, cpm_vars, vals):
        """
        PySAT supports warmstarting the solver with a feasible solution

        In PySAT, this is called setting the 'phases' or the 'polarities' of literals

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
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
