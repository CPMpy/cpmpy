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
from ..transformations.int2bool_onehot import int2bool_onehot, extract_boolvar, is_bool_model, to_bool_constraint, is_boolvar_constraint
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.variables import _BoolVarImpl, _IntVarImpl, NegBoolView, boolvar
from ..expressions.utils import is_any_list, is_int
from ..transformations.get_variables import get_variables, get_variables_model
from ..transformations.to_cnf import to_cnf

class CPM_pysat(SolverInterface):
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
            from pysat.card import CardEnc
            return True
        except ImportError as e:
            return False

    @staticmethod
    def pb_supported():
        try:
            from pypblib import pblib
            from pysat.pb import PBEnc
            from distutils.version import LooseVersion
            import pysat
            assert LooseVersion(pysat.__version__) >= LooseVersion("0.1.7.dev12"), "Upgrade PySAT version with command: pip3 install -U python-sat"
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
            raise Exception("CPM_pysat: only satisfaction, does not support an objective function")

        from pysat.formula import IDPool
        from pysat.solvers import Solver
        from pysat.card import CardEnc

        # determine subsolver
        if subsolver is None or subsolver == 'pysat':
            # default solver
            subsolver = "glucose4" # something recent...
        elif subsolver.startswith('pysat:'):
            subsolver = subsolver[6:] # strip 'pysat:'

        # initialise the native solver object
        self.pysat_vpool = IDPool()
        self.ivarmap = dict()

        if cpm_model is None:
            self.user_vars = []
            from pysat.formula import CNF
            cnf = CNF()
        # Model is bool variable based, there is no need for intvar transformations
        elif is_bool_model(cpm_model):
            # store original vars
            self.user_vars = get_variables_model(cpm_model)

            # create constraint model (list of clauses)
            cnf = self.make_cnf(cpm_model)
        # Model has int variables and needs to be encoded with boolean variables
        else:
            (self.ivarmap, bool_constraints) = int2bool_onehot(cpm_model)

            self.user_vars = get_variables(bool_constraints)

            # create constraint model (list of clauses)
            cnf = self._to_pysat_cnf(bool_constraints)
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


    def __add__(self, cpm_con):
        """
        Post a (list of) CPMpy constraints(=expressions) to the solver

        Note that we don't store the constraints in a cpm_model,
        we first transform the constraints into primitive constraints,
        then post those primitive constraints directly to the native solver

        :param cpm_con CPMpy constraint, or list thereof
        :type cpm_con (list of) Expression(s)
        """

        # flatten constraints and to cnf
        cpm_cons = to_cnf(cpm_con)
        for con in cpm_cons:
            if not is_boolvar_constraint(con):
                new_bool_constraints, new_ivarmap = to_bool_constraint(con, self.ivarmap)
                self.ivarmap.update(new_ivarmap)
                for new_bool_constraint in new_bool_constraints:
                    self._post_constraint(new_bool_constraint)
            else:
                self._post_constraint(con)

        return self

    def _post_constraint(self, cpm_expr):
        """
            Post a primitive CPMpy constraint to the native solver API
        """
        from pysat.card import CardEnc

        if isinstance(cpm_expr, _BoolVarImpl):
            # base case, just var or ~var
            self.pysat_solver.add_clause([self.solver_var(cpm_expr)])
        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == 'or':
                self.pysat_solver.add_clause(self.solver_vars(cpm_expr.args))
            else:
                raise NotImplementedError(
                    f"Automatic conversion of Operator {cpm_expr} to CNF not yet supported, please report on github.")
        elif isinstance(cpm_expr, Comparison):
            # only handle cardinality encodings (for now)
            if isinstance(cpm_expr.args[0], Operator) and cpm_expr.args[0].name == "sum" and all(
                    isinstance(v, _BoolVarImpl) for v in cpm_expr.args[0].args):
                lits = self.solver_vars(cpm_expr.args[0].args)
                bound = cpm_expr.args[1]

                clauses = []
                if cpm_expr.name == "<":
                    clauses += CardEnc.atmost(lits=lits, bound=bound - 1, vpool=self.pysat_vpool).clauses
                elif cpm_expr.name == "<=":
                    clauses += CardEnc.atmost(lits=lits, bound=bound, vpool=self.pysat_vpool).clauses
                elif cpm_expr.name == ">=":
                    clauses += CardEnc.atleast(lits=lits, bound=bound, vpool=self.pysat_vpool).clauses
                elif cpm_expr.name == ">":
                    clauses += CardEnc.atleast(lits=lits, bound=bound + 1, vpool=self.pysat_vpool).clauses
                elif cpm_expr.name == "==":
                    clauses += CardEnc.equals(lits=lits, bound=bound, vpool=self.pysat_vpool).clauses
                elif cpm_expr.name == "!=":
                    # special cases with bounding 'hardcoded' for clarity
                    if bound <= 0:
                        clauses += CardEnc.atleast(lits=lits, bound=bound + 1, vpool=self.pysat_vpool).clauses
                    elif bound >= len(lits):
                        clauses += CardEnc.atmost(lits=lits, bound=bound - 1, vpool=self.pysat_vpool).clauses
                    else:
                        ## add implication literal to atleast/atmost
                        is_atleast = self.solver_var(boolvar())
                        clauses += [atl + [-is_atleast] for atl in
                                    CardEnc.atleast(lits=lits, bound=bound + 1, vpool=self.pysat_vpool).clauses]

                        is_atmost = self.solver_var(boolvar())
                        clauses += [atm + [-is_atmost] for atm in
                                    CardEnc.atmost(lits=lits, bound=bound - 1, vpool=self.pysat_vpool).clauses]

                        ## add is_atleast or is_atmost
                        clauses.append([is_atleast, is_atmost])
                else:
                    raise NotImplementedError(f"Non-operator constraint {cpm_expr} not supported by CPM_pysat")

                # post the clauses
                self.pysat_solver.append_formula(clauses)
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
