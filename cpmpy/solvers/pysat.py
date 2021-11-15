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
from ..transformations.int2bool_onehot import int2bool_onehot, extract_boolvar, is_bool_model, to_bool_constraint, is_boolvar_constraint
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import *
from ..expressions.variables import _BoolVarImpl, NegBoolView, boolvar
from ..expressions.utils import is_int
from ..transformations.get_variables import get_variables, get_variables_model
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

    def __init__(self, cpm_model=None, solver=None):
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
            raise Exception("CPM_pysat: Install the python 'python-sat' package to use this solver interface (NOT the 'pysat' package!)")
        if cpm_model and cpm_model.objective is not None:
            raise Exception("CPM_pysat: only satisfaction, does not support an objective function")
        from pysat.formula import IDPool
        from pysat.solvers import Solver
        from pysat.card import CardEnc

        super().__init__(cpm_model, solver)

        # determine solvername, set cpmpy name
        solvername = solver
        if solver is None or solvername == 'pysat':
            # default solver
            solvername = "glucose4" # something recent...
        elif solvername.startswith('pysat:'):
            solvername = solvername[6:] # strip 'pysat:'
        self.name = "pysat:"+solvername

        # ID pool of variables
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
        directly to the internal pysat solver

        :param cpm_con CPMpy constraint, or list thereof
        :type cpm_con (list of) Expression(s)
        """

        # flatten constraints and to cnf
        cnf_cons = to_cnf(cpm_con)

        con_vars = get_variables(cnf_cons)

        # new variables should be added to user variables
        self.user_vars += [var for var in con_vars if var not in self.user_vars and var.is_bool()]

        new_constraints = []

        for constraint in cnf_cons:
            if not is_boolvar_constraint(constraint):
                new_bool_constraints, new_ivarmap = to_bool_constraint(constraint, self.ivarmap)
                new_constraints += new_bool_constraints
                self.user_vars += extract_boolvar(new_ivarmap)
                self.ivarmap.update(new_ivarmap)
            else:
                new_constraints.append(constraint)

        cnf = self._to_pysat_cnf(new_constraints)
        self.pysat_solver.append_formula(cnf)

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
            # assign value of original int model based on encoding
            if len(self.ivarmap) > 0:
                for var, val_bv_dict in self.ivarmap.items():
                    var._value = sum(value*bv.value() for value, bv in val_bv_dict.items())

        return self._solve_return(self.cpm_status)


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

    def _to_pysat_cnf(self, constraints):
        from pysat.formula import CNF
        from pysat.card import CardEnc

        # CNF object
        cnf = CNF()

        for con in to_cnf(constraints):

            # print(con.name, [(arg, type(arg)) for arg in con.args])
            # base case, just var or ~var
            if isinstance(con, _BoolVarImpl):
                cnf.append([ self.pysat_var(con) ])
            elif isinstance(con, Operator):
                if con.name == 'or':
                    cnf.append([ self.pysat_var(var) for var in con.args ])
                else:
                    raise NotImplementedError("Only 'or' operator supported by CPM_pysat for now (more possible with aiger, contact us on github")
            elif isinstance(con, Comparison):
                # only handle cardinality encodings
                left, right = con.args[0], con.args[1]
                bound = None
                lits, weights = None, None
                if not CPM_pysat.pb_supported():
                    raise ImportError("Please install PyPBLib: pip install pypblib")
                from pysat.pb import PBEnc

                if isinstance(left, Operator) and all(isinstance(v, _BoolVarImpl) for v in left.args) and is_int(right):
                    lits = [self.pysat_var(var) for var in left.args]
                    weights = [1]*len(lits)
                    bound = right

                # WEIGHTED !
                elif isinstance(left, Operator) and all(isinstance(v, _BoolVarImpl) or is_int(v) for v in left.args) and is_int(right):

                    lits = [self.pysat_var(v) for v in left.args if isinstance(v, _BoolVarImpl)]
                    weights = [v for v in left.args if is_int(v)]
                    bound = right
                else:
                    raise NotImplementedError(f"Comparison constraint {con} not supported by CPM_pysat")

                if con.name == "<":
                    atmost = PBEnc.leq(lits=lits, weights=weights, bound=bound-1, vpool=self.pysat_vpool)
                    if len(atmost.clauses) > 0:
                        cnf.extend(atmost.clauses)
                elif con.name == "<=":
                    atmost = PBEnc.leq(lits=lits, weights=weights, bound=bound,vpool=self.pysat_vpool)

                    if len(atmost.clauses) > 0:
                        cnf.extend(atmost.clauses)
                elif con.name == ">":
                    atleast = PBEnc.geq(lits=lits, weights=weights, bound=bound+1, vpool=self.pysat_vpool)

                    if len(atleast.clauses) > 0:
                        cnf.extend(atleast.clauses)
                elif con.name == ">=":
                    atleast = PBEnc.geq(lits=lits, weights=weights, bound=bound ,vpool=self.pysat_vpool)
                    if len(atleast.clauses) > 0:
                        cnf.extend(atleast.clauses)
                elif con.name == "==":
                    equals = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=self.pysat_vpool)
                    if len(equals.clauses) > 0:
                        cnf.extend(equals.clauses)
                elif con.name == "!=" and bound <= 0:
                    atleast = PBEnc.geq(lits=lits, weights=weights, bound=bound+1, vpool=self.pysat_vpool)
                    if len(atleast.clauses) > 0:
                        cnf.extend(atleast.clauses)
                elif con.name == "!=" and bound >= len(lits):
                    atmost = PBEnc.leq(lits=lits, weights=weights, bound=bound - 1, vpool=self.pysat_vpool)
                    if len(atmost.clauses) > 0:
                        cnf.extend(atmost.clauses)
                elif con.name == "!=":
                    # BUG with pblib solved in Pysat dev 0.1.7.dev12
                    is_atleast = self.pysat_var(boolvar())
                    is_atmost = self.pysat_var(boolvar())

                    atleast = PBEnc.geq(lits=lits, weights=weights, bound=bound+1, vpool=self.pysat_vpool).clauses
                    atleast_clauses = [cl + [-is_atleast] for cl in atleast]
                    if len(atleast) > 0:
                        cnf.extend(atleast_clauses)

                    atmost =  PBEnc.leq(lits=lits, weights=weights, bound=bound-1, vpool=self.pysat_vpool).clauses
                    atmost_clauses = [cl + [-is_atmost] for cl in atmost]
                    if len(atmost_clauses) > 0:
                        cnf.extend(atmost_clauses)

                    ## add is_atleast or is_atmost
                    cnf.append([is_atleast, is_atmost])

                else:
                    raise NotImplementedError(f"Comparison: {con} operator not supported by CPM_pysat")

            else:
                raise NotImplementedError(f"Other type Operator {con} not supported by CPM_pysat")

        return cnf

    def make_cnf(self, cpm_model):
        """
            Makes a pysat.formulae CNF out of 
            a CPMpy model (only supports clauses for now)

            Typically only needed for internal use
        """

        # check only BoolVarImpl (incl. NegBoolView)
        for var in get_variables_model(cpm_model):
            if not isinstance(var, _BoolVarImpl):
                raise NotImplementedError("Non-Boolean variables not (yet) supported. Reach out on github if you want to help implement a translation")

        assert all(constraint.is_bool() for constraint in cpm_model.constraints), f"Constraints \n{[constraint for constraint in cpm_model.constraints if not constraint.is_bool()]} should be a mapping to Boolean."

        return self._to_pysat_cnf(cpm_model.constraints)

