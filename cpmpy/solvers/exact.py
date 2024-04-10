#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## exact.py
##
"""
    Interface to Exact

    Exact solves decision and optimization problems formulated as integer linear programs. Under the hood, it converts integer variables to binary (0-1) variables and applies highly efficient propagation routines and strong cutting-planes / pseudo-Boolean conflict analysis.

    https://gitlab.com/JoD/exact

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_exact
"""
import sys  # for stdout checking
import time

import pkg_resources
from pkg_resources import VersionConflict

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import *
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..transformations.comparison import only_numexpr_equality
from ..transformations.flatten_model import flatten_constraint, flatten_objective, get_or_make_var
from ..transformations.get_variables import get_variables
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.linearize import linearize_constraint, only_positive_bv
from ..transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from ..transformations.normalize import toplevel_list
from ..expressions.globalconstraints import DirectConstraint
from ..exceptions import NotSupportedError
from ..expressions.utils import flatlist

import numpy as np
import numbers

class CPM_exact(SolverInterface):
    """
    Interface to the Python interface of Exact

    Requires that the 'exact' python package is installed:
    $ pip install exact
    See https://pypi.org/project/exact for more information.

    Creates the following attributes (see parent constructor for more):
        - xct_solver: the Exact instance used in solve() and solveAll()
        - assumption_dict: maps Exact variables to (Exact value, CPM assumption expression)
    to recover which expressions were in the core
        - solver_is_initialized: whether xct_solver is initialized
        - self.objective_given: whether an objective function is given to xct_solver
        - self.objective_minimize: the direction of the optimization (if false then maximize)
    as Exact can only minimize
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import exact
            import pkg_resources
            pkg_resources.require("exact>=1.1.5")
            return True
        except ImportError as e:
            return False
        except VersionConflict:
            warnings.warn(f"CPMpy requires Exact version >=1.1.5 is required but you have version {pkg_resources.get_distribution('exact').version}")
            return False


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        Exact solver object xct_solver.

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: None
        """
        if not self.supported():
            raise Exception("Install 'exact' as a Python package to use this solver interface")
        
        assert subsolver is None, "Exact does not allow subsolvers."

        from exact import Exact as xct

        # initialise the native solver object
        self.xct_solver = xct()
        self.xct_solver.setOption("inp-purelits", "0")    # no dominance breaking to preserve solutions
        self.xct_solver.setOption("inp-dombreaklim", "0") # no dominance breaking to preserve solutions

        # for solving with assumption variables,
        self.assumption_dict = None

        # objective can only be set once, so keep track of this
        self.solver_is_initialized = False
        self.objective_given = False
        self.objective_minimize = True

        # encoding for integer variables in Exact - one of "log", "onehot", "order"
        # for domain sizes > 10, the most efficient one is probably "log", so that is the default.
        # "onehot" is less efficient, but required to prune domains with Exact (TODO: implement this).
        self.encoding="log"

        # initialise everything else and post the constraints/objective
        super().__init__(name="exact", cpm_model=cpm_model)

    def _fillObjAndVars(self):
        if not self.xct_solver.hasSolution():
            self.objective_value_ = None
            for cpm_var in self.user_vars:
                cpm_var._value = None
            return

        # fill in variable values
        lst_vars = list(self.user_vars)
        exact_vals = self.xct_solver.getLastSolutionFor(self.solver_vars(lst_vars))
        for cpm_var, val in zip(lst_vars,exact_vals):
            cpm_var._value = bool(val) if isinstance(cpm_var, _BoolVarImpl) else val # xct value is always an int

        # translate objective
        self.objective_value_ = self.xct_solver.getObjectiveBounds()[1] # last upper bound to the objective
        if not self.objective_minimize:
            self.objective_value_ = -self.objective_value_

    def solve(self, time_limit=None, assumptions=None, **kwargs):
        """
            Call Exact

            Overwrites self.cpm_status

            :param assumptions: CPMpy Boolean variables (or their negation) that are assumed to be true.
                           For repeated solving, and/or for use with s.get_core(): if the model is UNSAT,
                           get_core() returns a small subset of assumption variables that are unsat together.
            :type assumptions: list of CPMpy Boolean variables

            :param time_limit: optional, time limit in seconds
            :type time_limit: int or float

            Additional keyword arguments:
            The Exact solver parameters are defined by https://gitlab.com/JoD/exact/-/blob/main/src/Options.hpp#L207

            :return: Bool:
                - True      if a solution is found (not necessarily optimal, e.g. could be after timeout)
                - False     if no solution is found
        """
        from exact import Exact as xct

        if not self.solver_is_initialized:
            assert not self.objective_given
            # NOTE: initialization of exact is also how it fixes the objective function.
            # So we cannot call it before self.objective() (e.g., in the constructor).
            # And if self.objective() is not called, we still need to call it before solving.
            # This is something Exact needs to fix at some point.
            self.xct_solver.init([],[])
            self.solver_is_initialized=True
            self.xct_solver.setOption("verbosity","0")
        self.xct_solver.clearAssumptions()

        # set additional keyword arguments
        for (kw, val) in kwargs.items():
            self.xct_solver.setOption(kw,str(val))

        # set assumptions
        if assumptions is not None:
            assert all(v.is_bool() for v in assumptions), "Non-Boolean assumptions given to Exact: " + str([v for v in assumptions if not v.is_bool()])
            assump_vals = [int(not isinstance(v, NegBoolView)) for v in assumptions]
            assump_vars = [self.solver_var(v._bv if isinstance(v, NegBoolView) else v) for v in assumptions]
            self.assumption_dict = {xct_var: (xct_val,cpm_assump) for (xct_var, xct_val, cpm_assump) in zip(assump_vars,assump_vals,assumptions)}
            for x,v in zip(assump_vars,assump_vals):
                self.xct_solver.setAssumption(x, [v])

        # call the solver, with parameters
        start = time.time()
        my_status = self.xct_solver.runFull(self.objective_given, time_limit if time_limit is not None else 0)
        end = time.time()

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = end - start

        # translate exit status
        if my_status == 0: # found unsatisfiability
            if self.objective_given and self.xct_solver.hasSolution():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status == 1: # found solution, but not optimality proven
            assert self.xct_solver.hasSolution()
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status == 2: # found inconsistency over assumptions
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status == 3: # found timeout
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:
            raise NotImplementedError(my_status)  # a new status type was introduced, please report on github
        
        self._fillObjAndVars()
        
        # True/False depending on self.cpm_status
        return self._solve_return(self.cpm_status)

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - call_from_model: whether the method is called from a CPMpy Model instance or not
                - any other keyword argument

            Returns: number of solutions found
        """
        if self.objective_given:
            raise NotSupportedError("Exact does not support finding all optimal solutions.")

        if not self.solver_is_initialized:
            # NOTE: initialization of exact is also how it fixes the objective function.
            # So we cannot call it before self.objective() (e.g., in the constructor).
            # And if self.objective() is not called, we still need to call it before solving.
            # This is something Exact needs to fix at some point.
            self.xct_solver.init([],[])
            self.solver_is_initialized=True
            self.xct_solver.setOption("verbosity","0")
        self.xct_solver.clearAssumptions()

        if(time_limit): self.xct_solver.setOption("timeout",str(time_limit))

        # set additional keyword arguments
        for (kw, val) in kwargs.items():
            self.xct_solver.setOption(kw,str(val))

        solsfound = 0
        while solution_limit == None or solsfound < solution_limit:
            # call the solver, with parameters
            my_status = self.xct_solver.runFull(False,time_limit if time_limit is not None else 0)
            assert my_status in [0,1,2,3], "Unexpected status code for Exact."
            if my_status == 0: # found unsatisfiability
                self._fillObjAndVars() # erases the solution
                break
            elif my_status == 1: # found solution, but not optimality proven
                assert self.xct_solver.hasSolution()
                solsfound += 1
                self.xct_solver.invalidateLastSol() # TODO: pass user vars to this function
                if display is not None:
                    self._fillObjAndVars()
                    if isinstance(display, Expression):
                        print(display.value())
                    elif isinstance(display, list):
                        print([v.value() for v in display])
                    else:
                        display()  # callback
            elif my_status == 2: # found inconsistency
                assert False, "Error: inconsistency during solveAll should not happen, please warn the developers of this bug"
            elif my_status == 3: # found timeout
                return solsfound

        return solsfound

    def has_objective(self):
        """
            Returns whether the solver has an objective function or not.
        """
        return self.objective_given


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var):  # shortcut, eases posting constraints
            return cpm_var

        # variables to be translated should be positive
        assert not isinstance(cpm_var, NegBoolView), "Internal error: found negative Boolean variables where only positive ones were expected, please report."

        # return it if it already exists
        if cpm_var in self._varmap:
            return self._varmap[cpm_var]

        # create if it does not exist
        revar = str(cpm_var)
        if isinstance(cpm_var, _BoolVarImpl):
            self.xct_solver.addVariable(revar,0,1)
        elif isinstance(cpm_var, _IntVarImpl):
            lb, ub = cpm_var.get_bounds()
            if max(abs(lb),abs(ub)) > 1e18:
                # larger than 64 bit should be passed by string
                self.xct_solver.addVariable(revar,str(lb), str(ub), self.encoding)
            else:
                self.xct_solver.addVariable(revar,lb,ub, self.encoding)
        else:
            raise NotImplementedError("Not a known var {}".format(cpm_var))
        self._varmap[cpm_var] = revar
        return revar


    def objective(self, expr, minimize):
        """
            Post the given expression to the solver as objective to minimize/maximize

            - expr: Expression, the CPMpy expression that represents the objective function
            - minimize: Bool, whether it is a minimization problem (True) or maximization problem (False)

            'objective()' can only be called once
        """
        if self.objective_given:
            NotImplementedError("Exact accepts setting the objective function only once.")

        self.objective_given = True
        self.objective_minimize = minimize

        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr)
        self += flat_cons  # add potentially created constraints
        self.user_vars.update(get_variables(flat_obj))  # add objvars to vars

        # make objective function or variable and post
        xct_coefs,xct_vars,xct_rhs = self._make_numexpr(flat_obj,0)
        if not self.objective_minimize:
            xct_coefs = [-x for x in xct_coefs]
        
        # TODO: make this a custom transformation?
        newcoefs = []
        newvars = []
        for c,v in zip(xct_coefs,xct_vars):
            if is_num(v):
                xct_rhs += c*v
            else:
                newcoefs += [int(c)]
                newvars += [v]

        # NOTE: initialization of exact is also how it fixes the objective function.
        # So we cannot call it before self.objective() (e.g., in the constructor).
        # And if self.objective() is not called, we still need to call it before solving.
        # This is something Exact needs to fix at some point.

        if max(max(abs(x) for x in newcoefs),xct_rhs) > 1e18:
            self.xct_solver.init([str(x) for x in newcoefs],newvars,str(xct_rhs))
        else:
            self.xct_solver.init(newcoefs,newvars,xct_rhs)
        self.solver_is_initialized = True
        self.xct_solver.setOption("verbosity","0")

        # TODO: arbitrary sized bool case

    def _make_numexpr(self, lhs, rhs):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression
        """

        xcoefs = []
        xvars = []
        xrhs = 0
        
        assert is_num(rhs), "RHS of inequality should be numeric after transformations: {}".format(rhs)
        xrhs += rhs

        if is_num(lhs):
            xrhs -= lhs
        elif isinstance(lhs, _NumVarImpl):
            xcoefs += [1]
            xvars += [self.solver_var(lhs)]
        elif lhs.name == "sum":
            xcoefs = [1]*len(lhs.args)
            xvars = self.solver_vars(lhs.args)
        elif lhs.name == "wsum":
            xcoefs += lhs.args[0]
            xvars += self.solver_vars(lhs.args[1])
        else:
            raise NotImplementedError("Exact: Unexpected lhs {} for expression {}".format(lhs.name,lhs))

        return xcoefs,xvars,xrhs


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

        # apply transformations, then post internally
        # expressions have to be linearized to fit in MIP model. See /transformations/linearize
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = decompose_in_tree(cpm_cons, supported=frozenset({'alldifferent'})) # Alldiff has a specialzed MIP decomp
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum"]))  # supports >, <, !=
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_implies(cpm_cons)  # anything that can create full reif should go above...
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum","wsum"}))  # the core of the MIP-linearization
        cpm_cons = only_positive_bv(cpm_cons)  # after linearisation, rewrite ~bv into 1-bv
        return cpm_cons

        # NOTE: the transformations that are still done specifically for Exact are two-fold:
        # 1) transform '==' and '<=' to '>='
        # 2) transform implications with negative conditions to ones with positive consequences
        #
        # 1) seems quite general and is a candidate to function as an independent transformation.
        # 2) seems very solver-specific.

    @staticmethod
    def fix(o):
        return o.item() if isinstance(o, np.generic) else o

    def _add_xct_constr(self, xct_coefs,xct_vars,uselb,lb,useub,ub):
        if any(not isinstance(x, numbers.Integral) for x in xct_coefs+[lb,ub]):
            raise NotImplementedError("Exact requires all values to be integral")
        maximum = max([abs(x) for x in xct_coefs]+[abs(lb),abs(ub)])
        if maximum > 1e18:
            self.xct_solver.addConstraint([str(x) for x in xct_coefs],xct_vars,uselb,str(lb),useub,str(ub))
        else:
            self.xct_solver.addConstraint([self.fix(x) for x in xct_coefs],xct_vars,uselb,self.fix(lb),useub,self.fix(ub))

    def _add_xct_reif(self,head,xct_coefs,xct_vars,lb):
        if any(not isinstance(x, numbers.Integral) for x in xct_coefs+[lb]):
            raise NotImplementedError("Exact requires all values to be integral")
        maximum = max([abs(x) for x in xct_coefs]+[abs(lb)])
        if maximum > 1e18:
            self.xct_solver.addReification(head,[str(x) for x in xct_coefs],xct_vars,str(lb))
        else:
            self.xct_solver.addReification(head,[self.fix(x) for x in xct_coefs],xct_vars,self.fix(lb))

    def _add_xct_reif_right(self,head, xct_coefs,xct_vars,xct_rhs):
        maximum = max([abs(x) for x in xct_coefs]+[abs(xct_rhs)])
        if maximum > 1e18:
            self.xct_solver.addRightReification(head,[str(x) for x in xct_coefs],xct_vars,str(xct_rhs))
        else:
            self.xct_solver.addRightReification(head,[self.fix(x) for x in xct_coefs],xct_vars,self.fix(xct_rhs))

    def _add_xct_reif_left(self,head, xct_coefs,xct_vars,xct_rhs):
        maximum = max([abs(x) for x in xct_coefs]+[abs(xct_rhs)])
        if maximum > 1e18:
            self.xct_solver.addLeftReification(head,[str(x) for x in xct_coefs],xct_vars,str(xct_rhs))
        else:
            self.xct_solver.addLeftReification(head,[self.fix(x) for x in xct_coefs],xct_vars,self.fix(xct_rhs))


    def __add__(self, cpm_expr_orig):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        :param cpm_expr_orig: CPMpy expression, or list thereof
        :type cpm_expr_orig: Expression or list of Expression

        :return: self
        """
        from exact import Exact as xct

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        for cpm_expr in self.transform(cpm_expr_orig):
            # Comparisons: only numeric ones as 'only_implies()' has removed the '==' reification for Boolean expressions
            # numexpr `comp` bvar|const
            if isinstance(cpm_expr, Comparison):
                lhs, rhs = cpm_expr.args
                xct_coefs, xct_vars, xct_rhs = self._make_numexpr(lhs,rhs)

                # linearize removed '<', '>' and '!='
                if cpm_expr.name == '<=':
                    self._add_xct_constr(xct_coefs, xct_vars, False, 0, True, xct_rhs)
                elif cpm_expr.name == '>=':
                    self._add_xct_constr(xct_coefs, xct_vars, True, xct_rhs, False, 0)
                elif cpm_expr.name == '==':
                    # a BoundedLinearExpression LHS, special case, like in objective
                    self._add_xct_constr(xct_coefs, xct_vars, True, xct_rhs, True, xct_rhs)
                else:
                    raise NotImplementedError(
                        "Constraint not supported by Exact '{}' {}".format(lhs.name, cpm_expr))

            elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
                # Indicator constraints
                # Take form bvar -> sum(x,y,z) >= rvar
                cond, sub_expr = cpm_expr.args
                assert isinstance(cond, _BoolVarImpl), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
                assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"

                lhs, rhs = sub_expr.args

                xct_coefs, xct_vars, xct_rhs = self._make_numexpr(lhs,rhs)

                if isinstance(cond, NegBoolView):
                    cond, bool_val = self.solver_var(cond._bv), False
                else:
                    cond, bool_val = self.solver_var(cond), True

                if sub_expr.name == "==":
                    if bool_val:
                        # a -> b==c
                        # a -> b>=c and a -> -b>=-c
                        self._add_xct_reif_right(cond, xct_coefs,xct_vars,xct_rhs)
                        self._add_xct_reif_right(cond, [-x for x in xct_coefs],xct_vars,-xct_rhs)
                    else:
                        # !a -> b==c
                        # !a -> b>=c and !a -> -b>=-c
                        #  a <- b<c  and  a <- -b<-c
                        #  a <- -b>=1-c  and  a <- b>=1+c
                        self._add_xct_reif_left(cond, xct_coefs,xct_vars,1+xct_rhs)
                        self._add_xct_reif_left(cond, [-x for x in xct_coefs],xct_vars,1-xct_rhs)
                elif sub_expr.name == ">=":
                    if bool_val:
                        # a -> b >= c
                        self._add_xct_reif_right(cond, xct_coefs,xct_vars,xct_rhs)
                    else:
                        # !a -> b>=c
                        #  a <- b<c
                        #  a <- -b>=1-c
                        self._add_xct_reif_left(cond, [-x for x in xct_coefs],xct_vars,1-xct_rhs)
                elif sub_expr.name == "<=":
                    if bool_val:
                        # a -> b=<c
                        # a -> -b>=-c
                        self._add_xct_reif_right(cond, [-x for x in xct_coefs],xct_vars,-xct_rhs)
                    else:
                        # !a -> b=<c
                        #  a <- b>c
                        #  a <- b>=1+c
                        self._add_xct_reif_left(cond, xct_coefs,xct_vars,1+xct_rhs)
                else:
                    raise NotImplementedError(
                    "Unexpected condition constraint for Exact '{}' {}".format(lhs.name,cpm_expr))

            # True or False
            elif isinstance(cpm_expr, BoolVal):
                self._add_xct_constr([], [], True, 0 if cpm_expr.args[0] else 1, False, 0)

            # a direct constraint, pass to solver
            elif isinstance(cpm_expr, DirectConstraint):
                cpm_expr.callSolver(self, self.xct_solver)
                return self

            else:
                raise NotImplementedError(cpm_expr)  # if you reach this... please report on github
            
        return self

    def get_core(self):
        from exact import Exact as xct
        """
            For use with s.solve(assumptions=[...]). Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

            CPMpy will return only those variables that are False (in the UNSAT core)

            Note that there is no guarantee that the core is minimal, though this interface does open up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!
        """
        assert self.xct_solver.hasCore(), "get_core(): requires a core to be present in the solver, i.e., UNSAT should have been reached at least once"
        assert self.assumption_dict is not None,  "get_core(): requires a list of assumption variables, e.g. s.solve(assumptions=[...])"

        # return cpm_variables corresponding to Exact core
        return [self.assumption_dict[i][1] for i in self.xct_solver.getLastCore()]


    def solution_hint(self, cpm_vars, vals):
        """
        Exact supports warmstarting the solver with a partial feasible assignment.
            Requires version >= 1.2.1
        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """

        cpm_vars = flatlist(cpm_vars)
        vals = flatlist(vals)
        assert (len(cpm_vars) == len(vals)), "Variables and values must have the same size for hinting"
        try:
            pkg_resources.require("exact>=1.1.5")
            self.xct_solver.setSolutionHints(self.solver_vars(cpm_vars), vals)
        except VersionConflict:
            raise NotSupportedError("Upgrade Exact version to >=1.2.1 to support solution hinting")