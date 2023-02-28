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

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import *
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..transformations.comparison import only_numexpr_equality
from ..transformations.flatten_model import flatten_constraint, flatten_objective, get_or_make_var
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, only_positive_bv
from ..transformations.reification import only_bv_implies, reify_rewrite
import numpy as np

class CPM_exact(SolverInterface):
    """
    Interface to the Python interface of Exact

    Requires that the 'exact' python package is installed:
    $ pip install exact
    See https://pypi.org/project/exact for more information.

    Creates the following attributes (see parent constructor for more):
    xct_solver: the Exact instance used in solve()
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import exact
            return True
        except ImportError as e:
            return False


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        Exact solver object xct_solver.

        xct_solver can be modified externally before
        calling solve(), a prime way to use more advanced solver features

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        """
        if not self.supported():
            raise Exception("Install 'exact' as a python package to use this solver interface")

        from exact import Exact as xct

        assert subsolver is None, "Exact does not allow subsolvers."

        # initialise the native solver object
        self.xct_solver = xct()
        self.xct_solver.setOption("inp-purelits", "0")    # no dominance breaking to preserve solutions
        self.xct_solver.setOption("inp-dombreaklim", "0") # no dominance breaking to preserve solutions

        # for solving with assumption variables,
        self.assumption_dict = None

        # objective can only be set once, so keep track of this
        self.solver_is_initialized = False
        self.has_objective = False
        self.objective_minimize = True

        # initialise everything else and post the constraints/objective
        super().__init__(name="exact", cpm_model=cpm_model)

    def getSolAndObj(self):
        if not self.xct_solver.hasSolution():
            self.objective_value_ = None
            return False

        # fill in variable values
        lst_vars = list(self.user_vars)
        exact_vals = self.xct_solver.getLastSolutionFor(self.solver_vars(lst_vars))
        for cpm_var, val in zip(lst_vars,exact_vals):
            cpm_var._value = bool(val) if isinstance(cpm_var, _BoolVarImpl) else val # xct value is always an int

        # translate objective
        self.objective_value_ = self.xct_solver.getObjectiveBounds()[1] # last upper bound to the objective
        if not self.objective_minimize:
            self.objective_value_ = -self.objective_value_

        return True

    def solve(self, time_limit=None, assumptions=None, **kwargs):
        # TODO: test this function
        """
            Call Exact

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - assumptions: list of CPMpy Boolean variables (or their negation) that are assumed to be true.
                           For repeated solving, and/or for use with s.get_core(): if the model is UNSAT,
                           get_core() returns a small subset of assumption variables that are unsat together.

            Additional keyword arguments:
            The Exact solver parameters are defined by https://gitlab.com/JoD/exact/-/blob/main/src/Options.hpp#L207
        """
        from exact import Exact as xct

        if not self.solver_is_initialized:
            assert not self.has_objective
            self.xct_solver.setOption("opt-boundupper", "0")
            self.xct_solver.init([],[])
            self.solver_is_initialized=True

        # TODO: set time limit?
        if time_limit is not None:
            self.xct_solver.setOption("timeout",str(time_limit))

        # set additional keyword arguments
        for (kw, val) in kwargs.items():
            self.xct_solver.setOption(kw,str(val))

        # set assumptions
        if assumptions is not None:
            assert(all(is_bool(v) for v in assumptions))
            assump_vals = [int(not isinstance(v, NegBoolView)) for v in assumptions]
            assump_vars = [self.solver_var(v._bv if isinstance(v, NegBoolView) else v) for v in assumptions]
            self.assumption_dict = {xct_var: (xct_val,cpm_assump) for (xct_var, xct_val, cpm_assump) in zip(assump_vars,assump_vals,assumptions)}
            self.xct_solver.setAssumptions(assump_vars, assump_vals);
            # NOTE: setAssumptions clears previous assumptions

        # call the solver, with parameters
        my_status = self.xct_solver.runFull(not self.has_objective)

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = 0 # TODO

        # translate exit status
        if my_status == 0:
            if self.xct_solver.hasSolution():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status == 1:
            assert self.xct_solver.hasSolution()
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status == 2:
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:
            raise NotImplementedError(my_status)  # a new status type was introduced, please report on github

        return self.getSolAndObj()

    def solveAll(self, display=None, time_limit=None, solution_limit=None, **kwargs):
        # TODO: test this function
        """
            Compute all solutions and optionally display the solutions.

            This is the generic implementation, solvers can overwrite this with
            a more efficient native implementation

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - any other keyword argument

            Returns: number of solutions found
        """

        assert not self.has_objective
        self.xct_solver.setOption("opt-boundupper", "0")

        if not self.solver_is_initialized:
            self.xct_solver.init([],[])
            self.solver_is_initialized=True

        # set time limit?
        if time_limit is not None:
            self.xct_solver.setOption("timeout",str(time_limit))

        # set additional keyword arguments
        for (kw, val) in kwargs.items():
            print(kw,val)
            print(type(self.xct_solver))
            self.xct_solver.setOption(kw,str(val))

        solsfound = 0
        while solution_limit == None or solsfound < solution_limit:
            status = 3
            while status == 3: # SolveState::INPROCESSED
                # call the solver, with parameters
                status = self.xct_solver.runOnce(not self.has_objective)
            assert status == 0 or status == 1, "Unexpected status code for Exact."
            if status == 0: # SolveState::UNSAT
                break
            else: # SolveState::SAT
                solsfound += 1
                self.getSolAndObj()
                # TODO: call callback / display?
                self.xct_solver.invalidateLastSol()

        return solsfound


    def solver_var(self, cpm_var, encoding="onehot"):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        assert not is_num(cpm_var)
        #if is_num(cpm_var):  # shortcut, eases posting constraints
            #return cpm_var

        # variables to be translated should be positive
        assert(not isinstance(cpm_var, NegBoolView))

        # return it if it already exists
        if cpm_var in self._varmap:
            return self._varmap[cpm_var]

        # create if it does not exist
        revar = str(cpm_var)
        if isinstance(cpm_var, _BoolVarImpl):
            self.xct_solver.addVariable(revar,0,1)
        elif isinstance(cpm_var, _IntVarImpl):
            lb = cpm_var.lb
            ub = cpm_var.ub
            if max(abs(lb),abs(ub)) > 1e18:
                # larger than 64 bit should be passed by string
                self.xct_solver.addVariable(revar,str(lb), str(ub), encoding)
            else:
                self.xct_solver.addVariable(revar,lb,ub, encoding)
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
        assert not self.has_objective, "Exact accepts an objective function only once."
        self.has_objective = True
        self.objective_minimize = minimize

        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr)
        self += flat_cons  # add potentially created constraints
        self.user_vars.update(get_variables(flat_obj))  # add objvars to vars

        # make objective function or variable and post
        xct_coefs,xct_vars,xct_rhs = self._make_numexpr(flat_obj,0)
        if not self.objective_minimize:
            xct_coefs = [-x for x in xct_coefs]

        self.xct_solver.init(xct_coefs,xct_vars)
        self.solver_is_initialized = True

    def _make_numexpr(self, lhs, rhs):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression
        """

        xcoefs = []
        xvars = []
        xrhs = 0

        if is_num(rhs):
            xrhs += rhs
        elif isinstance(rhs, _NumVarImpl):
            xcoefs += [-1]
            xvars += [self.solver_var(rhs)]
        else:
            raise NotImplementedError("Exact: Unexpected rhs {} for expression {}".format(rhs.name,rhs))

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
        elif lhs.name == "sub":
            assert len(lhs.args)==2
            xcoefs += [1, -1]
            xvars += [self.solver_var(lhs.args[0]), self.solver_var(lhs.args[1])]
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
        cpm_cons = flatten_constraint(cpm_expr)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        cpm_cons = only_bv_implies(cpm_cons)  # anything that can create full reif should go above...
        cpm_cons = linearize_constraint(cpm_cons)  # the core of the MIP-linearization
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        cpm_cons = only_positive_bv(cpm_cons)  # after linearisation, rewrite ~bv into 1-bv
        return cpm_cons

    @staticmethod
    def fix(o):
        return o.item() if isinstance(o, np.generic) else o

    def _add_xct_constr(self, xct_coefs,xct_vars,uselb,lb,useub,ub):
        maximum = max([max([abs(x) for x in xct_coefs]),abs(lb if uselb else 0),abs(ub if useub else 0)])
        if maximum > 1e18:
            self.xct_solver.addConstraint([str(x) for x in xct_coefs],xct_vars,uselb,str(lb),useub,str(ub))
        else:
            self.xct_solver.addConstraint([self.fix(x) for x in xct_coefs],xct_vars,uselb,self.fix(lb),useub,self.fix(ub))

    def _add_xct_reif(self,head,xct_coefs,xct_vars,lb):
        maximum = max([max([abs(x) for x in xct_coefs]),abs(lb)])
        if maximum > 1e18:
            self.xct_solver.addReification(head,[str(x) for x in xct_coefs],xct_vars,str(lb))
        else:
            self.xct_solver.addReification(head,[self.fix(x) for x in xct_coefs],xct_vars,self.fix(lb))

    def _add_xct_reif_right(self,head, xct_coefs,xct_vars,xct_rhs):
        maximum = max([max([abs(x) for x in xct_coefs]),abs(xct_rhs)])
        if maximum > 1e18:
            self.xct_solver.addRightReification(head,[str(x) for x in xct_coefs],xct_vars,str(xct_rhs))
        else:
            self.xct_solver.addRightReification(head,[self.fix(x) for x in xct_coefs],xct_vars,self.fix(xct_rhs))

    def _add_xct_reif_left(self,head, xct_coefs,xct_vars,xct_rhs):
        maximum = max([max([abs(x) for x in xct_coefs]),abs(xct_rhs)])
        if maximum > 1e18:
            self.xct_solver.addLeftReification(head,[str(x) for x in xct_coefs],xct_vars,str(xct_rhs))
        else:
            self.xct_solver.addLeftReification(head,[self.fix(x) for x in xct_coefs],xct_vars,self.fix(xct_rhs))


    def _post_constraint(self, cpm_expr, reifiable=False):
        """
            Post a supported CPMpy constraint directly to the underlying solver's API

            What 'supported' means depends on the solver capabilities, and in effect on what transformations
            are applied in `transform()`.

            Solvers can raise 'NotImplementedError' for any constraint not supported after transformation
        """
        from exact import Exact as xct

        # Comparisons: only numeric ones as 'only_bv_implies()' has removed the '==' reification for Boolean expressions
        # numexpr `comp` bvar|const
        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            xct_coefs, xct_vars, xct_rhs = self._make_numexpr(lhs,rhs)

            # Thanks to `only_numexpr_equality()` only supported comparisons should remain
            if cpm_expr.name == '<=':
                return self._add_xct_constr(xct_coefs, xct_vars, False, 0, True, xct_rhs)
            elif cpm_expr.name == '>=':
                return self._add_xct_constr(xct_coefs, xct_vars, True, xct_rhs, False, 0)
            elif cpm_expr.name == '==':
                # a BoundedLinearExpression LHS, special case, like in objective
                return self._add_xct_constr(xct_coefs, xct_vars, True, xct_rhs, True, xct_rhs)

            #elif lhs.name == 'mul':
                #assert len(lhs.args) == 2, "Exact only supports multiplication by a constant"
                #a, b = self.solver_vars(lhs.args)
                #assert is_num(b), "Exact only supports multiplication by a constant"
                #return self._add_xct_constr([b], [a], True, xct_rhs, False, xct_rhs)

            #elif lhs.name == 'div':
                #assert len(lhs.args), "Exact only supports division by a constant"
                #a, b = self.solver_vars(lhs.args)
                #assert is_num(b), "Exact only supports division by a constant"
                #return self._add_xct_constr([1], [a], True, xct_rhs*b, False, xct_rhs*b)

            raise NotImplementedError(
                "Constraint not supported by Exact '{}' {}".format(lhs.name, cpm_expr))

        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            # Indicator constraints
            # Take form bvar -> sum(x,y,z) >= rvar
            cond, sub_expr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
            assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"

            lhs, rhs = sub_expr.args
            assert isinstance(lhs, _NumVarImpl) or lhs.name == "sum" or lhs.name == "wsum",f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}"

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
                return
            elif sub_expr.name == ">=":
                if bool_val:
                    # a -> b >= c
                    return self._add_xct_reif_right(cond, xct_coefs,xct_vars,xct_rhs)
                else:
                    # !a -> b>=c
                    #  a <- b<c
                    #  a <- -b>=1-c
                    return self._add_xct_reif_left(cond, [-x for x in xct_coefs],xct_vars,1-xct_rhs)
            elif sub_expr.name == "<=":
                if bool_val:
                    # a -> b=<c
                    # a -> -b>=-c
                    return self._add_xct_reif_right(cond, [-x for x in xct_coefs],xct_vars,-xct_rhs)
                else:
                    # !a -> b=<c
                    #  a <- b>c
                    #  a <- b>=1+c
                    return self._add_xct_reif_left(cond, xct_coefs,xct_vars,1+xct_rhs)
            else:
                raise NotImplementedError(
                "Unexpected condition constraint for Exact '{}' {}".format(lhs.name,cpm_expr))

        # Global constraints
        else:
            self += cpm_expr.decompose()
            return

        raise NotImplementedError(cpm_expr)  # if you reach this... please report on github


    def get_core(self):
        raise NotImplementedError(
                "TODO: core extraction by Exact")

