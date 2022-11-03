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
        or-tools model and solver object (ort_model and ort_solver)

        ort_model and ort_solver can both be modified externally before
        calling solve(), a prime way to use more advanced solver features

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: None
        """
        if not self.supported():
            raise Exception("Install the python 'exact' package to use this solver interface")

        from exact import Exact as xct

        assert(subsolver is None)

        # initialise the native solver object
        self.xct_solver = xct()

        # for solving with assumption variables,
        # need to store mapping from ORTools Index to CPMpy variable
        self.assumption_dict = None

        # objective can only be set once, so keep track of this
        self.has_objective = False
        self.objective_minimize = True

        # initialise everything else and post the constraints/objective
        super().__init__(name="exact", cpm_model=cpm_model)


    def solve(self, time_limit=None, assumptions=None, **kwargs):
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

        # set time limit?
        if time_limit is not None:
            self.xct_solver.setOption("timeout",str(time_limit))

        if assumptions is not None:
            print("TODO: implement assumptions")
            assert(False)
            #ort_assum_vars = self.solver_vars(assumptions)
            ## dict mapping ortools vars to CPMpy vars
            #self.assumption_dict = {ort_var.Index(): cpm_var for (cpm_var, ort_var) in zip(assumptions, ort_assum_vars)}
            #self.ort_model.ClearAssumptions()  # because add just appends
            #self.ort_model.AddAssumptions(ort_assum_vars)
            ## workaround for a presolve with assumptions bug in ortools
            ## https://github.com/google/or-tools/issues/2649
            ## still present in v9.0
            #self.ort_solver.parameters.keep_all_feasible_solutions_in_presolve = True

        # set additional keyword arguments in sat_parameters.proto
        for (kw, val) in kwargs.items():
            print(kw,val)
            print(type(self.xct_solver))
            self.xct_solver.setOption(kw,str(val))

        # call the solver, with parameters
        my_status = self.xct_solver.runFull()
        has_sol = self.xct_solver.hasSolution()

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = 0 # TODO

        # translate exit status
        if my_status == 0:
            if has_sol:
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status == 2:
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:
            raise NotImplementedError(my_status)  # a new status type was introduced, please report on github

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            exact_vals = self.xct_solver.getLastSolutionFor([self.solver_var(cpm_var) for cpm_var in self.user_vars])
            i = 0
            for cpm_var in self.user_vars:
                cpm_var._value = exact_vals[i]
                if isinstance(cpm_var, _BoolVarImpl):
                    cpm_var._value = bool(cpm_var._value) # xct value is always an int
            i+=1

            # translate objective
            self.objective_value_ = self.xct_solver.getObjectiveBounds()[1] # last upper bound to the objective
            if not self.objective_minimize:
                self.objective_value_ = -self.objective_value_

        return has_sol

    def solveAll(self, display=None, time_limit=None, solution_limit=None, **kwargs):
        print("TODO: implement solveAll")
        assert(False)


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        assert not is_num(cpm_var)
        #if is_num(cpm_var):  # shortcut, eases posting constraints
            #return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return self.solver_var(cpm_var._bv).Not()

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
                self.xct_solver.addVariable(revar,str(lb), str(ub))
            else:
                self.xct_solver.addVariable(revar,lb,ub)
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
        assert(not self.has_objective)
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

    def _make_numexpr(self, lhs, rhs):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function
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
            raise NotImplementedError("Exact: Unexpected rhs expr {}".format(rhs))

        if is_num(lhs):
            xrhs -= lhs
        elif isinstance(lhs, _NumVarImpl):
            xcoefs += [1]
            xvars += [self.solver_var(lhs)]
        elif lhs.name == "sum":
            xcoefs += [1]*len(lhs.args)
            xvars += [self.solver_var(x) for x in lhs.args]
        elif lhs.expr_name == "wsum":
            xcoefs += [x[0] for x in lhs.args]
            xvars += [self.solver_var(x[1]) for x in lhs.args]
        else:
            raise NotImplementedError("Exact: Unexpected lhs expr {}".format(lhs))

        return xcoefs,xvars,xrhs


    def __add__(self, cpm_con):
        """
        Post a (list of) CPMpy constraints(=expressions) to the solver

        Note that we don't store the constraints in a cpm_model,
        we first transform the constraints into primitive constraints,
        then post those primitive constraints directly to the native solver

        :param cpm_con CPMpy constraint, or list thereof
        :type cpm_con (list of) Expression(s)
        """
        # add new user vars to the set
        self.user_vars.update(get_variables(cpm_con))

        # apply transformations, then post internally
        # expressions have to be linearized to fit in MIP model. See /transformations/linearize

        cpm_cons = flatten_constraint(cpm_con)
        cpm_cons = reify_rewrite(cpm_cons)
        cpm_cons = only_bv_implies(cpm_cons)
        cpm_cons = linearize_constraint(cpm_cons)
        cpm_cons = only_numexpr_equality(cpm_cons)
        cpm_cons = only_positive_bv(cpm_cons)

        for con in cpm_cons:
            self._post_constraint(con)

        return self

    def _add_xct_constr(self, xct_coefs,xct_vars,uselb,lb,useub,ub):
        maximum = max([max([abs(x) for x in xct_coefs]),abs(lb if uselb else 0),abs(ub if useub else 0)])
        if maximum > 1e18:
            self.xct_solver.addConstraint([str(x) for x in xct_coefs],xct_vars,uselb,str(lb),useub,str(ub))
        else:
            self.xct_solver.addConstraint(xct_coefs,xct_vars,uselb,lb,useub,ub)

    def _add_xct_reif(self,head,xct_coefs,xct_vars,lb):
        maximum = max([max([abs(x) for x in xct_coefs]),abs(lb)])
        if maximum > 1e18:
            self.xct_solver.addReification(head,[str(x) for x in xct_coefs],xct_vars,lb)
        else:
            self.xct_solver.addReification(head,xct_coefs,xct_vars,lb)


    def _post_constraint(self, cpm_expr, reifiable=False):
        """
            Post a primitive CPMpy constraint to the native solver API

            What 'primitive' means depends on the solver capabilities,
            more specifically on the transformations applied in `__add__()`

            Solvers do not need to support all constraints.
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

