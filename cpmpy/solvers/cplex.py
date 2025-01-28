#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## cplex.py
##
"""
    Interface to the python 'docplex.mp' package

    Requires that the 'docplex' python package is installed:

        $ pip install docplex
    
    CPLEX, standing as an acronym for ‘Complex Linear Programming Expert’,
    is a high-performance mathematical programming solver specializing in linear programming (LP),
    mixed integer programming (MIP), and quadratic programming (QP).

    Documentation of the solver's own Python API:
    https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_cplex

    ==============
    Module details
    ==============
"""

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import *
from ..expressions.utils import argvals, argval
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.globalconstraints import DirectConstraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, only_positive_bv
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from ..transformations.safening import no_partial_functions

class CPM_cplex(SolverInterface):
    """
    Interface to the CPLEX solver.
    Requires that the 'docplex' python package is installed:
    $ pip install docplex

    docplex documentation:
    https://ibmdecisionoptimization.github.io/docplex-doc/
    You will also need to install CPLEX Optimization Studio from IBM's website.
    There is a free community version available.
    https://www.ibm.com/products/ilog-cplex-optimization-studio
    See detailed installation instructions at:
    https://www.ibm.com/docs/en/icos/22.1.2?topic=2212-installing-cplex-optimization-studio
    Creates the following attributes (see parent constructor for more):
    - cplex_model: object, CPLEX model object
    """

    _domp = None  # Static attribute to hold the docplex.cp module

    @classmethod
    def get_domp(cls):
        if cls._domp is None:
            import docplex.mp as domp  # Import only once
            cls._domp = domp
        return cls._domp

    @staticmethod
    def supported():
        return CPM_cplex.installed() and CPM_cplex.license_ok()

    @staticmethod
    def installed():
        # try to import the package
        try:
            import docplex.mp as domp
            return True
        except ModuleNotFoundError:
            return False

    @staticmethod
    def license_ok():
        if not CPM_cplex.installed():
            warnings.warn(
                f"License check failed, python package 'docplex' is not installed! Please check 'CPM_cplex.installed()' before attempting to check license.")
            return False
        else:
            try:
                from docplex.mp.model import Model
                mdl = Model()
                mdl.solve()
                return True
            except Exception as e:
                warnings.warn(f"Problem encountered with CPLEX installation: {e}")

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: a CPMpy Model()
        - subsolver: None, not used
        """
        if not self.installed():
            raise Exception("CPM_cplex: Install the python package 'docplex' to use this solver interface.")
        elif not self.license_ok():
            raise Exception("CPM_cplex: A problem occured during license check. Make sure your installed the CPLEX Optimization Studio")
        import docplex.mp.model as dmm

        self.cplex_model = dmm.Model()
        super().__init__(name="cplex", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.cplex_model


    def solve(self, time_limit=None, **kwargs):
        """
            Call the cplex solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Examples of supported arguments include:
                - context (optional) – An instance of context to be used in instead of the context this model was built with.
                - cplex_parameters (optional) – A set of CPLEX parameters to use instead of the parameters defined as context.cplex_parameters. Accepts either a RootParameterGroup object (obtained by cloning the model’s parameters), or a dict of path-like names and values.
                - checker (optional) – a string which controls which type of checking is performed. Possible values are: - ‘std’ (the default) performs type checks on arguments to methods; checks that numerical arguments are numbers, but will not check for NaN or infinity. - ‘numeric’ checks that numerical arguments are valid numbers, neither NaN nor math.infinity - ‘full’ performs all possible checks, the union of ‘std’ and ‘numeric’ checks. - ‘off’ performs no checking at all. Disabling all checks might improve performance, but only when it is safe to do so.
                - log_output (optional) – if True, solver logs are output to stdout. If this is a stream, solver logs are output to that stream object. Overwrites the context.solver.log_output parameter.
                - clean_before_solve (optional) – a boolean (default is False). Solve normally picks up where the previous solve left, but if this flag is set to True, a fresh solve is started, forgetting all about previous solves..

            For a full list of parameters, please visit https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html?#docplex.mp.model.Model.solve
            and for cplex parameters: https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-topical-list-parameters
        """
        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        if time_limit is not None:
            self.cplex_model.set_time_limit(time_limit)

        cplex_objective = self.cplex_model.get_objective_expr()
        self.cplex_model.solve(**kwargs)
        # all available solve details:
        # https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.sdetails.html#docplex.mp.sdetails.SolveDetails

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.cplex_model.solve_details.time
        # translate solver exit status to CPMpy exit status
        cplex_status = self.cplex_model.solve_details.status
        print(cplex_status)
        if cplex_status == "Feasible":
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif "infeasible" in cplex_status:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif cplex_status == "Unknown":
            # can happen when timeout is reached...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        elif "optimal" in cplex_status:
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        elif cplex_status == "JobFailed":
            self.cpm_status.exitstatus = ExitStatus.ERROR
        elif cplex_status == "JobAborted":
            self.cpm_status.exitstatus = ExitStatus.NOT_RUN
        else:  # another? This can happen when error during solve. Error message will be in the status.
            raise NotImplementedError(cplex_status)  # if a new status type was introduced, please report on GitHub

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                solver_val = self.solver_var(cpm_var).solution_value
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = int(solver_val)
            # set _objective_value
            if self.has_objective():
                obj_val = cplex_objective.solution_value
                if round(obj_val) == obj_val: # it is an integer?:
                    self.objective_value_ = int(obj_val)
                else: #  can happen with DirectVar or when using floats as coefficients
                    self.objective_value_ = float(obj_val)

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None
        return has_sol

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var):  # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view
        if isinstance(cpm_var, NegBoolView):
            raise Exception("Negative literals should not be part of any equation. "
                            "See /transformations/linearize for more details")

        # create if it does not exit
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.cplex_model.binary_var(cpm_var.name)
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.cplex_model.integer_var(cpm_var.lb, cpm_var.ub, name=str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        # return from cache
        return self._varmap[cpm_var]


    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective
                are premanently posted to the solver)
        """
        # make objective function non-nested
        (flat_obj, flat_cons) = (flatten_objective(expr))
        self += flat_cons
        get_variables(flat_obj, collect=self.user_vars)  # add potentially created constraints

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        if minimize:
            self.cplex_model.set_objective('min',obj)
        else:
            self.cplex_model.set_objective('max', obj)

    def has_objective(self):
        return self.cplex_model.is_optimized()

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function
        """
        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # sum
        if cpm_expr.name == "sum":
            return self.cplex_model.sum_vars(self.solver_vars(cpm_expr.args))
        if cpm_expr.name == "sub":
            a,b = self.solver_vars(cpm_expr.args)
            return a - b
        # wsum
        if cpm_expr.name == "wsum":
            w, t = cpm_expr.args
            return self.cplex_model.scal_prod(self.solver_vars(t), w)

        if cpm_expr.name == 'pow':
            a, b = self.solver_vars(cpm_expr.args)
            assert b == 2, f"only quadratic expressions are allowed, {cpm_expr} not supported"
            return a**2

        raise NotImplementedError("CPLEX: Not a known supported numexpr {}".format(cpm_expr))


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
        cpm_cons = no_partial_functions(cpm_cons)
        supported = {"min", "max", "abs", "alldifferent"} # alldiff has a specialized MIP decomp in linearize
        cpm_cons = decompose_in_tree(cpm_cons, supported)
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_implies(cpm_cons)  # anything that can create full reif should go above...
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "sub", "min", "max", "abs", "pow"}))  # the core of the MIP-linearization
        cpm_cons = only_positive_bv(cpm_cons)  # after linearization, rewrite ~bv into 1-bv
        return cpm_cons

    def __add__(self, cpm_expr_orig):
      """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: self
      """
      # add new user vars to the set
      get_variables(cpm_expr_orig, collect=self.user_vars)

      # transform and post the constraints
      for cpm_expr in self.transform(cpm_expr_orig):

        # Comparisons: only numeric ones as 'only_implies()' has removed the '==' reification for Boolean expressions
        # numexpr `comp` bvar|const
        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            cplexrhs = self.solver_var(rhs)

            # Thanks to `only_numexpr_equality()` only supported comparisons should remain
            if cpm_expr.name == '<=':
                cplexlhs = self._make_numexpr(lhs)
                self.cplex_model.add_constraint(cplexlhs <= cplexrhs)
            elif cpm_expr.name == '>=':
                cplexlhs = self._make_numexpr(lhs)
                self.cplex_model.add_constraint(cplexlhs >= cplexrhs)
            elif cpm_expr.name == '==':
                if isinstance(lhs, _NumVarImpl) \
                        or (isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum' or lhs.name == "sub"
                                                           or lhs.name == "pow")):
                    # a BoundedLinearExpression LHS, special case, like in objective
                    cplexlhs = self._make_numexpr(lhs)
                    self.cplex_model.add_constraint(cplexlhs == cplexrhs)

                elif lhs.name == 'div':
                    #TODO should be linearized away since cplex doesn't use integer division
                    raise NotSupportedError(f"cplex only supports division by constants, but got {lhs.args[1]}")
                    a, b = self.solver_vars(lhs.args)
                    self.cplex_model.add_constraint(a / b == cplexrhs)

                else:
                    # General constraints
                    if lhs.name == 'min':
                        self.cplex_model.add_constraint(self.cplex_model.min(self.solver_vars(lhs.args)) == cplexrhs)
                    elif lhs.name == 'max':
                        self.cplex_model.add_constraint(self.cplex_model.max(self.solver_vars(lhs.args)) == cplexrhs)
                    elif lhs.name == 'abs':
                        self.cplex_model.add_constraint(self.cplex_model.abs(self.solver_var(lhs.args[0])) == cplexrhs)
                    else:
                        raise NotImplementedError(
                        "Not a known supported cplex comparison '{}' {}".format(lhs.name, cpm_expr))
            else:
                raise NotImplementedError(
                "Not a known supported cplex comparison '{}' {}".format(lhs.name, cpm_expr))

        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            # Indicator constraints
            # Take form bvar -> sum(x,y,z) >= rvar
            cond, sub_expr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
            assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"
            if isinstance(cond, NegBoolView):
                cond, trigger_val = self.solver_var(cond._bv), 0
            else:
                cond, trigger_val = self.solver_var(cond), 1

            lhs, rhs = sub_expr.args
            if isinstance(lhs, _NumVarImpl) or lhs.name == "sum" or lhs.name == "wsum":
                lin_expr = self._make_numexpr(lhs)
            else:
                raise Exception(f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}")
            if sub_expr.name == "<=":
                self.cplex_model.add_indicator(cond, lin_expr <= self.solver_var(rhs), trigger_val)
            elif sub_expr.name == ">=":
                self.cplex_model.add_indicator(cond, lin_expr >= self.solver_var(rhs), trigger_val)
            elif sub_expr.name == "==":
                self.cplex_model.add_indicator(cond, lin_expr == self.solver_var(rhs), trigger_val)
            else:
                raise Exception(f"Unknown linear expression {sub_expr} name")

        # True or False
        elif isinstance(cpm_expr, BoolVal):
            self.cplex_model.add_constraint(cpm_expr.args[0])

        # a direct constraint, pass to solver
        elif isinstance(cpm_expr, DirectConstraint):
            cpm_expr.callSolver(self, self.cplex_model)

        else:
            raise NotImplementedError(cpm_expr)  # if you reach this... please report on github

      return self

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

        if time_limit is not None:
            self.cplex_model.set_time_limit(time_limit)
        if not call_from_model:
            warnings.warn("Adding constraints to solver object to find all solutions, "
                          "solver state will be invalid after this call!")

        solution_count = 0
        while solution_limit is None or solution_count < solution_limit:
            cplex_result = self.solve(time_limit=time_limit, **kwargs)
            if not cplex_result:
                break
            solution_count += 1
            if self.has_objective():
                # only find all optimal solutions, we know the optimal value now.
                self.cplex_model.add_constraint(self.cplex_model.get_objective_expr() == self.objective_value_)
                self.cplex_model.remove_objective()  # it's a hard constraint now in stead.

            solvars = []
            vals = []
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                cpm_value = cpm_var._value
                solvars.append(sol_var)
                vals.append(cpm_value)
            # exclude previous solution
            self += any([v != v.value() for v in self.user_vars if v.value() is not None])
            if display is not None:
                if isinstance(display, Expression):
                    print(argval(display))
                elif isinstance(display, list):
                    print(argvals(display))
                else:
                    display()  # callback
        return solution_count

