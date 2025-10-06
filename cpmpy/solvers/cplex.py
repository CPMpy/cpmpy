#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## cplex.py
##
"""
    Interface to CPLEX Optimizer using the python 'docplex.mp' package

    CPLEX, standing as an acronym for ‘Complex Linear Programming Expert’,
    is a high-performance mathematical programming solver specializing in linear programming (LP),
    mixed integer programming (MIP), and quadratic programming (QP).

    Always use :func:`cp.SolverLookup.get("cplex") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============
    
    Requires that both the 'docplex' and the 'cplex' python packages are installed:

    .. code-block:: console

        $ pip install docplex cplex

    Detailed installation instructions available at:
    https://ibmdecisionoptimization.github.io/docplex-doc/getting_started_python.html

    You will also need to install CPLEX Optimization Studio from IBM's website.
    There is a free community version available.
    https://www.ibm.com/products/ilog-cplex-optimization-studio
    See detailed installation instructions at:
    https://www.ibm.com/docs/en/icos/22.1.2?topic=2212-installing-cplex-optimization-studio
    
    It also requires an active licence.
    Academic license:
    https://community.ibm.com/community/user/ai-datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students

    The rest of this documentation is for advanced users.

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
import warnings
from typing import Optional

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import *
from ..expressions.utils import argvals, argval, eval_comparison, flatlist, is_bool
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.globalconstraints import DirectConstraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, only_positive_bv, only_positive_bv_wsum
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from ..transformations.safening import no_partial_functions

class CPM_cplex(SolverInterface):
    """
    Interface to the CPLEX solver.

    Creates the following attributes (see parent constructor for more):

    - cplex_model: object, CPLEX model object

    The :class:`~cpmpy.expressions.globalconstraints.DirectConstraint`, when used, 
    calls a function on the ``cplex_model`` object.

    Documentation of the solver's own Python API:
    https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html
    """

    @staticmethod
    def supported():
        return CPM_cplex.installed() and CPM_cplex.license_ok()

    @staticmethod
    def installed():
        # try to import the package
        try:
            import docplex.mp as domp
        except ModuleNotFoundError as e:
            warnings.warn(f"CPM_cplex: Could not import docplex: {e}")
            return False
        try:
            import cplex
            return True
        except ModuleNotFoundError as e:
            warnings.warn(f"CPM_cplex: Could not import cplex: {e}")
            return False

    @staticmethod
    def license_ok():
        if not CPM_cplex.installed():
            warnings.warn(
                f"License check failed, python package 'docplex' or 'cplex' is not installed! Please check 'CPM_cplex.installed()' before attempting to check license.")
            return False
        else:
            try:
                from docplex.mp.model import Model
                mdl = Model()
                mdl.solve()
                return True
            except Exception as e:
                warnings.warn(f"Problem encountered with CPLEX installation: {e}")
                return False

    @staticmethod
    def version() -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        
        Two version numbers get returned: ``<docplex version>/<solver version>``
        """
        try:
            import pkg_resources
            import cplex
            cpx = cplex.Cplex()
            return f"{pkg_resources.get_distribution('docplex').version}/{cpx.get_version()}"
        except (pkg_resources.DistributionNotFound, ModuleNotFoundError):
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: a CPMpy Model()
        - subsolver: None, not used
        """
        if not self.installed():
            raise Exception("CPM_cplex: Install the python packages 'docplex' and 'cplex' to use this solver interface.")
        elif not self.license_ok():
            raise Exception("CPM_cplex: A problem occured during license check. Make sure your installed the CPLEX Optimization Studio and that you have an active license.")

        from docplex.mp.model import Model
        self.cplex_model = Model()
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
            - kwargs:      any keyword argument, sets parameters of solver object and cplex parameters

            Supported keyword arguments are all solve parameters and cplex parameters:
                - solve_parameters:
                    - context (optional) – context to use during solve
                    - checker (optional) – a string which controls which type of checking is performed. (type checks etc.)
                    - log_output (optional) – if True, solver logs are output to stdout.
                    - clean_before_solve (optional) – default False (iterative solving)
                - cplex_parameters:
                    - any cplex parameter, see https://www.ibm.com/docs/en/icos/22.1.2?topic=cplex-list-parameters
                    - a well-know parameter is the `threads` parameter, used to set the number of threads to use during solve

            For a full description of the parameters, please visit https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html?#docplex.mp.model.Model.solve

            After solving, all solve details can be accessed through self.cplex_model.solve_details:
            https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.sdetails.html#docplex.mp.sdetails.SolveDetails
        """
        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)
            
        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            self.cplex_model.set_time_limit(time_limit)
    
        # Handle special arguments
        solve_args = ["clean_before_solve", "checker", "log_output"]
        cplex_params = {}
        
        for arg in list(kwargs.keys()):
            if arg == "context":
                self.cplex_model.context = kwargs[arg]
                del kwargs[arg]
            elif arg not in solve_args:
                # Set as cplex parameter
                cplex_params[arg] = kwargs[arg] 
                del kwargs[arg]
        
        self.cplex_model.solve(cplex_parameters=cplex_params, **kwargs)
        
        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.cplex_model.solve_details.time

        # translate solver exit status to CPMpy exit status
        cplex_status = self.cplex_model.solve_details.status
        if cplex_status == "Feasible":
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif "infeasible" in cplex_status:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif cplex_status == "Unknown":
            # can happen when timeout is reached...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        elif "optimal" in cplex_status:            
            if self.has_objective(): # COP
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else: # CSP
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif cplex_status == "JobFailed":
            self.cpm_status.exitstatus = ExitStatus.ERROR
        elif "aborted" in cplex_status:
            self.cpm_status.exitstatus = ExitStatus.NOT_RUN
        else:  # another? This can happen when error during solve.
            raise NotImplementedError(f"Translation of cplex status {cplex_status} to CPMpy status not implemented")

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
                    cpm_var._value = round(solver_val)
            # set _objective_value
            if self.has_objective():
                obj_val = self.cplex_model.get_objective_expr().solution_value
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
            raise ValueError("Negative literals should not be part of any equation. "
                            "Should have been removed by the only_positive_bv() transformation. "
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
            
            .. note::
                technical side note: any constraints created during conversion of the objective
                are premanently posted to the solver
        """
        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr)
        flat_obj = only_positive_bv_wsum(flat_obj)  # remove negboolviews
        get_variables(flat_obj, collect=self.user_vars)  # add potentially created variables
        self.add(flat_cons)

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

        # wsum
        if cpm_expr.name == "wsum":
            w, t = cpm_expr.args
            return self.cplex_model.scal_prod(self.solver_vars(t), w)

        if cpm_expr.name == "sub":
            a,b = self.solver_vars(cpm_expr.args)
            return a - b
        raise NotImplementedError("CPLEX: Not a known supported numexpr {}".format(cpm_expr))


    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the :ref:`Adding a new solver` docs on readthedocs for more information.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        # apply transformations, then post internally
        # expressions have to be linearized to fit in MIP model. See /transformations/linearize
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div"})  # linearize expects safe exprs
        supported = {"min", "max", "abs", "alldifferent"} # alldiff has a specialized MIP decomp in linearize
        cpm_cons = decompose_in_tree(cpm_cons, supported, csemap=self._csemap)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum', 'sub']), csemap=self._csemap)  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]), csemap=self._csemap)  # supports >, <, !=
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)  # anything that can create full reif should go above...
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "sub", "min", "max", "abs", "mul"}), csemap=self._csemap)  # CPLEX supports quadratic constraints and division by constants
        cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)  # after linearization, rewrite ~bv into 1-bv
        return cpm_cons

    def add(self, cpm_expr_orig):
      """
        Eagerly add a constraint to the underlying solver.

        Any CPMpy expression given is immediately transformed (through `transform()`)
        and then posted to the solver in this function.

        This can raise `NotImplementedError` for any constraint not supported after transformation

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
                        or (isinstance(lhs, Operator) and (lhs.name in {'sum', 'wsum', 'sub'})):
                    # a BoundedLinearExpression LHS, special case, like in objective
                    cplexlhs = self._make_numexpr(lhs)
                    self.cplex_model.add_constraint(cplexlhs == cplexrhs)

                elif lhs.name == 'mul':
                    raise NotImplementedError(f'CPLEX only supports quadratic constraints that define a convex region, i.e. quadratic equalities are not supported: {cpm_expr}')

                else:
                    # Global functions
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
                cond, trigger_val = self.solver_var(cond._bv), False
            else:
                cond, trigger_val = self.solver_var(cond), True

            lhs, rhs = sub_expr.args
            if isinstance(lhs, _NumVarImpl) or (lhs.name in {'sum', 'wsum', 'sub'}):
                lin_expr = self._make_numexpr(lhs)
            else:
                raise ValueError(f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}")
            constraint = eval_comparison(sub_expr.name, lin_expr, self.solver_var(rhs))
            self.cplex_model.add_indicator(cond, constraint, trigger_val)

        # True or False
        elif isinstance(cpm_expr, BoolVal):
            if cpm_expr.args[0]: # just true
                pass # do nothing
            else: # just false
                a = self.cplex_model.binary_var()
                self.cplex_model.add_constraint(a - a >= 1) # create a constraint that is always false

        # a direct constraint, pass to solver
        elif isinstance(cpm_expr, DirectConstraint):
            cpm_expr.callSolver(self, self.cplex_model)

        else:
            raise NotImplementedError(cpm_expr)  # if you reach this... please report on github

      return self
    __add__ = add  # avoid redirect in superclass

    def solution_hint(self, cpm_vars, vals):
        """
        CPLEX supports warmstarting the solver with a (in)feasible solution.
        This is done using MIP starts which provide the solver with a starting point
        for the branch-and-bound algorithm.

        The solution hint does NOT need to satisfy all constraints, it should just provide 
        reasonable default values for the variables. It can decrease solving times substantially, 
        especially when solving a similar model repeatedly.

        To learn more about solution hinting in CPLEX, see:
        https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html#docplex.mp.model.Model.add_mip_start

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
        # Flatten nested lists to handle test cases like solution_hint([a,[b]], [[[False]], True])
        cpm_vars = flatlist(cpm_vars)
        vals = flatlist(vals)
        
        # Validate input lengths
        if len(cpm_vars) != len(vals):
            raise ValueError(f"Number of variables ({len(cpm_vars)}) and values ({len(vals)}) must match")
        
        self.cplex_model.clear_mip_starts()

        # Create a MIP start solution using the proper docplex API
        if len(cpm_vars) > 0:
            warmstart = self.cplex_model.new_solution()
            for cpm_var, val in zip(cpm_vars, vals):
                # Convert boolean values to numeric (True -> 1, False -> 0) for docplex
                if is_bool(val):
                    val = int(val)
                warmstart.add_var_value(self.solver_var(cpm_var), val)

            self.cplex_model.add_mip_start(warmstart)

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            This is the generic implementation, solvers can overwrite this with
            a more efficient native implementation

            Arguments:
                display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                time_limit: stop after this many seconds (default: None)
                solution_limit: stop after this many solutions (default: None)
                any other keyword argument

            Returns: number of solutions found
        """
        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)

        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            self.cplex_model.set_time_limit(time_limit)

        if solution_limit is None:
            raise Exception(
                "CPLEX does not support searching for all solutions. If you really need all solutions, "
                "try setting solution limit to a large number")

        # Ask for multiple solutions
        self.cplex_model.context.cplex_parameters.mip.limits.populate = solution_limit
        self.cplex_model.context.cplex_parameters.mip.pool.intensity = 4 # (optional) max effort for finding solutions
        
        # For optimization problems, ensure we only get optimal solutions
        if self.has_objective():
            self.cplex_model.context.cplex_parameters.mip.pool.absgap = 0.0  # Only optimal solutions
            self.cplex_model.context.cplex_parameters.mip.pool.relgap = 0.0  # Only optimal solutions

        # Handle special arguments (same as in solve())
        solve_args = ["clean_before_solve", "checker", "log_output"]
        cplex_params = {}
        
        for arg in list(kwargs.keys()):
            if arg == "context":
                self.cplex_model.context = kwargs[arg]
                del kwargs[arg]
            elif arg not in solve_args:
                # Set as cplex parameter
                cplex_params[arg] = kwargs[arg] 
                del kwargs[arg]

        solutions_pool = self.cplex_model.populate_solution_pool(cplex_parameters=cplex_params, **kwargs)

        optimal_val = None
        opt_sol_count = 0

        # clear user vars if no solution found
        if solutions_pool is None:
            self.objective_value_ = None
            for var in self.user_vars:
                var._value = None

        else:
            for i, solution in enumerate(solutions_pool):
                if i == solution_limit:
                    break

                sol_obj_val = solution.get_objective_value()
                if optimal_val is None:
                    optimal_val = sol_obj_val
                if optimal_val is not None:
                    # sub-optimal solutions
                    if sol_obj_val != optimal_val:
                        continue
                opt_sol_count += 1

                # Translate solution to variables
                for cpm_var in self.user_vars:
                    solver_val = solution.get_value(self.solver_var(cpm_var))
                    if cpm_var.is_bool():
                        cpm_var._value = solver_val >= 0.5
                    else:
                        cpm_var._value = int(solver_val)

                # Translate objective
                if self.has_objective():
                    self.objective_value_ = sol_obj_val

                if display is not None:
                    if isinstance(display, Expression):
                        print(argval(display))
                    elif isinstance(display, list):
                        print(argvals(display))
                    else:
                        display()  # callback

        # Reset pool search mode to default
        self.cplex_model.context.cplex_parameters.mip.limits.populate = 1
        self.cplex_model.context.cplex_parameters.mip.pool.intensity = 0
        self.cplex_model.context.cplex_parameters.mip.pool.absgap = 1e-6  # Default value
        self.cplex_model.context.cplex_parameters.mip.pool.relgap = 1e-4  # Default value

        cplex_status = self.cplex_model.solve_details.status
        if cplex_status == "JobFailed": # something went wrong
            self.cpm_status.exitstatus = ExitStatus.ERROR
        elif "aborted" in cplex_status: # solve got prematurely aborted
            self.cpm_status.exitstatus = ExitStatus.NOT_RUN
        elif opt_sol_count: # found at least a single optimal solution
            if cplex_status == "populate solution limit exceeded": # reached solution limit
                # Sanity check
                if opt_sol_count == solution_limit: # reach set solution limit
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
                else:
                    raise ValueError(f"cplex returned status {cplex_status}, but solution count {opt_sol_count} doesn't match set limit of {solution_limit}. Please report on GitHub.")
            elif "all reachable solutions enumerated" in cplex_status or cplex_status == "integer optimal solution": # found all solutions
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            elif cplex_status == "Unknown" or cplex_status == "time limit exceeded": # reached time limit
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            else:
                raise NotImplementedError(f"Translation of cplex status {cplex_status} to CPMpy status not implemented. Please report on GitHub")
        else:
            if cplex_status == "Unknown": # reached time limit
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            elif cplex_status == "integer infeasible": # Did not find any solution
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            else: 
                raise NotImplementedError(f"Translation of cplex status {cplex_status} to CPMpy status not implemented. Please report on GitHub")

        return opt_sol_count