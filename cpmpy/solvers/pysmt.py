#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## pysmt.py
##
"""
    Interface to pySMT's API

    pySMT is a solver-agnostic library for SMT formulae manipulation and solving.
    It provides a unified interface to multiple SMT solvers including Z3, MathSAT,
    CVC4, Yices, Boolector, Picosat, and CUDD.

    Always use :func:`cp.SolverLookup.get("pysmt") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'pysmt' python package is installed:

    .. code-block:: console

        $ pip install pysmt

    Additionally, you need to install at least one SMT solver backend:

    .. code-block:: console

        $ pysmt-install --z3
        # or
        $ pysmt-install --msat
        # or install all available solvers
        $ pysmt-install --all

    See detailed installation instructions at:
    https://pysmt.readthedocs.io/en/latest/getting_started.html

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_pysmt
"""

from typing import Optional
import warnings

from importlib.metadata import version, PackageNotFoundError

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl
from ..expressions.utils import is_num, is_any_list, is_bool, is_int, is_boolexpr, eval_comparison
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.safening import no_partial_functions
from ..transformations.flatten_model import flatten_objective


class CPM_pysmt(SolverInterface):
    """
    Interface to pySMT's API.

    Creates the following attributes (see parent constructor for more):

    - pysmt_solver: object, pySMT's Solver() or Optimizer() object
    - pysmt_env: object, pySMT's environment

    Documentation of the solver's own Python API:
    https://pysmt.readthedocs.io/en/latest/api_ref.html
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pysmt
            # Check if at least one solver is available using the factory
            from pysmt.shortcuts import get_env
            env = get_env()
            return env.factory.has_solvers()
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e

    @classmethod
    def version(cls) -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        try:
            return version('pysmt')
        except PackageNotFoundError:
            return None

    @staticmethod
    def solvernames(installed:bool=True):
        """
            Returns solvers supported by pySMT (on your system).

            Arguments:
                installed (boolean): whether to filter the solvernames to those installed on your system (default: True)

            Returns:
                list of solver names
        """
        if not CPM_pysmt.supported():
            warnings.warn("pySMT is not installed or no solver backend is available on this system.")
            return []
        
        if installed:
            # Use factory to get all available solvers
            from pysmt.shortcuts import get_env
            env = get_env()
            # Get all solver names from the factory
            available = env.factory.all_solvers()
            return list(available.keys()) if hasattr(available, 'keys') else list(available)
        else:
            # Return all possible solvers pySMT supports
            return ["z3", "msat", "cvc4", "cvc5", "yices", "btor", "picosat", "bdd"]

    @classmethod
    def solverversion(cls, subsolver: str) -> Optional[str]:
        """
        Returns the version of the requested subsolver.

        Arguments:
            subsolver (str): name of the subsolver

        Returns:
            Version number of the subsolver if installed, else None
        """
        if not cls.supported():
            return None
        
        try:
            from pysmt.shortcuts import Solver
            with Solver(name=subsolver) as solver:
                # Try to get version from solver object
                if hasattr(solver, 'version'):
                    return solver.version
                # Some solvers expose version through their underlying solver
                if hasattr(solver, 'z3'):
                    import z3
                    return z3.get_version_string()
                return None
        except Exception:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: str, name of a subsolver (optional)
                     Available: "z3", "msat", "cvc4", "cvc5","yices", "btor", "picosat", "bdd"
                     If None, pySMT will auto-select an available solver
        """
        if not self.supported():
            raise Exception("CPM_pysmt: Install the python package 'pysmt' and at least one solver backend (e.g., 'pysmt-install --z3') to use this solver interface.")

        from pysmt.shortcuts import Solver, Optimizer, get_env

        # Get available solvers using factory
        env = get_env()
        available_solvers = env.factory.all_solvers()
        available = list(available_solvers.keys()) if hasattr(available_solvers, 'keys') else list(available_solvers)
        
        if not available:
            raise ValueError("No pySMT solver backends are available. Install at least one solver (e.g., 'pysmt-install --z3')")
        
        # Check if optimization is needed
        needs_optimization = cpm_model and cpm_model.has_objective()
        
        # Determine subsolver
        if subsolver is None:
            if needs_optimization:
                # For optimization, prefer solvers that support optimization (z3, msat)
                if "z3" in available:
                    subsolver = "z3"
                elif "cvc5" in available:
                    subsolver = "cvc5"
                else:
                    # Try to find any solver that supports optimization
                    available_optimizers = env.factory.all_optimizers()
                    opt_names = list(available_optimizers.keys()) if hasattr(available_optimizers, 'keys') else list(available_optimizers)
                    if opt_names:
                        subsolver = opt_names[0]
                    else:
                        subsolver = available[0]  # Use first available (will raise error if no optimizer)
            else:
                # Use preference list or first available
                try:
                    preference_list = env.factory.preferences if hasattr(env.factory, 'preferences') else None
                    if preference_list:
                        for pref_solver in preference_list:
                            if pref_solver in available:
                                subsolver = pref_solver
                                break
                except:
                    pass
                if subsolver is None:
                    subsolver = available[0] if available else None
        
        if subsolver not in available:
            raise ValueError(f"pySMT solver '{subsolver}' is not available. Available solvers: {available}")

        self.subsolver_name = subsolver
        
        # Create solver or optimizer
        if needs_optimization:
            # Check if optimizer is available for this solver
            available_optimizers = env.factory.all_optimizers()
            opt_names = list(available_optimizers.keys()) if hasattr(available_optimizers, 'keys') else list(available_optimizers)
            if subsolver in opt_names:
                self.pysmt_solver = Optimizer(name=subsolver)
                self._optimization_supported = True
            else:
                raise NotSupportedError(f"Optimizer not available for solver '{subsolver}'. Available optimizers: {opt_names}")
        else:
            self.pysmt_solver = Solver(name=subsolver)
            self._optimization_supported = False

        # Store environment for creating symbols
        self.pysmt_env = get_env()
        
        # Handle for objective (if using optimizer)
        self.obj_handle = None
        
        # Check if solver provides access to assertions (if not, we need to track them ourselves)
        self._needs_constraint_tracking = not hasattr(self.pysmt_solver, 'assertions')
        # Track pySMT constraints for SMT-LIB export only if solver doesn't provide assertions
        self._pysmt_constraints = [] if self._needs_constraint_tracking else None

        # initialise everything else and post the constraints/objective
        super().__init__(name="pysmt:"+subsolver, cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.pysmt_solver

    def solve(self, time_limit=None, assumptions=None, **kwargs):
        """
            Call the pySMT solver

            Arguments:
                time_limit (float, optional):       maximum solve time in seconds
                assumptions:                        list of CPMpy Boolean variables (or their negation) that are assumed to be true.
                **kwargs:                           any keyword argument, sets parameters of solver object

            Note: pySMT parameter setting depends on the underlying solver backend.
            See pySMT documentation for details.
        """
        from pysmt.exceptions import SolverReturnedUnknownResultError

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            # Convert to milliseconds (pySMT expects milliseconds)
            self.pysmt_solver.options = getattr(self.pysmt_solver, 'options', {})
            # Set timeout based on solver type
            if hasattr(self.pysmt_solver, 'set_timeout'):
                self.pysmt_solver.set_timeout(int(time_limit * 1000))

        # Handle assumptions
        pysmt_assum_vars = []
        self.assumption_dict = {}
        if assumptions is not None:
            pysmt_assum_vars = self.solver_vars(assumptions)
            self.assumption_dict = {pysmt_var: cpm_var for (cpm_var, pysmt_var) in zip(assumptions, pysmt_assum_vars)}

        # Set solver parameters
        for (key, value) in kwargs.items():
            if hasattr(self.pysmt_solver, 'set_option'):
                self.pysmt_solver.set_option(key, value)

        # Call the solver or optimizer
        try:
            if self.has_objective():
                # Use optimize() for optimization problems
                from pysmt.shortcuts import Minus, Int
                if hasattr(self.pysmt_solver, 'optimize'):
                    # For maximization, negate the objective (optimize minimizes)
                    goal = self._objective_expr_pysmt if self._minimize else Minus(Int(0), self._objective_expr_pysmt)
                    model, cost = self.pysmt_solver.optimize(goal)
                    my_status = model is not None
                    self._optimizer_model = model
                    self._optimizer_cost = cost
                else:
                    raise NotSupportedError(f"Optimization not supported by solver '{self.subsolver_name}'. Use an optimizer-enabled solver.")
            else:
                my_status = self.pysmt_solver.solve()
        except SolverReturnedUnknownResultError:
            my_status = None

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        # Try to get runtime from solver statistics
        if hasattr(self.pysmt_solver, 'get_statistics'):
            stats = self.pysmt_solver.get_statistics()
            if 'time' in stats:
                self.cpm_status.runtime = stats['time']
            else:
                self.cpm_status.runtime = 0
        else:
            self.cpm_status.runtime = 0

        # Translate solver exit status to CPMpy exit status
        if my_status is True:
            if self.has_objective():  # COP
                # For optimization, check if we used optimizer and got a result
                if hasattr(self, '_optimizer_model') and self._optimizer_model is not None:
                    # Optimizer found a solution - report as OPTIMAL if cost is valid
                    if hasattr(self, '_optimizer_cost') and self._optimizer_cost is not None:
                        self.cpm_status.exitstatus = ExitStatus.OPTIMAL
                    else:
                        self.cpm_status.exitstatus = ExitStatus.FEASIBLE
                else:
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            else:  # CSP
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status is False:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status is None:
            # unknown/timeout
            # Check if we have a model anyway
            try:
                model = self.pysmt_solver.get_model()
                if model:
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
                else:
                    self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            except:
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:
            raise NotImplementedError(f"Unknown solver status: {my_status}")

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            try:
                # Get model - use optimizer model if available, otherwise get from solver
                if hasattr(self, '_optimizer_model') and self._optimizer_model is not None:
                    model = self._optimizer_model
                    # Use optimizer cost for objective value
                    if hasattr(self, '_optimizer_cost') and self._optimizer_cost is not None:
                        if self._minimize:
                            self.objective_value_ = int(self._optimizer_cost.constant_value()) if hasattr(self._optimizer_cost, 'constant_value') else int(self._optimizer_cost)
                        else:
                            # Negate back since we negated for optimization
                            cost_val = int(self._optimizer_cost.constant_value()) if hasattr(self._optimizer_cost, 'constant_value') else int(self._optimizer_cost)
                            self.objective_value_ = -cost_val
                else:
                    model = self.pysmt_solver.get_model()
                
                # fill in variable values
                for cpm_var in self.user_vars:
                    sol_var = self.solver_var(cpm_var)
                    if isinstance(cpm_var, _BoolVarImpl):
                        val = model.get_value(sol_var)
                        cpm_var._value = val.constant_value() if hasattr(val, 'constant_value') else bool(val)
                    elif isinstance(cpm_var, _IntVarImpl):
                        val = model.get_value(sol_var)
                        cpm_var._value = int(val.constant_value()) if hasattr(val, 'constant_value') else int(val)
                
                # translate objective, for optimisation problems only (if not already set)
                if self.has_objective() and self.objective_value_ is None and self.obj_handle:
                    try:
                        obj_val = model.get_value(self.obj_handle)
                        self.objective_value_ = int(obj_val.constant_value()) if hasattr(obj_val, 'constant_value') else int(obj_val)
                    except:
                        # Fallback: evaluate objective expression
                        obj_expr = self._pysmt_expr(self._objective_expr)
                        obj_val = model.get_value(obj_expr)
                        self.objective_value_ = int(obj_val.constant_value()) if hasattr(obj_val, 'constant_value') else int(obj_val)
            except Exception as e:
                warnings.warn(f"Error retrieving solution values: {e}")
                for cpm_var in self.user_vars:
                    cpm_var._value = None
        else:  # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None

        return has_sol

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        from pysmt.shortcuts import Symbol, FreshSymbol
        from pysmt.typing import BOOL, INT

        if is_num(cpm_var):  # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            from pysmt.shortcuts import Not
            return Not(self.solver_var(cpm_var._bv))

        # create if it does not exist
        if cpm_var not in self._varmap:
            self.user_vars.add(cpm_var)
            if isinstance(cpm_var, _BoolVarImpl):
                revar = Symbol(str(cpm_var), BOOL)
            elif isinstance(cpm_var, _IntVarImpl):
                revar = Symbol(str(cpm_var), INT)
                # Set bounds as constraints
                lb_expr = self._pysmt_expr(cpm_var.lb)
                ub_expr = self._pysmt_expr(cpm_var.ub)
                self.pysmt_solver.add_assertion(revar >= lb_expr)
                self.pysmt_solver.add_assertion(revar <= ub_expr)
            else:
                raise NotImplementedError(f"Not a known var {cpm_var}")
            self._varmap[cpm_var] = revar

        return self._varmap[cpm_var]

    def has_objective(self):
        return self._optimization_supported and self.obj_handle is not None

    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective
            are permanently posted to the solver)
        """
        # Check if solver is an optimizer by checking if it has optimize method
        if not hasattr(self.pysmt_solver, 'optimize'):
            raise NotSupportedError(f"Optimization not supported. Use Optimizer for optimization problems. Initialize with a model that has an objective, or recreate solver with optimization support.")

        # Store expression for evaluation
        self._objective_expr = expr

        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr, csemap=self._csemap)
        self += flat_cons  # add potentially created constraints
        get_variables(flat_obj, collect=self.user_vars)  # add objvars to vars

        # Convert flattened objective to pySMT expression
        obj = self._pysmt_expr(flat_obj)
        
        # Store objective expression - pySMT Optimizer uses optimize() method during solve
        # For maximization, we'll negate the objective
        self._objective_expr_pysmt = obj
        self._minimize = minimize
        self.obj_handle = obj  # Store for evaluation

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
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel=frozenset({"div", "mod"}))
        supported = {"alldifferent", "xor", "ite"}  # pySMT accepts these
        cpm_cons = decompose_in_tree(cpm_cons, supported, csemap=self._csemap)
        return cpm_cons

    def add(self, cpm_expr):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

            Arguments:

                cpm_expr (Expression or list of Expression): CPMpy expression, or list thereof

            Returns:
                self
        """
        # add new user vars to the set
        get_variables(cpm_expr, collect=self.user_vars)
        
        # transform and post the constraints
        for cpm_con in self.transform(cpm_expr):
            # translate each expression tree, then post straight away
            pysmt_con = self._pysmt_expr(cpm_con)
            self.pysmt_solver.add_assertion(pysmt_con)
            # Track pySMT constraints for SMT-LIB export only if solver doesn't provide assertions
            if self._needs_constraint_tracking:
                self._pysmt_constraints.append(pysmt_con)

        return self
    __add__ = add  # avoid redirect in superclass

    @classmethod
    def _cpm_to_pysmt_expr_impl(cls, cpm_expr, var_resolver, handle_direct_constraint=None):
        """
        Shared implementation for converting CPMpy expressions to pySMT expressions.
        
        Arguments:
            cpm_expr: CPMpy expression to convert
            var_resolver: Callable that takes a CPMpy variable and returns a pySMT symbol
            handle_direct_constraint: Optional callable for handling DirectConstraint expressions
        
        Returns:
            pySMT formula
        """
        from pysmt.shortcuts import Symbol, And, Or, Not, Implies, Plus, Times, Minus, Div, Pow, Ite, Int, Bool, FreshSymbol, AllDifferent, Xor, GE, LE, GT, LT, Equals
        from pysmt.typing import BOOL, INT
        from pysmt.fnode import FNode
        
        if is_num(cpm_expr):
            # translate numpy to python native
            if is_bool(cpm_expr):
                return Bool(bool(cpm_expr))
            elif is_int(cpm_expr):
                return Int(int(cpm_expr))
            return Int(int(cpm_expr))  # Convert float to int for integer variables

        elif is_any_list(cpm_expr):
            # arguments can be lists
            return [cls._cpm_to_pysmt_expr_impl(expr, var_resolver, handle_direct_constraint) for expr in cpm_expr]

        elif isinstance(cpm_expr, BoolVal):
            return Bool(cpm_expr.args[0])

        elif isinstance(cpm_expr, _NumVarImpl):
            return var_resolver(cpm_expr)

        # Operators
        elif isinstance(cpm_expr, Operator):
            arity, _ = Operator.allowed[cpm_expr.name]
            
            if cpm_expr.name == 'and':
                return And(cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint))
            elif cpm_expr.name == 'or':
                return Or(cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint))
            elif cpm_expr.name == '->':
                args = cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint)
                return Implies(args[0], args[1])
            elif cpm_expr.name == 'not':
                return Not(cls._cpm_to_pysmt_expr_impl(cpm_expr.args[0], var_resolver, handle_direct_constraint))
            elif cpm_expr.name == 'sum':
                args = cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint)
                if not args:
                    return Int(0)
                elif len(args) == 1:
                    return args[0]
                else:
                    return Plus(args)
            elif cpm_expr.name == 'wsum':
                weights = cpm_expr.args[0]
                vars = cls._cpm_to_pysmt_expr_impl(cpm_expr.args[1], var_resolver, handle_direct_constraint)
                terms = [Times(Int(w), v) for w, v in zip(weights, vars)]
                if not terms:
                    return Int(0)
                elif len(terms) == 1:
                    return terms[0]
                else:
                    return Plus(terms)
            elif cpm_expr.name == 'sub':
                args = cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint)
                return Minus(args[0], args[1])
            elif cpm_expr.name == 'mul':
                args = cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint)
                return Times(args[0], args[1])
            elif cpm_expr.name == 'div':
                args = cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint)
                return Div(args[0], args[1])
            elif cpm_expr.name == 'mod':
                args = cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint)
                # Mod is not in shortcuts, need to import from formula
                from pysmt.formula import FormulaManager
                from pysmt.environment import get_env
                mgr = get_env().formula_manager
                return mgr.Mod(args[0], args[1])
            elif cpm_expr.name == 'pow':
                args = cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint)
                if not is_num(cpm_expr.args[1]):
                    raise NotSupportedError(f"pySMT only supports power with constant exponent, got {cpm_expr}")
                return Pow(args[0], args[1])
            elif cpm_expr.name == '-':
                arg = cls._cpm_to_pysmt_expr_impl(cpm_expr.args[0], var_resolver, handle_direct_constraint)
                return Minus(Int(0), arg)
            else:
                raise NotImplementedError(f"Operator {cpm_expr.name} not (yet) implemented for pySMT")

        # Comparisons
        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint)
            
            # Handle boolean comparisons - check if original args were boolean
            lhs_bexpr = is_boolexpr(cpm_expr.args[0])
            rhs_bexpr = is_boolexpr(cpm_expr.args[1])
            
            if cpm_expr.name in ["==", "!="]:
                if lhs_bexpr and not rhs_bexpr:
                    lhs = Ite(lhs, Int(1), Int(0))
                elif rhs_bexpr and not lhs_bexpr:
                    rhs = Ite(rhs, Int(1), Int(0))
            else:
                if lhs_bexpr:
                    lhs = Ite(lhs, Int(1), Int(0))
                if rhs_bexpr:
                    rhs = Ite(rhs, Int(1), Int(0))
            
            return eval_comparison(cpm_expr.name, lhs, rhs)

        # Global constraints
        elif isinstance(cpm_expr, GlobalConstraint):
            if cpm_expr.name == 'alldifferent':
                return AllDifferent(cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint))
            elif cpm_expr.name == 'xor':
                args = cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint)
                if len(args) == 1:
                    return args[0]
                result = Xor(args[0], args[1])
                for a in args[2:]:
                    result = Xor(result, a)
                return result
            elif cpm_expr.name == 'ite':
                args = cls._cpm_to_pysmt_expr_impl(cpm_expr.args, var_resolver, handle_direct_constraint)
                return Ite(args[0], args[1], args[2])
            else:
                raise ValueError(f"Global constraint {cpm_expr} should be decomposed already")

        # Direct constraint
        elif isinstance(cpm_expr, DirectConstraint):
            if handle_direct_constraint is not None:
                return handle_direct_constraint(cpm_expr)
            else:
                raise NotImplementedError(f"DirectConstraint not supported without handle_direct_constraint callback")

        raise NotImplementedError(f"pySMT: constraint not (yet) supported: {cpm_expr}")

    def _pysmt_expr(self, cpm_expr):
        """
            Recursively translate CPMpy expressions to pySMT expressions.

            pySMT supports nested expressions, so we can translate expression trees directly.
        """
        return self._cpm_to_pysmt_expr_impl(cpm_expr, var_resolver=self.solver_var, 
                                        handle_direct_constraint=lambda expr: expr.callSolver(self, self.pysmt_solver))

    def get_core(self):
        """
            For use with :func:`s.solve(assumptions=[...]) <solve()>`. Only meaningful if the solver returned UNSAT.
            In that case, get_core() returns a small subset of assumption variables that are unsat together.
        """
        if self.cpm_status.exitstatus != ExitStatus.UNSATISFIABLE:
            raise ValueError("Can only extract core from UNSAT model")
        if not hasattr(self, 'assumption_dict') or len(self.assumption_dict) == 0:
            raise ValueError("Assumptions must be set using s.solve(assumptions=[...])")
        
        try:
            core = self.pysmt_solver.get_unsat_core()
            return [self.assumption_dict[pysmt_var] for pysmt_var in core if pysmt_var in self.assumption_dict]
        except Exception as e:
            raise NotSupportedError(f"Unsat core extraction not supported by {self.subsolver_name}: {e}")

    def _build_smtlib_formula(self, cpm_constraints=None):
        """
        Internal helper method to build a pySMT formula from all constraints in the solver.
        
        Arguments:
            cpm_constraints: Optional list of CPMpy constraints to convert.
                           If provided, these will be converted and collected.
                           If None, tries to get assertions from the solver.
        
        Returns:
            A pySMT formula representing all constraints and variable bounds.
        """
        from pysmt.shortcuts import And, GE, LE, TRUE
        
        # Ensure all variables are known to solver
        self.solver_vars(list(self.user_vars))
        
        # Collect all constraint formulas
        constraint_formulas = []
        
        # Add variable bounds for integer variables
        for cpm_var in self.user_vars:
            if isinstance(cpm_var, _IntVarImpl):
                pysmt_var = self.solver_var(cpm_var)
                # Add bounds as constraints
                constraint_formulas.append(GE(pysmt_var, self._pysmt_expr(cpm_var.lb)))
                constraint_formulas.append(LE(pysmt_var, self._pysmt_expr(cpm_var.ub)))
        
        # Get constraints from solver or convert provided CPMpy constraints
        if cpm_constraints is not None:
            # Convert provided CPMpy constraints
            for constraint in cpm_constraints:
                pysmt_expr = self._pysmt_expr(constraint)
                constraint_formulas.append(pysmt_expr)
        else:
            # Try to get assertions from the native solver
            # Note: get_assertions() raises NotImplementedError by default in pySMT's base Solver class.
            # Solvers that inherit from IncrementalTrackingSolver have an 'assertions' property instead.
            # However, not all solvers track assertions (e.g., SmtLibSolver), so we need a fallback.
            assertions_found = False
            try:
                if hasattr(self.pysmt_solver, 'assertions'):
                    # Use assertions property (available on IncrementalTrackingSolver and its subclasses)
                    assertions = self.pysmt_solver.assertions
                    if assertions and len(assertions) > 0:
                        # Extract formula from assertion tuples (Z3Solver returns tuples for named assertions)
                        for assertion in assertions:
                            if isinstance(assertion, tuple):
                                # Z3Solver with unsat cores returns (key, named, formula) tuples
                                formula = assertion[2] if len(assertion) > 2 else assertion[0]
                            else:
                                formula = assertion
                            constraint_formulas.append(formula)
                        assertions_found = True
            except (AttributeError, NotImplementedError):
                pass
            
            # If assertions weren't available from solver, use tracked pySMT constraints
            # This fallback is necessary for solvers that don't track assertions (e.g., SmtLibSolver)
            if not assertions_found:
                if self._needs_constraint_tracking and self._pysmt_constraints:
                    # Use already-transformed pySMT constraints (no need to convert again)
                    constraint_formulas.extend(self._pysmt_constraints)
                elif hasattr(self, 'cpm_model') and self.cpm_model is not None:
                    # Last resort: use model constraints if available (convert them)
                    for constraint in self.cpm_model.constraints:
                        pysmt_expr = self._pysmt_expr(constraint)
                        constraint_formulas.append(pysmt_expr)
        
        # Create a single formula from all constraints
        if constraint_formulas:
            formula = And(constraint_formulas) if len(constraint_formulas) > 1 else constraint_formulas[0]
        else:
            formula = TRUE()
        
        return formula

    def _formula_to_smtlib_string(self, formula, logic=None, objective_expr=None, minimize=True, daggify=True):
        """
        Internal helper method to convert a pySMT formula to an SMT-LIB script string.
        
        Arguments:
            formula: pySMT formula to convert
            logic: Optional SMT-LIB logic name
            objective_expr: Optional pySMT expression for objective
            minimize: Whether objective is minimization (True) or maximization (False)
            daggify: Whether to use DAG representation
        
        Returns:
            SMT-LIB script string
        """
        from io import StringIO
        from pysmt.smtlib.script import SmtLibCommand, smtlibscript_from_formula
        from pysmt.smtlib import commands as smtcmd
        from pysmt.smtlib.annotations import Annotations
        
        # Use pySMT's helper function to create SMT-LIB script from formula
        script = smtlibscript_from_formula(formula, logic=logic)
        
        # Add annotations for variable names
        annotations = Annotations()
        for cpm_var in self.user_vars:
            if cpm_var in self._varmap:
                pysmt_var = self._varmap[cpm_var]
                # Get the CPMpy variable name (use str() as fallback)
                cpm_name = str(cpm_var)
                # Add :named annotation with the CPMpy variable name
                annotations.add(pysmt_var, 'named', cpm_name)
        
        # Set annotations on the script
        script.annotations = annotations
        
        # Handle objective if present
        if objective_expr is not None:
            opt_type = smtcmd.MINIMIZE if minimize else smtcmd.MAXIMIZE
            opt_cmd = SmtLibCommand(opt_type, [objective_expr, []])
            
            # Insert objective command before check-sat if present
            check_sat_idx = None
            for i, cmd in enumerate(script.commands):
                if cmd.name == smtcmd.CHECK_SAT:
                    check_sat_idx = i
                    break
            
            if check_sat_idx is not None:
                script.commands.insert(check_sat_idx, opt_cmd)
            else:
                script.commands.append(opt_cmd)
        
        # Serialize to string
        buf = StringIO()
        script.serialize(buf, daggify=daggify)
        return buf.getvalue()

    def to_smtlib(self, fname: Optional[str] = None, logic: Optional[str] = None, daggify: bool = True) -> str:
        """
        Export the current solver state to SMT-LIB format.
        
        This method exports all constraints and variables currently in the solver to SMT-LIB format.
        It can be called at any time after constraints have been added to the solver.
        
        Arguments:
            fname: Optional file path where the SMT-LIB file will be written.
                  If None, returns the SMT-LIB string without writing to file.
            logic: Optional SMT-LIB logic name (e.g., "QF_LIA" for quantifier-free linear integer arithmetic).
                   If None, pySMT will try to infer an appropriate logic.
            daggify: If True, uses DAG (directed acyclic graph) representation for shared subexpressions.
                     If False, uses tree representation. DAG is more compact but may be less readable.
        
        Returns:
            String containing the SMT-LIB representation of the current solver state.
        
        Example:
        
        .. code-block:: python
        
            import cpmpy as cp
            
            s = cp.SolverLookup.get("pysmt")
            x = cp.intvar(0, 10)
            s += x >= 5
            s.to_smtlib("model.smt2")  # Write to file
            
            # Or get as string
            smtlib_str = s.to_smtlib()
            print(smtlib_str)
        """
        # Build formula from all constraints
        formula = self._build_smtlib_formula()
        
        # Get objective expression if present
        objective_expr = None
        minimize = True
        if self.has_objective() and hasattr(self, '_objective_expr_pysmt'):
            objective_expr = self._objective_expr_pysmt
            minimize = self._minimize
        
        # Convert to SMT-LIB string using shared helper
        result = self._formula_to_smtlib_string(formula, logic=logic, objective_expr=objective_expr, 
                                                minimize=minimize, daggify=daggify)
        
        # Write to file if requested
        if fname is not None:
            with open(fname, 'w') as f:
                f.write(result)
        
        return result

