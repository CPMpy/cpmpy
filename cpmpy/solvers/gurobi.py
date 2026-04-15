#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## gurobi.py
##
"""
    Interface to Gurobi Optimizer's Python API.

    Gurobi Optimizer is a highly efficient commercial solver for Integer Linear Programming (and more).

    Always use :func:`cp.SolverLookup.get("gurobi") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'gurobipy' python package is installed:

    .. code-block:: console

        $ pip install gurobipy
    
    Gurobi Optimizer requires an active licence (for example a free academic license)
    You can read more about available licences at https://www.gurobi.com/downloads/

    See detailed installation instructions at:
    https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_gurobi

    ==============
    Module details
    ==============
"""

import cpmpy as cp
from typing import Optional, List
import copy
import warnings

from .solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal, is_boolexpr
from ..expressions.utils import argvals, argval, is_any_list, is_num, is_bool, get_bounds
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..transformations.comparison import only_numexpr_equality
from ..transformations.flatten_model import flatten_constraint, flatten_objective, get_or_make_var_or_list
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, linearize_reified_variables, only_positive_bv, only_positive_bv_wsum, decompose_linear, decompose_linear_objective
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from ..transformations.safening import no_partial_functions, safen_objective
from ..transformations.negation import recurse_negation, push_down_negation

try:
    import gurobipy as gp
    GRB_ENV = None
except ImportError:
    pass


class CPM_gurobi(SolverInterface):
    """
    Interface to Gurobi's Python API

    Creates the following attributes (see parent constructor for more):
    
    - ``grb_model``: object, TEMPLATE's model object

    The :class:`~cpmpy.expressions.globalconstraints.DirectConstraint`, when used, calls a function on the ``grb_model`` object.
    
    Documentation of the solver's own Python API:
    https://docs.gurobi.com/projects/optimizer/en/current/reference/python.html
    """

    supported_global_constraints = frozenset({"min", "max", "abs", "mul", "pow"})
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        return CPM_gurobi.installed() and CPM_gurobi.license_ok()

    @staticmethod
    def installed():
        try:
            import gurobipy as gp
            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e

    @staticmethod
    def license_ok():
        if not CPM_gurobi.installed():
            warnings.warn(f"License check failed, python package 'gurobipy' is not installed! Please check 'CPM_gurobi.installed()' before attempting to check license.")
            return False
        try:
            import gurobipy as gp
            global GRB_ENV
            if GRB_ENV is None:
                # initialise the native gurobi model object
                GRB_ENV = gp.Env(params={"OutputFlag": 0})
                GRB_ENV.start()
            return True
        except Exception as e:
            warnings.warn(f"Problem encountered with Gurobi license: {e}")
            return False
        
    @staticmethod
    def version() -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("gurobipy")
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: a CPMpy Model()
            subsolver: None, not used
        """
        if not self.installed():
            raise ModuleNotFoundError("CPM_gurobi: Install the python package 'cpmpy[gurobi]' to use this solver interface.") 
        elif not self.license_ok():
            raise ModuleNotFoundError("CPM_gurobi: No license found or a problem occured during license check. Make sure your license is activated!")
        import gurobipy as gp

        # TODO: subsolver could be a GRB_ENV if a user would want to hand one over
        self.grb_model = gp.Model(env=GRB_ENV)

        # initialise everything else and post the constraints/objective
        # it is sufficient to implement add() and minimize/maximize() below
        super().__init__(name="gurobi", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.grb_model


    def solve(self, time_limit:Optional[float]=None, solution_callback=None, **kwargs):
        """
            Call the gurobi solver

            Arguments:
                time_limit (float, optional):  maximum solve time in seconds
                solution_callback:             Gurobi callback function
                **kwargs:                      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            Examples of gurobi supported arguments include:

            - ``Threads`` : int
            - ``MIPFocus`` : int
            - ``ImproveStartTime`` : bool
            - ``FlowCoverCuts`` : int

            For a full list of gurobi parameters, please visit https://www.gurobi.com/documentation/9.5/refman/parameters.html#sec:Parameters
        """
        from gurobipy import GRB

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)
        
        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            self.grb_model.setParam("TimeLimit", time_limit)

        # call the solver, with parameters
        for param, val in kwargs.items():
            self.grb_model.setParam(param, val)

        # write LP file for debugging
        self.grb_model.write("/tmp/model.lp")

        _ = self.grb_model.optimize(callback=solution_callback)
        grb_objective = self.grb_model.getObjective()

        grb_status = self.grb_model.Status

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.grb_model.runtime

        # translate exit status
        if grb_status == GRB.OPTIMAL:
            # COP
            if self.has_objective():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            # CSP
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif grb_status == GRB.INFEASIBLE:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif grb_status == GRB.TIME_LIMIT:
            if self.grb_model.SolCount == 0:
                # can be sat or unsat
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        else:  # another?
            raise NotImplementedError(
                f"Translation of gurobi status {grb_status} to CPMpy status not implemented")  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                solver_val = self.solver_var(cpm_var).X
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = int(solver_val)
            # set _objective_value
            if self.has_objective():
                grb_obj_val = grb_objective.getValue()
                if round(grb_obj_val) == grb_obj_val: # it is an integer?:
                    self.objective_value_ = int(grb_obj_val)
                else: #  can happen with DirectVar or when using floats as coefficients
                    self.objective_value_ =  float(grb_obj_val)

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None

        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var): # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view. Should be eliminated in linearize
        if isinstance(cpm_var, NegBoolView):
            return 1 - self.solver_var(cpm_var._bv)

        # create if it does not exit
        if cpm_var not in self._varmap:
            from gurobipy import GRB
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.grb_model.addVar(vtype=GRB.BINARY, name=cpm_var.name)
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.grb_model.addVar(cpm_var.lb, cpm_var.ub, vtype=GRB.INTEGER, name=str(cpm_var))
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
        from gurobipy import GRB

        # save user variables
        get_variables(expr, self.user_vars)

        # transform objective
        obj, safe_cons = safen_objective(expr)
        obj, decomp_cons = decompose_linear_objective(obj,
                                                      supported=self.supported_global_constraints,
                                                      supported_reified=self.supported_reified_global_constraints,
                                                      csemap=self._csemap)
        obj, flat_cons = flatten_objective(obj, csemap=self._csemap, supported={"pow", "mul"})
        obj = only_positive_bv_wsum(obj)  # remove negboolviews

        self.add(safe_cons + decomp_cons + flat_cons)

        # make objective function or variable and post
        grb_obj = self._make_numexpr(obj)
        if minimize:
            self.grb_model.setObjective(grb_obj, sense=GRB.MINIMIZE)
        else:
            self.grb_model.setObjective(grb_obj, sense=GRB.MAXIMIZE)
        self.grb_model.update()

    def has_objective(self):
        return self.grb_model.getObjective().size() != 0  # TODO: check if better way to do this...

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function
        """
        import gurobipy as gp

        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # sum
        if cpm_expr.name == "sum":
            return gp.quicksum(self.solver_vars(cpm_expr.args))
        if cpm_expr.name == "sub":
            a,b = self.solver_vars(cpm_expr.args)
            return a - b
        # wsum
        if cpm_expr.name == "wsum":
            return gp.quicksum(w * self.solver_var(var) for w, var in zip(*cpm_expr.args))

        raise NotImplementedError("gurobi: Not a known supported numexpr {}".format(cpm_expr))

    verbose = False
    general_constraints = {"max", "min", "abs", "and", "or"}

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
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel=frozenset(["mod", "div", "element"]))  # linearize and decompose expect safe exprs
        cpm_cons = decompose_linear(cpm_cons,
                                    supported=self.supported_global_constraints,
                                    supported_reified=self.supported_reified_global_constraints,
                                    csemap=self._csemap)

        all_cons = []  # accumulate all constraints (including side-effect ones from != and reify)


        def add(cpm_expr):
            """Recursively create a Gurobi constraint from a CPMpy expression."""

            def get_or_make_var(cpm_expr, define=True):
                """Get or make (Boolean/integer) var `b` which represent the expression. Add defining constraints b == cpm_expr if `define=True`."""
                if cpm_expr in self._csemap:
                    return self._csemap[cpm_expr]

                r = cp.boolvar() if is_boolexpr(cpm_expr) else cp.intvar(*cpm_expr.get_bounds())
                self._csemap[cpm_expr] = r
                if define:
                    add(r == cpm_expr)
                return r

            def with_args(cpm_expr, args):
                """Copy expression and replace args"""
                cpm_expr = copy.copy(cpm_expr)
                cpm_expr.update_args(args)
                return cpm_expr

            def propagate_boolconst(name, args):
                """Propagate boolean constants in and/or: and(0,...)=0, or(1,...)=1, filter neutral elements."""
                match name:
                    case "and" | "or":
                        # TODO single loop should be possible
                        absorb = 0 if name == "and" else 1  # absorbing element
                        args = [a for a in args if not (is_num(a) and a == 1 - absorb)]
                        if any(is_num(a) and a == absorb for a in args):
                            return absorb
                        return args
                    case _:
                        return args

            def add_general_constraint(f, depth, reified=False, y=None):
                """Add the general constraint `f(x)` in gurobi's required form, `y=f(x)` with `x` a list of variables (Boolean/integer, depending on the general constraint type)"""
                # require only variables
                # f(x1, x2, ..) === f(y1, y2, ..), y1=x1, y2=x2, ..
                args = [reify(add_(arg, depth, reified=True), depth) for arg in f.args]
                # require non-constants
                args = propagate_boolconst(f.name, args)
                if is_num(args):  # may have become fixed (e.g. `and(x1, 0, x2) === 0`)
                    return args
                else:
                    # require the form: y = f(x)
                    if y is None:
                        assert reified or f.name in {"and", "or"}, f"Unexpected numexpr {f} encountered at root level"
                        y = get_or_make_var(f, define=False) if reified else 1

                    # y, y = f(x)
                    f = with_args(f, args)
                    all_cons.append(y == f)  # add directly so that Comparison does not have to deal with it
                    # TODO could be done by add(y == f)?
                    return y

            def reify(cpm_expr, depth):
                """Return something which can be an argument for a general constraint or linear (indicator) constraint"""
                if self.verbose: print(f"{'  ' * depth}reify", cpm_expr, getattr(cpm_expr, 'name', None))

                if is_num(cpm_expr):
                    return cpm_expr
                elif cpm_expr in self._csemap:  # required to prevent infinite recursion
                    return self._csemap[cpm_expr]
                elif isinstance(cpm_expr, NegBoolView):
                    return get_or_make_var(cpm_expr)
                elif isinstance(cpm_expr, (_BoolVarImpl, _IntVarImpl)):
                    return cpm_expr
                elif is_boolexpr(cpm_expr):
                    if cpm_expr.name == "->":
                        # TODO check
                        # Convert p -> q to or(~p, q) to avoid circular reification
                        # (reifying via manual implies would cause CSE to create tautologies)
                        a, b = cpm_expr.args
                        return reify(Operator("or", [recurse_negation(a), b]), depth)
                    else:
                        return get_or_make_var(cpm_expr)
                else:
                    return get_or_make_var(cpm_expr)

            def add_comparison(cpm_expr, depth, reified=False):
                a, b = cpm_expr.args

                match cpm_expr.name:
                    case "==" if not reified and isinstance(a, (int, _NumVarImpl)) and isinstance(b, Operator) and b.name in self.general_constraints:
                        # already of the form: y = f(x)
                        add_general_constraint(b, depth, reified=reified, y=a)
                        return True
                    case "==" if isinstance(a, _BoolVarImpl) and is_boolexpr(b) and not isinstance(b, _NumVarImpl):
                        # BV == boolexpr === BV <-> boolexpr: post as bi-implications
                        add(a.implies(b))
                        add((~a).implies(recurse_negation(b)))
                        con = True
                    case "==" | "<=" | ">=":
                        a, b = add_(a, depth, reified=True), add_(b, depth, reified=True)
                        con = with_args(cpm_expr, [a, b])
                    case "!=":
                        # One-directional indicator split: d=1 -> a>b, e=1 -> a<b
                        cpm_expr, = push_down_negation([cpm_expr], toplevel=not reified)
                        if cpm_expr.name == "!=":
                            a, b = cpm_expr.args
                            if reified:
                                return add_((a > b) | (a < b), depth, reified=reified)
                            else:
                                z = cp.boolvar()
                                # return add_(z.implies(a > b) & (~z).implies(a < b), depth, reified=reified)
                                add(z.implies(a > b))
                                add((~z).implies(a < b))
                                return True
                        else:  # push_down_negation may have changed e.g. != into ==
                            return add_(cpm_expr, depth, reified=reified)
                    case ">":
                        return add_(a >= b + 1, depth, reified=reified)
                    case "<":
                        return add_(a <= b - 1, depth, reified=reified)
                    case _:
                        raise Exception(f"Expected comparator to be ==,<=,>= in Comparison expression {cpm_expr}, but was {cpm_expr.name}")

                return reify(con, depth) if reified else con

            def linearize(cpm_expr, depth):
                """Ensure expression is linear (no mul/pow) by reifying non-linear parts into aux vars."""
                if self.verbose: print(f"{'  ' * depth}lin", cpm_expr)

                # Only comparisons (except !=) can be indicator bodies directly;
                # everything else (!=, BoolVars, and/or/etc.) needs reification into a BV
                can_be_linear = isinstance(cpm_expr, Comparison) and cpm_expr.name != "!="
                cpm_expr = add_(cpm_expr, depth, reified=not can_be_linear)
                if isinstance(cpm_expr, _NumVarImpl):
                    return cpm_expr >= 1
                elif isinstance(cpm_expr, Comparison):
                    def linearize_expr(expr):
                        if is_num(expr) or isinstance(expr, _NumVarImpl):
                            return expr
                        elif isinstance(expr, Operator):
                            match expr.name:
                                case "sum":
                                    return with_args(expr, [reify(a_i, depth) for a_i in expr.args])
                                case "wsum":
                                    w, x = expr.args
                                    return with_args(expr, [w, [reify(x_i, depth) for x_i in x]])
                                case _:
                                    return reify(expr, depth)
                        else:
                            return reify(expr, depth)

                    a, b = cpm_expr.args
                    return with_args(cpm_expr, [linearize_expr(a), linearize_expr(b)])
                else:
                    return reify(cpm_expr, depth) >= 1

            def raise_unsupported(cpm_expr):
                raise NotSupportedError("CPM_gurobi: Unsupported constraint", cpm_expr)
              
            def add_(cpm_expr, depth, reified=False):
                """Transforms a cpm_expr into something supported by Gurobi's (non-linear) expression tree"""

                indent = "  " * depth
                depth += 1
                if self.verbose: print(f"{indent}Con:", cpm_expr, type(cpm_expr), "reif" if reified else "root")

                if is_num(cpm_expr):
                    return int(cpm_expr)
                elif isinstance(cpm_expr, _NumVarImpl):
                    return cpm_expr
                elif isinstance(cpm_expr, (Operator, GlobalFunction)):
                    match cpm_expr.name:
                        case "->":  # Gurobi indicator constraint: (Var == 0|1) >> (LinExpr sense LinExpr)
                            a, b = cpm_expr.args
                            # if isinstance(b, (_NumVarImpl, Comparison)):  # TODO requires reification?
                            if isinstance(b, Comparison):
                                p = a if isinstance(a, NegBoolView) else reify(a, depth)
                                if is_num(p):  # propagate fixed antecedent
                                    return add_(b, depth, reified=reified) if a else True
                                assert isinstance(p, _BoolVarImpl)
                                q = linearize(b, depth)
                                if is_num(q):
                                    return True if q else add_(p, depth, reified=reified)
                                assert isinstance(q, Comparison), f"Expected linear constraint, but got {q}"  # not required to be a canonical comparison
                                return with_args(cpm_expr, [p, q])
                            else:
                                return add_((~a) | b, depth, reified=reified)
                        case "not":  # not is not handled by gurobi
                            a, = cpm_expr.args
                            return add_(recurse_negation(a), depth, reified=True)
                        case "-" | "sub" | "sum" | "mul" | "pow" | "div":  # Expression tree nodes (w/ args)
                            assert cpm_expr.name != "div", "TODO"
                            return with_args(cpm_expr, [add_(a, depth, reified=True) for a in cpm_expr.args])
                        case "wsum":  # Just for efficiency, don't call add on the weights
                            ws, xs = cpm_expr.args
                            return with_args(cpm_expr, [ws, [add_(x, depth, reified=True) for x in xs]])
                        case name if name in self.general_constraints:  # general constraints are not handled by the expression tree, so they will be reified
                            return add_general_constraint(cpm_expr, depth, reified=reified)
                        case _:
                            raise_unsupported(cpm_expr)
                elif isinstance(cpm_expr, Comparison):
                    return add_comparison(cpm_expr, depth, reified=reified)
                elif isinstance(cpm_expr, DirectConstraint):
                    cpm_expr.callSolver(self, self.grb_model)
                    return True
                else:
                    raise_unsupported(cpm_expr)

            result = add_(cpm_expr, 0)
            all_cons.append(result)
            return result

        for c in cpm_cons:
            add(c)
        cpm_cons = all_cons

        with open("/tmp/model.txt", "w") as f:
            f.write("# Original constraints:\n")
            for c in toplevel_list(cpm_expr):
                f.write(str(c) + "\n")
            f.write("\n# Transformed constraints:\n")
            for c in cpm_cons:
                f.write(str(c) + "\n")

        return cpm_cons

    def add(self, cpm_expr_orig):
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
        import gurobipy as gp

        def add(cpm_expr):
            def add_(cpm_expr, depth):
                indent = "  " * depth
                if self.verbose: print(f"{indent}Add:", cpm_expr, type(cpm_expr))

                depth += 1

                if isinstance(cpm_expr, int):
                    return cpm_expr
                elif isinstance(cpm_expr, _NumVarImpl):
                    return self.solver_var(cpm_expr)
                elif isinstance(cpm_expr, (Operator, GlobalFunction)):
                    match cpm_expr.name:
                        case "->":  # Gurobi indicator constraint: (Var == 0|1) >> (LinExpr sense LinExpr)
                            a, b = cpm_expr.args
                            is_pos = not isinstance(a, NegBoolView)
                            p = add_(a if is_pos else a._bv, depth)
                            return (p == int(is_pos)) >> add_(b, depth)
                        case "-":
                            return -add_(cpm_expr.args[0], depth=depth)
                        case "sum":
                            return sum(add_(arg, depth) for arg in cpm_expr.args)
                        case "wsum":
                            return sum(weight * add_(arg, depth) for weight, arg in zip(cpm_expr.args[0], cpm_expr.args[1]))
                        case "sub":
                            return add_(cpm_expr.args[0], depth) - add_(cpm_expr.args[1], depth)
                        case "mul":
                            return add_(cpm_expr.args[0], depth) * add_(cpm_expr.args[1], depth)
                        case "pow":
                            return add_(cpm_expr.args[0], depth) ** add_(cpm_expr.args[1], depth)
                        case general_constraint_name:
                            assert general_constraint_name in self.general_constraints
                            args = [add_(a, depth) for a in cpm_expr.args]
                            match general_constraint_name:
                                case "or":
                                    return gp.or_(*args)
                                case "and":
                                    return gp.and_(*args)
                                case "abs":
                                    return gp.abs_(*args)
                                case "min":
                                    return gp.min_(*args)
                                case "max":
                                    return gp.max_(*args)
                elif isinstance(cpm_expr, Comparison):
                    a, b = cpm_expr.args
                    match cpm_expr.name:
                        case "==":
                            a, b = add_(a, depth), add_(b, depth)
                            if isinstance(a, gp.NLExpr) or isinstance(b, gp.NLExpr):
                                # NLExpr must be on RHS: y == f(x)
                                if isinstance(b, gp.NLExpr):
                                    y = a if isinstance(a, gp.Var) else self.grb_model.addVar(lb=a, ub=a)
                                    return y == b
                                else:
                                    y = b if isinstance(b, gp.Var) else self.grb_model.addVar(lb=b, ub=b)
                                    return y == a
                            elif isinstance(a, (int, gp.LinExpr, gp.QuadExpr, gp.Var, gp.GenExpr)):
                                return a == b
                            else:
                                raise Exception(f"Unexpected expression in {cpm_expr}, {type(a)}")
                        case "<=" | ">=":
                            a, b = add_(a, depth), add_(b, depth)
                            # Gurobi requires NLExpr in y=f(x) form; reify to aux var
                            # TODO check if can be re-used in ==
                            # TODO use reify
                            def _reify_nl(expr):
                                if isinstance(expr, gp.NLExpr):
                                    y = self.grb_model.addVar(lb=-gp.GRB.INFINITY)
                                    self.grb_model.addConstr(y == expr)
                                    return y
                                return expr
                            a, b = _reify_nl(a), _reify_nl(b)
                            return (a <= b) if cpm_expr.name == "<=" else (a >= b)
                        case _:
                            raise Exception(f"Expected comparator to be ==,<=,>= in Comparison expression {cpm_expr}, but was {cpm_expr.name}")
                elif isinstance(cpm_expr, DirectConstraint):
                    cpm_expr.callSolver(self, self.grb_model)
                    return True
                else:
                    raise NotImplementedError(f"add_() not implemented for {cpm_expr} of type {type(cpm_expr)} w/ name {getattr(cpm_expr, 'name', None)}")

            grb_expr = add_(cpm_expr, 0)
            if isinstance(grb_expr, (bool, int)):
                if not grb_expr:
                    self.grb_model.addConstr(0 >= 1)  # infeasible
                return
            elif isinstance(grb_expr, (gp.Var, gp.LinExpr)):
                # If add() returned a Gurobi Var (not a constraint), wrap it as >= 1
                grb_con = grb_expr >= 1
            elif isinstance(grb_expr, gp.TempConstr):
                grb_con = grb_expr
            else:
                grb_con = self.grb_model.addVar(lb=1, ub=1) == grb_expr
            if self.verbose: self.grb_model.update()
            if self.verbose: print("OUT", grb_con)
            self.native_model.addConstr(grb_con)


        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        for cpm_expr in self.transform(cpm_expr_orig):
            add(cpm_expr)

        return self
    __add__ = add  # avoid redirect in superclass

    def solution_hint(self, cpm_vars:List[_NumVarImpl], vals:List[int|bool]):
        """
        Gurobi supports warmstarting the solver with a (in)feasible solution.
        The provided value will affect branching heurstics during solving, making it more likely the final solution will contain the provided assignment.

        To learn more about solution hinting in gurobi, see:
        https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/variable.html#varhintval

        Optionally, you can also set the relative priority of the hint, using:
        
        .. code-block:: python

            solver.solver_var(cpm_var).setAttr("VarHintPri", <priority>)

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
        for cpm_var, val in zip(cpm_vars, vals):
            self.solver_var(cpm_var).setAttr("VarHintVal", val)

    def solveAll(self, display:Optional[Callback]=None, time_limit:Optional[float]=None, solution_limit:Optional[int]=None, call_from_model=False, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            This is the generic implementation, solvers can overwrite this with
            a more efficient native implementation

            Arguments:
                display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                time_limit: stop after this many seconds (default: None)
                solution_limit: stop after this many solutions (default: None)
                call_from_model: whether the method is called from a CPMpy Model instance or not
                any other keyword argument

            Returns: number of solutions found
        """
        from gurobipy import GRB

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)

        if time_limit is not None:
            self.grb_model.setParam("TimeLimit", time_limit)

        if solution_limit is None:
            raise Exception(
                "Gurobi does not support searching for all solutions. If you really need all solutions, "
                "try setting solution limit to a large number")

        # Force gurobi to keep searching in the tree for optimal solutions
        sa_kwargs = {"PoolSearchMode":2, "PoolSolutions":solution_limit}

        # solve the model
        self.solve(time_limit=time_limit, **sa_kwargs, **kwargs)

        optimal_val = None
        solution_count = self.grb_model.SolCount
        opt_sol_count = 0

        # clear user vars if no solution found
        if solution_count == 0:
            self.objective_value_ = None
            for var in self.user_vars:
                var._value = None

        for i in range(solution_count):
            # Specify which solution to query
            self.grb_model.setParam("SolutionNumber", i)
            sol_obj_val = self.grb_model.PoolObjVal
            if optimal_val is None:
                optimal_val = sol_obj_val
            if optimal_val is not None:
                # sub-optimal solutions
                if sol_obj_val != optimal_val:
                    break
            opt_sol_count += 1

            # Translate solution to variables
            for cpm_var in self.user_vars:
                solver_val = self.solver_var(cpm_var).Xn
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = int(solver_val)

            # Translate objective
            if self.has_objective():
                self.objective_value_ = self.grb_model.PoolObjVal

            if display is not None:
                if isinstance(display, Expression):
                    print(display.value())
                elif is_any_list(display):
                    print(argvals(display))
                else:
                    assert callable(display), f"Expected display argument to be an Expression, list thereof or a function, but got {display} of type {type(display)}"
                    display()  # callback

        # Reset pool search mode to default
        self.grb_model.setParam("PoolSearchMode", 0)

        if opt_sol_count:
            if opt_sol_count == solution_limit:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE 
            else:
                grb_status = self.grb_model.Status
                if grb_status == GRB.TIME_LIMIT: # reached time limit
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
                else: # found all solutions   
                    self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        # if unsat or timout with no solution, .solve() will have already set the state accordingly (so nothing to update)

        return opt_sol_count
