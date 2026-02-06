#!/usr/bin/env python
"""
    Interface to the SCIP's python "PySCIPOpt" package

    First install the ScipOptSuite on your machine, follow:
    https://scipopt.org/index.php#download

    Then install the 'pyscipopt' python package:
        $ pip install pyscipopt
    (more information on https://github.com/scipopt/PySCIPOpt)
    
    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_scip

    ==============
    Module details
    ==============
"""
import warnings
from typing import Optional

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import *
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar, boolvar
from ..expressions.globalconstraints import DirectConstraint, GlobalConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..transformations.comparison import only_numexpr_equality
from ..transformations.decompose_global import decompose_in_tree, decompose_objective
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, only_positive_bv, only_positive_bv_wsum
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_bv_reifies, only_implies, reify_rewrite
from ..transformations.safening import no_partial_functions, safen_objective
from ..expressions.utils import is_true_cst, is_false_cst


class CPM_scip(SolverInterface):
    """
    Interface to SCIP's API

    Requires that the SCIPOptSuite and 'pyscipopt' python package is installed
    See detailed installation instructions at the top of this file.

    Creates the following attributes (see parent constructor for more):
    - scip_model: object, SCIP's Model object

    Detailed documentation on the Model():
    https://scipopt.github.io/PySCIPOpt/docs/html/classpyscipopt_1_1scip_1_1Model.html

    The `DirectConstraint`, when used, calls a function on the `scip_model` object.
    """

    # Globals we keep (decompose_in_tree) and how they are translated:
    # - "xor": kept; linearize passes it through; we translate to addConsXor() in add().
    # - "abs": GlobalFunction supported natively (PySCIPOpt addCons(abs(x) <= k)).
    # SCIP has no native AllDifferent, Circuit, Table, Cumulative, etc.; others are decomposed by decompose_in_tree.
    supported_global_constraints = frozenset({"xor", "abs"})
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pyscipopt as scip
            return True
        except Exception:
            return False

    @classmethod
    def version(cls) -> Optional[str]:
        """Returns the installed version of the solver's Python API (pyscipopt)."""
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("pyscipopt")
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: a CPMpy Model()
        """
        if not self.supported():
            raise ModuleNotFoundError(
                "CPM_scip: Install SCIPOptSuite and the python package 'pyscipopt' to use this solver interface.")
        assert subsolver is None, "SCIP does not support subsolvers"
        import pyscipopt as scip

        self.scip_model = scip.Model("From CPMpy")
        self.scip_model.setParam("display/verblevel", 0) # remove solver logs from output
        # initialise everything else and post the constraints/objective
        # it is sufficient to implement __add__() and minimize/maximize() below
        super().__init__(name="scip", cpm_model=cpm_model)

    @property
    def native_model(self):
        """Returns the solver's underlying native model (SCIP Model) for direct solver access."""
        return self.scip_model

    def solve(self, time_limit=None, solution_callback=None, **kwargs):
        """
            Call the SCIP solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional). Persists across solve() calls until overridden.
            - kwargs:      any keyword argument, sets parameters of solver object (e.g. "limits/time" overrides time_limit).

            Arguments that correspond to solver parameters (passed via setParams):
            Examples: limits/nodes=1000, limits/solutions=1, parallel/maxnthreads=4,
                display/verblevel=4, separating/maxrounds=0.
            For a full list see https://www.scipopt.org/doc/html/PARAMETERS.php
        """
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            self.scip_model.setParam("limits/time", float(time_limit))

        if solution_callback is not None:
            raise NotSupportedError("SCIP: solution callback not (yet?) implemented")

        # apply any solver parameters (can override time limit if e.g. "limits/time" in kwargs)
        self.scip_model.setParams(kwargs)
        _ = self.scip_model.optimize()

        scip_status = self.scip_model.getStatus()

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.scip_model.getSolvingTime()

        unknown_stati = {"unknown", "userinterrupt", "unbounded", "inforunbd", "terminate"}

        # translate exit status
        if scip_status == "optimal":
            if self.has_objective():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif scip_status == "infeasible":  # proven unsat
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        else:
            n_sols = self.scip_model.getSols()
            n_sols = len(n_sols) if isinstance(n_sols, list) else n_sols
            if scip_status.endswith("limit"):  # timelimit, nodelimit, etc.
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE if n_sols > 0 else ExitStatus.UNKNOWN
            elif n_sols > 0:  # at least one feasible solution found, not proven optimal
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            elif scip_status in unknown_stati:
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            else:
                raise NotImplementedError(
                    f"Translation of scip status {scip_status} to CPMpy status not implemented")  # a non-mapped status type, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        # use best solution object so values are correct (getVal can be wrong after transform)
        self.objective_value_ = None
        if has_sol:
            best_sol = self.scip_model.getBestSol()
            for cpm_var in self.user_vars:
                if cpm_var not in self._varmap:
                    continue  # variable never posted (e.g. unused); skip to avoid addVar after solve
                scip_var = self._varmap[cpm_var]
                if best_sol is not None:
                    solver_val = self.scip_model.getSolVal(best_sol, scip_var)
                else:
                    solver_val = self.scip_model.getVal(scip_var)
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = int(solver_val)
            # set _objective_value
            if self.has_objective():
                self.objective_value_ = self.scip_model.getObjVal()
        else:
            for cpm_var in self.user_vars:
                cpm_var._value = None

        # From SCIP's Mark Turner:
        # SCIP transforms the problem that you are actual solving into something it believes
        # is easier to solve (presolve techniques etc). All solutions when found are then 
        # transformed back to the original space, because that is obviously what the user 
        # modelled and can interpret. Allowing the user to change the model during solving 
        # gets messy when balancing these things however, so it's forbidden. 
        # What self.scip_model.freeTransform() does is to remove all information from the 
        # transformed space, and just leave the original model. All solutions are kept, 
        # and the user can now change the model as they will. The downside is that potentially 
        # useful information for speeding up the next optimisation call is thrown out.
        self.scip_model.freeTransform()

        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var): # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view (not supported as first-class var; use 1-bv in constraints)
        if isinstance(cpm_var, NegBoolView):
            raise NotSupportedError(
                "Negative literals should not be part of any equation. See /transformations/linearize for more details"
            )

        # create if it does not exit
        if cpm_var not in self._varmap:
            import pyscipopt as scip
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.scip_model.addVar(vtype='B', name=cpm_var.name)
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.scip_model.addVar(lb=cpm_var.lb, ub=cpm_var.ub, vtype='I', name=cpm_var.name)
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
                are permanently posted to the solver)
        """
        import pyscipopt as scip

        get_variables(expr, collect=self.user_vars)

        obj, safe_cons = safen_objective(expr)
        obj, decomp_cons = decompose_objective(obj,
                                               supported=self.supported_global_constraints,
                                               supported_reified=self.supported_reified_global_constraints,
                                               csemap=self._csemap)
        obj, flat_cons = flatten_objective(obj, csemap=self._csemap)
        obj = only_positive_bv_wsum(obj)

        self.add(safe_cons + decomp_cons + flat_cons)

        scip_obj = self._make_numexpr(obj)
        if minimize:
            self.scip_model.setObjective(scip_obj, sense='minimize')
        else:
            self.scip_model.setObjective(scip_obj, sense='maximize')

    def has_objective(self):
        try:
            obj = self.scip_model.getObjective()
            return obj is not None and (hasattr(obj, 'terms') and len(obj.terms) != 0)
        except (AttributeError, TypeError):
            return False

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function
        """
        import pyscipopt as scip

        if is_num(cpm_expr):
            return cpm_expr

        # negated bool (e.g. in objective): 1 - bv
        if isinstance(cpm_expr, NegBoolView):
            return 1 - self.solver_var(cpm_expr._bv)

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # sum
        if cpm_expr.name == "sum":
            return scip.quicksum(self.solver_vars(cpm_expr.args))
        if cpm_expr.name == "sub":
            a,b = self.solver_vars(cpm_expr.args)
            return a - b
        # wsum
        if cpm_expr.name == "wsum":
            return scip.quicksum(w * self.solver_var(var) for w, var in zip(*cpm_expr.args))

        # abs (GlobalFunction; PySCIPOpt supports abs() in constraints/objective)
        if isinstance(cpm_expr, GlobalFunction) and cpm_expr.name == "abs":
            return abs(self._make_numexpr(cpm_expr.args[0]))

        raise NotImplementedError("scip: Not a known supported numexpr {}".format(cpm_expr))


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
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div", "element"})
        cpm_cons = decompose_in_tree(cpm_cons,
                                     supported=self.supported_global_constraints | {"alldifferent"},
                                     supported_reified=self.supported_reified_global_constraints,
                                     csemap=self._csemap)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum', 'sub']), csemap=self._csemap)
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]) | self.supported_global_constraints, csemap=self._csemap)
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum", "sub", "mul", "div", "sum!=", "wsum!="}) | self.supported_global_constraints, csemap=self._csemap)
        cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)
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

        :param cpm_expr_orig: CPMpy expression, or list thereof
        :type cpm_expr_orig: Expression or list of Expression

        :return: self
        """
        get_variables(cpm_expr_orig, collect=self.user_vars)
        # Ensure every user var has a solver variable (so we get values after solve even if
        # the constraint was simplified away and the var never appears in transformed constraints)
        for cpm_var in self.user_vars:
            if not is_num(cpm_var):
                self.solver_var(cpm_var)

        for cpm_expr in self.transform(cpm_expr_orig):
            if isinstance(cpm_expr, Comparison):
                lhs, rhs = cpm_expr.args
                lhs_is_operator = isinstance(lhs, Operator)
                sciprhs = self.solver_var(rhs)

                if cpm_expr.name == '!=':
                    # sum(x) != k  =>  (lhs <= k-1) or (lhs >= k+1); use SCIP's native disjunction (no extra binary)
                    sciplhs = self._make_numexpr(lhs)
                    self.scip_model.addConsDisjunction([
                        sciplhs <= sciprhs - 1,
                        sciplhs >= sciprhs + 1
                    ])
                    continue

                if cpm_expr.name == '<=':
                    if (lhs_is_operator and lhs.name == "sum" and all(a.is_bool() and not isinstance(a, NegBoolView) for a in lhs.args)):
                        if rhs == 1:
                            self.scip_model.addConsSOS1(self.solver_vars(lhs.args))
                        else:
                            self.scip_model.addConsCardinality(self.solver_vars(lhs.args), int(rhs))
                    else:
                        sciplhs = self._make_numexpr(lhs)
                        self.scip_model.addCons(sciplhs <= sciprhs)

                elif cpm_expr.name == '>=':
                    sciplhs = self._make_numexpr(lhs)
                    self.scip_model.addCons(sciplhs >= sciprhs)
                elif cpm_expr.name == '==':
                    if isinstance(lhs, _NumVarImpl) \
                            or (lhs_is_operator and (lhs.name == 'sum' or lhs.name == 'wsum' or lhs.name == "sub")) \
                            or (isinstance(lhs, GlobalFunction) and lhs.name == "abs"):
                        sciplhs = self._make_numexpr(lhs)
                        self.scip_model.addCons(sciplhs == sciprhs)
                    elif lhs_is_operator and lhs.name == 'mul':
                        scp_vars = self.solver_vars(lhs.args)
                        scp_lhs = scp_vars[0] * scp_vars[1]
                        for v in scp_vars[2:]:
                            scp_lhs *= v
                        self.scip_model.addCons(scp_lhs == sciprhs)
                    elif lhs_is_operator and lhs.name == 'div':
                        a, b = self.solver_vars(lhs.args)
                        self.scip_model.addCons(a / b == sciprhs)
                    else:
                        raise NotImplementedError(
                            "Not a known supported scip comparison '{}' {}".format(getattr(lhs, 'name', lhs), cpm_expr))
                else:
                    raise NotImplementedError(
                        "Not a known supported scip comparison '{}' {}".format(cpm_expr.name, cpm_expr))

            elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
                cond, sub_expr = cpm_expr.args
                assert isinstance(cond, _BoolVarImpl), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
                assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"

                lhs, rhs = sub_expr.args
                lhs_is_globalfunc = isinstance(lhs, GlobalFunction)
                lhs_is_operator = isinstance(lhs, Operator)
                # cond -> (abs(x) <= rhs) is equivalent to cond -> (x <= rhs and x >= -rhs)
                if lhs_is_globalfunc and lhs.name == "abs" and sub_expr.name == "<=":
                    (arg,) = lhs.args
                    self.add([cond.implies(arg <= rhs), cond.implies(arg >= -rhs)])
                    continue
                # cond -> (abs(x) >= rhs) means when cond, x >= rhs or x <= -rhs
                if lhs_is_globalfunc and lhs.name == "abs" and sub_expr.name == ">=":
                    (arg,) = lhs.args
                    self.add([cond.implies((arg >= rhs) | (arg <= -rhs))])
                    continue
                # cond -> (abs(x) == rhs) means when cond, x == rhs or x == -rhs
                if lhs_is_globalfunc and lhs.name == "abs" and sub_expr.name == "==":
                    (arg,) = lhs.args
                    self.add([cond.implies((arg == rhs) | (arg == -rhs))])
                    continue
                if lhs_is_globalfunc and lhs.name == "abs":
                    raise NotImplementedError(
                        f"Reified abs with {sub_expr.name} not supported in SCIP"
                    )

                assert isinstance(lhs, _NumVarImpl) or (lhs_is_operator and lhs.name in ("sum", "wsum")), f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}"
                assert is_num(rhs), f"linearize should only leave constants on rhs of comparison but got {rhs}"

                if sub_expr.name == ">=":
                    if lhs_is_operator and lhs.name == "sum":
                        lhs = Operator("wsum", [[-1] * len(lhs.args), lhs.args])
                    elif lhs_is_operator and lhs.name == "wsum":
                        lhs = Operator("wsum", [[-w for w in lhs.args[0]], lhs.args[1]])
                    else:
                        lhs = Operator("wsum", [[-1], [lhs]])
                    sub_expr = lhs <= -rhs

                if sub_expr.name == "<=":
                    lhs, rhs = sub_expr.args
                    lin_expr = self._make_numexpr(lhs)
                    if isinstance(cond, NegBoolView):
                        self.scip_model.addConsIndicator(lin_expr <= rhs,
                                                         binvar=self.solver_var(cond._bv), activeone=False)
                    else:
                        self.scip_model.addConsIndicator(lin_expr <= rhs,
                                                         binvar=self.solver_var(cond), activeone=True)
                elif sub_expr.name == "==":
                    self.add([cond.implies(lhs <= rhs), cond.implies(lhs >= rhs)])
                else:
                    raise Exception(f"Unknown linear expression {sub_expr} name")

            elif isinstance(cpm_expr, BoolVal):
                if cpm_expr.args[0] is False:
                    bv = self.solver_var(boolvar())
                    self.scip_model.addCons(bv <= -1)

            elif isinstance(cpm_expr, DirectConstraint):
                cpm_expr.callSolver(self, self.scip_model)

            elif isinstance(cpm_expr, GlobalConstraint):
                if cpm_expr.name == "xor":
                    # SCIP addConsXor(vars, rhs): rhs is the value the xor must equal (True for top-level)
                    args = [a for a in cpm_expr.args if not is_false_cst(a)]
                    if not args:
                        # only false constants: xor() is False -> no solution
                        if not hasattr(self, "_scip_infeasible_aux"):
                            self._scip_infeasible_aux = self.scip_model.addVar(vtype='B', lb=0, ub=0, name='_false')
                            self.scip_model.addCons(self._scip_infeasible_aux >= 1)
                        continue
                    if any(is_true_cst(a) for a in args):
                        if not hasattr(self, "_scip_true_var"):
                            self._scip_true_var = self.scip_model.addVar(vtype='B', lb=1, ub=1, name='true')
                        args = [self._scip_true_var if is_true_cst(a) else self.solver_var(a) for a in args]
                    else:
                        args = self.solver_vars(args)
                    self.scip_model.addConsXor(args, True)
                else:
                    raise NotImplementedError(
                        f"SCIP does not translate global constraint '{cpm_expr.name}' natively; "
                        f"supported globals: {sorted(self.supported_global_constraints)}. "
                        "It should have been decomposed by transform(); please report if you see this."
                    )

            else:
                raise NotImplementedError(cpm_expr)

        return self
    __add__ = add

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            This is the generic implementation, solvers can overwrite this with
            a more efficient native implementation

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - call_from_model: whether the method is called from a CPMpy Model instance or not
                - any other keyword argument

            Returns: number of solutions found
        """

        warnings.warn("Solution enumeration is not implemented in PyScipOPT, defaulting to CPMpy's naive implementation")
        """
            Issues to track for future reference:
                https://github.com/scipopt/PySCIPOpt/issues/549 and
                https://github.com/scipopt/PySCIPOpt/issues/248
        """
        
        return super().solveAll(display, time_limit, solution_limit, call_from_model, **kwargs)

