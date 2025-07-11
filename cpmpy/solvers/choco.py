#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## choco.py
##
"""
    Interface to Choco solver's Python API.

    Choco-solver is an open-source Java library for Constraint Programming (see https://choco-solver.org/).
    It comes with many features such as various types of variables, various state-of-the-art constraints, various search strategies, etc.

    Always use :func:`cp.SolverLookup.get("choco") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'pychoco' python package is installed:

    .. code-block:: console

        $ pip install pychoco

    Detailed installation instructions available at:
    
    - https://pypi.org/project/pychoco/
    - https://pychoco.readthedocs.io/en/latest/

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_choco

    ==============
    Module details
    ==============
"""
import time

import numpy as np

import warnings
import pkg_resources
from pkg_resources import VersionConflict

from ..transformations.normalize import toplevel_list
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView, intvar
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.utils import is_num, is_int, is_boolexpr, is_any_list, get_bounds, argval, argvals, STAR
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.linearize import canonical_comparison
from ..transformations.safening import no_partial_functions
from ..transformations.reification import reify_rewrite
from ..exceptions import ChocoBoundsException, NotSupportedError


class CPM_choco(SolverInterface):
    """
    Interface to the Choco solver python API

    Creates the following attributes (see parent constructor for more):
    
    - ``chc_model`` : the pychoco.Model() created by _model()
    - ``chc_solver`` : the choco Model().get_solver() instance used in solve()

    Documentation of the solver's own Python API:
    
    - https://pypi.org/project/pychoco/
    - https://pychoco.readthedocs.io/en/latest/

    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            # check if pychoco is installed
            import pychoco as chc
            # check it's the correct version
            # CPMPy uses features only available from 0.2.1
            pkg_resources.require("pychoco>=0.2.1")
            return True
        except ModuleNotFoundError:
            return False
        except VersionConflict: # unsupported version of pychoco
            warnings.warn(f"CPMpy uses features only available from Pychoco version 0.2.1, "
                          f"but you have version {pkg_resources.get_distribution('pychoco').version}.")
            return False
        except Exception as e:
            raise e

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        choco model and solver object (chc_model and chc_solver)

        chc_model and chc_solver can both be modified externally before
        calling solve(), a prime way to use more advanced solver features

        Arguments:
            cpm_model: Model(), a CPMpy Model() (optional)
            subsolver: None
        """
        if not self.supported():
            raise Exception("CPM_choco: Install the python package 'pychoco' to use this solver interface.")

        import pychoco as chc

        assert (subsolver is None), "Choco does not support any subsolver"

        # initialise the native solver objects
        self.chc_model = chc.Model()

        # for the objective
        self.obj = None
        self.minimize_obj = None
        self.helper_var = None
        # for solving with assumption variables, TO-CHECK

        # initialise everything else and post the constraints/objective
        super().__init__(name="choco", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.chc_model

    def solve(self, time_limit=None, **kwargs):
        """
            Call the Choco solver

            Arguments:
                time_limit (float, optional):   maximum solve time in seconds 
                kwargs:                         any keyword argument, sets parameters of solver object

        """
        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # call the solver, with parameters
        self.chc_solver = self.chc_model.get_solver()

        start = time.time()
        
        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            self.chc_solver.limit_time(str(time_limit) + "s")

        if self.has_objective():
            sol = self.chc_solver.find_optimal_solution(maximize= not self.minimize_obj,
                                                        objective=self.solver_var(self.obj),
                                                        **kwargs)
        else:
            sol = self.chc_solver.find_solution()
        end = time.time()

        # new status, get runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = end - start

        # translate exit status
        # A) Found a solution
        if sol is not None:
            # COP
            if self.has_objective():
                if time_limit is None or self.cpm_status.runtime < time_limit: # solved to optimality
                    self.cpm_status.exitstatus = ExitStatus.OPTIMAL
                else: # solved, but optimality not proven
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            # CSP
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        # B) Found unsat
        elif time_limit is None or self.cpm_status.runtime < time_limit: # proven unsat
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        # C) Timeout
        else:
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN  # can happen when timeout is reached...


        # True/False depending on self.chc_status
        has_sol = sol is not None

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                value = sol.get_int_val(self.solver_var(cpm_var))
                if isinstance(cpm_var, _BoolVarImpl):
                    cpm_var._value = bool(value)
                else:
                    cpm_var._value = value

            # translate objective
            if self.has_objective():
                self.objective_value_ = sol.get_int_val(self.solver_var(self.obj))
        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None

        return has_sol

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            Compute all (optimal) solutions, map them to CPMpy and optionally display the solutions.

            Arguments:
                display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                solution_limit: stop after this many solutions (default: None)
                time_limit (float, optional):   maximum solve time in seconds

            Returns: 
                number of solutions found
        """

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # create solver object
        self.chc_solver = self.chc_model.get_solver()

        # set time limit (if given)
        if time_limit is not None:
            self.chc_solver.limit_time(str(time_limit) + "s")

        start = time.time()
        if self.has_objective():
            sols = self.chc_solver.find_all_optimal_solutions(maximize=not self.minimize_obj,
                                                              solution_limit=solution_limit,
                                                              objective=self.solver_var(self.obj),
                                                              **kwargs)
        else:
            sols = self.chc_solver.find_all_solutions(solution_limit=solution_limit)
        end = time.time()

        # new status, get runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = end - start

        if len(sols): # solutions found
            if (len(sols) == solution_limit): # matched the set limit (if given)
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            elif (time_limit is None) or (self.cpm_status.runtime < time_limit): # found all solutions
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else: # reached timeout
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        else: # no solutions found
            if (time_limit is None) or (self.cpm_status.runtime < time_limit): # unsat problem
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            else: # timeout
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN

        # if no solutions, clear values of variables
        if len(sols) == 0:
            for var in self.user_vars:
                var._value = None

        # display if needed
        if display is not None:
            for sol in sols:
                # map the solution to user vars
                for cpm_var in self.user_vars:
                    value = sol.get_int_val(self.solver_var(cpm_var))
                    if isinstance(cpm_var, _BoolVarImpl):
                        cpm_var._value = bool(value)
                    else:
                        cpm_var._value = value
                # print the desired display
                if isinstance(display, Expression):
                    print(argval(display))
                elif isinstance(display, list):
                    print(argvals(display))
                else:
                    display()  # callback

        return len(sols)

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var):  # shortcut, eases posting constraints
            if not is_int(cpm_var):
                raise ValueError(f"Choco only accepts integer constants, got {cpm_var} of type {type(cpm_var)}")
            if cpm_var < -2147483646 or cpm_var > 2147483646:
                raise ChocoBoundsException(
                    "Choco does not accept integer literals with bounds outside of range (-2147483646..2147483646)")
            return int(cpm_var)

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return self.chc_model.bool_not_view(self.solver_var(cpm_var._bv))

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.chc_model.boolvar(name=str(cpm_var.name))
            elif isinstance(cpm_var, _IntVarImpl):
                if cpm_var.lb < -2147483646 or cpm_var.ub > 2147483646:
                    raise ChocoBoundsException(
                        "Choco does not accept variables with bounds outside of range (-2147483646..2147483646)")
                revar = self.chc_model.intvar(cpm_var.lb, cpm_var.ub, name=str(cpm_var.name))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        return self._varmap[cpm_var]

    def objective(self, expr, minimize):
        """
            Post the given expression to the solver as objective to minimize/maximize

            Arguments:
                expr: Expression, the CPMpy expression that represents the objective function
                minimize: Bool, whether it is a minimization problem (True) or maximization problem (False)

            ``objective()`` can be called multiple times, only the last one is stored

            .. note::
                technical side note: constraints created during conversion of the objective
                are premanently posted to the solver. Choco accepts variables to maximize or minimize
                so it is needed to post constraints and create auxiliary variables
        """

        # make objective function non-nested
        obj_var = intvar(*get_bounds(expr))
        self += obj_var == expr

        self.obj = obj_var
        self.minimize_obj = minimize  # Choco has as default to maximize

    def has_objective(self):
        return self.obj is not None


    def _to_var(self, val):
        from pychoco.variables.intvar import IntVar
        if is_int(val):
            # Choco accepts only int32, not int64
            if val < -2147483646 or val > 2147483646:
                raise ChocoBoundsException(
                    "Choco does not accept integer literals with bounds outside of range (-2147483646..2147483646)")
            return self.chc_model.intvar(int(val), int(val))  # convert to "variable"
        elif isinstance(val, _NumVarImpl):
            return self.solver_var(val)  # use variable
        else:
            raise ValueError(f"Cannot convert {val} of type {type(val)} to Choco variable, expected int or NumVarImpl")

        # elif isinstance(val, IntVar):
        #     return val
        # return None

    def _to_vars(self, vals):
        if is_any_list(vals):
            return [self._to_vars(v) for v in vals]
        return self._to_var(vals)

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
        supported = {"min", "max", "abs", "count", "element", "alldifferent", "alldifferent_except0", "allequal",
                     "table", 'negative_table', "short_table", "regular", "InDomain", "cumulative", "circuit", "gcc", "inverse", "nvalue", "increasing",
                     "decreasing","strictly_increasing","strictly_decreasing","lex_lesseq", "lex_less", "among", "precedence"}

        cpm_cons = no_partial_functions(cpm_cons)
        cpm_cons = decompose_in_tree(cpm_cons, supported, supported, csemap=self._csemap) # choco supports any global also (half-) reified
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
        cpm_cons = canonical_comparison(cpm_cons)
        cpm_cons = reify_rewrite(cpm_cons, supported = supported | {"sum", "wsum"}, csemap=self._csemap)  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]), csemap=self._csemap)  # support >, <, !=

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

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: self
        """
        # add new user vars to the set
        get_variables(cpm_expr, collect=self.user_vars)
        # ensure all vars are known to solver

        # transform and post the constraints
        for con in self.transform(cpm_expr):
            c = self._get_constraint(con)
            if c is not None: # Reification constraints are not posted
                c.post()

        return self
    __add__ = add  # avoid redirect in superclass

    def _get_constraint(self, cpm_expr):
        """
        Get a solver's constraint by a supported CPMpy constraint

        :param cpm_expr: CPMpy expression
        :type cpm_expr: Expression

        """

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        if isinstance(cpm_expr, Operator):
            # 'and'/n, 'or'/n, '->'/2
            if cpm_expr.name == 'and':
                return self.chc_model.and_(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == 'or':
                return self.chc_model.or_(self.solver_vars(cpm_expr.args))

            elif cpm_expr.name == "->":
                cond, subexpr = cpm_expr.args
                if isinstance(cond, _BoolVarImpl) and isinstance(subexpr, _BoolVarImpl):
                    return self.chc_model.or_(self.solver_vars([~cond, subexpr]))
                elif isinstance(cond, _BoolVarImpl):
                    return self._get_constraint(subexpr).implied_by(self.solver_var(cond))
                elif isinstance(subexpr, _BoolVarImpl):
                    return self._get_constraint(cond).implies(self.solver_var(subexpr))
                else:
                    ValueError(f"Unexpected implication: {cpm_expr}")

            else:
                raise NotImplementedError("Not a known supported Choco Operator '{}' {}".format(
                    cpm_expr.name, cpm_expr))

        # Comparisons: both numeric and boolean ones
        # numexpr `comp` bvar|const
        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            op = "=" if cpm_expr.name == "==" else cpm_expr.name
            if is_boolexpr(lhs) and is_boolexpr(rhs): #boolean equality -- Reification
                if isinstance(lhs, _BoolVarImpl) and isinstance(lhs, _BoolVarImpl):
                    return self.chc_model.all_equal(self.solver_vars([lhs, rhs]))
                elif isinstance(lhs, _BoolVarImpl):
                    return self._get_constraint(rhs).reify_with(self.solver_var(lhs))
                elif isinstance(rhs, _BoolVarImpl):
                    return self._get_constraint(lhs).reify_with(self.solver_var(rhs))
                else:
                    raise ValueError(f"Unexpected reification {cpm_expr}")

            elif isinstance(lhs, _NumVarImpl):
                return self.chc_model.arithm(self.solver_var(lhs), op, self.solver_var(rhs))
            elif isinstance(lhs, Operator) and lhs.name in {'sum','wsum','sub'}:
                if lhs.name == 'sum':
                    return self.chc_model.sum(self.solver_vars(lhs.args), op, self.solver_var(rhs))
                elif lhs.name == "sub":
                    a, b = self.solver_vars(lhs.args)
                    return self.chc_model.arithm(a, "-", b, op, self.solver_var(rhs))
                elif lhs.name == 'wsum':
                    wgt, x = lhs.args
                    w = np.array(wgt).tolist()
                    x = self.solver_vars(lhs.args[1])
                    return self.chc_model.scalar(x, w, op, self.solver_var(rhs))

            elif cpm_expr.name == '==':

                chc_rhs = self._to_var(rhs) # result is always var
                all_vars = {"min", "max", "abs", "div", "mod", "element", "nvalue"}
                if lhs.name in all_vars:

                    chc_args = self._to_vars(lhs.args)

                    if lhs.name == 'min': # min(vars) = var
                        return self.chc_model.min(chc_rhs, chc_args)
                    elif lhs.name == 'max': # max(vars) = var
                        return self.chc_model.max(chc_rhs, chc_args)
                    elif lhs.name == 'abs': # abs(var) = var
                        assert len(chc_args) == 1, f"Expected one argument of abs constraint, but got {chc_args}"
                        return self.chc_model.absolute(chc_rhs, chc_args[0])
                    elif lhs.name == "div": # var / var = var
                        dividend, divisor = chc_args
                        return self.chc_model.div(dividend, divisor, chc_rhs)
                    elif lhs.name == 'mod': # var % var = var
                        dividend, divisor = chc_args
                        return self.chc_model.mod(dividend, divisor, chc_rhs)
                    elif lhs.name == "element": # varsvar[var] = var
                        # TODO: actually, Choco also supports ints[var] = var, but no mix of var and int in array
                        arr, idx = chc_args
                        return self.chc_model.element(chc_rhs, arr, idx)
                    elif lhs.name == "nvalue": # nvalue(vars) = var
                        # TODO: should look into leaving nvalue <= arg so can post atmost_nvalues here
                        return self.chc_model.n_values(chc_args, chc_rhs)

                elif lhs.name == 'count': # count(vars, var/int) = var
                    arr, val = lhs.args
                    return self.chc_model.count(self.solver_var(val), self._to_vars(arr), chc_rhs)
                elif lhs.name == "among":
                    arr, vals = lhs.args
                    return self.chc_model.among(chc_rhs, self._to_vars(arr), vals)
                elif lhs.name == 'mul': # var * var/int = var/int
                    a,b = self.solver_vars(lhs.args)
                    if isinstance(a, int):
                        a,b = b,a # int arg should always be second
                    return self.chc_model.times(a,b, self.solver_var(rhs))
                elif lhs.name == 'pow': # var ^ int = var
                    chc_rhs = self._to_var(rhs)
                    return self.chc_model.pow(*self.solver_vars(lhs.args),chc_rhs)



                raise NotImplementedError(
                    "Not a known supported Choco left-hand-side '{}' {}".format(lhs.name, cpm_expr))

        # base (Boolean) global constraints
        elif isinstance(cpm_expr, GlobalConstraint):

            # many globals require all variables as arguments
            if cpm_expr.name in {"alldifferent", "alldifferent_except0", "allequal", "circuit", "inverse","increasing","decreasing","strictly_increasing","strictly_decreasing","lex_lesseq","lex_less"}:
                chc_args = self._to_vars(cpm_expr.args)
                if cpm_expr.name == 'alldifferent':
                    return self.chc_model.all_different(chc_args)
                elif cpm_expr.name == 'alldifferent_except0':
                    return self.chc_model.all_different_except_0(chc_args)
                elif cpm_expr.name == 'allequal':
                    return self.chc_model.all_equal(chc_args)
                elif cpm_expr.name == "circuit":
                    return self.chc_model.circuit(chc_args)
                elif cpm_expr.name == "inverse":
                    return self.chc_model.inverse_channeling(*chc_args)
                elif cpm_expr.name == "increasing":
                    return self.chc_model.increasing(chc_args,0)
                elif cpm_expr.name == "decreasing":
                    return self.chc_model.decreasing(chc_args,0)
                elif cpm_expr.name == "strictly_increasing":
                    return self.chc_model.increasing(chc_args,1)
                elif cpm_expr.name == "strictly_decreasing":
                    return self.chc_model.decreasing(chc_args,1)
                elif cpm_expr.name in ["lex_lesseq", "lex_less"]:
                    if cpm_expr.name == "lex_lesseq":
                        return self.chc_model.lex_less_eq(*chc_args)
                    return self.chc_model.lex_less(*chc_args)
# Ready for when it is fixed in pychoco (https://github.com/chocoteam/pychoco/issues/30)
#                elif cpm_expr.name == "lex_chain_less":
#                    return self.chc_model.lex_chain_less(chc_args)

            # but not all
            elif cpm_expr.name == 'table':
                assert (len(cpm_expr.args) == 2)  # args = [array, table]
                array, table = self.solver_vars(cpm_expr.args)
                return self.chc_model.table(array, table)
            elif cpm_expr.name == 'negative_table':
                assert (len(cpm_expr.args) == 2)  # args = [array, table]
                array, table = self.solver_vars(cpm_expr.args)
                return self.chc_model.table(array, table, False)
            elif cpm_expr.name == 'short_table':
                assert (len(cpm_expr.args) == 2)  # args = [array, table]
                array, table = cpm_expr.args
                table = np.array(table)
                table[table == STAR] = np.nan
                table = table.astype(float) # nan's require float dtype
                # Choco requires a wildcard value not present in dom of args,
                # take value lower than anything else
                chc_star = int(min(np.nanmin(table), *get_bounds(array)[0]) -1) # should be an int
                chc_table = np.nan_to_num(table, nan=chc_star).astype(int).tolist()
                return self.chc_model.table(self.solver_vars(array), chc_table, universal_value=chc_star, algo="STR2+")
            elif cpm_expr.name == "regular":
                from pychoco.objects.automaton.finite_automaton import FiniteAutomaton
                array, transitions, start, accepting = cpm_expr.args
                for i, (lb, ub) in enumerate(zip(*get_bounds(array))):
                    if lb < 0 or ub > 65535:
                        raise NotSupportedError(f"Choco regular only supports variables within domain 0..65535, got {array[i]} with bounds {lb}..{ub}")
                # convert to Automaton Choco object
                automaton = FiniteAutomaton()
                for node, i in cpm_expr.node_map.items(): automaton.add_state()
                for src, label, dst in transitions:
                    automaton.add_transition(cpm_expr.node_map[src], cpm_expr.node_map[dst], label)
                automaton.set_initial_state(cpm_expr.node_map[start])
                automaton.set_final(*[cpm_expr.node_map[a] for a in accepting])
                return self.chc_model.regular(self._to_vars(array), automaton)
            
            elif cpm_expr.name == 'InDomain':
                assert len(cpm_expr.args) == 2  # args = [array, list of vals]
                expr, table = self.solver_vars(cpm_expr.args)
                return self.chc_model.member(expr, table)
            elif cpm_expr.name == "cumulative":
                start, dur, end, demand, cap = cpm_expr.args
                # start, end, demand and cap should be var
                start, end, demand, cap = self._to_vars([start, end, demand, cap])
                # duration can be var or int
                dur = self.solver_vars(dur)
                # Create task variables. Choco can create them only one by one
                tasks = [self.chc_model.task(s, d, e) for s, d, e in zip(start, dur, end)]
                return self.chc_model.cumulative(tasks, demand, cap)
            elif cpm_expr.name == "precedence":
                return self.chc_model.int_value_precede_chain(self._to_vars(cpm_expr.args[0]), cpm_expr.args[1])
            elif cpm_expr.name == "gcc":
                vars, vals, occ = cpm_expr.args
                return self.chc_model.global_cardinality(self._to_vars(vars), self.solver_vars(vals), self._to_vars(occ), cpm_expr.closed)
            else:
                raise NotImplementedError(f"Unknown global constraint {cpm_expr}, should be decomposed! If you reach this, please report on github.")

        # unlikely base case: Boolean variable
        elif isinstance(cpm_expr, _BoolVarImpl):
            return self.chc_model.and_([self.solver_var(cpm_expr)])

        # unlikely base case: True or False
        elif isinstance(cpm_expr, BoolVal):
            # Choco does not allow to post True or False. Post "certainly True or False" constraints instead
            if cpm_expr.args[0] is True:
                return None
            else:
                if self.helper_var is None:
                    self.helper_var = self.chc_model.intvar(0, 0)
                return self.chc_model.arithm(self.helper_var, "<", 0)

        # a direct constraint, pass to solver
        elif isinstance(cpm_expr, DirectConstraint):
            c = cpm_expr.callSolver(self, self.chc_model)
            return c

        # else
        raise NotImplementedError(cpm_expr)  # if you reach this... please report on github
