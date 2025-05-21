# /usr/bin/env python
"""
Interface to Pindakaas (`pindakaas`) API.

Requires that the `pindakaas` library python package is installed:

    $ pip install pindakaas

`pindakaas` is a library to transform pseudo-Boolean and integer constraints into conjunctive normal form.
See https://github.com/pindakaashq/pindakaas.

This solver can be used if the model only has PB constraints.

===============
List of classes
===============

.. autosummary::
    :nosignatures:

    CPM_pindakaas

"""
import inspect
import time
from datetime import timedelta

from ..exceptions import NotSupportedError
from ..expressions.core import BoolVal, Comparison
from ..expressions.variables import NegBoolView, _BoolVarImpl, _IntVarImpl
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint
from ..transformations.normalize import simplify_boolean, toplevel_list
from ..transformations.reification import only_bv_reifies, only_implies
from .solver_interface import ExitStatus, SolverInterface


class CPM_pindakaas(SolverInterface):
    """
    Interface to Pindakaas' Python API.

    Creates the following attributes (see parent constructor for more):

    - ``pdk_solver``: the `pindakaas` solver or formula object
    """

    # TODO add link to docs Documentation of the solver's own Python API: ...

    @staticmethod
    def supported():
        try:
            import pindakaas as pdk

            # check subsolvers via `solver.*` modules
            CPM_pindakaas.subsolvers = dict(
                (name.lower(), solver)
                for name, solver in inspect.getmembers(pdk.solver, inspect.isclass)
            )
            return True
        except ModuleNotFoundError:
            return False

    @staticmethod
    def solvernames():
        if CPM_pindakaas.supported():
            return list(CPM_pindakaas.subsolvers)

    def __init__(self, cpm_model=None, subsolver=None):
        name = "pindakaas"
        if not self.supported():
            raise ImportError(
                f"CPM_{name}: Install the Pindakaas python library `pindakaas` (e.g. `pip install pindakaas`) package to use this solver interface"
            )
        if cpm_model and cpm_model.objective_ is not None:
            raise NotSupportedError(
                f"CPM_{name}: only satisfaction, does not support an objective function"
            )

        import pindakaas as pdk

        try:
            # Set subsolver or use CNF if None
            self.pdk_solver = (
                # pdk.CNF()
                pdk.solver.CaDiCaL()
                if subsolver is None
                else CPM_pindakaas.subsolvers.get[subsolver]
            )
        except KeyError:
            raise ValueError(
                f"Expected subsolver to be `None` or one of {CPM_pindakaas.subsolvers()}, but was {subsolver}"
            )

        self.unsatisfiable = False  # `pindakaas` might determine unsat before solving
        super().__init__(name=name, cpm_model=cpm_model)

    @property
    def native_model(self):
        self.pdk_solver

    def solve(self, time_limit=None, assumptions=None):
        if self.unsatisfiable:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            return self._solve_return(self.cpm_status)

        if time_limit is not None and time_limit <= 0:
            raise ValueError("Time limit must be positive")

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        import pindakaas as pdk

        t = time.time()

        # If no subsolver selected, use CaDiCaL as default CNF solver
        if isinstance(self.pdk_solver, pdk.CNF):
            assert False
            cadical = pdk.solver.CaDiCaL()
            for c in self.pdk_solver:
                cadical += c
            self.pdk_solver = cadical

        time_limit = None if time_limit is None else timedelta(seconds=time_limit)
        assumptions = None if assumptions is None else self.solver_vars(assumptions)

        with self.pdk_solver.solve(
            time_limit=time_limit, assumptions=assumptions
        ) as result:
            self.cpm_status.runtime = time.time() - t

            # translate pindakaas result status to cpmpy status
            match result.status:
                case pdk.solver.Status.SATISFIED:
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
                case pdk.solver.Status.UNSATISFIABLE:
                    self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
                case pdk.solver.Status.UNKNOWN:
                    self.cpm_status.exitstatus = ExitStatus.UNKNOWN
                case _:
                    raise NotImplementedError(
                        f"Pindakaas returned an unkown type of result status: {result}"
                    )

            # # True/False depending on self.cpm_status
            has_sol = self._solve_return(self.cpm_status)

            # translate solution values (of user specified variables only)
            if has_sol:
                # fill in variable values
                for cpm_var in self.user_vars:
                    if cpm_var.name in self._varmap:
                        lit = self.solver_var(cpm_var)
                        cpm_var._value = result.value(lit)
                        if cpm_var._value is None:
                            cpm_var._value = True  # dummy value
                    else:  # if pindakaas does not know the literal, it will error
                        cpm_var._value = None
            else:  # clear values of variables
                for cpm_var in self.user_vars:
                    cpm_var._value = None

        return has_sol

    def solver_var(self, cpm_var):
        if isinstance(cpm_var, NegBoolView):  # negative literal
            # get inner variable and return its negated solver var
            return ~self.solver_var(cpm_var._bv)
        elif isinstance(cpm_var, _BoolVarImpl):  # positive literal
            # insert if new
            if cpm_var.name not in self._varmap:
                (self._varmap[cpm_var.name],) = self.pdk_solver.new_vars(1)
            return self._varmap[cpm_var.name]
        else:
            raise NotImplementedError(
                f"{self.name}: unexpected variable {cpm_var} of type {type(cpm_var)} not supported"
            )

    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the `pindakaas` solver supports.

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = decompose_in_tree(cpm_cons)
        cpm_cons = simplify_boolean(cpm_cons)
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_implies(cpm_cons)
        cpm_cons = linearize_constraint(
            cpm_cons, supported=frozenset({"sum", "wsum", "and", "or"})
        )
        return cpm_cons

    def add(self, cpm_expr_orig):
        import pindakaas as pdk

        if self.unsatisfiable:
            return self

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        try:
            for cpm_expr in self.transform(cpm_expr_orig):
                self._add(cpm_expr)
        except pdk.Unsatisfiable:
            self.unsatisfiable = True

        return self

    __add__ = add

    def _add(self, cpm_expr, conditions=[]):
        """Add for a single, transformed expression, implied by conditions (mostly for internal use)"""
        import pindakaas as pdk

        if isinstance(cpm_expr, BoolVal):
            # base case: Boolean value
            if cpm_expr.args[0] is False:
                self.pdk_solver.add_clause(conditions)

        elif isinstance(cpm_expr, _BoolVarImpl):  # (implied) literal
            self.pdk_solver.add_clause(conditions + [self.solver_var(cpm_expr)])

        elif cpm_expr.name == "or":  # (implied) clause
            self.pdk_solver.add_clause(conditions + self.solver_vars(cpm_expr.args))

        elif cpm_expr.name == "->":  # implication
            a0, a1 = cpm_expr.args
            self._add(a1, conditions=conditions + [~self.solver_var(a0)])

        elif isinstance(cpm_expr, Comparison):  # Bool linear
            lhs, k = cpm_expr.args
            match lhs.name:
                case "sum":
                    literals = lhs.args
                    coefficients = [1] * len(literals)
                case "wsum":
                    coefficients, literals = lhs.args
                case _:
                    raise ValueError(
                        f"Trying to encode non (Boolean) linear constraint: {cpm_expr}"
                    )

            lhs = sum(c * l for c, l in zip(coefficients, self.solver_vars(literals)))

            match cpm_expr.name:
                case "<=":
                    self.pdk_solver += lhs <= k
                case ">=":
                    self.pdk_solver += lhs >= k
                case "==":
                    self.pdk_solver += lhs == k
                case _:
                    raise ValueError(
                        f"Unsupported comparator for constraint: {cpm_expr}"
                    )
        else:
            raise NotSupportedError(f"{self.name}: Unsupported constraint {cpm_expr}")
