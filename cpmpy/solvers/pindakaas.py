# /usr/bin/env python
"""
Interface to the Pindakaas solver's Python API.

Pindakaas is an open-source Rust library for encoding propositional and pseudo-Boolean constraints to SAT, with support for incremental solving and assumptions.

Always use :func:`cp.SolverLookup.get("pindakaas") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

============
Installation
============

Requires that the 'pindakaas' optional dependency is installed:

.. code-block:: console

    $ pip install cpmpy[pindakaas]

Detailed installation instructions available at:

- https://pypi.org/project/pindakaas/
- https://github.com/pindakaashq/pindakaas

The rest of this documentation is for advanced users.

===============
List of classes
===============

.. autosummary::
    :nosignatures:

    CPM_pindakaas

==============
Module details
==============
"""
import importlib
import time
from datetime import timedelta

from ..exceptions import NotSupportedError
from ..expressions.core import BoolVal, Comparison
from ..expressions.variables import NegBoolView, _BoolVarImpl
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

    Documentation of the solver's own Python API:

    - https://pypi.org/project/pindakaas/
    - https://github.com/pindakaashq/pindakaas

    """

    @staticmethod
    def supported():
        return importlib.util.find_spec("pindakaas") is not None

    @staticmethod
    def supported():
        try:
            import pindakaas

            return True
        except ModuleNotFoundError:
            return False

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Initialize Pindakaas interface.

        - `pdk_solver`: The pindakaas back-end which will encode and post constraints for the SAT solver
        - `unsatisfiable`: If a constraint is found to be unsatisfiable during the encoding phase, this flag is set to `True` to prevent further encoding efforts
        """
        name = "pindakaas"
        if not self.supported():
            raise ImportError(
                f"CPM_{name}: Install the Pindakaas python library `pindakaas` (e.g. `pip install cpmpy[pindakaas]`) package to use this solver interface"
            )
        if cpm_model and cpm_model.objective_ is not None:
            raise NotSupportedError(
                f"CPM_{name}: only satisfaction, does not support an objective function"
            )

        import pindakaas as pdk

        assert (
            subsolver is None
        ), "Pindakaas does not support any subsolvers for the moment"
        self.pdk_solver = pdk.solver.CaDiCaL()
        self.unsatisfiable = False  # `pindakaas` might determine unsat before solving
        self.core = None  # latest UNSAT core
        super().__init__(name=name, cpm_model=cpm_model)

    @property
    def native_model(self):
        self.pdk_solver

    def solve(self, time_limit=None, assumptions=None):
        """
        Solve the encoded CPMpy model given optional time limit and assumptions, returning whether a solution was found.

        :param time_limit: optional, time limit in seconds
        :param assumptions: optional, a list of assumptions (Boolean variables which should hold for this solve call)
        """
        if self.unsatisfiable:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            return self._solve_return(self.cpm_status)

        if time_limit is not None and time_limit <= 0:
            raise ValueError("Time limit must be positive")

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        time_limit = None if time_limit is None else timedelta(seconds=time_limit)
        solver_assumptions = (
            None if assumptions is None else self.solver_vars(assumptions)
        )

        t = time.time()
        with self.pdk_solver.solve(
            time_limit=time_limit,
            assumptions=solver_assumptions,
        ) as result:
            self.cpm_status.runtime = time.time() - t

            # translate pindakaas result status to cpmpy status
            import pindakaas as pdk

            if result.status == pdk.solver.Status.SATISFIED:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            elif result.status == pdk.solver.Status.UNSATISFIABLE:
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            elif result.status == pdk.solver.Status.UNKNOWN:
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            else:
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
                self.core = None
            else:  # clear values of variables
                for cpm_var in self.user_vars:
                    cpm_var._value = None
                # we have to save the unsat core here, as the result object does not live beyond this solve call
                if assumptions is not None:
                    self.core = [
                        x
                        for x, s_x in zip(assumptions, solver_assumptions)
                        if result.failed(s_x)
                    ]

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
                self._add_transformed(cpm_expr)
        except pdk.Unsatisfiable:
            self.unsatisfiable = True

        return self

    __add__ = add  # avoid redirect in superclass

    def _add_transformed(self, cpm_expr, conditions=[]):
        """Add for a single, *transformed* expression, implied by conditions."""
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
            self._add_transformed(a1, conditions=conditions + [~self.solver_var(a0)])

        elif isinstance(cpm_expr, Comparison):  # Bool linear
            # lhs is a sum/wsum, right hand side a constant int
            lhs, rhs = cpm_expr.args
            if lhs.name == "sum":
                literals = lhs.args
                coefficients = [1] * len(literals)
            elif lhs.name == "wsum":
                coefficients, literals = lhs.args
            else:
                raise ValueError(
                    f"Trying to encode non (Boolean) linear constraint: {cpm_expr}"
                )

            lhs = sum(c * l for c, l in zip(coefficients, self.solver_vars(literals)))

            if cpm_expr.name == "<=":
                con = lhs <= rhs
            elif cpm_expr.name == ">=":
                con = lhs >= rhs
            elif cpm_expr.name == "==":
                con = lhs == rhs
            else:
                raise ValueError(f"Unsupported comparator for constraint: {cpm_expr}")

            self.pdk_solver.add_encoding(con, conditions=conditions)
        else:
            raise NotSupportedError(f"{self.name}: Unsupported constraint {cpm_expr}")

    def get_core(self):
        assert (
            self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE
            and self.core is not None
        ), "get_core(): requires a previous solve call with assumption variables and an UNSATISFIABLE result"
        return self.core
