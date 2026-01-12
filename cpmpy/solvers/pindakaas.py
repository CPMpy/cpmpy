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

    $ pip install pindakaas

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

import time
from datetime import timedelta
from typing import Optional, List

from ..exceptions import NotSupportedError
from ..expressions.core import BoolVal, Comparison
from ..expressions.utils import eval_comparison
from ..expressions.variables import NegBoolView, _BoolVarImpl, _IntVarImpl
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint
from ..transformations.get_variables import get_variables
from ..transformations.int2bool import _decide_encoding, _encode_int_var, int2bool
from ..transformations.linearize import linearize_constraint
from ..transformations.normalize import simplify_boolean, toplevel_list
from ..transformations.reification import only_bv_reifies, only_implies
from ..transformations.safening import no_partial_functions
from .solver_interface import ExitStatus, SolverInterface


class CPM_pindakaas(SolverInterface):
    """
    Interface to Pindakaas' Python API.

    Creates the following attributes (see parent constructor for more):

    - ``pdk_solver``: The Pindakaas solver back-end which encodes and solves models through the SAT sub-solver
    - ``ivarmap``: a mapping from integer variables to their encoding for `int2bool`
    - ``encoding``: the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary"). Set to "auto" but can be changed in the solver object.
    - `unsatisfiable`: if a constraint is found to be unsatisfiable during the encoding phase, this flag is set to `True` to prevent further encoding efforts
    - ``core``: if the problem is unsatisfiable, the unsatisfiable core, else `None`


    Documentation of the solver's own Python API:

    - https://pypi.org/project/pindakaas/
    - https://github.com/pindakaashq/pindakaas

    """

    supported_global_constraints = frozenset()
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        try:
            import pindakaas

            return True
        except ModuleNotFoundError:
            return False

    @staticmethod
    def version() -> Optional[str]:
        """Return the installed version of the solver's Python API."""
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("pindakaas")
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Initialize Pindakaas interface.

        """
        name = "pindakaas"
        if not self.supported():
            raise ImportError(
                f"CPM_{name}: Install the Pindakaas python library `pindakaas` (e.g. `pip install pindakaas`) package to use this solver interface"
            )
        if cpm_model and cpm_model.objective_ is not None:
            raise NotSupportedError(f"CPM_{name}: only satisfaction, does not support an objective function")

        import pindakaas as pdk

        assert subsolver is None, "Pindakaas does not support any subsolvers for the moment"
        self.ivarmap = dict()  # for the integer to boolean encoders
        self.encoding = "auto"
        self.pdk_solver = pdk.solver.CaDiCaL()
        self.unsatisfiable = False  # `pindakaas` might determine unsat before solving
        self.core = None  # latest UNSAT core
        super().__init__(name=name, cpm_model=cpm_model)

    @property
    def native_model(self):
        return self.pdk_solver

    def _int2bool_user_vars(self):
        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # the user vars are only the Booleans (e.g. to ensure solveAll behaves consistently)
        user_vars = set()
        for x in self.user_vars:
            if isinstance(x, _BoolVarImpl):
                user_vars.add(x)
            else:
                # extends set with encoding variables of `x`
                user_vars.update(self.ivarmap[x.name].vars())
        return user_vars

    def solve(self, time_limit:Optional[float]=None, assumptions:Optional[List[_BoolVarImpl]]=None):
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

        self.user_vars = self._int2bool_user_vars()

        if time_limit is not None:
            time_limit = timedelta(seconds=time_limit)
        solver_assumptions = None if assumptions is None else self.solver_vars(assumptions)

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
                raise NotImplementedError(f"Pindakaas returned an unkown type of result status: {result}")

            # True/False depending on self.cpm_status
            has_sol = self._solve_return(self.cpm_status)

            # translate solution values (of user specified variables only)
            if has_sol:
                # fill in variable values
                for cpm_var in self.user_vars:
                    # essentially `.solver_var`, but failing if new vars are added
                    if isinstance(cpm_var, NegBoolView):
                        lit = ~self._varmap[cpm_var._bv.name]
                    elif isinstance(cpm_var, _BoolVarImpl):
                        lit = self._varmap[cpm_var.name]
                    else:
                        raise ValueError(
                            f"Integer variables should have been encoded using `int2bool` transformation, but {cpm_var} is integer, please report on GitHub"
                        )
                    value = result.value(lit)
                    assert value is not None, (
                        "All user variables should have been assigned, but {cpm_var} (literal {lit}) was not."
                    )
                    cpm_var._value = value
                self.core = None
                # Now assign the user integer variables using their encodings
                # `ivarmap` also contains auxiliary variable, but they will be assigned 'None' as their encoding variables are assigned `None`
                for enc in self.ivarmap.values():
                    enc._x._value = enc.decode()

            else:  # clear values of variables
                for cpm_var in self.user_vars:
                    cpm_var._value = None
                # we have to save the unsat core here, as the result object does not live beyond this solve call
                if assumptions is not None:
                    self.core = [x for x, s_x in zip(assumptions, solver_assumptions) if result.failed(s_x)]

        return has_sol

    def solver_var(self, cpm_var):
        if isinstance(cpm_var, NegBoolView):  # negative literal
            # get inner variable and return its negated solver var
            return ~self.solver_var(cpm_var._bv)
        elif isinstance(cpm_var, _BoolVarImpl):  # positive literal
            # insert if new
            if cpm_var.name not in self._varmap:
                self._varmap[cpm_var.name] = self.pdk_solver.new_var()
            return self._varmap[cpm_var.name]
        elif isinstance(cpm_var, _IntVarImpl):  # intvar
            if cpm_var.name not in self.ivarmap:
                enc, cons = _encode_int_var(
                    self.ivarmap, cpm_var, _decide_encoding(cpm_var, None, encoding=self.encoding)
                )
                self += cons
            else:
                enc = self.ivarmap[cpm_var.name]
            return self.solver_vars(enc.vars())
        else:
            raise TypeError(f"Unexpected type: {cpm_var}")

    def transform(self, cpm_expr):
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"div", "mod", "element"})
        cpm_cons = decompose_in_tree(
            cpm_cons,
            supported=self.supported_global_constraints | {"alldifferent"},  # alldiff has a specialized MIP decomp in linearize
            supported_reified=self.supported_reified_global_constraints,
            csemap=self._csemap,
        )
        cpm_cons = simplify_boolean(cpm_cons)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)
        cpm_cons = linearize_constraint(
            cpm_cons, supported=frozenset({"sum", "wsum", "->", "and", "or"}), csemap=self._csemap
        )
        cpm_cons = int2bool(cpm_cons, self.ivarmap, encoding=self.encoding)
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
                self._post_constraint(cpm_expr)
        except pdk.Unsatisfiable:
            self.unsatisfiable = True

        return self

    __add__ = add  # avoid redirect in superclass

    def _add_clause(self, clause, conditions=[]):
        """Add a clause implied by conditions; both arguments are lists of CPMpy literals."""
        if not isinstance(clause, list):
            raise TypeError

        self.pdk_solver.add_clause(self.solver_vars([~c for c in conditions] + clause))

    def _post_constraint(self, cpm_expr, conditions=[]):
        if not isinstance(conditions, list):
            raise TypeError

        """Add a single, *transformed* constraint, implied by conditions."""
        import pindakaas as pdk

        if isinstance(cpm_expr, BoolVal):
            # base case: Boolean value
            if cpm_expr.args[0] is False:
                self._add_clause([], conditions=conditions)

        elif isinstance(cpm_expr, _BoolVarImpl):  # (implied) literal
            self._add_clause([cpm_expr], conditions=conditions)

        elif cpm_expr.name == "or":  # (implied) clause
            self._add_clause(cpm_expr.args, conditions=conditions)

        elif cpm_expr.name == "->":  # implication
            a0, a1 = cpm_expr.args
            self._post_constraint(a1, conditions=conditions + [a0])

        elif isinstance(cpm_expr, Comparison):  # Bool linear
            assert cpm_expr.name in {"<=", ">=", "=="}, (
                f"Unsupported comparator {cpm_expr.name} for constraint should have been transformed: {cpm_expr}"
            )

            # lhs is a sum/wsum, right hand side a constant int
            lhs, rhs = cpm_expr.args
            if isinstance(lhs, _BoolVarImpl):
                literals = [lhs]
                coefficients = [1]
            elif lhs.name == "sum":
                literals = lhs.args
                coefficients = [1] * len(literals)
            elif lhs.name == "wsum":
                coefficients, literals = lhs.args
            else:
                raise ValueError(f"Trying to encode non (Boolean) linear constraint: {cpm_expr}")

            lhs = sum(c * l for c, l in zip(coefficients, self.solver_vars(literals)))

            try:
                # normalization may raise `pdk.Unsatisfiable`
                self.pdk_solver.add_encoding(
                    eval_comparison(cpm_expr.name, lhs, rhs),
                    # seems pindakaas conditions are the wrong way around
                    conditions=self.solver_vars([~c for c in conditions]),
                )
            except pdk.Unsatisfiable as e:
                if conditions:
                    # trivial unsat with conditions does not count; posts ~conditions
                    # `add_clause` may raise `pdk.Unsatisfiable` too, but the conditions are added to the clause, so no need to catch
                    self._add_clause([], conditions=conditions)
                else:
                    # no conditions means truly unsatisfiable
                    raise e
        else:
            raise NotSupportedError(f"{self.name}: Unsupported constraint {cpm_expr}")

    def get_core(self):
        assert self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE and self.core is not None, (
            "get_core(): requires a previous solve call with assumption variables and an UNSATISFIABLE result"
        )
        return self.core
