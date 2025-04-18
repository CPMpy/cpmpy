# /usr/bin/env python
"""
Interface to Pindakaas (`pkd`) API.

Requires that the `pkd` library python package is installed:

    $ pip install pindakaas

`pkd` is a library to transform pseudo-Boolean and integer constraints into conjunctive normal form.
See https://github.com/pindakaashq/pindakaas.

This solver can be used if the model only has PB constraints.

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

from ..exceptions import NotSupportedError
from ..expressions.core import BoolVal, Comparison, Operator
from ..expressions.globalconstraints import DirectConstraint
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
    Interface to TEMPLATE's API.

    Requires that the 'TEMPLATEpy' python package is installed:
    $ pip install TEMPLATEpy

    See detailed installation instructions at:
    <URL to detailed solver installation instructions, if any>

    Creates the following attributes (see parent constructor for more):
    - tpl_model: object, TEMPLATE's model object
    """

    @staticmethod
    def supported():
        """Return if solver is installed."""
        # check import without importing
        return importlib.util.find_spec("spam") is not None

    @staticmethod
    def solvernames():
        """Return solvers supported by `pkd` on your system."""
        return ["cadical"]

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Construct the native solver object.

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: str, name of a subsolver (optional)
        """
        name = "pindakaas"
        if not self.supported():
            raise Exception(
                f"CPM_{name}: Install the Pindakaas python library `pindakaas` (e.g. `pip install pindakaas`) package to use this solver interface"
            )
        if cpm_model and cpm_model.objective_ is not None:
            raise NotSupportedError(
                f"CPM_{name}: only satisfaction, does not support an objective function"
            )

        import pindakaas as pkd

        self.pkl_solver = pkd.Cadical

        # initialise everything else and post the constraints/objective
        self.unsatisfiable = False
        super().__init__(name=name, cpm_model=cpm_model)

    @property
    def native_model(self):
        """Returns the solver's underlying native model (for direct solver access)."""
        raise NotSupportedError(
            f"{self.name}: sub-solvers not yet supported, encode-only"
        )

    def solve(self, time_limit=None, assumptions=None):
        """
        Call the `pkd` back-end SAT solver.

        Arguments:
        - time_limit:  maximum solve time in seconds (float, optional)
        - kwargs:      any keyword argument, sets parameters of solver object

        Arguments that correspond to solver parameters:
        # [GUIDELINE] Please document key solver arguments that the user might wish to change
        #       for example: assumptions=[x,y,z], log_output=True, var_ordering=3, num_cores=8, ...
        # [GUIDELINE] Add link to documentation of all solver parameters
        """
        if self.unsatisfiable:
            return False

        if assumptions is not None:
            raise NotSupportedError(f"{self.name}: assumptions currently unsupported")
        if time_limit is not None:
            raise NotSupportedError(f"{self.name}: time not supported yet")

        # ensure all vars are known to solver
        user_vars = self.solver_vars(list(self.user_vars))

        t = time.time()
        assert hasattr(
            self.pkl_solver, "solve"
        ), f"Pindakaas ClauseDatabase did not have solve:\n{self.pkl_solver}"
        my_status = self.pkl_solver.solve(user_vars)

        self.cpm_status.runtime = time.time() - t

        # translate exit status
        if my_status is True:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status is False:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status is None:
            # can happen when timeout is reached
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:  # another?
            raise NotImplementedError(
                my_status
            )  # a new status type was introduced, please report on github

        # # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                if cpm_var.name in self._varmap:
                    lit = self.solver_var(cpm_var)
                    cpm_var._value = self.pkl_solver.value(lit)
                    if cpm_var._value is None:
                        cpm_var._value = True  # dummy value
                else:
                    cpm_var._value = (
                        None  # pindakaas does not know this literal and will error
                    )
        else:  # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None

        return has_sol

    def solver_var(self, cpm_var):
        """
        Create solver variable for cpmpy variable or returns from cache if previously created.

        Transforms cpm_var into CNF literal using self.pkl_solver
        (positive or negative integer)
        """
        if isinstance(cpm_var, NegBoolView):  # negative literal
            # get inner variable and return its negated solver var
            return ~self.solver_var(cpm_var._bv)
        elif isinstance(cpm_var, _BoolVarImpl):  # positive literal
            # insert if new
            if cpm_var.name not in self._varmap:
                self._varmap[cpm_var.name] = self.pkl_solver.add_variable()
            return self._varmap[cpm_var.name]
        else:
            raise NotImplementedError(
                f"{self.name}: variable {cpm_var} of type {type(cpm_var)} not supported"
            )

    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports.

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the 'Adding a new solver' docs on readthedocs for more information.

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
            cpm_cons, supported=frozenset({"sum", "wsum", "and", "or", "bv"})
        )
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

        What 'supported' means depends on the solver capabilities, and in effect on what transformations
        are applied in `transform()`.

        """
        import pindakaas as pkd

        if self.unsatisfiable:
            return self

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        try:
            for cpm_expr in self.transform(cpm_expr_orig):
                if cpm_expr.name == "or":
                    self.pkl_solver.add_clause(self.solver_vars(cpm_expr.args))

                elif cpm_expr.name == "->":  # BV -> BE only thanks to only_bv_reifies
                    a0, a1 = cpm_expr.args
                    self._add_bool_linear(a1, conditions=[~a0])

                elif isinstance(cpm_expr, Comparison):
                    self._add_bool_linear(cpm_expr)

                elif isinstance(cpm_expr, BoolVal):
                    # base case: Boolean value
                    if cpm_expr.args[0] is False:
                        self.pkl_solver.add_clause([])

                elif isinstance(cpm_expr, _BoolVarImpl):
                    # base case, just var or ~var
                    self.pkl_solver.add_clause([self.solver_var(cpm_expr)])

                # a direct constraint, pass to solver
                elif isinstance(cpm_expr, DirectConstraint):
                    raise NotImplementedError("TODO")
                    cpm_expr.callSolver(self, self.pysat_solver)

                else:
                    raise NotImplementedError(
                        f"{self.name}: Non supported constraint {cpm_expr}"
                    )
        except pkd.Unsatisfiable:
            self.unsatisfiable = True

        return self

    """ Unpack implied literal, clause, sum, or weighted sum """

    def _add_bool_linear(self, cpm_expr, conditions=[]):
        import pindakaas as pkd

        literals = None
        coefficients = None
        comparator = None
        k = None
        if isinstance(cpm_expr, _BoolVarImpl):
            literals = [cpm_expr]
        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "or":
            literals = cpm_expr.args
        elif isinstance(cpm_expr, Comparison):
            lhs, k = cpm_expr.args
            if lhs.name == "sum":
                literals = lhs.args
            elif lhs.name == "wsum":
                coefficients, literals = lhs.args
            else:
                raise ValueError(
                    f"Trying to encode non (Boolean) linear constraint: {cpm_expr}"
                )
            if cpm_expr.name == "<=":
                comparator = pkd.Comparator.LessEq
            elif cpm_expr.name == ">=":
                comparator = pkd.Comparator.GreaterEq
            elif cpm_expr.name == "==":
                comparator = pkd.Comparator.Equal
            else:
                raise ValueError(f"Unsupported comparator: {cpm_expr.name}")

        self.pkl_solver.add_linear(
            self.solver_vars(literals),
            coefficients=coefficients,
            comparator=comparator,
            k=k,
            conditions=self.solver_vars(conditions),
        )
