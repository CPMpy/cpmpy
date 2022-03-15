"""
    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        SolverInterface
        SolverStatus
        ExitStatus

    ==================
    Module description
    ==================
    Contains the abstract class `SolverInterface` for defining solver interfaces,
    as well as a class `SolverStatus` that collects solver statistics,
    and the `ExitStatus` class that represents possible exist statuses.

    Each solver has its own class that inherits from `SolverInterface`.

"""
from ..expressions.core import Expression
from ..expressions.utils import is_num, is_any_list
from ..expressions.python_builtins import any,all
#
#==============================================================================
from enum import Enum
import time

#==============================================================================
from cpmpy.transformations.get_variables import get_variables_model


class SolverInterface(object):
    """
        Abstract class for defining solver interfaces. All classes implementing
        the ``SolverInterface``
    """

    # REQUIRED functions:

    @staticmethod
    def supported():
        """
            Check for support in current system setup. Return True if the system
            has package installed or supports solver, else returns False.

        Returns:
            [bool]: Solver support by current system setup.
        """
        return False

    def __init__(self, name="dummy", cpm_model=None, subsolver=None):
        """
            Initalize solver interface

            - name: str: name of this solver
            - subsolver: string: not used/allowed here
            - cpm_model: CPMpy Model() object, optional: will post its constraints/objective

            Creates the following attributes:
            - name: str, name of the solver
            - cpm_status: SolverStatus(), the CPMpy status after a `solve()`
            - objective_value_: the value of the objective function after solving (or None)
            - user_vars: set(), variables in the original (non-transformed) model,
                           for reverse mapping the values after `solve()`
            - _varmap: dict(), maps cpmpy variables to native solver variables
        """
        assert(subsolver is None)

        self.name = name
        self.cpm_status = SolverStatus(self.name) # status of solving this model
        self.objective_value_ = None

        # initialise variable handling
        self.user_vars = set()  # variables in the original (non-transformed) model
        self._varmap = dict()  # maps cpmpy variables to native solver variables

        # rest uses own API
        if cpm_model is not None:
            # post all constraints at once, implemented in __add__()
            self += cpm_model.constraints

            # post objective
            if cpm_model.objective_ is not None:
                if cpm_model.objective_is_min:
                    self.minimize(cpm_model.objective_)
                else:
                    self.maximize(cpm_model.objective_)

    # instead of overloading minimize/maximize, better just overload 'objective()'
    def minimize(self, expr):
        """
            Post the given expression to the solver as objective to minimize

            `minimize()` can be called multiple times, only the last one is stored
        """
        return self.objective(expr, minimize=True)

    def maximize(self, expr):
        """
            Post the given expression to the solver as objective to maximize

            `maximize()` can be called multiple times, only the last one is stored
        """
        return self.objective(expr, minimize=False)

    # REQUIRED functions to mimic `Model` interface:
    def objective(self, expr, minimize):
        """
            Post the given expression to the solver as objective to minimize/maximize

            - expr: Expression, the CPMpy expression that represents the objective function
            - minimize: Bool, whether it is a minimization problem (True) or maximization problem (False)

            'objective()' can be called multiple times, only the last one is stored
        """
        raise NotImplementedError("Solver does not support objective functions")

    def __add__(self, cpm_cons):
        """
            Adds a constraint to the solver, eagerly (e.g. instantly passed to API)
        """
        raise NotImplementedError("Solver does not support eagerly adding constraints")


    def status(self):
        return self.cpm_status

    def solve(self, model, time_limit=None):
        """
            Build the CPMpy model into solver-supported model ready for solving
            and returns the answer (True/False/objective.value())

            Overwrites self.cpm_status

        :param model: CPMpy model to be parsed.
        :type model: Model

        :param time_limit: optional, time limit in seconds
        :type time_limit: int or float

        :return: Bool:
            - True      if a solution is found (not necessarily optimal, e.g. could be after timeout)
            - False     if no solution is found
        """
        return False

    def objective_value(self):
        """
            Returns the value of the objective function of the latest solver run on this model

        :return: an integer or 'None' if it is not run, or a satisfaction problem
        """
        return self.objective_value_

    def solver_var(self, cpm_var):
        """
           Creates solver variable for cpmpy variable
           or returns from cache if previously created
        """
        return None

    def solver_vars(self, cpm_vars):
        """
           Like `solver_var()` but for arbitrary shaped lists/tensors
        """
        if is_any_list(cpm_vars):
            return [self.solver_vars(v) for v in cpm_vars]
        return self.solver_var(cpm_vars)

    def _post_constraint(self, cpm_expr):
        """
            Post a primitive CPMpy constraint to the native solver API

            What 'primitive' means depends on the solver capabilities,
            more specifically on the transformations applied in `__add__()`

            Solvers do not need to support all constraints.
        """
        return None

    # OPTIONAL functions

    def solveAll(self, display=None, time_limit=None, solution_limit=None, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            This is the generic implementation, solvers can overwrite this with
            a more efficient native implementation

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - any other keyword argument

            Returns: number of solutions found
        """
        # XXX: check that no objective function??
        solution_count = 0
        while self.solve(time_limit=time_limit, **kwargs):
            # display if needed
            if display is not None:
                if isinstance(display, Expression):
                    print(display.value())
                elif isinstance(display, list):
                    print([v.value() for v in display])
                else:
                    display() # callback

            # count and stop
            solution_count += 1
            if solution_count == solution_limit:
                break

            # add nogood on the user variables
            self += any([v != v.value() for v in self.user_vars])

        return solution_count

    def solution_hint(self, cpm_vars, vals):
        """
        For warmstarting the solver with a variable assignment

        Typically implemented in SAT-based solvers

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
        raise NotImplementedError("Solver does not support solution hinting")

    def get_core(self):
        """
        For use with s.solve(assumptions=[...]). Only meaningful if the solver returned UNSAT.

        Typically implemented in SAT-based solvers
        
        Returns a small subset of assumption literals that are unsat together.
        (a literal is either a `_BoolVarImpl` or a `NegBoolView` in case of its negation, e.g. x or ~x)
        Setting these literals to True makes the model UNSAT, setting any to False makes it SAT
        """
        raise NotImplementedError("Solver does not support unsat core extraction")


    # shared helper functions

    def _solve_return(self, cpm_status, objective_value=None):
        """
            Take a CPMpy Model and SolverStatus object and return
            the proper answer (True/False/objective_value)

        :param cpm_status: status extracted from the solver
        :type cpm_status: SolverStatus

        :param objective_value: None or Int, as computed by solver [DEPRECATED]

        :return: Bool
            - True      if a solution is found (not necessarily optimal, e.g. could be after timeout)
            - False     if no solution is found
        """
        return (cpm_status.exitstatus == ExitStatus.OPTIMAL or \
                cpm_status.exitstatus == ExitStatus.FEASIBLE)


#==============================================================================
class ExitStatus(Enum):
    """
    Exit status of the solver

    Attributes:

        `NOT_RUN`: Has not been run

        `OPTIMAL`: Optimal solution to an optimisation problem found

        `FEASIBLE`: Feasible solution to a satisfaction problem found,
                    or feasible (but not proven optimal) solution to an
                    optimisation problem found

        `UNSATISFIABLE`: No satisfying solution exists

        `ERROR`: Some error occured (solver should have thrown Exception)

        `UNKNOWN`: Outcome unknown, for example when timeout is reached
    """
    NOT_RUN = 1
    OPTIMAL = 2
    FEASIBLE = 3
    UNSATISFIABLE = 4
    ERROR = 5
    UNKNOWN = 6

#==============================================================================
class SolverStatus(object):
    """
        Status and statistics of a solver run
    """
    exitstatus: ExitStatus
    runtime: time

    def __init__(self, name):
        self.solver_name = name
        self.exitstatus = ExitStatus.NOT_RUN
        self.runtime = None

    def __repr__(self):
        return "{} ({} seconds)".format(self.exitstatus, self.runtime)
