"""
    Generic interface, solver status and exit status.

    Contains the abstract :class:`SolverInterface` for defining solver interfaces,
    as well as a class :class:`SolverStatus` that collects solver statistics,
    and the :class:`ExitStatus` class that represents possible exist statuses.

    Each solver has its own class that inherits from :class:`SolverInterface`.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        SolverInterface
        SolverStatus
        ExitStatus

"""
from typing import Optional, List, Callable, TypeAlias
import warnings
import time
from enum import Enum

from ..exceptions import NotSupportedError
from ..expressions.core import Expression
from ..expressions.variables import _NumVarImpl
from ..transformations.get_variables import get_variables
from ..expressions.utils import is_any_list
from ..expressions.python_builtins import any
from ..transformations.normalize import toplevel_list

Callback: TypeAlias = Expression | List[Expression] | Callable # type alias to use in solveAll

class SolverInterface(object):
    """
        Abstract class for defining solver interfaces. All classes implementing
        the ``SolverInterface``
    """

    supported_global_constraints: frozenset[str] = frozenset()  # global constraints supported by the solver (e.g., AllDifferent...)
    supported_reified_global_constraints: frozenset[str] = frozenset()  # global constraints supported in reified context

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
    
    @classmethod
    def version(cls) -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        raise NotImplementedError("Implementation of 'version' is missing in solver interface. This should be fixed. If encountered, please report on GitHub.")

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
        self._csemap = dict()  # maps cpmpy expressions to solver expressions

        # rest uses own API
        if cpm_model is not None:
            # post all constraints at once, implemented in `add()`
            self += cpm_model.constraints

            # post objective
            if cpm_model.objective_ is not None:
                if cpm_model.objective_is_min:
                    self.minimize(cpm_model.objective_)
                else:
                    self.maximize(cpm_model.objective_)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        raise NotImplementedError("Solver does not support direct solver access. Look at the solver's API for "
                                  "alternative native objects to access directly.")

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

            Arguments:
                expr: Expression, the CPMpy expression that represents the objective function
                minimize: Bool, whether it is a minimization problem (True) or maximization problem (False)

            ``objective()`` can be called multiple times, only the last one is stored
        """
        raise NotImplementedError("Solver does not support objective functions")

    def status(self):
        return self.cpm_status

    def solve(self,time_limit:Optional[float]=None):
        """
            Call the underlying solver.

            Overwrites self.cpm_status

            :param time_limit: optional, time limit in seconds

            :return: Bool:
                - True      if a solution is found (not necessarily optimal, e.g. could be after timeout)
                - False     if no solution is found
        """
        return False

    def has_objective(self):
        """
            Returns whether the solver has an objective function or not.
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
        return toplevel_list(cpm_expr)  # replace by the transformations your solver needs

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

        # transform and post the constraints
        for con in self.transform(cpm_expr):
            raise NotImplementedError("solver add(): abstract function, overwrite")

        return self
    
    # needed here for subclasses that don't do the more direct `__add__ = add` in their class
    def __add__(self, cpm_expr):
        return self.add(cpm_expr)


    # OPTIONAL functions

    def solveAll(self, display:Optional[Callback]=None, time_limit:Optional[float]=None, solution_limit:Optional[int]=None, call_from_model=False, **kwargs):
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

            Returns: 
                number of solutions found
        """
        if self.has_objective():
            raise NotSupportedError(f"Solver of type {self} does not support finding all optimal solutions!")

        if not call_from_model:
            warnings.warn("Adding constraints to solver object to find all solutions, "
                          "solver state will be invalid after this call!")
            
        self.cpm_status = SolverStatus(self.name)

        solution_count = 0
        start = time.time()
        while ((time_limit is None) or (time_limit > 0)) and self.solve(time_limit=time_limit, **kwargs):
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
            self += any([v != v.value() for v in self.user_vars if v.value() is not None])

            if time_limit is not None: # update remaining time
                time_limit -= self.status().runtime
        end = time.time()

        # update solver status
        self.cpm_status.runtime = end - start
        if solution_count:
            if solution_count == solution_limit:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            elif self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE:
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        # else: <- is implicit since nothing needs to update
        #     if self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE:
        #         self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        #     elif self.cpm_status.exitstatus == ExitStatus.UNKNOWN:
        #         self.cpm_status.exitstatus = ExitStatus.UNKNOWN

        return solution_count

    def solution_hint(self, cpm_vars:List[_NumVarImpl], vals:List[int|bool]):
        """
        For warmstarting the solver with a variable assignment

        Typically implemented in SAT-based solvers

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
        raise NotSupportedError("Solver does not support solution hinting")

    def get_core(self):
        """
        For use with :func:`s.solve(assumptions=[...]) <solve()>`. Only meaningful if the solver returned UNSAT.

        Typically implemented in SAT-based solvers
        
        Returns a small subset of assumption literals that are unsat together.
        (a literal is either a :class:`~cpmpy.expressions.variables._BoolVarImpl` or a :class:`~cpmpy.expressions.variables.NegBoolView` in case of its negation, e.g. x or ~x)
        Setting these literals to True makes the model UNSAT, setting any to False makes it SAT
        """
        raise NotSupportedError("Solver does not support unsat core extraction")


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
