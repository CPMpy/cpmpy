#!/usr/bin/env python
"""
    Template file for a new solver interface

    Replace <TEMPLATE> by the solver's name, and implement the missing pieces
    The functions are ordered in a way that could be convenient to 
    start from the top and continue in that order

    WARNING: do not include the python package at the top of the file,
    as CPMpy should also work without this solver installed.
    To ensure that, include it inside supported() and other functions that need it...
"""
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.normalize import toplevel_list

"""
    Interface to TEMPLATE's API

    <some information on the solver>

    Documentation of the solver's own Python API:
    <URL to docs or source code>

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_template
"""
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl
from ..expressions.utils import is_num, is_any_list
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint

class CPM_template(SolverInterface):
    """
    Interface to TEMPLATE's API

    Requires that the 'TEMPLATEpy' python package is installed:
    $ pip install TEMPLATEpy

    See detailed installation instructions at:
    <URL to detailed solver installation instructions, if any>

    Creates the following attributes (see parent constructor for more):
    - tpl_model: object, TEMPLATE's model object
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import TEMPLATEpy as gp
            return True
        except ImportError:
            return False


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: str, name of a subsolver (optional)
        """
        if not self.supported():
            raise Exception("CPM_TEMPLATE: Install the python package 'TEMPLATEpy'")

        import TEMPLATEpy

        assert(subsolver is None) # unless you support subsolvers, see pysat or minizinc

        # initialise the native solver object
        self.tpl_model = TEMPLATEpy.Model("cpmpy")

        # initialise everything else and post the constraints/objective
        super().__init__(name="TEMPLATE", cpm_model=cpm_model)


    def solve(self, time_limit=None, **kwargs):
        """
            Call the TEMPLATE solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            <Please document key solver arguments that the user might wish to change
             for example: log_output=True, var_ordering=3, num_cores=8, ...>
            <Add link to documentation of all solver parameters>
        """

        if time_limit is not None:
            raise NotImplementedError("TEMPLATE: TODO, implement time_limit")

        # call the solver, with parameters
        my_status = self.TEMPLATE_solver.solve(**kwargs)

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.TEMPLATE_solver.time()

        # translate exit status
        if my_status is True:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status is False:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status is None:
            # can happen when timeout is reached...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:  # another?
            raise NotImplementedError(my_status)  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                cpm_var._value = None # if not in solution
                #cpm_var._value = self.TEMPLATEpy.value(sol_var)
                raise NotImplementedError("TEMPLATE: back-translating the solution values")

            # translate objective, for optimisation problems only
            if self.has_objective():
                self.objective_value_ = self.TEMPLATE_solver.ObjectiveValue()

        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var): # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return TEMPLATEpy.negate(self.solver_var(cpm_var._bv))

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = TEMPLATEpy.NewBoolVar(str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = TEMPLATEpy.NewIntVar(cpm_var.lb, cpm_var.ub, str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        # return from cache
        return self._varmap[cpm_var]


    # if TEMPLATE does not support objective functions, you can delete objective()/_make_numexpr()
    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective

            are permanently posted to the solver)
        """
        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr)
        self += flat_cons # add potentially created constraints
        self.user_vars.update(get_variables(flat_obj)) # add objvars to vars

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        if minimize:
            TEMPLATEpy.Minimize(obj)
        else:
            TEMPLATEpy.Maximize(obj)

    def has_objective(self):
        return TEMPLATEpy.hasObjective()

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function
        """
        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # if the solver only supports a decision variable as argument to its minimize/maximize then
        # well, smth generic like
        #(obj_var, obj_cons) = get_or_make_var(cpm_expr)
        #self += obj_cons
        #return self.solver_var(obj_var)

        # else if the solver support e.g. a linear expression as objective, built it here
        # something like
        #if isinstance(cpm_expr, Operator):
        #    if cpm_expr.name == 'sum':
        #        return sum(self.solver_vars(cpm_expr.args)) # if TEMPLATEpy supports this
        #    elif cpm_expr.name == 'wsum':
        #        w = cpm_expr.args[0]
        #        x = self.solver_vars(cpm_expr.args[1])
        #        return sum(wi*xi for wi,xi in zip(w,x)) # if TEMPLATEpy supports this

        raise NotImplementedError("TEMPLATE: Not a known supported numexpr {}".format(cpm_expr))


    # `__add__()` first calls `transform()`
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
        # apply transformations
        # XXX chose the transformations your solver needs, see cpmpy/transformations/
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = decompose_in_tree(cpm_cons, supported={"AllDifferent"})
        cpm_cons = flatten_constraint(cpm_expr)  # flat normal form
        #cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        #cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        # ...
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

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: self
        """

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        for cpm_expr in self.transform(cpm_expr_orig):

            if isinstance(cpm_con, _BoolVarImpl):
                # base case, just var or ~var
                self.TEMPLATE_solver.add_clause([ self.solver_var(cpm_con) ])
            elif cpm_con.name == 'or':
                self.TEMPLATE_solver.add_clause(self.solver_vars(cpm_con.args))
            elif hasattr(cpm_expr, 'decompose'):
                # global constraint not known, try posting generic decomposition
                # side-step `__add__()` as the decomposition can contain non-user (auxiliary) variables
                for con in self.transform(cpm_expr.decompose()):
                    self._post_constraint(con)

            raise NotImplementedError("TEMPLATE: constraint not (yet) supported", cpm_con)

        return self

    # Other functions from SolverInterface that you can overwrite:
    # solveAll, solution_hint, get_core

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            A shorthand to (efficiently) compute all (optimal) solutions, map them to CPMpy and optionally display the solutions.

            If the problem is an optimization problem, returns only optimal solutions.

           Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - call_from_model: whether the method is called from a CPMpy Model instance or not
                - any other keyword argument

            Returns: number of solutions found
        """

        # check if objective function
        if self.has_objective():
            raise Exception("TEMPLATE does not support finding all optimal solutions")

        # A. Example code if solver supports callbacks
        if is_any_list(display):
            callback = lambda : print([var.value() for var in display])
        else:
            callback = display

        self.solve(time_limit, callback=callback, enumerate_all_solutions=True, **kwargs)
        return self.TEMPLATE_solver.SolutionCount()

        # B. Example code if solver does not support callbacks
        self.solve(time_limit, enumerate_all_solutions=True, **kwargs)
        solution_count = 0
        for solution in self.TEMPLATE_solver.GetSolutions():
            solution_count += 1
            # Translate solution to variables
            for cpm_var in self.user_vars:
                cpm_var._value = solution.value(solver_var)

            if display is not None:
                if isinstance(display, Expression):
                    print(display.value())
                elif isinstance(display, list):
                    print([v.value() for v in display])
                else:
                    display()  # callback

        return solution_count
