#!/usr/bin/env python
"""
    Interface to z3's API

    Z3 is a highly versatile and effective theorem prover from Microsoft.
    Underneath, it is an SMT solver with a wide scala of theory solvers.
    We will interface to the finite-domain integer related parts of the API

    Documentation of the solver's own Python API:
    https://z3prover.github.io/api/html/namespacez3py.html

    Terminology note: a 'model' for z3 is a solution!

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_z3
"""
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.variables import _BoolVarImpl, NegBoolView
from ..expressions.utils import is_num, is_any_list
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint

class CPM_z3(SolverInterface):
    """
    Interface to z3's API

    Requires that the 'z3-solver' python package is installed:
    $ pip install z3-solver

    See detailed installation instructions at:
    https://github.com/Z3Prover/z3#python

    Creates the following attributes (see parent constructor for more):
    z3_solver: object, z3's Solver() object
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import z3
            return True
        except ImportError as e:
            return False


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: None
        """
        if not self.supported():
            raise Exception("CPM_z3: Install the python package 'z3-solver'")

        import z3

        assert(subsolver is None) # unless you support subsolvers, see pysat or minizinc

        # initialise the native solver object
        self.z3_solver = z3.Solver()

        # initialise everything else and post the constraints/objective
        super().__init__(name="z3", cpm_model=cpm_model)


    def solve(self, time_limit=None, assumptions=[], **kwargs):
        """
            Call the z3 solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - assumptions: list of CPMpy Boolean variables (or their negation) that are assumed to be true.
                           For repeated solving, and/or for use with s.get_core(): if the model is UNSAT,
                           get_core() returns a small subset of assumption variables that are unsat together.
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
                - ... (no common examples yet)
            The full list doesn't seem to be documented online, you have to run its help() function:
            ```
            import z3
            z3.Solver().help()
            ```

            Warning! Some parameternames in z3 have a '.' in their name,
            such as (arbitrarily chosen): 'sat.lookahead_simplify'
            You have to construct a dictionary of keyword arguments upfront:
            ```
            params = {"sat.lookahead_simplify": True}
            s.solve(**params)
            ```
        """

        if time_limit is not None:
            # z3 expects milliseconds in int
            self.z3_solver.set(timeout=int(time_limit*1000))

        # call the solver, with parameters
        for (key,value) in kwargs.items():
            self.z3_solver.set(key, value)
        # TODO: how to do optimisation?
        # TODO: assumptions?
        my_status = repr(self.z3_solver.check(assumptions))

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        st = self.z3_solver.statistics()
        self.cpm_status.runtime = st.get_key_value('time')

        # translate exit status
        if my_status == "sat":
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status == "unsat":
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status == "unknown":
            # can happen when timeout is reached...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:  # another?
            raise NotImplementedError(my_status)  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            sol = self.z3_solver.model() # the solution (called model in z3)
            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                cpm_var._value = sol[sol_var]

            # TODO
            # translate objective, for optimisation problems only
            #if self.TEMPLATE_solver.HasObjective():
            #    self.objective_value_ = self.TEMPLATE_solver.ObjectiveValue()
        else:
            for cpm_var in self.user_vars:
                cpm_var._value = None # XXX, maybe all solvers should do this...

        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        import z3

        if is_num(cpm_var): # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return z3.Not(self.solver_var(cpm_var._bv))

        # create if it does not exit
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = z3.Bool(str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = z3.Int(str(cpm_var))
                # set bounds
                self.z3_solver.add(revar >= cpm_var.lb)
                self.z3_solver.add(revar <= cpm_var.ub)
            else:
                raise NotImplementedError("Not a know var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        return self._varmap[cpm_var]


    # if TEMPLATE does not support objective functions, you can delete objective()/_make_numexpr()
    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective
            are premanently posted to the solver)
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

        raise NotImplementedError("TEMPLATE: Not a know supported numexpr {}".format(cpm_expr))


    def __add__(self, cpm_con):
        """
        Post a (list of) CPMpy constraints(=expressions) to the solver

        Note that we don't store the constraints in a cpm_model,
        we first transform the constraints into primitive constraints,
        then post those primitive constraints directly to the native solver

        :param cpm_con CPMpy constraint, or list thereof
        :type cpm_con (list of) Expression(s)
        """
        # add new user vars to the set
        self.user_vars.update(get_variables(cpm_con))

        # apply transformations, then post internally
        # XXX chose the transformations your solver needs, see cpmpy/transformations/
        cpm_cons = flatten_constraint(cpm_con)
        for con in cpm_cons:
            self._post_constraint(con)

        return self

    def _post_constraint(self, cpm_con):
        """
            Post a primitive CPMpy constraint to the native solver API

            What 'primitive' means depends on the solver capabilities,
            more specifically on the transformations applied in `__add__()`

            Solvers do not need to support all constraints.
        """
        if isinstance(cpm_con, _BoolVarImpl):
            # base case, just var or ~var
            self.z3_solver.add(self.solver_var(cpm_con))
        #elif isinstance(cpm_con, Operator) and cpm_con.name == 'or':
        #    self.z3_solver.add_clause([ self.solver_var(var) for var in cpm_con.args ]) # TODO, soon: .add_clause(self.solver_vars(cpm_con.args))
        else:
            raise NotImplementedError("Z3: constraint not (yet) supported", cpm_con)

    # Other functions from SolverInterface that you can overwrite:
    # solveAll, solution_hint, get_core
