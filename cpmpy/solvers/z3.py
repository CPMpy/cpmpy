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
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.variables import _BoolVarImpl, NegBoolView, _NumVarImpl, _IntVarImpl
from ..expressions.python_builtins import min, max,any, all
from ..expressions.utils import is_num, is_any_list, is_bool
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint, get_or_make_var

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


    def __init__(self, cpm_model=None, subsolver="sat"):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: None
        """
        if not self.supported():
            raise Exception("CPM_z3: Install the python package 'z3-solver'")

        import z3

        if subsolver is None:
            subsolver = "sat"
        assert "sat" in subsolver or "opt" in subsolver, "Z3 only has a satisfaction or optimization sub-solver."

        # initialise the native solver object
        if "sat" in subsolver:
            self.z3_solver = z3.Solver()
        if "opt" in subsolver:
            self.z3_solver = z3.Optimize()

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
        import z3

        if time_limit is not None:
            # z3 expects milliseconds in int
            self.z3_solver.set(timeout=int(time_limit*1000))


        z3_assum_vars = self.solver_vars(assumptions)
        self.assumption_dict = {z3_var : cpm_var for (cpm_var, z3_var) in zip(assumptions, z3_assum_vars)}


        # call the solver, with parameters
        for (key,value) in kwargs.items():
            self.z3_solver.set(key, value)

        my_status = repr(self.z3_solver.check(*z3_assum_vars))

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        st = self.z3_solver.statistics()
        if 'time' not in st.keys():
            self.cpm_status.runtime = 0
        else:
            self.cpm_status.runtime = st.get_key_value('time')

        # translate exit status
        if my_status == "sat":
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            if isinstance(self.z3_solver, z3.Optimize):
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
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
                if isinstance(cpm_var, _BoolVarImpl):
                    cpm_var._value = bool(sol[sol_var])
                elif isinstance(cpm_var, _NumVarImpl):
                    cpm_var._value = sol[sol_var].as_long()

            # translate objective, for optimisation problems only
            if isinstance(self.z3_solver, z3.Optimize) and \
                    len(self.z3_solver.objectives()) != 0:
                obj = self.z3_solver.objectives()[0]
                self.objective_value_ = sol.evaluate(obj)

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
            # we assume al variables are user variables (because nested expressions)
            self.user_vars.add(cpm_var)
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
        import z3
        # objective can be a nested expression for z3
        assert isinstance(self.z3_solver, z3.Optimize), "Use the z3 optimizer for optimization problems"
        obj = self._z3_expr(expr)
        if minimize:
            self.z3_solver.minimize(obj)
        else:
            self.z3_solver.maximize(obj)

    def __add__(self, cpm_con):
        """
        Post a (list of) CPMpy constraints(=expressions) to the solver

        Note that we don't store the constraints in a cpm_model,
        we first transform the constraints into primitive constraints,
        then post those primitive constraints directly to the native solver

        :param cpm_con CPMpy constraint, or list thereof
        :type cpm_con (list of) Expression(s)
        """
        # Z3 supports nested expressions, so no transformations needed
        # that also means we don't need to extract user variables here
        # we store them directly in `solver_var()` itself.
        self._post_constraint(cpm_con)

        return self

    def _post_constraint(self, cpm_expr):
        """
            Post a primitive CPMpy constraint to the native solver API

            Z3 supports nested expressions so translate expression tree and post to solver API directly
        """
        if is_any_list(cpm_expr):
            for con in cpm_expr:
                self._post_constraint(con)

        # translate each expression tree, then post straight away
        z3_cons = self._z3_expr(cpm_expr)
        if is_any_list(z3_cons):
            for z3_con in z3_cons:
                self.z3_solver.add(z3_con)
        else:
            return self.z3_solver.add(z3_cons)

    def _z3_expr(self, cpm_con, reify=False):
        """
            Z3 supports nested expressions,
            so we recursively translate our expressions to theirs.

            Accepts single constraints or a list thereof, return type changes accordingly.

        """
        import z3

        if is_num(cpm_con):
            # translate numpy to python native
            if is_bool(cpm_con):
                return bool(cpm_con)
            return int(cpm_con)

        elif is_any_list(cpm_con):
            return [self._z3_expr(con) for con in cpm_con]
            
        elif isinstance(cpm_con, _NumVarImpl):
            return self.solver_var(cpm_con)

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        elif isinstance(cpm_con, Operator):
            # 'and'/n, 'or'/n, 'xor'/n, '->'/2
            if cpm_con.name == 'and':
                return z3.And(self._z3_expr(cpm_con.args))
            elif cpm_con.name == 'or':
                return z3.Or(self._z3_expr(cpm_con.args))
            elif cpm_con.name == '->':
                return z3.Implies(*self._z3_expr(cpm_con.args, reify=True))

            # 'sum'/n, 'wsum'/2
            elif cpm_con.name == 'sum':
                return z3.Sum(self._z3_expr(cpm_con.args))
            elif cpm_con.name == 'wsum':
                w = cpm_con.args[0]
                x = self._z3_expr(cpm_con.args[1])
                return z3.Sum([wi*xi for wi,xi in zip(w,x)])

            # 'sub'/2, 'mul'/2, 'div'/2, 'pow'/2, 'mod'/2
            elif cpm_con.name == 'sub':
                lhs , rhs = self._z3_expr(cpm_con.args)
                return lhs - rhs
            elif cpm_con.name == "mul":
                assert len(cpm_con.args) == 2, "Currently only support multiplication with 2 vars"
                lhs , rhs = self._z3_expr(cpm_con.args)
                return lhs * rhs
            elif cpm_con.name == "div":
                lhs , rhs = self._z3_expr(cpm_con.args)
                return lhs / rhs
            elif cpm_con.name == "pow":
                lhs , rhs = self._z3_expr(cpm_con.args)
                return lhs ** rhs
            elif cpm_con.name == "mod":
                lhs , rhs = self._z3_expr(cpm_con.args)
                return lhs % rhs

            # '-'/1
            elif cpm_con.name == "-":
                return -self._z3_expr(cpm_con.args[0])

            else:
                raise NotImplementedError(f"Operator {cpm_con} not (yet) implemented for Z3, please report on github if you need it")

        # Comparisons (just translate the subexpressions and re-post)
        elif isinstance(cpm_con, Comparison):

            lhs, rhs = cpm_con.args

            # 'abs'/1
            if isinstance(rhs, Operator) and rhs.name == "abs":
                arg = rhs.args[0]
                return self._z3_expr(Comparison(cpm_con.name, lhs, max([arg, -arg])))
            if isinstance(lhs, Operator) and lhs.name == "abs":
                arg = lhs.args[0]
                return self._z3_expr(Comparison(cpm_con.name, max([arg, -arg]), rhs))

            if isinstance(lhs, GlobalConstraint) and lhs.name == "element":
                arr, idx = lhs.args
                return self._z3_expr(all([(idx == i).implies(Comparison(cpm_con.name, arr[i], rhs)) for i in range(len(arr))]))
            if isinstance(rhs, GlobalConstraint) and rhs.name == "element":
                arr, idx = rhs.args
                return self._z3_expr(all([(idx == i).implies(Comparison(cpm_con.name, lhs, arr[i])) for i in range(len(arr))]))

            if cpm_con.name == "==":
                if isinstance(lhs, GlobalConstraint) and lhs.name == "max":
                    if reify:
                        raise NotImplementedError(f"Reification of {cpm_con} not supported yet")
                    return z3.And(self._z3_expr(any(a == rhs for a in lhs.args)),
                                  self._z3_expr(all([a <= rhs for a in lhs.args])))
                if isinstance(rhs, GlobalConstraint) and rhs.name == "max":
                    if reify:
                        raise NotImplementedError(f"Reification of {cpm_con} not supported yet")
                    return z3.And(self._z3_expr(any(lhs == a for a in rhs.args)),
                                  self._z3_expr(all([lhs >= a for a in rhs.args])))
                if isinstance(lhs, GlobalConstraint) and lhs.name == "min":
                    if reify:
                        raise NotImplementedError(f"Reification of {cpm_con} not supported yet")
                    return z3.And(self._z3_expr(any(a == rhs for a in lhs.args)),
                                  self._z3_expr(all([a >= rhs for a in lhs.args])))
                if isinstance(rhs, GlobalConstraint) and rhs.name == "min":
                    if reify:
                        raise NotImplementedError(f"Reification of {cpm_con} not supported yet")
                    return z3.And(self._z3_expr(any(lhs == a for a in rhs.args)),
                                  self._z3_expr(all([lhs <= a for a in rhs.args])))

                lhs, rhs = self._z3_expr(cpm_con.args)
                return (lhs == rhs)


            if isinstance(lhs, GlobalConstraint) and lhs.name in ("min", "max"):
                new_var, cons = get_or_make_var(lhs)
                return z3.And(self._z3_expr(all(cons)), self._z3_expr(Comparison(cpm_con.name, new_var, rhs)))
            if isinstance(rhs, GlobalConstraint) and rhs.name in ("min", "max"):
                new_var, cons = get_or_make_var(rhs)
                return z3.And(self._z3_expr(all(cons)), self._z3_expr(Comparison(cpm_con.name, lhs, new_var)))

            # other comparisons
            lhs, rhs = self._z3_expr(cpm_con.args)
            # post the comparison
            if cpm_con.name == '<=':
                return (lhs <= rhs)
            elif cpm_con.name == '<':
                return (lhs < rhs)
            elif cpm_con.name == '>=':
                return (lhs >= rhs)
            elif cpm_con.name == '>':
                return (lhs > rhs)
            elif cpm_con.name == '!=':
                return (lhs != rhs)


        # TODO:
        # table

        # rest: base (Boolean) global constraints
        elif cpm_con.name == 'xor':
            z3_args = self._z3_expr(cpm_con.args)
            z3_cons = z3.Xor(z3_args[0], z3_args[1])
            for a in z3_args[2:]:
                z3_cons = z3.Xor(z3_cons, a)
            return z3_cons
        elif cpm_con.name == 'alldifferent':
            return z3.Distinct(self._z3_expr(cpm_con.args))

        # global constraints
        return self._z3_expr(all(cpm_con.decompose()))

        raise NotImplementedError("Z3: constraint not (yet) supported", cpm_con)

    # Other functions from SolverInterface that you can overwrite:
    # solveAll, solution_hint, get_core

    def get_core(self):
        """
            For use with s.solve(assumptions=[...]). Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

            CPMpy will return only those variables that are False (in the UNSAT core)

            Note that there is no guarantee that the core is minimal, though this interface does upon up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!
        """
        assert (self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE), "Can only extract core form UNSAT model"
        assert (len(self.assumption_dict) > 0), "Assumptions must be set using s.solve(assumptions=[...])"

        return [self.assumption_dict[z3_var] for z3_var in self.z3_solver.unsat_core()]