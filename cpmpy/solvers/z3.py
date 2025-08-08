#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## z3.py
##
"""
    Interface to Z3's Python API.

    Z3 is a highly versatile and effective theorem prover from Microsoft.
    Underneath, it is an SMT solver with a wide scala of theory solvers.
    We will interface to the finite-domain integer related parts of the API.
    (see https://github.com/Z3Prover/z3)

    .. warning::
        For incrementally solving an optimisation function, instantiate the solver object
        with a model that has an objective function, e.g. ``s = cp.SolverLookup.get("z3", Model(maximize=1))``.

    Always use :func:`cp.SolverLookup.get("z3") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'z3-solver' python package is installed:

    .. code-block:: console
    
        $ pip install z3-solver

    See detailed installation instructions at:
    https://github.com/Z3Prover/z3#python

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_z3

    ==============
    Module details
    ==============
"""
from typing import Optional
import pkg_resources

from cpmpy.transformations.get_variables import get_variables
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.variables import _BoolVarImpl, NegBoolView, _NumVarImpl, _IntVarImpl, intvar
from ..expressions.utils import is_num, is_any_list, is_bool, is_int, is_boolexpr, eval_comparison
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.normalize import toplevel_list
from ..transformations.safening import no_partial_functions


class CPM_z3(SolverInterface):
    """
    Interface to Z3's Python API.

    Creates the following attributes (see parent constructor for more):
        
    - ``z3_solver``: object, z3's Solver() object

    The :class:`~cpmpy.expressions.globalconstraints.DirectConstraint`, when used, calls a function in the `z3` namespace and ``z3_solver.add()``'s the result.

    Documentation of the solver's own Python API:
    https://z3prover.github.io/api/html/namespacez3py.html

    .. note::
        Terminology note: a 'model' for z3 is a solution!
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import z3
            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e
        
    @classmethod
    def version(cls) -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        try:
            return pkg_resources.get_distribution('z3-solver').version
        except pkg_resources.DistributionNotFound:
            return None
        
    def __init__(self, cpm_model=None, subsolver="sat"):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: None
        """
        if not self.supported():
            raise Exception("CPM_z3: Install the python package 'z3-solver' to use this solver interface.")

        import z3

        if subsolver is None:
            if cpm_model and cpm_model.has_objective():
                subsolver = "opt"
            else:
                subsolver = "sat"
        assert "sat" in subsolver or "opt" in subsolver, "Z3 only has a satisfaction or optimization sub-solver."

        # initialise the native solver object
        if "sat" in subsolver:
            self.z3_solver = z3.Solver()
        if "opt" in subsolver:
            self.z3_solver = z3.Optimize()

        # handle of objective (as returned by solver)
        self.obj_handle = None

        # initialise everything else and post the constraints/objective
        super().__init__(name="z3", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.z3_solver


    def solve(self, time_limit=None, assumptions=[], **kwargs):
        """
            Call the z3 solver

            Arguments:
                time_limit (float, optional):       maximum solve time in seconds
                assumptions:                        list of CPMpy Boolean variables (or their negation) that are assumed to be true.
                                                    For repeated solving, and/or for use with :func:`s.get_core() <get_core()>`: if the model is UNSAT,
                                                    get_core() returns a small subset of assumption variables that are unsat together.
                **kwargs:                           any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:

            - ... (no common examples yet)

            The full list doesn't seem to be documented online, you have to run its help() function:
            
            .. code-block:: python

                import z3
                z3.Solver().help()

            .. warning::
                Warning! Some parameternames in z3 have a '.' in their name,
                such as (arbitrarily chosen): ``sat.lookahead_simplify``
                You have to construct a dictionary of keyword arguments upfront:
                
                .. code-block:: python

                    params = {"sat.lookahead_simplify": True}
                    s.solve(**params)
        """
        import z3

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
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
            if self.has_objective(): # COP
                # check if optimal solution found and proven, i.e. bounds are equal
                lower_bound = self.z3_solver.lower(self.obj_handle)
                upper_bound = self.z3_solver.upper(self.obj_handle)
                if lower_bound == upper_bound: # found optimal
                    self.cpm_status.exitstatus = ExitStatus.OPTIMAL
                else: # suboptimal / not proven
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            else: # CSP
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status == "unsat":
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status == "unknown":
            try:
                model = self.z3_solver.model()
                if model: # a solution was found, just not the optimal one (or not proven)
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
                # can happen when timeout is reached...
                else:
                    self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            # can happen when timeout is reached...
            except z3.Z3Exception as e: # no model has been initialized, not even an empty one
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
            if self.has_objective():
                obj = self.z3_solver.objectives()[0]
                self.objective_value_ = sol.evaluate(obj).as_long()

        else:  # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None

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


    def has_objective(self):
        import z3
        return isinstance(self.z3_solver, z3.Optimize) and len(self.z3_solver.objectives()) != 0

    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            ``objective()`` can be called multiple times, only the last one is stored

            .. note::
                technical side note: any constraints created during conversion of the objective
                are premanently posted to the solver
        """
        import z3
        # objective can be a nested expression for z3
        if not isinstance(self.z3_solver, z3.Optimize):
            raise NotSupportedError("Use the z3 optimizer for optimization problems")

        if isinstance(expr, GlobalFunction): # not supported by Z3
            obj_var = intvar(*expr.get_bounds())
            self += expr == obj_var
            expr = obj_var

        obj = self._z3_expr(expr)
        if minimize:
            self.obj_handle = self.z3_solver.minimize(obj)
        else:
            self.obj_handle = self.z3_solver.maximize(obj)


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
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"div", "mod"})
        supported = {"alldifferent", "xor", "ite"}  # z3 accepts these reified too
        cpm_cons = decompose_in_tree(cpm_cons, supported, supported, csemap=self._csemap)
        return cpm_cons

    def add(self, cpm_expr):
        """
            Z3 supports nested expressions so translate expression tree and post to solver API directly

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
        # all variables are user variables, handled in `solver_var()`
        # unless their constraint gets simplified away, so lets collect them anyway
        get_variables(cpm_expr, collect=self.user_vars)

        # transform and post the constraints
        for cpm_con in self.transform(cpm_expr):
            # translate each expression tree, then post straight away
            z3_con = self._z3_expr(cpm_con)
            self.z3_solver.add(z3_con)

        return self
    __add__ = add  # avoid redirect in superclass

    def _z3_expr(self, cpm_con):
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
            elif is_int(cpm_con):
                return z3.IntVal(int(cpm_con))
            return float(cpm_con)

        elif is_any_list(cpm_con):
            # arguments can be lists
            return [self._z3_expr(con) for con in cpm_con]

        elif isinstance(cpm_con, BoolVal):
            return cpm_con.args[0]

        elif isinstance(cpm_con, _NumVarImpl):
            return self.solver_var(cpm_con)

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        elif isinstance(cpm_con, Operator):
            arity, _ = Operator.allowed[cpm_con.name]
            # 'and'/n, 'or'/n, '->'/2
            if cpm_con.name == 'and':
                return z3.And(self._z3_expr(cpm_con.args))
            elif cpm_con.name == 'or':
                return z3.Or(self._z3_expr(cpm_con.args))
            elif cpm_con.name == '->':
                return z3.Implies(*self._z3_expr(cpm_con.args))
            elif cpm_con.name == 'not':
                return z3.Not(self._z3_expr(cpm_con.args[0]))

            # 'sum'/n, 'wsum'/2
            elif cpm_con.name == 'sum':
                return z3.Sum(self._z3_expr(cpm_con.args))
            elif cpm_con.name == 'wsum':
                w = cpm_con.args[0]
                x = self._z3_expr(cpm_con.args[1])
                return z3.Sum([wi*xi for wi,xi in zip(w,x)])

            # 'sub'/2, 'mul'/2, 'div'/2, 'pow'/2, 'm2od'/2
            elif arity == 2 or cpm_con.name == "mul":
                assert len(cpm_con.args) == 2, "Currently only support multiplication with 2 vars"
                x, y = self._z3_expr(cpm_con.args)
                if isinstance(x, z3.BoolRef):
                    x = z3.If(x, 1, 0)
                if isinstance(y, z3.BoolRef):
                    y = z3.If(y, 1, 0)

                if cpm_con.name == 'sub':
                    return x - y
                elif cpm_con.name == "mul":
                    return x * y
                elif cpm_con.name == "div":
                    # z3 rounds towards negative infinity, need this hack when result is negative
                    return z3.If(z3.And(x >= 0, y >= 0), x / y,
                           z3.If(z3.And(x <= 0, y <= 0), -x / -y,
                           z3.If(z3.And(x >= 0, y <= 0), -(x / -y),
                           z3.If(z3.And(x <= 0, y >= 0), -(-x / y), 0))))

                elif cpm_con.name == "pow":
                    if not is_num(cpm_con.args[1]):
                        # tricky in Z3 not all power constraints are decidable
                        # solver will return 'unknown', even if theory is satisfiable.
                        # https://stackoverflow.com/questions/70289335/power-and-logarithm-in-z3
                        # raise error to be consistent with other solvers
                        raise NotSupportedError(f"Z3 only supports power constraint with constant exponent, got {cpm_con}")
                    return x ** y
                elif cpm_con.name == "mod":
                    # minimic modulo with integer division (round towards o)
                    return z3.If(z3.And(x >= 0), x % y,-(-x % y))

            # '-'/1
            elif cpm_con.name == "-":
                if is_boolexpr(cpm_con.args[0]):
                    return -z3.If(self._z3_expr(cpm_con.args[0]), 1, 0)
                return -self._z3_expr(cpm_con.args[0])

            else:
                raise NotImplementedError(f"Operator {cpm_con} not (yet) implemented for Z3, "
                                          f"please report on github if you need it")

        # Comparisons (just translate the subexpressions and re-post)
        elif isinstance(cpm_con, Comparison):
            lhs, rhs = cpm_con.args

            lhs_bexpr = is_boolexpr(lhs)
            rhs_bexpr = is_boolexpr(rhs)

            lhs, rhs = self._z3_expr(cpm_con.args)

            if cpm_con.name == "==" or cpm_con.name == "!=":
                # z3 supports bool <-> bool comparison but not bool <-> arith
                if lhs_bexpr and not rhs_bexpr:
                    # upcast lhs to integer
                    lhs = z3.If(lhs, 1, 0)
                elif rhs_bexpr and not lhs_bexpr:
                    # upcase rhs to integer
                    rhs = z3.If(rhs, 1, 0)
            else:
                # other comparisons are not supported on boolexpr
                if lhs_bexpr: # upcast lhs
                    lhs = z3.If(lhs, 1, 0)
                if rhs_bexpr: # upcase rhs
                    rhs = z3.If(rhs, 1, 0)

            # post the comparison
            return eval_comparison(cpm_con.name, lhs, rhs)

        # rest: base (Boolean) global constraints
        elif isinstance(cpm_con, GlobalConstraint):
            # TODO:
            # table

            if cpm_con.name == 'alldifferent':
                return z3.Distinct(self._z3_expr(cpm_con.args))
            elif cpm_con.name == 'xor':
                z3_args = self._z3_expr(cpm_con.args)
                if len(z3_args) == 1: # just the arg
                    return z3_args[0]
                z3_cons = z3.Xor(z3_args[0], z3_args[1])
                for a in z3_args[2:]:
                    z3_cons = z3.Xor(z3_cons, a)
                return z3_cons
            elif cpm_con.name == 'ite':
                return z3.If(self._z3_expr(cpm_con.args[0]), self._z3_expr(cpm_con.args[1]),
                             self._z3_expr(cpm_con.args[2]))

            raise ValueError(f"Global constraint {cpm_con} should be decomposed already, please report on github.")

        # a direct constraint, make with z3 (will be posted to it by calling function)
        elif isinstance(cpm_con, DirectConstraint):
            return cpm_con.callSolver(self, z3)

        raise NotImplementedError("Z3: constraint not (yet) supported", cpm_con)

    # Other functions from SolverInterface that you can overwrite:
    # solveAll, solution_hint, get_core

    def get_core(self):
        """
            For use with :func:`s.solve(assumptions=[...]) <solve()>`. Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

            CPMpy will return only those variables that are False (in the UNSAT core)

            Note that there is no guarantee that the core is minimal, though this interface does upon up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!
        """
        assert (self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE), "Can only extract core form UNSAT model"
        assert (len(self.assumption_dict) > 0), "Assumptions must be set using s.solve(assumptions=[...])"

        return [self.assumption_dict[z3_var] for z3_var in self.z3_solver.unsat_core()]



