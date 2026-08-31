#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## cvc5.py
##
"""
    Interface to CVC5's Python API.

    cvc5 is an open-source automatic theorem prover for Satisfiability Modulo Theories (SMT) problems that
    supports a large number of theories and their combination. It is the successor of CVC4 and is intended
    to be an open and extensible SMT engine. cvc5 is a joint project led by Stanford University and the
    University of Iowa.(see https://cvc5.github.io/)

    This implementation makes use of cvc5's "pythonic" API, closely replicating the Z3 API.

    Always use :func:`cp.SolverLookup.get("cvc5") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'cvc5' python package is installed:

    .. code-block:: console

        $ pip install cvc5

    See detailed installation instructions at:
    https://cvc5.github.io/docs/latest/api/python/python.html

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_cvc5

    ==============
    Module details
    ==============
"""
from typing import Optional, Iterable

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal, NestedBoolExprLike
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint
from ..expressions.globalfunctions import GlobalFunction
from ..expressions.variables import _BoolVarImpl, NegBoolView, _NumVarImpl
from ..expressions.utils import is_num, is_any_list, is_bool, is_int, is_boolexpr, eval_comparison
from ..transformations.get_variables import get_variables
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.normalize import toplevel_list
from ..transformations.safening import no_partial_functions


class CPM_cvc5(SolverInterface):
    """
    Interface to cvc5's Python API.

    Creates the following attributes (see parent constructor for more):

    - ``cvc5_solver``: object, cvc5's Solver() object

    The :class:`~cpmpy.expressions.globalconstraints.DirectConstraint`, when used, calls a function in the `cvc5` namespace and ``cvc5_solver.add()``'s the result.

    Documentation of the solver's own Python API:
    https://cvc5.github.io/docs/latest/api/python/pythonic/pythonic.html

    .. note::
        Terminology note: a 'model' for cvc5 is a solution!
    """

    supported_global_constraints = frozenset({"alldifferent", "xor", "ite", "div", "mul", "mod", "pow"})
    supported_reified_global_constraints = supported_global_constraints

    @staticmethod
    def supported():
        # try to import the package
        try:
            import cvc5
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
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version('cvc5')
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None, unsat_cores: bool = False):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: Model(), a CPMpy Model() (optional)
            subsolver: None
            unsat_cores (bool): enable CVC5 unsat-core tracking so :func:`get_core() <get_core>` works
                                after :func:`solve(assumptions=...) <solve>`. Off by default: this option
                                must be set before the first constraint is posted, and it enables proof
                                production which can slow the solver down.
        """
        if not self.supported():
            raise ModuleNotFoundError("CPM_cvc5: Install the python package 'cpmpy[cvc5]' to use this solver interface.")

        import cvc5.pythonic as cvc5

        assert subsolver is None

        if cpm_model and cpm_model.objective_ is not None:
            raise NotSupportedError("CPM_cvc5: only satisfaction, does not support an objective function")

        # initialise the native solver object
        self.cvc5_solver = cvc5.Solver()
        self._unsat_cores = unsat_cores
        if unsat_cores:
            # must be set before the first constraint is posted
            self.cvc5_solver.set("produce-unsat-cores", "true")

        # initialise everything else and post the constraints
        super().__init__(name="cvc5", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.cvc5_solver


    def solve(self, time_limit:Optional[float]=None, assumptions:Optional[Iterable[_BoolVarImpl]]=None, **kwargs):
        """
            Call the cvc5 solver

            Arguments:
                time_limit (float, optional):       maximum solve time in seconds
                assumptions:                        iterable (e.g. list, set, tuple) of CPMpy Boolean variables (or their negation) that are assumed to be true.
                                                    For repeated solving, and/or for use with :func:`s.get_core() <get_core()>`: if the model is UNSAT,
                                                    get_core() returns a small subset of assumption variables that are unsat together.
                **kwargs:                           any keyword argument, sets parameters of solver object

            An overview of the cvc5 solver parameters can found at
            https://cvc5.github.io/docs/latest/options.html

            You can use any of these parameters as keyword argument to `solve()` and they will
            be forwarded to the solver. Examples include:

            =============================   ============
            Argument                        Description
            =============================   ============
            ``rlimit-per``                    set resource limit
            ``random-seed``                   random seed
            ``compute-partitions``            number of parallel workers (default=0)
            =============================   ============
        """

        import cvc5.pythonic as cvc5

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            # cvc5 expects milliseconds in int
            self.cvc5_solver.set(**{"tlimit-per": int(time_limit * 1000)})

        if assumptions is not None:
            assumptions = list(assumptions)  # iterable to ordered list
            cvc5_assum_vars = self.solver_vars(assumptions)
            self.assumption_dict = {cvc5_var: cpm_var for (cpm_var, cvc5_var) in zip(assumptions, cvc5_assum_vars)}
        else:
            cvc5_assum_vars = []

        # call the solver, with parameters
        for (key, value) in kwargs.items():
            self.cvc5_solver.setOption(key, value)

        # check assumption variables
        my_status = repr(self.cvc5_solver.check(*cvc5_assum_vars))

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        try:
            st = self.cvc5_solver.statistics()
            self.cpm_status.runtime = float(st['global::totalTime']["value"][:-2]) / 1000
        except Exception:
            self.cpm_status.runtime = 0

        # translate exit status
        if my_status == "sat":
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status == "unsat":
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status == "unknown":
            try:
                model = self.cvc5_solver.model()
                if model: # a solution was found, just not the optimal one (or not proven)
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
                # can happen when timeout is reached...
                else:
                    self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            # can happen when timeout is reached...
            except cvc5.Exception: # no model has been initialized, not even an empty one
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:  # another?
            raise NotImplementedError(my_status)  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            sol = self.cvc5_solver.model() # the solution (called model in cvc5)
            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                if cpm_var.is_bool():
                    cpm_var._value = bool(sol[sol_var])
                else:
                    cpm_var._value = sol[sol_var].as_long()
        else:  # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var.clear()

        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
            or returns a constant if the variable is a constant
        """
        import cvc5.pythonic as cvc5

        if isinstance(cpm_var, _NumVarImpl):

            name = cpm_var.name
            revar = self._varmap.get(name)
            if revar is not None:
                return revar

            # not yet created, make a new solver var
            # cvc5 requires a native str (numpy.str_ from named arrays is rejected)
            cvc5_name = str(name)
            if cpm_var.is_bool():
                if isinstance(cpm_var, NegBoolView):
                    revar = cvc5.Not(self.solver_var(cpm_var._bv))
                else:
                    revar = cvc5.Bool(cvc5_name)
            else:
                revar = cvc5.Int(cvc5_name)
                # set bounds
                self.cvc5_solver.add(revar >= cpm_var.lb)
                self.cvc5_solver.add(revar <= cpm_var.ub)
            self._varmap[name] = revar
            return revar

        if is_int(cpm_var):  # shortcut, eases posting constraints
            return cpm_var

        raise NotImplementedError("Not a known var {}".format(cpm_var))


    def objective(self, expr, minimize=True):
        """
            CVC5 only supports satisfaction problems.
        """
        raise NotSupportedError("CVC5 only supports satisfaction problems.")

    def transform(self, cpm_expr: NestedBoolExprLike) -> list[Expression]:
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the :ref:`Adding a new solver` docs on readthedocs for more information.

            Arguments:
                cpm_expr (NestedBoolExprLike): CPMpy expression, or list thereof

            Returns:
                list[Expression]: transformed constraints
        """

        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel=frozenset({"div", "mod"}))
        cpm_cons = decompose_in_tree(cpm_cons,
                                     supported=self.supported_global_constraints,
                                     supported_reified=self.supported_reified_global_constraints,
                                     csemap=self._csemap)
        return cpm_cons

    def add(self, cpm_expr: NestedBoolExprLike) -> "CPM_cvc5":
        """
            CVC5 supports nested expressions so translate expression tree and post to solver API directly

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

            Arguments:
                cpm_expr (NestedBoolExprLike): CPMpy expression, or list thereof

            Returns:
                self
        """
        # all variables are user variables, handled in `solver_var()`
        # unless their constraint gets simplified away, so lets collect them anyway
        get_variables(cpm_expr, collect=self.user_vars)

        # transform and post the constraints
        for cpm_con in self.transform(cpm_expr):
            # translate each expression tree, then post straight away
            cvc5_con = self._cvc5_expr(cpm_con)
            self.cvc5_solver.add(cvc5_con)

        return self
    __add__ = add  # avoid redirect in superclass

    def _as_int(self, cvc5, expr):
        """Upcast a cvc5 Boolean expression to an integer 0/1 term."""
        if isinstance(expr, cvc5.BoolRef):
            return cvc5.If(expr, 1, 0)
        return expr

    def _cvc5_expr(self, cpm_con):
        """
            CVC5 supports nested expressions,
            so we recursively translate our expressions to theirs.

            Accepts single constraints or a list thereof, return type changes accordingly.

        """
        import cvc5.pythonic as cvc5

        if is_num(cpm_con):
            # translate numpy to python native
            if is_bool(cpm_con):
                return bool(cpm_con)
            elif is_int(cpm_con):
                return cvc5.IntVal(int(cpm_con))
            # cvc5 truncates Python floats to integers in Int comparisons;
            # post non-integral constants as reals so thresholds like x >= 0.1 stay exact
            return cvc5.RealVal(str(float(cpm_con)))

        elif is_any_list(cpm_con):
            # arguments can be lists
            return [self._cvc5_expr(con) for con in cpm_con]

        elif isinstance(cpm_con, BoolVal):
            return cpm_con.args[0]

        elif isinstance(cpm_con, _NumVarImpl):
            return self.solver_var(cpm_con)

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        elif isinstance(cpm_con, Operator):
            arity, _ = Operator.allowed[cpm_con.name]
            # 'and'/n, 'or'/n, '->'/2
            # cvc5 requires And/Or to have at least 2 children (unlike Z3)
            if cpm_con.name == 'and':
                args = self._cvc5_expr(cpm_con.args)
                return args[0] if len(args) == 1 else cvc5.And(args)
            elif cpm_con.name == 'or':
                args = self._cvc5_expr(cpm_con.args)
                return args[0] if len(args) == 1 else cvc5.Or(args)
            elif cpm_con.name == '->':
                return cvc5.Implies(*self._cvc5_expr(cpm_con.args))
            elif cpm_con.name == 'not':
                return cvc5.Not(self._cvc5_expr(cpm_con.args[0]))

            # 'sum'/n, 'wsum'/2
            elif cpm_con.name == 'sum':
                return cvc5.Sum([self._as_int(cvc5, a) for a in self._cvc5_expr(cpm_con.args)])
            elif cpm_con.name == 'wsum':
                w = cpm_con.args[0]
                x = [self._as_int(cvc5, a) for a in self._cvc5_expr(cpm_con.args[1])]
                return cvc5.Sum([wi * xi for wi, xi in zip(w, x)])

            # 'sub'/2
            elif cpm_con.name == 'sub':
                x, y = self._cvc5_expr(cpm_con.args)
                return self._as_int(cvc5, x) - self._as_int(cvc5, y)

            # '-'/1
            elif cpm_con.name == "-":
                if is_boolexpr(cpm_con.args[0]):
                    return -cvc5.If(self._cvc5_expr(cpm_con.args[0]), 1, 0)
                return -self._cvc5_expr(cpm_con.args[0])

            else:
                raise NotImplementedError(f"Operator {cpm_con} not (yet) implemented for CVC5, "
                                          f"please report on github if you need it")

        # Comparisons (just translate the subexpressions and re-post)
        elif isinstance(cpm_con, Comparison):
            lhs, rhs = cpm_con.args

            lhs_bexpr = is_boolexpr(lhs)
            rhs_bexpr = is_boolexpr(rhs)

            lhs, rhs = self._cvc5_expr(cpm_con.args)

            if cpm_con.name == "==" or cpm_con.name == "!=":
                # cvc5 supports bool <-> bool comparison but not bool <-> arith
                if lhs_bexpr and not rhs_bexpr:
                    # upcast lhs to integer
                    lhs = cvc5.If(lhs, 1, 0)
                elif rhs_bexpr and not lhs_bexpr:
                    # upcast rhs to integer
                    rhs = cvc5.If(rhs, 1, 0)
            else:
                # other comparisons are not supported on boolexpr
                if lhs_bexpr:  # upcast lhs
                    lhs = cvc5.If(lhs, 1, 0)
                if rhs_bexpr:  # upcast rhs
                    rhs = cvc5.If(rhs, 1, 0)

            # post the comparison
            return eval_comparison(cpm_con.name, lhs, rhs)

        elif isinstance(cpm_con, GlobalFunction):
            if cpm_con.name == "mod":
                # mimic modulo with integer division (round towards 0)
                x, y = self._cvc5_expr(cpm_con.args)
                return cvc5.If(cvc5.And(x >= 0), x % y, -(-x % y))

            elif cpm_con.name == "mul":
                x, y = self._cvc5_expr(cpm_con.args)
                return self._as_int(cvc5, x) * self._as_int(cvc5, y)

            elif cpm_con.name == "div":
                # cvc5 rounds towards negative infinity, need this hack when result is negative
                x, y = self._cvc5_expr(cpm_con.args)
                return cvc5.If(cvc5.And(x >= 0, y >= 0), x / y,
                       cvc5.If(cvc5.And(x <= 0, y <= 0), -x / -y,
                       cvc5.If(cvc5.And(x >= 0, y <= 0), -(x / -y),
                       cvc5.If(cvc5.And(x <= 0, y >= 0), -(-x / y), 0))))

            elif cpm_con.name == "pow":
                x, y = self._cvc5_expr(cpm_con.args)
                if not is_num(cpm_con.args[1]):
                    # tricky in cvc5 not all power constraints are decidable
                    # solver will return 'unknown', even if theory is satisfiable.
                    # raise error to be consistent with other solvers
                    raise NotSupportedError(f"CVC5 only supports power constraint with constant exponent, got {cpm_con}")
                return x ** y

            raise NotImplementedError(f"Global function {cpm_con} not (yet) implemented for CVC5")

        # rest: base (Boolean) global constraints
        elif isinstance(cpm_con, GlobalConstraint):
            if cpm_con.name == 'alldifferent':
                if len(cpm_con.args) > 1:
                    return cvc5.Distinct(self._cvc5_expr(cpm_con.args))
                else:
                    return True
            elif cpm_con.name == 'xor':
                cvc5_args = self._cvc5_expr(cpm_con.args)
                if len(cvc5_args) == 1:  # just the arg
                    return cvc5_args[0]
                cvc5_cons = cvc5.Xor(cvc5_args[0], cvc5_args[1])
                for a in cvc5_args[2:]:
                    cvc5_cons = cvc5.Xor(cvc5_cons, a)
                return cvc5_cons
            elif cpm_con.name == 'ite':
                return cvc5.If(self._cvc5_expr(cpm_con.args[0]), self._cvc5_expr(cpm_con.args[1]),
                             self._cvc5_expr(cpm_con.args[2]))

            raise ValueError(f"Global constraint {cpm_con} should be decomposed already, please report on github.")

        # a direct constraint, make with cvc5 (will be posted to it by calling function)
        elif isinstance(cpm_con, DirectConstraint):
            return cpm_con.callSolver(self, cvc5)

        raise NotImplementedError("CVC5: constraint not (yet) supported", cpm_con)

    def get_core(self):
        """
            For use with :func:`s.solve(assumptions=[...]) <solve()>`. Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

            Requires constructing the solver with ``unsat_cores=True``, e.g.
            ``cp.SolverLookup.get("cvc5", unsat_cores=True)``.

            CPMpy will return only those variables that are False (in the UNSAT core)

            Note that there is no guarantee that the core is minimal, though this interface does upon up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!
        """
        if not self._unsat_cores:
            raise NotSupportedError("CPM_cvc5: get_core() requires unsat_cores=True in the constructor "
                                    "(e.g. cp.SolverLookup.get('cvc5', unsat_cores=True))")
        assert (self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE), "Can only extract core form UNSAT model"
        assert (len(self.assumption_dict) > 0), "Assumptions must be set using s.solve(assumptions=[...])"

        return [self.assumption_dict[cvc5_var] for cvc5_var in self.cvc5_solver.unsat_core()
                if cvc5_var in self.assumption_dict]
