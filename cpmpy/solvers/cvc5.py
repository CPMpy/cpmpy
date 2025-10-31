#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## cvc5.py
##
"""
    Interface to CVC5's Python API.

    cvc5 is an open-source automatic theorem prover for Satisfiability Modulo Theories (SMT) problems that
    supports a large number of theories and their combination. It is the successor of CVC4 and is intended 
    to be an open and extensible SMT engine. (see https://cvc5.github.io/)

    This implemantation makes use of cvc5's "pythonic" API, closely replicating the Z3 API.

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


class CPM_cvc5(SolverInterface):
    """
    Interface to cvc5's Python API.

    Creates the following attributes (see parent constructor for more):
        
    - ``cvc5_solver``: object, cvc5's Solver() object

    The :class:`~cpmpy.expressions.globalconstraints.DirectConstraint`, when used, calls a function in the `cvc5` namespace and ``cvc5_solver.add()``'s the result.

    Documentation of the solver's own Python API:
    https://cvc5.github.io/api/python/pythonic/cvc5.html

    .. note::
        Terminology note: a 'model' for cvc5 is a solution!
    """

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
        try:
            return pkg_resources.get_distribution('cvc5').version
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
            raise Exception("CPM_cvc5: Install the python package 'cvc5' to use this solver interface.")

        import cvc5.pythonic as cvc5


        # initialise the native solver object
        self.cvc5_solver = cvc5.Solver()
   
        # initialise everything else and post the constraints
        super().__init__(name="cvc5", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.cvc5_solver


    def solve(self, time_limit=None, assumptions=[], **kwargs):
        """
            Call the cvc5 solver

            Arguments:
                time_limit (float, optional):       maximum solve time in seconds
                assumptions:                        list of CPMpy Boolean variables (or their negation) that are assumed to be true.
                                                    For repeated solving, and/or for use with :func:`s.get_core() <get_core()>`: if the model is UNSAT,
                                                    get_core() returns a small subset of assumption variables that are unsat together.
                **kwargs:                           any keyword argument, sets parameters of solver object

            An overview of the cvc5 solver parameters can found at 
            https://cvc5.github.io/docs/cvc5-1.0.8/options.html

            You can use any of these parameters as keyword argument to `solve()` and they will
            be forwarded to the solver. Examples include:

            =============================   ============
            Argument                        Description
            =============================   ============
            `` rlimit-per``                   set resource limit
            ``random_seed``                   random seed
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
            self.cvc5_solver.set(**{"tlimit-per":int(time_limit*1000)})

        # process assumption variables
        cvc5_assum_vars = self.solver_vars(assumptions)
        self.assumption_dict = {cvc5_var : cpm_var for (cpm_var, cvc5_var) in zip(assumptions, cvc5_assum_vars)}

        # call the solver, with parameters
        for (key,value) in kwargs.items():
            self.cvc5_solver.setOption(key, value)

        # check assumption variables
        my_status = repr(self.cvc5_solver.check(*cvc5_assum_vars))

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        st = self.cvc5_solver.statistics()
        self.cpm_status.runtime = float(st['global::totalTime']["value"][:-2])/1000

        # translate exit status
        if my_status == "sat":
            if self.has_objective(): # COP
                # check if optimal solution found and proven, i.e. bounds are equal
                lower_bound = self.cvc5_solver.lower(self.obj_handle)
                upper_bound = self.cvc5_solver.upper(self.obj_handle)
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
                model = self.cvc5_solver.model()
                if model: # a solution was found, just not the optimal one (or not proven)
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
                # can happen when timeout is reached...
                else:
                    self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            # can happen when timeout is reached...
            except cvc5.Exception as e: # no model has been initialized, not even an empty one
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
                if isinstance(cpm_var, _BoolVarImpl):
                    cpm_var._value = bool(sol[sol_var])
                elif isinstance(cpm_var, _NumVarImpl):
                    cpm_var._value = sol[sol_var].as_long()

            # translate objective, for optimisation problems only
            if self.has_objective():
                obj = self.cvc5_solver.objectives()[0]
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
        import cvc5.pythonic as cvc5

        if is_num(cpm_var): # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return cvc5.Not(self.solver_var(cpm_var._bv))

        # create if it does not exit
        if cpm_var not in self._varmap:
            # we assume al variables are user variables (because nested expressions)
            self.user_vars.add(cpm_var)
            if isinstance(cpm_var, _BoolVarImpl):
                revar = cvc5.Bool(str(str(cpm_var))) # TODO: first str call shouldn't return a np.str_
            elif isinstance(cpm_var, _IntVarImpl):
                revar = cvc5.Int(str(str(cpm_var)))
                # set bounds
                self.cvc5_solver.add(revar >= cpm_var.lb)
                self.cvc5_solver.add(revar <= cpm_var.ub)
            else:
                raise NotImplementedError("Not a know var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        return self._varmap[cpm_var]

    def objective(self, expr, minimize=True):
        raise NotSupportedError("CVC5 only supports satisfaction problems.")

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
        supported = {"alldifferent", "xor", "ite"}  # TODO: check which else
        cpm_cons = decompose_in_tree(cpm_cons, supported, supported, csemap=self._csemap)
        return cpm_cons

    def add(self, cpm_expr):
        """
            CVC5 supports nested expressions so translate expression tree and post to solver API directly

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        Arguments:
            cpm_expr (Expression or list of Expression): CPMpy expression, or list thereof
        
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
            return float(cpm_con)

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
            if cpm_con.name == 'and':
                return cvc5.And(self._cvc5_expr(cpm_con.args))
            elif cpm_con.name == 'or':
                return cvc5.Or(self._cvc5_expr(cpm_con.args))
            elif cpm_con.name == '->':
                return cvc5.Implies(*self._cvc5_expr(cpm_con.args))
            elif cpm_con.name == 'not':
                return cvc5.Not(self._cvc5_expr(cpm_con.args[0]))

            # 'sum'/n, 'wsum'/2
            elif cpm_con.name == 'sum':
                # Check for boolean variables and replace with auxiliary integer variables
                cvc5_args = []
                for arg in cpm_con.args:
                    cvc5_arg = self._cvc5_expr(arg)
                    # Check if this is a boolean variable that needs to be converted
                    if isinstance(cvc5_arg, cvc5.BoolRef):
                        # Create auxiliary integer variable with domain [0, 1]
                        aux_var_name = f"__bool_aux_{arg}"
                        aux_int_var = cvc5.Int(aux_var_name)
                        # Add bounds constraint
                        self.cvc5_solver.add(aux_int_var >= 0)
                        self.cvc5_solver.add(aux_int_var <= 1)
                        # Add channeling constraint: bool_var ↔ (aux_int_var == 1)
                        # This is equivalent to: (bool_var → aux_int_var == 1) ∧ (~bool_var → aux_int_var == 0)
                        self.cvc5_solver.add(cvc5.Implies(cvc5_arg, aux_int_var == 1))
                        self.cvc5_solver.add(cvc5.Implies(cvc5.Not(cvc5_arg), aux_int_var == 0))
                        cvc5_args.append(aux_int_var)
                    else:
                        cvc5_args.append(cvc5_arg)
                return cvc5.Sum(cvc5_args)
            elif cpm_con.name == 'wsum':
                w = cpm_con.args[0]
                # Check for boolean variables in the second argument and replace with auxiliary integer variables
                cvc5_args = []
                for arg in cpm_con.args[1]:
                    cvc5_arg = self._cvc5_expr(arg)
                    # Check if this is a boolean variable that needs to be converted
                    if isinstance(cvc5_arg, cvc5.BoolRef):
                        # Create auxiliary integer variable with domain [0, 1]
                        aux_var_name = f"__bool_aux_{arg}"
                        aux_int_var = cvc5.Int(aux_var_name)
                        # Add bounds constraint
                        self.cvc5_solver.add(aux_int_var >= 0)
                        self.cvc5_solver.add(aux_int_var <= 1)
                        # Add channeling constraint: bool_var ↔ (aux_int_var == 1)
                        # This is equivalent to: (bool_var → aux_int_var == 1) ∧ (~bool_var → aux_int_var == 0)
                        self.cvc5_solver.add(cvc5.Implies(cvc5_arg, aux_int_var == 1))
                        self.cvc5_solver.add(cvc5.Implies(cvc5.Not(cvc5_arg), aux_int_var == 0))
                        cvc5_args.append(aux_int_var)
                    else:
                        cvc5_args.append(cvc5_arg)
                return cvc5.Sum([wi*xi for wi,xi in zip(w,cvc5_args)])

            # 'sub'/2, 'mul'/2, 'div'/2, 'pow'/2, 'm2od'/2
            elif arity == 2 or cpm_con.name == "mul":
                assert len(cpm_con.args) == 2, "Currently only support multiplication with 2 vars"
                x, y = self._cvc5_expr(cpm_con.args)
                if isinstance(x, cvc5.BoolRef):
                    x = cvc5.If(x, 1, 0)
                if isinstance(y, cvc5.BoolRef):
                    y = cvc5.If(y, 1, 0)

                if cpm_con.name == 'sub':
                    return x - y
                elif cpm_con.name == "mul":
                    return x * y
                elif cpm_con.name == "div":
                    # cvc5 rounds towards negative infinity, need this hack when result is negative
                    return cvc5.If (cvc5.And(x >= 0, y >= 0), x / y,
                         cvc5.If (cvc5.And(x <= 0, y <= 0), -x / -y,
                         cvc5.If (cvc5.And(x >= 0, y <= 0), -(x / -y),
                         cvc5.If (cvc5.And(x <= 0, y >= 0), -(-x / y), 0))))

                elif cpm_con.name == "pow":
                    if not is_num(cpm_con.args[1]):
                        # tricky in cvc5 not all power constraints are decidable
                        # solver will return 'unknown', even if theory is satisfiable.
                        # https://stackoverflow.com/questions/70289335/power-and-logarithm-in-z3
                        # raise error to be consistent with other solvers
                        raise NotSupportedError(f"CVC5 only supports power constraint with constant exponent, got {cpm_con}")
                    return x ** y
                elif cpm_con.name == "mod":
                    # minimic modulo with integer division (round towards o)
                    return cvc5.If (cvc5.And(x >= 0), x % y,-(-x % y))

            # '-'/1
            elif cpm_con.name == "-":
                if is_boolexpr(cpm_con.args[0]):
                    return  cvc5.If(self._cvc5_expr(cpm_con.args[0]), 1, 0)
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
                    # upcase rhs to integer
                    rhs = cvc5.If(rhs, 1, 0)
            else:
                # other comparisons are not supported on boolexpr
                if lhs_bexpr: # upcast lhs
                    lhs = cvc5.If(lhs, 1, 0)
                if rhs_bexpr: # upcase rhs
                    rhs = cvc5.If(rhs, 1, 0)

            # post the comparison
            return eval_comparison(cpm_con.name, lhs, rhs)

        # rest: base (Boolean) global constraints
        elif isinstance(cpm_con, GlobalConstraint):
            # TODO:

            if cpm_con.name == 'alldifferent':
                if len(cpm_con.args) > 1:
                    return cvc5.Distinct(self._cvc5_expr(cpm_con.args))
                else:
                    return True
            elif cpm_con.name == 'xor':
                cvc5_args = self._cvc5_expr(cpm_con.args)
                if len(cvc5_args) == 1: # just the arg
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

    # CVC5 currently to does not provide access to unsat cores through the "pythonic" Python API
    # def get_core(self):
    #     """
    #         For use with :func:`s.solve(assumptions=[...]) <solve()>`. Only meaningful if the solver returned UNSAT. In that case, get_core() returns a small subset of assumption variables that are unsat together.

    #         CPMpy will return only those variables that are False (in the UNSAT core)

    #         Note that there is no guarantee that the core is minimal, though this interface does upon up the possibility to add more advanced Minimal Unsatisfiabile Subset algorithms on top. All contributions welcome!
    #     """
    #     assert (self.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE), "Can only extract core form UNSAT model"
    #     assert (len(self.assumption_dict) > 0), "Assumptions must be set using s.solve(assumptions=[...])"

    #     return [self.assumption_dict[cvc5_var] for cvc5_var in self.cvc5_solver.unsat_core()]



