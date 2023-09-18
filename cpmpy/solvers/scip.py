#!/usr/bin/env python
"""
    Interface to the SCIP's python "PySCIPOpt" package

    First install the ScipOptSuite on your machine, follow:
    https://scipopt.org/index.php#download

    Then install the 'pyscipopt' python package:
        $ pip install pyscipopt
    
    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_scip

    ==============
    Module details
    ==============
"""

from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import *
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.globalconstraints import DirectConstraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, only_positive_bv
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_bv_implies, reify_rewrite


class CPM_scip(SolverInterface):
    """
    Interface to SCIP's API

    Requires that the SCIPOptSuite and 'pyscipopt' python package is installed
    See detailed installation instructions at the top of this file.

    Creates the following attributes (see parent constructor for more):
    - scip_model: object, SCIP's Model object

    Detailed documentation on the Model():
    https://scipopt.github.io/PySCIPOpt/docs/html/classpyscipopt_1_1scip_1_1Model.html

    The `DirectConstraint`, when used, calls a function on the `scip_model` object.
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pyscipopt as scip
            return True
        except:
            return False

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: a CPMpy Model()
        """
        if not self.supported():
            raise Exception(
                "CPM_scip: Install SCIPOptSuite and the python package 'pyscipopt'!")
        import pyscipopt as scip

        self.scip_model = scip.Model("From CPMpy")

        # initialise everything else and post the constraints/objective
        # it is sufficient to implement __add__() and minimize/maximize() below
        super().__init__(name="scip", cpm_model=cpm_model)


    def solve(self, time_limit=None, solution_callback=None, **kwargs):
        """
            Call the SCIP solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            Examples of scip supported arguments include:
                - TODO

            For a full list of scip parameters, please visit 
            https://www.scipopt.org/doc/html/PARAMETERS.php
        """
        #import pyscipopt as scip

        if time_limit is not None:
            # in seconds
            self.scip_model.setParam("limits/time", time_limit)

        if solution_callback is not None:
            raise Exception("SCIP: solution callback not (yet?) implemented")

        # call the solver, with parameters
        #for param, val in kwargs.items():
        #    self.scip_model.setParam(param, val)
        self.scip_model.setParams(kwargs)

        _ = self.scip_model.optimize()
        scip_objective = self.scip_model.getObjective()

        scip_status = self.scip_model.getStatus()

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.scip_model.getSolvingTime()

        # translate exit status
        if scip_status == "optimal":
            if self.has_objective():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif scip_status == "infeasible":
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif scip_status.endswith("limit"):  # timelimit, nodelimit, ...
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif scip_status == "unknown":
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        # TODO: "userinterrupt" and check if there is a solution?
        else: # there are many more, e.g. one for each limit
            raise NotImplementedError(
                f"Translation of scip status {scip_status} to CPMpy status not implemented")  # a non-mapped status type, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                solver_val = self.scip_model.getVal(self.solver_var(cpm_var))
                if cpm_var.is_bool(): 
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = int(solver_val)
            # set _objective_value
            if self.has_objective():
                self.objective_value_ = scip_objective.getObjVal()

        self.scip_model.freeTransform()  # Mark Turner from SCIP told me you need to do this if you want to support adding additional vars/constraints later...

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
            raise Exception("Negative literals should not be part of any equation. See /transformations/linearize for more details")

        # create if it does not exit
        if cpm_var not in self._varmap:
            import pyscipopt as scip
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.scip_model.addVar(vtype='B', name=cpm_var.name)
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.scip_model.addVar(lb=cpm_var.lb, ub=cpm_var.ub, vtype='I', name=cpm_var.name)
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        # return from cache
        return self._varmap[cpm_var]


    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective
                are premanently posted to the solver)
        """
        import pyscipopt as scip

        # make objective function non-nested
        (flat_obj, flat_cons) = (flatten_objective(expr))
        self += flat_cons
        get_variables(flat_obj, collect=self.user_vars)  # add potentially created constraints

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        if minimize:
            self.scip_model.setObjective(obj, sense='minimize')
        else:
            self.scip_model.setObjective(obj, sense='maximize')

    def has_objective(self):
        return len(self.scip_model.getObjective().terms) != 0

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function
        """
        import pyscipopt as scip

        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # sum
        if cpm_expr.name == "sum":
            return scip.quicksum(self.solver_vars(cpm_expr.args))
        if cpm_expr.name == "sub":
            a,b = self.solver_vars(cpm_expr.args)
            return a - b
        # wsum
        if cpm_expr.name == "wsum":
            return scip.quicksum(w * self.solver_var(var) for w, var in zip(*cpm_expr.args))

        raise NotImplementedError("scip: Not a known supported numexpr {}".format(cpm_expr))


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
        # apply transformations, then post internally
        # expressions have to be linearized to fit in MIP model. See /transformations/linearize
        cpm_cons = toplevel_list(cpm_expr)
        supported = {"alldifferent"}  # alldiff has a specialized MIP decomp in linearize
        cpm_cons = decompose_in_tree(cpm_cons, supported)
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        cpm_cons = only_bv_implies(cpm_cons)  # anything that can create full reif should go above...
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum","sub", "mul", "div"})) # the core of the MIP-linearization
        cpm_cons = only_positive_bv(cpm_cons)  # after linearization, rewrite ~bv into 1-bv
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
      import pyscipopt as scip

      # add new user vars to the set
      get_variables(cpm_expr_orig, collect=self.user_vars)

      # transform and post the constraints
      for cpm_expr in self.transform(cpm_expr_orig):

        # Comparisons: only numeric ones as 'only_bv_implies()' has removed the '==' reification for Boolean expressions
        # numexpr `comp` bvar|const
        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            sciprhs = self.solver_var(rhs)

            # Thanks to `only_numexpr_equality()` only supported comparisons should remain
            if cpm_expr.name == '<=':
                sciplhs = self._make_numexpr(lhs)
                self.scip_model.addCons(sciplhs <= sciprhs)
            elif cpm_expr.name == '>=':
                sciplhs = self._make_numexpr(lhs)
                self.scip_model.addCons(sciplhs >= sciprhs)
            elif cpm_expr.name == '==':
                if isinstance(lhs, _NumVarImpl) \
                        or (isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum' or lhs.name == "sub")):
                    # a BoundedLinearExpression LHS, special case, like in objective
                    sciplhs = self._make_numexpr(lhs)
                    self.scip_model.addCons(sciplhs == sciprhs)

                elif lhs.name == 'mul':
                    assert len(lhs.args) == 2, "scip only supports multiplication with 2 variables"
                    a, b = self.solver_vars(lhs.args)
                    #self.scip_model.setParam("NonConvex", 2)  # gurobi leftover
                    self.scip_model.addCons(a * b == sciprhs)

                elif lhs.name == 'div':
                    assert is_num(lhs.args[1]), "scip only supports division by constants"
                    a, b = self.solver_vars(lhs.args)
                    self.scip_model.addLConstr(a / b == sciprhs)

                else:
                    raise NotImplementedError(
                        "Not a known supported scip comparison '{}' {}".format(lhs.name, cpm_expr))

                    # SCIP does have 'addConsAnd', 'addConsOr', 'addConsXor'
                    #   'addConsCardinality' (atmost constant nr non-zero)
                    #   'addConsSOS1' (at most 1 non-zero?) 'addConsSOS2'

                    """
                    # General constraints, gurobi leftover...
                    # sciprhs should be a variable for scip in the subsequent, fake it
                    if is_num(sciprhs):
                        sciprhs = self.solver_var(intvar(lb=sciprhs, ub=sciprhs))

                    if lhs.name == 'min':
                        self.scip_model.addGenConstrMin(sciprhs, self.solver_vars(lhs.args))
                    elif lhs.name == 'max':
                        self.scip_model.addGenConstrMax(sciprhs, self.solver_vars(lhs.args))
                    elif lhs.name == 'abs':
                        self.scip_model.addGenConstrAbs(sciprhs, self.solver_var(lhs.args[0]))
                    elif lhs.name == 'pow':
                        x, a = self.solver_vars(lhs.args)
                        assert a == 2, "scip: 'pow', only support quadratic constraints (x**2)"
                        self.scip_model.addGenConstrPow(x, sciprhs, a)
                    else:
                        raise NotImplementedError(
                        "Not a known supported scip comparison '{}' {}".format(lhs.name, cpm_expr))
                    """
            else:
                raise NotImplementedError(
                "Not a known supported scip comparison '{}' {}".format(lhs.name, cpm_expr))

        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            # Indicator constraints
            # Takes form bvar -> sum(x,y,z) >= rvar
            cond, sub_expr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
            assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"
            if isinstance(cond, NegBoolView):
                cond, bool_val = self.solver_var(cond._bv), False
            else:
                cond, bool_val = self.solver_var(cond), True

            lhs, rhs = sub_expr.args
            if isinstance(lhs, _NumVarImpl) or lhs.name == "sum" or lhs.name == "wsum":
                lin_expr = self._make_numexpr(lhs)
            else:
                raise Exception(f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}")
            if sub_expr.name == "<=":  # XXX, docs say only this one?
                if bool_val: # positive Boolean
                    self.scip_model.addConsIndicator(lin_expr <= self.solver_var(rhs), cond)
                else: # negation of `cond` variable is the indicator
                    raise Exception(f"This code uncertain to work, test at your own risk")
                    self.scip_model.addConsIndicator(lin_expr <= self.solver_var(rhs), ~cond)
            elif sub_expr.name == ">=":
                raise Exception(f"This code uncertain to work, test at your own risk")
                if bool_val: # positive Boolean
                    self.scip_model.addConsIndicator(lin_expr >= self.solver_var(rhs), cond)
                else: # negation of `cond` variable is the indicator
                    self.scip_model.addConsIndicator(lin_expr >= self.solver_var(rhs), ~cond)
            elif sub_expr.name == "==":
                raise Exception(f"This code uncertain to work, test at your own risk")
                if bool_val: # positive Boolean
                    self.scip_model.addConsIndicator(lin_expr == self.solver_var(rhs), cond)
                else: # negation of `cond` variable is the indicator
                    self.scip_model.addConsIndicator(lin_expr == self.solver_var(rhs), ~cond)
            else:
                raise Exception(f"Unknown linear expression {sub_expr} name")

        # True or False
        elif isinstance(cpm_expr, BoolVal):
            self.scip_model.addConstr(cpm_expr.args[0])

        # a direct constraint, pass to solver
        elif isinstance(cpm_expr, DirectConstraint):
            cpm_expr.callSolver(self, self.scip_model)

        else:
            raise NotImplementedError(cpm_expr)  # if you reach this... please report on github

      return self

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
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

            Returns: number of solutions found
        """
        raise NotImplementedError(f"Not actually implemented (yet?)")

        """ OLD GUROBI code
        if time_limit is not None:
            self.scip_model.setParam("TimeLimit", time_limit)

        if solution_limit is None:
            raise Exception(
                "scip does not support searching for all solutions. If you really need all solutions, try setting solution limit to a large number")

        # Force scip to keep searching in the tree for optimal solutions
        sa_kwargs = {"PoolSearchMode":2, "PoolSolutions":solution_limit}

        # solve the model
        self.solve(time_limit=time_limit, **sa_kwargs, **kwargs)

        optimal_val = None
        solution_count = self.scip_model.SolCount
        opt_sol_count = 0

        for i in range(solution_count):
            # Specify which solution to query
            self.scip_model.setParam("SolutionNumber", i)
            sol_obj_val = self.scip_model.PoolObjVal
            if optimal_val is None:
                optimal_val = sol_obj_val
            if optimal_val is not None:
                # sub-optimal solutions
                if sol_obj_val != optimal_val:
                    break
            opt_sol_count += 1

            # Translate solution to variables
            for cpm_var in self.user_vars:
                solver_val = self.solver_var(cpm_var).Xn
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = int(solver_val)

            # Translate objective
            if self.has_objective():
                self.objective_value_ = self.scip_model.PoolObjVal

            if display is not None:
                if isinstance(display, Expression):
                    print(display.value())
                elif isinstance(display, list):
                    print([v.value() for v in display])
                else:
                    display()  # callback

        # Reset pool search mode to default
        self.scip_model.setParam("PoolSearchMode", 0)

        return opt_sol_count
        """
