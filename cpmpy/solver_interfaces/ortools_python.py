#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## ortools_python.py
##
"""
    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        ORToolsPython

    ==================
    Module description
    ==================

    ==============
    Module details
    ==============
"""
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions import *
from ..globalconstraints import *
from ..variables import *
from ..model_tools.get_variables import get_variables, vars_expr
from ..model_tools.flatten_model import flatten_model, flatten_constraint, get_or_make_var, negated_normal

class ORToolsPython(SolverInterface):
    """
    Interface to the python 'ortools' API

    Requires that the 'ortools' python package is installed:
    $ pip install ortools

    Creates the following attributes:
    _model: the ortools.sat.python.cp_model.CpModel() created by _model()
    _solver: the ortools cp_model.CpSolver() instance used in solve()
    """

    def __init__(self):
        self.name = "ortools"

    def supported(self):
        try:
            import ortools
            return True
        except ImportError as e:
            return False

    def solve(self, cpm_model, num_workers=1):
        if not self.supported():
            raise Exception("Install the python 'ortools' package to use this '{}' solver interface".format(self.name))
        from ortools.sat.python import cp_model as ort

        # store original vars (before flattening)
        original_vars = get_variables(cpm_model)

        # create model
        self.ort_model = self.make_model(cpm_model)
        # solve the instance
        self.ort_solver = ort.CpSolver()
        self.ort_solver.parameters.num_search_workers = num_workers # increase for more efficiency (parallel)
        self.ort_status = self.ort_solver.Solve(self.ort_model)

        # translate status
        my_status = SolverStatus()
        my_status.solver_name = self.name
        if self.ort_status == ort.FEASIBLE:
            my_status.exitstatus = ExitStatus.FEASIBLE
        elif self.ort_status == ort.OPTIMAL:
            my_status.exitstatus = ExitStatus.OPTIMAL
        elif self.ort_status == ort.INFEASIBLE:
            my_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif self.ort_status == ort.MODEL_INVALID:
            raise Exception("OR-Tools says: model invalid")
        else: # ort.UNKNOWN or another
            raise NotImplementedError(self.ort_status) # a new status type was introduced, please report on github
        my_status.runtime = self.ort_solver.WallTime()

        if self.ort_status == ort.FEASIBLE or self.ort_status == ort.OPTIMAL:
            # fill in variables
            for var in original_vars:
                var._value = self.ort_solver.Value(self.varmap[var])

        return my_status


    def make_model(self, cpm_model):
        """
            Makes the ortools.sat.python.cp_model formulation out of 
            a CPMpy model (will do flattening and other trnasformations)
        """
        from ortools.sat.python import cp_model as ort

        # Constraint programming engine
        self._model = ort.CpModel()

        # Transform into flattened model
        flat_model = flatten_model(cpm_model)

        # Create corresponding solver variables
        self.varmap = dict() # cppy var -> solver var
        for var in get_variables(flat_model):
            self.add_to_varmap(var)

        # Post the (flat) constraint expressions to the solver
        for con in flat_model.constraints:
            self.post_constraint(con)

        # Post the objective
        if flat_model.objective is None:
            pass # no objective, satisfaction problem
        else:
            obj = self.ort_numexpr(flat_model.objective)
            if flat_model.objective_max:
                self._model.Maximize(obj)
            else:
                self._model.Minimize(obj)

        return self._model


    def add_to_varmap(self, cpm_var):
        if isinstance(cpm_var, BoolVarImpl):
            revar = self._model.NewBoolVar(str(cpm_var))
        elif isinstance(cpm_var, IntVarImpl):
            revar = self._model.NewIntVar(cpm_var.lb, cpm_var.ub, str(cpm_var))
        self.varmap[cpm_var] = revar


    def post_constraint(self, cpm_expr):
        """
            Constraints are expected to be in 'flat normal form' (see flatten_model.py)

            While the normal form is divided in 'base', 'comparison' and 'reified', we
            here regroup it per CPMpy class

            Returns the posted ortools 'Constraint', so that it can be used in reification
            e.g. self.post_constraint(smth).onlyEnforceIf(self.ort_var(bvar))
        """
        # Base case: Boolean variable
        if isinstance(cpm_expr, BoolVarImpl):
            return self._model.AddBoolOr( [self.ort_var(cpm_expr)] )
        
        # Comparisons: including base (vars), numeric comparison and reify/imply comparison
        elif isinstance(cpm_expr, Comparison):
            lhs,rhs = cpm_expr.args

            if isinstance(lhs, BoolVarImpl) and cpm_expr.name == '==':
                # base: bvar == bvar|const
                lvar,rvar = map(self.ort_var, (lhs,rhs))
                return self._model.Add(lvar == rvar)

            elif lhs.is_bool() and cpm_expr.name == '==':
                # reified case: boolexpr == var
                lexpr = cpm_expr.args[0]
                rvar = cpm_expr.args[1]
                # split in boolexpr -> var and var -> boolexpr
                self.post_constraint(lexpr.implies(rvar))
                self.post_constraint(rvar.implies(lexpr))

            else:
                # numeric (non-reify) comparison case
                rvar = self.ort_var(rhs)
                # lhs can be numexpr
                if isinstance(lhs, NumVarImpl):
                    # simplest LHS case, a var
                    newlhs = self.ort_var(lhs)
                else:
                    if isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum'):
                        # a BoundedLinearExpression LHS, special case, like in objective
                        newlhs = self.ort_numexpr(lhs) 
                    elif cpm_expr.name == '==':
                        newlhs = None
                        if lhs.name == 'abs':
                            return self._model.AddAbsEquality(rvar, self.ort_var(lhs.args[0]))
                        elif lhs.name == 'mul':
                            return self._model.AddMultiplicationEquality(rvar, self.ort_var_or_list(lhs.args))
                        elif lhs.name == 'mod':
                            #self._model.AddModuloEquality(rvar, ...)
                            raise NotImplementedError("modulo")
                        elif lhs.name == 'min':
                            return self._model.AddMinEquality(rvar, self.ort_var_or_list(lhs.args))
                        elif lhs.name == 'max':
                            return self._model.AddMaxEquality(rvar, self.ort_var_or_list(lhs.args))
                        elif lhs.name == 'element':
                            # arr[idx]==rvar (arr=arg0,idx=arg1), ort: (idx,arr,target)
                            return self._model.AddElement(self.ort_var(lhs.args[1]), self.ort_var_or_list(lhs.args[0]), rvar)
                        else:
                            raise NotImplementedError("Not a know supported ORTools left-hand-side '{}' {}".format(lhs.name, cpm_expr))
                    else:
                        # other equality than == 
                        # example: x*y > 10 :: x*y == aux, aux > 10
                        # creat the equality (will handle appropriate bounds)
                        (newvar, cons) = get_or_make_var(lhs)
                        self.add_to_varmap(newvar)
                        for con in cons:
                            # post the flattened constraints, including the 'lhs == newvar' one
                            # if this contains new auxiliary variables we will crash
                            self.post_constraint(con)
                        newlhs = newvar

                if newlhs is None:
                    pass # is already posted directly, eg a '=='
                elif cpm_expr.name == '==':
                    return self._model.Add( newlhs == rvar)
                elif cpm_expr.name == '!=':
                    return self._model.Add( newlhs != rvar )
                elif cpm_expr.name == '<=':
                    return self._model.Add( newlhs <= rvar )
                elif cpm_expr.name == '<':
                    return self._model.Add( newlhs < rvar )
                elif cpm_expr.name == '>=':
                    return self._model.Add( newlhs >= rvar )
                elif cpm_expr.name == '>':
                    return self._model.Add( newlhs > rvar )

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == '->' and \
             (not isinstance(cpm_expr.args[0], BoolVarImpl) or \
              not isinstance(cpm_expr.args[1], BoolVarImpl)):
                # reified case: var -> boolexpr, boolexpr -> var
                if isinstance(cpm_expr.args[0], BoolVarImpl):
                    # var -> boolexpr, natively supported by or-tools
                    bvar = self.ort_var(cpm_expr.args[0])
                    return self.post_constraint(cpm_expr.args[1]).OnlyEnforceIf(bvar)
                else:
                    # boolexpr -> var, have to convert to ~var -> ~boolexpr
                    negbvar = self.ort_var(cpm_expr.args[1]).Not()
                    negleft = negated_normal(cpm_expr.args[0])
                    return self.post_constraint(negleft).OnlyEnforceIf(negbvar)

            else:
                # base 'and'/n, 'or'/n, 'xor'/n, '->'/2
                args = [self.ort_var(v) for v in cpm_expr.args]

                if cpm_expr.name == 'and':
                    return self._model.AddBoolAnd(args)
                elif cpm_expr.name == 'or':
                    return self._model.AddBoolOr(args)
                elif cpm_expr.name == 'xor':
                    return self._model.AddBoolXor(args)
                elif cpm_expr.name == '->':
                    return self._model.AddImplication(args[0],args[1])
                else:
                    raise NotImplementedError("Not a know supported ORTools Operator '{}' {}".format(cpm_expr.name, cpm_expr))

        # rest: base (Boolean) global constraints
        else:
            args = [self.ort_var_or_list(v) for v in cpm_expr.args]

            if cpm_expr.name == 'alldifferent':
                return self._model.AddAllDifferent(args) 
            # TODO: NOT YET MAPPED: AllowedAssignments, Automaton, Circuit, Cumulative,
            #    ForbiddenAssignments, Inverse?, NoOverlap, NoOverlap2D,
            #    ReservoirConstraint, ReservoirConstraintWithActive
            else:
                # global constraint not known, try generic decomposition
                dec = cpm_expr.decompose()
                if not dec is None:
                    flatdec = flatten_constraint(dec)

                    # collect and create new variables
                    for var in vars_expr(flatdec):
                        if not var in self.varmap:
                            self.add_to_varmap(var)
                    # post decomposition
                    for con in flatdec:
                        self.post_constraint(con)
                    # XXX how to deal with reification of such a global??
                    # TODO: we would have to catch this at the time of the reification... (outer call)
                    return None # will throw error if used in reification...
                else:
                    raise NotImplementedError(cpm_expr) # if you reach this... please report on github

    def ort_var(self, cpm_var):
        if is_num(cpm_var):
            return cpm_var

        # decision variables, check in varmap
        if isinstance(cpm_var, NegBoolView):
            return self.varmap[cpm_var._bv].Not()
        elif isinstance(cpm_var, NumVarImpl): # BoolVarImpl is subclass of NumVarImpl
            return self.varmap[cpm_var]

        raise NotImplementedError("Not a know var {}".format(cpm_var))

    def ort_var_or_list(self, cpm_expr):
        if is_any_list(cpm_expr):
            return [self.ort_var_or_list(sub) for sub in cpm_expr]
        return self.ort_var(cpm_expr)


    def ort_numexpr(self, cpm_expr):
        """
            ORTools subexpressions (for in objective function and comparisons)
            Accepted by ORTools:
            - Decision variable: Var
            - Linear: sum([Var])                                   (CPMpy class 'Operator', name 'sum')
                      wsum([Const],[Var])                          (CPMpy class 'Operator', name 'wsum')
        """
        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, NumVarImpl): # BoolVarImpl is subclass of NumVarImpl
            return self.ort_var(cpm_expr)

        # sum or (to be implemented: wsum)
        if isinstance(cpm_expr, Operator):
            args = [self.ort_var(v) for v in cpm_expr.args]
            if cpm_expr.name == 'sum':
                return sum(args) # OR-Tools supports this

        raise NotImplementedError("Not a know supported ORTools expression {}".format(cpm_expr))
