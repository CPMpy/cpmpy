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
from ..model_tools.get_variables import get_variables, vars_expr
from ..model_tools.flatten_model import *

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
        else:
            raise NotImplementedError(my_status) # a new status type was introduced, please report on github
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
        modelvars = get_variables(flat_model)
        for var in modelvars:
            if isinstance(var, BoolVarImpl):
                revar = self._model.NewBoolVar(str(var.name))
            elif isinstance(var, IntVarImpl):
                revar = self._model.NewIntVar(var.lb, var.ub, str(var.name))
            self.varmap[var] = revar

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
            return ort_var(cpm_expr)

        # sum or (to be implemented: wsum)
        if isinstance(cpm_expr, Operator):
            args = [self.ort_var(v) for v in cpm_expr.args]
            if expr.name == 'sum':
                return sum(args)

        raise NotImplementedError("Not a know supported ORTools expression {}".format(cpm_expr))


    def post_constraint(self, cpm_expr):
        """
            Constraints are expected to be in 'flat normal form' (see flatten_model.py)

            While the normal form is divided in 'base', 'comparison' and 'reified', we
            here regroup it per CPMpy class
        """
        # Base case: Boolean variable
        if isinstance(cpm_expr, BoolVarImpl):
            self._model.AddBoolOr( [self.ort_var(cpm_expr)] )
        
        # Comparisons: including base (vars), numeric comparison and reify/imply comparison
        elif isinstance(cpm_expr, Comparison):
            lhs,rhs = cpm_expr.args

            if isinstance(lhs, BoolVarImpl) and cpm_expr.name == '==':
                # base: bvar == bvar|const
                lvar,rvar = map(self.ort_var, (lhs,rhs))
                self._model.Add(lvar == rvar)

            elif lhs.is_bool() and cpm_expr.name == '==':
                # reified case: boolexpr == var
                raise NotImplementedError # TODO

            else:
                # numeric (non-reify) comparison case
                # lhs can be numexpr
                newlhs = self.ort_numexpr(lhs) 
                rvar = self.ort_var(rhs)
                if expr.name == '==':
                    self._model.Add( newlhs == rvar)
                elif expr.name == '!=':
                    self._model.Add( newlhs != rvar )
                elif expr.name == '<=':
                    self._model.Add( newlhs <= rvar )
                elif expr.name == '<':
                    self._model.Add( newlhs < rvar )
                elif expr.name == '>=':
                    self._model.Add( newlhs >= rvar )
                elif expr.name == '>':
                    self._model.Add( newlhs > rvar )

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == '->' and \
             (not isinstance(cpm_expr.args[0], BoolVarImpl) or \
              not isinstance(cpm_expr.args[1], BoolVarImpl)):
                # reified case: boolexpr -> var, var -> boolexpr
                """
            # two special cases:
            #    '->' with .onlyEnforceIf()
            #    'xor' does not have subexpression form
            # all others: add( subexpression )
            if expr.name == '->':
                args = [self.convert_subexpr(e) for e in expr.args]
                if isinstance(expr.args[0], BoolVarImpl):
                    # regular implication
                    self._model.AddImplication(args[0], args[1])
                else:
                    # XXX needs proper implementation of half-reification
                    print("May not actually work")
                    self._model.Add( args[0] ).OnlyEnforceIf(args[1])
                """
                raise NotImplementedError # TODO

            else:
                # base 'and'/n, 'or'/n, 'xor'/n, '->'/2
                args = [self.ort_var(v) for v in cpm_expr.args]

                if cpm_expr.name == 'and':
                    self._model.AddBoolAnd(args)
                elif cpm_expr.name == 'or':
                    self._model.AddBoolOr(args)
                elif cpm_expr.name == 'xor':
                    self._model.AddBoolXor(args)
                elif cpm_expr.name == '->':
                    self._model.AddImplication(arg[0],args[1])

            raise NotImplementedError("Not a know supported ORTools Operator {}".format(cpm_expr))

        # rest: base (Boolean) global constraints
        else:
            # TODO: could be list of vars...
            args = [self.ort_var_or_list(v) for v in cpm_expr.args]

            if cpm_expr.name == 'alldifferent':
                self._model.AddAllDifferent(args) 
            # NOT YET MAPPED: AllowedAssignments, Automaton, Circuit, Cumulative,
            #    ForbiddenAssignments, Inverse?, NoOverlap, NoOverlap2D,
            #    ReservoirConstraint, ReservoirConstraintWithActive
            else:
                # global constraint not known, try generic decomposition
                raise NotImplementedError("Untested old code!")
                dec = cpm_expr.decompose()
                if not dec is None:
                    flatdec = flatten_constraint(dec)

                    # collect and create new variables
                    flatvars = vars_expr(flatdec)
                    for var in flatvars:
                        if not var in self.varmap:
                            # new variable
                            if isinstance(var, BoolVarImpl):
                                revar = self._model.NewBoolVar(str(var.name))
                            elif isinstance(var, IntVarImpl):
                                revar = self._model.NewIntVar(var.lb, var.ub, str(var.name))
                            self.varmap[var] = revar
                    # post decomposition
                    for con in flatdec:
                        self.post_constraint(con)
                else:
                    raise NotImplementedError(cpm_expr) # if you reach this... please report on github


    def also_old(self):
        if True:
            pass
        elif isinstance(expr, Element):
            # A0[A1] == Var --> AddElement(A1, A0, Var)
            args = [self.convert_subexpr(e) for e in expr.args]
            # TODO: make 'Var'...
            return self._model.AddElement(args[1], args[0], None)

        # rest: global constraints
        elif expr.name == 'min' or expr.name == 'max':
            args = [self.convert_subexpr(e) for e in expr.args]
            lb = min(a.lb() if isinstance(arg, NumVarImpl) else a for a in args)
            ub = max(a.ub() if isinstance(arg, NumVarImpl) else a for a in args)
            aux = self._model.NewIntVar(lb, ub, "aux")
            if expr.name == 'min':
                self._model.AddMinEquality(aux, args) 
            else:
                self._model.AddMaxEquality(aux, args) 

        else:
            # global constraint not known, try generic decomposition
            dec = expr.decompose()
            if not dec is None:
                flatdec = flatten_constraint(dec)

                # collect and create new variables
                flatvars = vars_expr(flatdec)
                for var in flatvars:
                    if not var in self.varmap:
                        # new variable
                        if isinstance(var, BoolVarImpl):
                            revar = self._model.NewBoolVar(str(var.name))
                        elif isinstance(var, IntVarImpl):
                            revar = self._model.NewIntVar(var.lb, var.ub, str(var.name))
                        self.varmap[var] = revar
                # post decomposition
                for con in flatdec:
                    self.post_constraint(con)
            else:
                raise NotImplementedError(dec) # if you reach this... please report on github
        


    def old(self):
        if isinstance(expr, Operator):
            # bool: 'and'/n, 'or'/n, 'xor'/n, '->'/2
            # unary int: '-', 'abs'
            # binary int: 'sub', 'mul', 'div', 'mod', 'pow'
            # nary int: 'sum'
            args = [self.convert_subexpr(e) for e in expr.args]
            if expr.name == 'and':
                return all(args)
            elif expr.name == 'or':
                return any(args)
            elif expr.name == 'xor':
                raise Exception("or-tools translation: XOR probably illegal as subexpression")
            elif expr.name == '->':
                # when part of subexpression: can not use .OnlyEnforceIf() (I think)
                # so convert to -a | b
                return args[0].Not() | args[1]
            elif expr.name == '-':
                return -args[0]
            elif expr.name == 'abs':
                return abs(args[0])
            if expr.name == 'sub':
                return args[0] - args[1]
            elif expr.name == 'mul':
                return args[0] * args[1]
            elif expr.name == 'div':
                return args[0] / args[1]
            elif expr.name == 'mod':
                return args[0] % args[1]
            elif expr.name == 'pow':
                return args[0] ** args[1]
            elif expr.name == 'sum':
                return sum(args)

        elif isinstance(expr, Comparison):
            #allowed = {'==', '!=', '<=', '<', '>=', '>'}
            # recursively convert arguments (subexpressions)
            lvar = self.convert_subexpr(expr.args[0])
            rvar = self.convert_subexpr(expr.args[1])
            if expr.name == '==':
                return (lvar == rvar)
            elif expr.name == '!=':
                return (lvar != rvar)
            elif expr.name == '<=':
                return (lvar <= rvar)
            elif expr.name == '<':
                return (lvar < rvar)
            elif expr.name == '>=':
                return (lvar >= rvar)
            elif expr.name == '>':
                return (lvar > rvar)

        raise NotImplementedError(expr) # should not reach this... please report on github
        # there might be an Element expression here... need to add flatten rule then?
