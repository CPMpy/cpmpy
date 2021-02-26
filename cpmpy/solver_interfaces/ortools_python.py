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
from ..expressions import Comparison, Expression, Operator, Element
from ..globalconstraints import *
from ..model_tools.get_variables import get_variables
from ..model_tools.flatten_model import *

from itertools import cycle

def zipcycle(vars1, vars2):
    v1 = [vars1] if not is_any_list(vars1) else vars1
    v2 = [vars2] if not is_any_list(vars2) else vars2
    return zip(v1, cycle(v2)) if len(v2) < len(v1) else zip(cycle(v1), v2)

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

    def make_model(self, cpm_model):
        from ortools.sat.python import cp_model as ort

        # Constraint programming engine
        self._model = ort.CpModel()

        # transform into flattened model
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

        # post the (flat) constraint expressions to the solver
        for con in flat_model.constraints:
            self.post_expression(con)

        # the objective
        if flat_model.objective is None:
            pass # no objective, satisfaction problem
        else:
            obj = self.convert_expression(flat_model.objective)
            if flat_model.objective_max:
                self._model.Maximize(obj)
            else:
                self._model.Minimize(obj)

        return self._model

    def solve(self, cpm_model, num_workers=1):
        if not self.supported():
            raise "Install the python 'ortools' package to use this '{}' solver interface".format(self.name)
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
            raise NotImplementedError # a new status type was introduced, please report on github
        my_status.runtime = self.ort_solver.WallTime()

        if self.ort_status == ort.FEASIBLE or self.ort_status == ort.OPTIMAL:
            # fill in variables
            for var in original_vars:
                var._value = self.ort_solver.Value(self.varmap[var])

        return my_status

    # for subexpressions (variables, lists and linear expressions)
    def convert_expression(self, expr):
        # python constants
        if is_num(expr):
            return expr

        # list
        if is_any_list(expr):
            return [self.convert_expression(e) for e in expr]

        # decision variables, check in varmap
        if isinstance(expr, NumVarImpl): # BoolVarImpl is subclass of NumVarImpl
            return self.varmap[expr]

        # ~B
        # XXX should we implement NegBoolView instead?
        if isinstance(expr, Comparison) and expr.name == '==' and \
           isinstance(expr.args[0], BoolVarImpl) and expr.args[1] == 0:
            return self.varmap[expr.args[0]].Not()

        if isinstance(expr, Operator):
            # bool: 'and'/n, 'or'/n, 'xor'/n, '->'/2
            # unary int: '-', 'abs'
            # binary int: 'sub', 'mul', 'div', 'mod', 'pow'
            # nary int: 'sum'
            args = [self.convert_expression(e) for e in expr.args]
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

        raise NotImplementedError # should not reach this... please report on github
        # there might be an Element expression here... need to add flatten rule then?

    def post_expression(self, expr):
        # base case
        if isinstance(expr, BoolVarImpl):
            print(type(expr), expr, self.varmap[expr])
            self._model.AddBoolOr( [self.varmap[expr]] )
        
        # standard expressions: comparison, operator, element
        if isinstance(expr, Comparison):
            # recursively convert arguments (subexpressions)
            args = [self.convert_expression(e) for e in expr.args]
            #allowed = {'==', '!=', '<=', '<', '>=', '>'}
            #XXX refactor decomposition into constructor of Comparison()?
            for lvar, rvar in zipcycle(args[0], args[1]):
                if expr.name == '==':
                    # XXX this might break for 'reification' of constraints...
                    # will have to add two-sided .OnlyEnforceIf() then?
                    # if you get an error here, please report on github
                    self._model.Add(lvar == rvar)
                elif expr.name == '!=':
                    self._model.Add( lvar != rvar )
                elif expr.name == '<=':
                    self._model.Add( lvar <= rvar )
                elif expr.name == '<':
                    self._model.Add( lvar < rvar )
                elif expr.name == '>=':
                    self._model.Add( lvar >= rvar )
                elif expr.name == '>':
                    self._model.Add( lvar > rvar )

        elif isinstance(expr, Operator):
            # bool: 'and'/n, 'or'/n, 'xor'/n, '->'/2
            # unary int: '-', 'abs'
            # binary int: 'sub', 'mul', 'div', 'mod', 'pow'
            # nary int: 'sum'

            # two special cases:
            #    '->' with .onlyEnforceIf()
            #    'xor' does not have subexpression form
            # all others: add( subexpression )
            if expr.name == '->':
                # XXX we might have to do some sort of flattening here if RHS is not BoolVar...
                # or-tools will raise error if invalid LHS (or RHS)
                args = [self.convert_expression(e) for e in expr.args]
                self._model.Add( args[0] ).OnlyEnforceIf(args[1])
            elif expr.name == 'xor':
                args = [self.convert_expression(e) for e in expr.args]
                self._model.AddBoolXor(args)
            else:
                self._model.Add( self.convert_expression(expr) )

        elif isinstance(expr, Element):
            # A0[A1] == A2 --> AddElement(A1, A0, A2)
            args = [self.convert_expression(e) for e in expr.args]
            return self._model.AddElement(args[1], args[0], args[2])
        

        # rest: global constraints
        elif expr.name == 'alldifferent':
            args = [self.convert_expression(e) for e in expr.args]
            self._model.AddAllDifferent(args) 
        elif expr.name == 'min' or expr.name == 'max':
            args = [self.convert_expression(e) for e in expr.args]
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
                self.post_expression(flatten_constraint(dec))
            else:
                raise NotImplementedError # if you reach this... please report on github
        

