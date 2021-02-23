from . import *
from ..expressions import *
from ..variables import *
from .minizinc_text import *
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

    def make_model(self, cppy_model):
        from ortools.sat.python import cp_model as ort

        # Constraint programming engine
        self._model = ort.CpModel()

        # Create corresponding solver variables
        self.varmap = dict() # cppy var -> solver var
        self.revarmap = dict() # reverse: solver var -> cppy var
        modelvars = get_variables(cppy_model)
        for var in modelvars:
            if isinstance(var, BoolVarImpl):
                revar = self._model.NewBoolVar(str(var.name))
            elif isinstance(var, IntVarImpl):
                revar = self._model.NewIntVar(var.lb, var.ub, str(var.name))
            self.varmap[var] = revar
            self.revarmap[revar] = var

        # post the constraint expressions to the solver
        # TODO: assumes 'flat' constraints (no subexpressions)
        for con in cppy_model.constraints:
            self.post_expression(con)

        # the objective
        if cppy_model.objective is None:
            pass # no objective, satisfaction problem
        else:
            obj = self.convert_expression(cppy_model.objective)
            if cppy_model.objective_max:
                self._model.Maximize(obj)
            else:
                self._model.Minimize(obj)

        return self._model

    def solve(self, cppy_model, num_workers=1):
        if not self.supported():
            raise "Install the python 'ortools' package to use this '{}' solver interface".format(self.name)
        from ortools.sat.python import cp_model as ort

        # create model (TODO: how to start from other model?)
        self._model = self.make_model(cppy_model)
        # solve the instance
        self._solver = ort.CpSolver()
        self._solver.parameters.num_search_workers = num_workers # increase for more efficiency (parallel)
        self._status = self._solver.Solve(self._model)

        # translate status
        solstats = SolverStats()
        if self._status == ort.FEASIBLE:
            solstats.status = ExitStatus.FEASIBLE
        elif self._status == ort.OPTIMAL:
            solstats.status = ExitStatus.OPTIMAL
        else:
            raise NotImplementedError
        solstats.runtime = self._solver.WallTime()

        if self._status == ort.FEASIBLE or self._status == ort.OPTIMAL:
            # fill in variables
            for var in self.varmap:
                var._value = self._solver.Value(self.varmap[var])

        return solstats

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
            # TODO bool: 'and'/n, 'or'/n, 'xor'/n, '->'/2
            # TODO -> for non-bool input with .OnlyEnforceIf()...
            # unary int: '-', 'abs'
            # binary int: 'sub', 'mul', 'div', 'mod', 'pow'
            # nary int: 'sum'
            args = [self.convert_expression(e) for e in expr.args]
            if expr.name == '-':
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


        print(type(expr),expr)
        raise NotImplementedError

    def post_expression(self, expr):
        # recursively convert arguments (subexpressions)
        args = [self.convert_expression(e) for e in expr.args]
        
        # standard expressions: comparison, operator, element
        if isinstance(expr, Comparison):
            #allowed = {'==', '!=', '<=', '<', '>=', '>'}
            #TODO refactor as rewrite rule?
            for lvar, rvar in zipcycle(args[0], args[1]):
                if expr.name == '==':
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
            #printmap = {'and': '/\\', 'or': '\\/',
            #            'sum': '+', 'sub': '-',
            #            'mul': '*', 'div': '/', 'pow': '^'}
            if expr.name == 'or':
                self._model.AddBoolOr(args)
            elif expr.name == 'and':
                self._model.AddBoolAnd(args)
            else:
                print(expr.name, type(expr), expr)
                raise NotImplementedError

        elif isinstance(expr, Element):
            subtype = "int"
            # TODO: need better bool check... is_bool() or type()?
            if all((v == 1) is v for v in iter(expr.args[0])):
                subtype = "bool"
            print(expr.name, type(expr), expr)
            raise NotImplementedError
        

        # rest: global constraints
        elif expr.name == 'alldifferent':
           self._model.AddAllDifferent(args) 


        elif expr.name.endswith('circuit'): # circuit, subcircuit
            print(expr.name, type(expr), expr)
            raise NotImplementedError

        else:
            # TODO: what is default action? how to catch if not supported?
            print(expr.name, type(expr), expr)
            raise NotImplementedError
        
        

