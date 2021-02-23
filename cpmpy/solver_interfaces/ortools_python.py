from . import *
from ..expressions import *
from ..variables import *
from .minizinc_text import *

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

    def make_model(self, cpmpy_model):
        from ortools.sat.python import cp_model as ort

        # Constraint programming engine
        self._model = ort.CpModel()

        # will store the variables here
        self.vardict = dict()

        # make the constraint expressions (and create the vars)
        for con in cpmpy_model.constraints:
            self.post_expression(con)

        # the objective
        # TODO
        print(cpmpy_model.objective)

        return self._model

    def solve(self, cpmpy_model, num_workers=1):
        if not self.supported():
            raise "Install the python 'ortools' package to use this '{}' solver interface".format(self.name)
        from ortools.sat.python import cp_model as ort

        # create model (TODO: how to start from other model?)
        self._model = self.make_model(cpmpy_model)

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
        # TODO, runtime?

        if self._status == ort.FEASIBLE or self._status == ort.OPTIMAL:
            # TODO smth with enumerating the python vars and filling them
            # TODO, use a decorator for .value again so that can look like propety but is function
            # fill in variables
            modelvars = get_variables(model)
            #for name,var in self.vardict:
            #    pass
            for var in modelvars:
                var.set_value(self._solver.Value(var)) # not sure this will work

        return solstats

    def post_expression(self, expr):
        #if is_any_list(expr):

        #if not isinstance(expr, Expression):

        #args_str = [self.convert_expression(e) for e in expr.args]

        # standard expressions: comparison, operator, element
        if isinstance(expr, Comparison):
            pass


        if isinstance(expr, Operator):
            # some names differently (the infix names!)
            printmap = {'and': '/\\', 'or': '\\/',
                        'sum': '+', 'sub': '-',
                        'mul': '*', 'div': '/', 'pow': '^'}
            op_str = expr.name
            if op_str in printmap:
                op_str = printmap[op_str]
            pass


        if isinstance(expr, Element):
            subtype = "int"
            # TODO: need better bool check... is_bool() or type()?
            if all((v == 1) is v for v in iter(expr.args[0])):
                subtype = "bool"
            pass
        

        # rest: global constraints
        if expr.name == 'alldifferent':
           self._model.AddAllDifferent(expr) 


        if expr.name.endswith('circuit'): # circuit, subcircuit
            pass

        # TODO: what is default action? how to catch if not supported?
        
        

