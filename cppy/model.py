from .expressions import *
from .solver_interfaces import *
# from . import *

import numpy as np
# import os
from pathlib import Path

class Model(object):
    """
    CPpy Model object, contains the constraint and objective expression trees

    Arguments of constructor:
    *args: Expression object(s) or list(s) of Expression objects
    minimize: Expression object representing the objective to minimize
    maximize: Expression object representing the objective to maximize

    At most one of minimize/maximize can be set, if none are set, it is assumed to be a satisfaction problem
    """
    def __init__(self, *args, minimize=None, maximize=None):
        assert ((minimize is None) or (maximize is None)), "can not set both minimize and maximize"
        # list of constraints (arguments of top-level conjunction)
        root_constr = self.make_and_from_list(args)
        if root_constr.name == 'and':
            # unpack top-level conjuction
            self.constraints = root_constr.args
        else:
            # wrap in list
            self.constraints = [root_constr]

        # an expresion or None
        self.objective = None
        self.objective_max = None

        if not maximize is None:
            self.objective = maximize
            self.objective_max = True
        if not minimize is None:
            self.objective = minimize
            self.objective_max = False

    def make_and_from_list(self, args):
        """ recursively reads a list of Expression and returns the 'And' conjunctive of the elements in the list """
        lst = list(args) # make mutable copy of type list
        # do recursive where needed, with overwrite
        for (i, expr) in enumerate(lst):
            if isinstance(expr, list):
                lst[i] = self.make_and_from_list(expr)
        if len(lst) == 1:
            return lst[0]
        return Operator("and", lst)

    def __repr__(self):
        cons_str = ""
        for c in self.constraints:
            cons_str += "    {}\n".format(c)

        obj_str = ""
        if not self.objective is None:
            if self.objective_max:
                obj_str = "maximize "
            else:
                obj_str = "minimize "
        obj_str += str(self.objective)
            
        return "Constraints:\n{}Objective: {}".format(cons_str, obj_str)
    
    # solver: name of supported solver or any SolverInterface object
    def solve(self, solver=None):
        """ Send the model to a solver and get the result

        'solver': None (default) or in [s.name in get_supported_solvers()] or a SolverInterface object
        verifies that the solver is supported on the current system
        """
        # get supported solvers
        supsolvers = get_supported_solvers()
        if solver is None: # default is first
            solver = supsolvers[0]
        elif not isinstance(solver, SolverInterface):
            solvername = solver
            for s in supsolvers:
                if s.name == solvername:
                    solver = s
                    break # break and hence 'solver' is correct object

            if not isinstance(solver, SolverInterface) or not solver.supported():
                raise Exception("'{}' is not in the list of supported solvers and not a SolverInterface object".format(solver))

        return solver.solve(self)

    def to_cnf(self):
        # 1. consider all subformulas
        sub_formulas = []
        # sub_formulas = [self.constraints]
        # print(self)
        for c in self.constraints:
            arg_sub_f = c.subformula()
            # a. Subformulas
            if arg_sub_f != None and len(arg_sub_f) != 0:
                for f in arg_sub_f:
                    sub_formulas.append(f)
            # b. Just regular literal 
            else:
                sub_formulas.append(c)
        # return sub_formulas
        # 2. introduce new variable for each subformula
        # TODO: add link to recognize original formula
        new_formulas = []

        for formula in sub_formulas:
            bi = BoolVar()
            new_formulas.append(implies(formula, bi) & implies(bi, formula))

        cnf_formula = []
        # 3. conjunct all substituations and the substitution for phi
        for formula in new_formulas:
            # TODO: transform formula to cnf
            cnf_formula.append(formula.to_cnf())
        # all substitutions can be transformed into CNF
        return cnf_formula

    def cnf_to_pysat(self, cnf, output = None):
        # TODO 1. use the boolvar counter => translate to number

        pysat_clauses = []

        for c in cnf:
            clause = []
            for lit in c:
                # TODO: do something here with negations
                clause.append(lit.name)
            pysat_clauses.append(clause)

        if(output != None):
            try:
                with(output, "w+") as f:
                    # TODO write the clauses
                    f.write(f"c {output}")
                    f.write(f"p cnf {len(get_variables(self))} {len(pysat_clauses)}")
                    for clause in pysat_clauses:
                        f.write(" ".join(clause) + " 0")

            except OSError as err:
                print("OS Error: {0}".format(err))
            finally:
                return pysat_clauses
        return pysat_clauses

# TODO: HOTFIX should be corrected ....
def implies(a, b):
    # both constant
    if type(a) == bool and type(b) == bool:
        return (~a | b)
    # one constant
    if a is True:
        return b
    if a is False:
        return True
    if b is True:
        return True
    if b is False:
        return ~a

    return Operator('->', [a.boolexpr(), b.boolexpr()])

def BoolVar(shape=None):
    if shape is None or shape == 1:
        return BoolVarImpl()
    length = np.prod(shape)
    
    # create base data
    data = np.array([BoolVarImpl() for _ in range(length)]) # repeat new instances
    # insert into custom ndarray
    return NDVarArray(shape, dtype=object, buffer=data)