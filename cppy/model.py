from .expressions import *
from .solver_interfaces import *
from . import *
from copy import copy

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
        # https://en.wikipedia.org/wiki/Tseytin_transformation
        # 1. consider all subformulas
        sub_formulas = []
        new_vars = []

        print("Constraints:")
        for i, c in enumerate(self.constraints):
            # print(i, c)

            is_parent_formula = True

            stack = [c]
            added_vars = []
            added_subformulas = []
            while(len(stack) > 0):
                formula = stack[0]
                del stack[0]
                # ignore the basic case

                if is_int(formula) or is_var(formula):
                    continue
                    #  isinstance(formula, Comparison):
                    # sub_f = formula.subformula()

                # new_args = []
                for arg in formula.args:
                    if is_int(arg) or is_var(arg):
                        continue
                    else:
                        stack.append(arg)

                        # added_subformulas
                        bi = BoolVar()
                        added_subformulas.append((arg, bi ))
                        added_vars.append(bi)

                if is_parent_formula:
                    # create substitution variable for original constraint
                    bi = BoolVar()
                    added_vars.append(bi)

                    # add substitution variable as replacement for constraint
                    sub_formulas.append(bi)
                    added_subformulas.append((formula, bi ))

                    is_parent_formula = False

                    new_vars.append(bi)

            # 3. conjunct all substituations and the substitution for phi
            added_subformulas.sort(key=lambda x: len(str(x[0])))

            for i, (formula, bi) in enumerate(added_subformulas):
                new_formula = copy(formula)
                new_args = []

                for arg in new_formula.args:
                    if is_int(arg) or is_var(arg):
                        new_args.append(arg)
                    else:
                        found = False
                        for j, (formula_j, bj) in enumerate(reversed(added_subformulas[:i])):
                            # TODO replace equality by python "equals" 
                            if formula_j == arg:
                                new_args.append(bj)
                                found= True
                                break
                        if not found:
                            new_args.append(arg)

                new_formula.args = new_args
                new_f1 = implies(new_formula, bi)
                # formula => bi
                new_f2 = implies(bi, new_formula)
                # add new substituted formulas
                sub_formulas.append(new_f1)
                sub_formulas.append(new_f2)
            

        cnf_formulas = []
        for i, formula in enumerate(sub_formulas):
            print(i, formula)

            # all substitutions can be transformed into CNF
            # TODO: transform formula to cnf
            cnf_formula = formula.to_cnf()
            cnf_formulas.append(cnf_formula)

        return cnf_formulas, new_vars

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
