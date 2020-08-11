from ..model import *
from ..expressions import *
from ..variables import *
"""
 Model transformation, read-only
 Returns an (ordered by appearance) list of all variables in the model
"""
def tseitin(constraints):
    transforms = []
    for formula in constraints:
        _, tr = tseitin_transform(formula)
        transforms.append((formula, tr))

    return transforms

def tseitin_transform(expr):
    # base cases
    print(expr, type(expr), "\n")
    if isinstance(expr, BoolVarImpl):
        return (expr, [])

    if not isinstance(expr, Expression) or isinstance(expr, NumVarImpl):
        return (None, [])

    if isinstance(expr, list):
        return (expr[0], [])

    if isinstance(expr, Expression):
        A, expr1 = tseitin_transform(expr.args[0])

        if len(expr.args) == 2:
            # TODO add tseitin transformation recursion here too ? 
            B, expr2 = tseitin_transform(expr.args[1])
        elif len(expr.args) > 2:
            if expr.name == "and" or expr.name == "&":
                B, expr2 = tseitin_transform(expr.args[1] & expr.args[2:])
            elif expr.name == "or" or expr.name == "|" :
                B, expr2 = tseitin_transform(expr.args[1] | expr.args[2:])
            else:
                raise "Case not handled"
        else:
            raise "Case not handled"

        C = BoolVarImpl()
        print(expr, A, B, C)
        if expr.name == "and" or expr.name == "&":
            cnf_expr =  [( ~A | ~B | C), (A | ~C), (B | ~C)]

        if expr.name == "or" or expr.name == "|" :
            cnf_expr = [( ~A | ~B | ~C), (A | C), (B | C)]

        if expr.name  == "->":
            # Implication is treated as if it were "or": A -> B <=> ~A or B
            cnf_expr = [( A | ~B | ~C), (~A | C), (B | C)]
        
        if len(expr1) > 0:
            cnf_expr += expr1
        if len(expr2) > 0:
            cnf_expr += expr2

        return C, cnf_expr
       
def to_cnf(constraints):
    # https://en.wikipedia.org/wiki/Tseytin_transformation
    # 1. consider all subformulas
    sub_formulas = []
    new_vars = []

    print("Constraints:")
    for i, c in enumerate(constraints):
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
            # print(len(sub_formulas)+1, new_formula)

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

def cnf_to_pysat(cnf, output = None):
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

