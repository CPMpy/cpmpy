from ..model import *
from ..expressions import *
from ..variables import *
"""
 Do tseitin transform on list of constraints
"""
def tseitin(constraints):
    # 'constraints' should be list, but lets add two special cases
    if isinstance(constraints, Model):
        # transform model's constraints
        return tseitin(constraints.constraints)
    if isinstance(constraints, Expression):
        # transform expression directly
        return tseitin_transform(constraints)
    
    cnf = []
    for expr in constraints:
        if isinstance(expr, Operator) and expr.name == "or":
            # special case: OR constraint, shortcut to disjunction of subexprs
            subvarcnfs = [tseitin_transform(subexpr) for subexpr in expr.args]
            cnf.append( Operator("or", [v for (v,cnf) in subvarcnfs]) )
            cnf += [clause for (v,cnf) in subvarcnfs for clause in cnf]
        # TODO: special case: AND constraint, flatten into toplevel conjunction
        else:
            newvar, newcnf = tseitin_transform(expr)
            cnf.append(newvar)
            cnf += newcnf
    return cnf

def tseitin_transform(expr):
    # base cases
    if isinstance(expr, BoolVarImpl):
        return (expr, [])

    if not isinstance(expr, Expression) or isinstance(expr, NumVarImpl):
        return (None, [])

    # can't be reached??
    if isinstance(expr, list):
        return (expr[0], [])

    print("inexpr:", expr, type(expr), "\n")
    subvarcnfs = [tseitin_transform(subexpr) for subexpr in expr.args]
    # init cnf as the conjunction of cnf of its subexpressions
    cnf = [clause for (_,subcnf) in subvarcnfs for clause in subcnf]
    subvars = [subvar for (subvar,_) in subvarcnfs]

    if isinstance(expr, Operator):
        # special case: unary -, negate single argument var
        if expr.name == '-':
            return (~subvars[0], cnf)

    implemented = ['and', 'or', '->']
    if not expr.name in implemented:
        raise "Tseitin: Operator '"+self.name+"' not implemented"

    Aux = BoolVarImpl()
    if expr.name == "and":
        cnf.append( Operator("or", [Aux] + [~var for var in subvars]) )
        for var in subvars:
            cnf.append( ~Aux | var )

    if expr.name == "or":
        cnf.append( Operator("or", [~Aux] + [var for var in expr.args]) )
        for var in subvars:
            cnf.append( Aux | ~var )

    if expr.name  == "->":
        # Implication is treated as if it were "or": A -> B <=> ~A or B
        A = subvars[0]
        B = subvars[1]
        cnf = [(~Aux | ~A | B), (Aux | A), (Aux | ~B)]
        
    return Aux, cnf

       
# OLD
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

