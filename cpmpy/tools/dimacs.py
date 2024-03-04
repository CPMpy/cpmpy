import cpmpy as cp

from cpmpy.expressions.variables import _NumVarImpl, _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Operator, Comparison

from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.get_variables import get_variables

"""
This file implements helper functions for exporting CPMpy models from and to DIMACS format.
DIMACS is a textual format to represent CNF problems.
The header of the file should be formatted as `p cnf <n_vars> <n_constraints>
If the number of variables and constraints are not given, it is inferred by the parser.

Each remaining line of the file is formatted as a list of integers.
An integer represents a Boolean variable and a negative Boolean variable is represented using a `-` sign.
"""


def write_dimacs(model, fname=None):
    """
        Writes CPMpy model to DIMACS format
        Uses the "to_cnf" transformation from CPMpy

        # TODO: implement pseudoboolean constraints in to_cnf
        :param model: a CPMpy model
        :fname: optional, file name to write the DIMACS output to
    """

    constraints = toplevel_list(model.constraints)
    constraints = to_cnf(constraints)

    vars = get_variables(constraints)
    mapping = {v : i+1 for i, v in enumerate(vars)}

    out = f"p cnf {len(vars)} {len(constraints)}\n"
    for cons in constraints:

        if isinstance(cons, _BoolVarImpl):
            cons = Operator("or", [cons])

        if isinstance(cons, Operator) and cons.name == "->":
            # implied constraint
            cond, subexpr = cons.args
            assert isinstance(cond, _BoolVarImpl)

            # implied boolean variable, convert to unit clause
            if isinstance(subexpr, _BoolVarImpl):
                subexpr = Operator("or", [subexpr])

            # implied clause, convert to clause
            if isinstance(subexpr, Operator) and subexpr.name == "or":
                cons = Operator("or", [~cond]+subexpr.args)
            else:
                raise ValueError(f"Unknown format for CNF-constraint: {cons}")

        if isinstance(cons, Comparison):
            raise NotImplementedError(f"Pseudo-boolean constraints not (yet) supported!")

        assert isinstance(cons, Operator) and cons.name == "or", f"Should get a clause here, but got {cons}"

        # write clause to cnf format
        ints = []
        for v in cons.args:
            if isinstance(v, NegBoolView):
                ints.append(str(-mapping[v._bv]))
            elif isinstance(v, _BoolVarImpl):
                ints.append(str(mapping[v]))
            else:
                raise ValueError(f"Expected Boolean variable in clause, but got {v} which is of type {type(v)}")

        out += " ".join(ints + ["0"]) + "\n"

    if fname is not None:
        with open(fname, "w") as f:
            f.write(out)

    return out


def read_dimacs(fname, sep=None):
    """
        Read a CPMpy model from a DIMACS formatted file
        If the header is omitted in the file, the number of variables and constraints are inferred.
        :param: fname: the name of the DIMACS file
        :param: sep: optional, separator used in the DIMACS file, will try to infer if None
    """

    m = cp.Model()

    with open(fname, "r") as f:

        lines = f.readlines()
        for i, line in enumerate(lines): # DIMACS allows for comments, skip comment lines
            if line.startswith("p cnf"):
                break
            else:
                assert line.startswith("c"), f"Expected comment on line {i}, but got {line}"

        cnf = "\n".join(lines[i+1:]) # part of file containing clauses

        bvs = []

        negate = False
        clause = []
        for char in cnf:
            if char == "0": # end of clause, add to model and reset
                print(f"End of clause: {clause}, adding to model")
                m += cp.any(clause)
                clause = []

            elif char.isnumeric():  # found Boolvar
                var_idx = int(char)
                if abs(var_idx) >= len(bvs):  # var does not exist yet, create
                    bvs += [cp.boolvar() for _ in range(abs(var_idx) - len(bvs))]
                bv = bvs[var_idx-1]

                clause.append(bv if negate is False else ~bv)
                negate = False # reset negation

            elif char == "-": # negation of next Boolvar
                negate = True

            else: # whitespace, newline...
                pass

    return m





