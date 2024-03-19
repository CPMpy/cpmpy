import cpmpy as cp

from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Operator, Comparison

from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.get_variables import get_variables

import re

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

        if not (isinstance(cons, Operator) and cons.name == "or"):
            raise NotImplementedError(f"Unsupported constraint {cons}")

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


def read_dimacs(fname):
    """
        Read a CPMpy model from a DIMACS formatted file
        If the number of variables and constraints is not present in the header, they are inferred.
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
        txt_clauses = re.split(r"\n* \n*0", cnf) # clauses end with ` 0` but can have arbitrary newlines

        for txt_ints in txt_clauses:
            if txt_ints is None or len(txt_ints.strip()) == 0:
                continue # empty clause or weird format

            clause = []
            ints = [int(idx.strip()) for idx in txt_ints.split(" ") if len(idx.strip())]

            for i in ints:
                if abs(i) >= len(bvs):  # var does not exist yet, create
                    bvs += [cp.boolvar() for _ in range(abs(i) - len(bvs))]
                bv = bvs[abs(i) - 1]
                clause.append(bv if i > 0 else ~bv)

            m += cp.any(clause)

    return m





