import cpmpy as cp

from cpmpy.expressions.variables import _NumVarImpl, _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Operator, Comparison

from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.get_variables import get_variables

"""
This file implements helper functions for exporting CPMpy models from and to .cnf format.
"""


def write_cnf(model, fname=None):
    """
        Writes CPMpy model to .cnf format
        Uses the "to_cnf" transformation from CPMpy

        # TODO: implement pseudoboolean constraints in to_cnf
        :param model: a CPMpy model
        :fname: optional, file name to write the cnf output to
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


def read_cnf(fname, sep=None):
    """
        Read a CPMpy model from a .cnf formatted file
        :param: fname: the name of the .cnf
        :param: sep: optional, separator used in the .cnf file, will try to infer if None
    """

    m = cp.Model()

    with open(fname, "r") as f:

        line = f.readline().strip()

        if sep is None:
            if "\t" in line: sep = "\t"
            elif " " in line: sep =" "
            else: raise ValueError(f"Unknown separator, got line {line}")

        assert line[0] == "p", f"The header of a cnf file should be formatted as 'p cnf ..., but got {line}"
        if sep is None:
            sep = line[1]
        p, fmt, *_ = line.split(sep)

        bvs = []

        while 1:
            line = f.readline()
            if line is None or len(line) <= 0:
                break

            str_idxes = line.strip().split(sep)
            clause = []
            for i, var_idx in enumerate(map(int, str_idxes)):
                if abs(var_idx) >= len(bvs): # var does not exist yet, create
                    bvs += [cp.boolvar() for _ in range(abs(var_idx)- len(bvs))]

                if var_idx > 0: # boolvar
                    clause.append(bvs[var_idx-1])
                elif var_idx < 0: # neg boolvar
                    clause.append(~bvs[(-var_idx)-1])
                elif var_idx == 0: # end of clause
                    assert i == len(str_idxes)-1, f"Can only have '0' at end of a clause, but got 0 at index {i} in clause {str_idxes}"
            m += cp.any(clause)

    return m





