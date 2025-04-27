"""
Convert integer linear constraints to pseudo-boolean constraints
"""

from typing import List
import cpmpy as cp
from abc import ABC, abstractmethod
from ..expressions.variables import _BoolVarImpl, _IntVarImpl
from ..expressions.core import Comparison, Operator, BoolVal
from ..transformations.get_variables import get_variables
from ..expressions.core import Expression


def int2bool(cpm_lst: List[Expression], ivarmap=None, encoding="auto"):
    """
    Convert integer linear constraints to pseudo-boolean constraints
    """
    assert encoding in ("auto", "direct"), "Only auto or direct encoding is supported"
    if ivarmap is None:
        ivarmap = dict()

    cpm_out = []
    for expr in cpm_lst:
        vs = get_variables(expr)
        # skip all Boolean expressions
        if all(isinstance(v, _BoolVarImpl) for v in vs):
            cpm_out.append(expr)
            continue

        # we also need to support b -> subexpr
        # where subexpr's transformation is identical to non-reified expr
        # we do this with a special flag
        if expr.name == "->":
            # PAY ATTENTION: we will overwrite expr by the rhs of the ->
            b = expr.args[0]
            expr = expr.args[1]
            cpm_out.append(b == b)  # ensure b is in the model
        else:
            b = BoolVal(True)

        # now replace intvars with their encoding
        if isinstance(expr, Comparison):
            # special case: lhs is a single intvar
            lhs, rhs = expr.args
            if type(lhs) is _IntVarImpl:
                lhs, cons = int2bool_encode(ivarmap, lhs, encoding)
                cpm_out += cons
                constraints = lhs.encode_comparison(expr.name, rhs)
            elif lhs.name == "sum":
                if len(lhs.args) == 1:
                    assert (
                        type(lhs.args[0]) is _IntVarImpl
                    ), "Expected single intvar in sum"
                    lhs, cons = int2bool_encode(ivarmap, lhs.args[0], encoding)
                    cpm_out += cons
                    constraints = lhs.encode_comparison(expr.name, rhs)
                else:
                    # need to translate to wsum and insert encoding of terms
                    newweights = []
                    newvars = []
                    for v in lhs.args:
                        if type(v) is _IntVarImpl:
                            enc, cons = int2bool_encode(ivarmap, v, encoding)
                            cpm_out += cons
                            ws, vs = enc.encode_term()
                            newweights.extend(ws)
                            newvars.extend(vs)
                        else:
                            newweights.append(1)
                            newvars.append(v)
                    # make the new comparison over the new wsum
                    constraints = [
                        Comparison(
                            expr.name, Operator("wsum", (newweights, newvars)), rhs
                        )
                    ]
            elif lhs.name == "wsum":
                # if its a wsum, insert encoding of terms
                newweights = []
                newvars = []
                for w, v in zip(*lhs.args):
                    if type(v) is _IntVarImpl:
                        # get list of weights/vars to add
                        enc, cons = int2bool_encode(ivarmap, v, encoding)
                        cpm_out += cons
                        ws, vs = enc.encode_term(w)
                        newweights.extend(ws)
                        newvars.extend(vs)
                    else:
                        newweights.append(w)
                        newvars.append(v)
                constraints = [
                    Comparison(expr.name, Operator("wsum", (newweights, newvars)), rhs)
                ]

            else:
                raise NotImplementedError(
                    f"int2bool: comparison with lhs {lhs} not (yet?) supported"
                )

        else:
            raise NotImplementedError(
                f"int2bool: non-comparison {expr} not (yet?) supported"
            )

        # make the new comparison over the new wsum
        cpm_out.extend([b.implies(constraint) for constraint in constraints])

    return cpm_out


def int2bool_wsum(expr: Expression, ivarmap, encoding="auto"):
    """
    Convert a weighted sum to a pseudo-boolean constraint

    Accepts only bool/int/sum/wsum expressions

    Returns (newexpr, newcons)
    """
    vs = get_variables(expr)
    # skip all Boolean expressions
    if all(isinstance(v, _BoolVarImpl) for v in vs):
        return expr, []

    # check if all variables are in the ivarmap, add constraints if not
    newcons = []
    for v in vs:
        if type(v) is _IntVarImpl and v.name not in ivarmap:
            enc, cons = int2bool_encode(ivarmap, v, encoding)
            newcons += cons

    if isinstance(expr, _IntVarImpl):
        ws, vs = ivarmap[expr.name][0].encode_term()
        return Operator("wsum", (ws, vs)), newcons

    # rest: sum or wsum
    if expr.name == "sum":
        w = [1] * len(expr.args)
        v = expr.args
    elif expr.name == "wsum":
        w, v = expr.args
    else:
        raise NotImplementedError(
            f"int2bool_wsum: non-sum/wsum expression {expr} not supported"
        )

    new_w, new_v = [], []
    for wi, vi in zip(w, v):
        if type(vi) is _IntVarImpl:
            # get list of weights/vars to add
            ws, vs = ivarmap[vi.name][0].encode_term(wi)
            new_w.extend(ws)
            new_v.extend(vs)
        else:
            new_w.append(wi)
            new_v.append(vi)

    return Operator("wsum", (new_w, new_v)), newcons


def int2bool_encode(ivarmap, v, encoding="auto"):
    """
    Return encoding of `v` and its domain constraints (if newly encoded)
    """
    if isinstance(v, _BoolVarImpl):
        return v, []
    elif v.name in ivarmap:
        return ivarmap[v.name][0], []
    else:
        enc = IntVarEncDirect(v)
        ivarmap[v.name] = (enc, v)
        return (enc, enc.encode_self())


class IntVarEnc(ABC):
    """
    Abstract base class for integer variable encodings.
    """

    def __init__(self, varname):
        self.varname = varname

    @abstractmethod
    def vars(self):
        """
        Return the Boolean variables in the encoding.
        """
        pass

    def decode(self, vals):
        """
        Decode the Boolean values to the integer value.
        """
        pass

    @abstractmethod
    def encode_self(self):
        """
        Return consistency constraints for the encoding.

        Returns:
            List[Expression]: a list of constraints
        """
        pass

    @abstractmethod
    def encode_comparison(self, op, rhs):
        """
        Encode a comparison over the variable: self <op> rhs

        Args:
            op: The comparison operator ("==", "!=", "<", "<=", ">", ">=")
            rhs: The right-hand side value to compare against

        Returns:
            List[Expression]: a list of constraints
        """
        pass

    @abstractmethod
    def encode_term(self, w=1):
        """
        Encode w*self as a weighted sum of Boolean variables

        Args:
            w: The weight to multiply the variable by

        Returns:
            tuple: (weights, variables) where weights is a list of weights and
                  variables is a list of Boolean variables
        """
        pass


class IntVarEncDirect(IntVarEnc):
    """
    Direct (or sparse or one-hot) encoding of an integer variable.

    Uses a Boolean 'equality' variable for each value in the domain.
    """

    def __init__(self, v):
        super().__init__(v.name)
        self.offset = v.lb
        n = v.ub + 1 - v.lb  # number of Boolean variables
        if n == 0:
            assert False, "Unsat"
        elif n == 1:
            self.bvars = cp.cpm_array([cp.boolvar(name=f"EncDir({v.name})")])
            # self.bvars = cp.cpm_array([BoolVal(True)])
        else:
            self.bvars = cp.boolvar(shape=n, name=f"EncDir({v.name})")

    def vars(self):
        return self.bvars

    def decode(self):
        """
        Decode integer assignment from its encoding's assignment
        """
        for i, v in enumerate([var.value() for var in self.vars()]):
            if v is None:
                return None
            elif v is True:
                return self.offset + i
        raise ValueError(f"The direct encoding was assigned all-false: {self.vars()}")

    def encode_self(self):
        """
        Return consistency constraints

        Variable x has exactly one value from domain,
        so only one of the Boolean variables can be True
        """
        # if len(self.bvars) == 1:  # TODO fix linearize for cp.sum([BoolVal(True)])
        #     return []
        return [cp.sum(self.bvars) == 1]

    def eq(self, d):
        """
        Return a literal whether x==d
        """
        i = d - self.offset
        if i in range(len(self.bvars)):
            return self.bvars[i]
        else:  # don't use try IndexError since negative values wrap
            return cp.BoolVal(False)

    def encode_comparison(self, op, rhs):
        """
        Encode a comparison over the variable: self <op> rhs
        """

        if op == "==":
            # one yes, hence also rest no, if rhs is not in domain will set all to no
            # return [b if i==(rhs-self.offset) else ~b for i,b in enumerate(self.bvars)]
            # return [self.eq(rhs) for i, b in enumerate(self.bvars)]
            return [self.eq(rhs)]
        elif op == "!=":
            return [~self.eq(rhs)]
        elif op == "<":
            # all higher-or-equal values are False
            return list(~self.bvars[rhs - self.offset :])
        elif op == "<=":
            # all higher values are False
            return list(~self.bvars[rhs - self.offset + 1 :])
        elif op == ">":
            # all lower values are False
            return list(~self.bvars[: rhs - self.offset + 1])
        elif op == ">=":
            # all lower-or-equal values are False
            return list(~self.bvars[: rhs - self.offset])
        else:
            raise NotImplementedError(f"int2bool: comparison with op {op} unknown")

    def encode_term(self, w=1):
        """
        Rewrite term w*self to terms [w1, w2 ,...]*[bv1, bv2, ...]
        """
        o = self.offset
        return [w * (o + i) for i in range(len(self.bvars))], self.bvars


# TODO: class IntVarEncOrder(IntVarEnc)
# TODO: class IntVarEncLog(IntVarEnc)
