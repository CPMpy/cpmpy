"""Convert integer linear constraints to pseudo-boolean constraints."""

from typing import List
import cpmpy as cp
from abc import ABC, abstractmethod
from ..expressions.variables import _BoolVarImpl, _IntVarImpl
from ..expressions.core import Comparison, Operator
from ..transformations.get_variables import get_variables
from ..expressions.core import Expression


def int2bool(cpm_lst: List[Expression], ivarmap, encoding="auto"):
    """Convert integer linear constraints to pseudo-boolean constraints."""
    assert encoding in (
        "auto",
        "direct",
    ), "Only auto or direct encoding is supported"

    cpm_out = []
    for expr in cpm_lst:
        constraints, domain_constraints = _encode_expr(ivarmap, expr, encoding)
        cpm_out += domain_constraints + constraints
    return cpm_out


def _encode_expr(ivarmap, expr, encoding):
    """Return encoded constraints and root-level constraint (e.g. domain constraints exactly-one, ..)."""
    constraints = []
    domain_constraints = []

    vs = get_variables(expr)
    # skip all Boolean expressions
    if all(isinstance(v, _BoolVarImpl) for v in vs):
        return ([expr], [])

    if expr.name == "->":
        # Encode implication recursively
        p, consequent = expr.args
        constraints, domain_constraints = _encode_expr(
            ivarmap, consequent, encoding
        )
        return (
            [p.implies(constraint) for constraint in constraints],
            domain_constraints + [p == p],  # keep `p` in model
        )
    elif isinstance(expr, Comparison):
        lhs, rhs = expr.args
        if type(lhs) is _IntVarImpl:
            return _encode_comparison(ivarmap, lhs, expr.name, rhs, encoding)
        elif lhs.name == "sum":
            if len(lhs.args) == 1:
                return _encode_comparison(
                    ivarmap, lhs.args[0], expr.name, rhs, encoding
                )
            else:
                return _encode_linear(
                    ivarmap, lhs.args, expr.name, rhs, encoding=encoding
                )
        elif lhs.name == "wsum":
            return _encode_linear(
                ivarmap,
                lhs.args[1],
                expr.name,
                rhs,
                weights=lhs.args[0],
                encoding=encoding,
            )
        else:
            raise NotImplementedError(
                f"int2bool: comparison with lhs {lhs} not (yet?) supported"
            )

    else:
        raise NotImplementedError(
            f"int2bool: non-comparison {expr} not (yet?) supported"
        )

    # make the new comparison over the new wsum
    return (constraints, domain_constraints)


def _encode_int_var(ivarmap, x, encoding="auto"):
    """Return encoding of `x` and its domain constraints (if newly encoded)."""
    if isinstance(x, _BoolVarImpl):
        return x, []
    elif x.name in ivarmap:
        return ivarmap[x.name][0], []
    else:
        x_enc = IntVarEncDirect(x)
        ivarmap[x.name] = (x_enc, x)
        return (x_enc, x_enc.encode_domain_constraint())


def _encode_linear(ivarmap, xs, cmp, rhs, weights=None, encoding="auto"):
    """
    Convert a weighted sum to a pseudo-boolean constraint.

    Accepts only bool/int/sum/wsum expressions

    Returns (newexpr, newcons)
    """
    if weights is None:
        weights = len(xs) * [1]

    pb_weights = []  # PB weights
    pb_literals = []  # PB literals
    domain_constraints = []
    for w, x in zip(weights, xs):
        if isinstance(x, _BoolVarImpl):
            pb_weights += [w]
            pb_literals += [x]
        else:
            x_enc, x_cons = _encode_int_var(ivarmap, x, encoding)
            domain_constraints += x_cons
            ws, xs = x_enc.encode_term(w)
            pb_weights += ws
            pb_literals += list(xs)  # convert from numpy array TODO better way?

    # Revert back to sum if we happen to have constructed one
    if all(w == 1 for w in pb_weights):
        comparison = Comparison(cmp, Operator("sum", pb_literals), rhs)
    else:
        comparison = Comparison(cmp, Operator("wsum", (pb_weights, pb_literals)), rhs)

    return [comparison], domain_constraints


def _encode_comparison(ivarmap, lhs, op, rhs, encoding="auto"):
    enc, domain_constraints = _encode_int_var(ivarmap, lhs, encoding)
    return enc.encode_comparison(op, rhs), domain_constraints


class IntVarEnc(ABC):
    """Abstract base class for integer variable encodings."""

    def __init__(self, varname):
        """Create encoding of integer variable `v`."""
        self.varname = varname

    @abstractmethod
    def vars(self):
        """Return the Boolean variables in the encoding."""
        pass

    def decode(self, vals):
        """Decode the Boolean values to the integer value."""
        pass

    @abstractmethod
    def encode_domain_constraint(self):
        """
        Return domain constraints for the encoding.

        Returns:
            List[Expression]: a list of constraints
        """
        pass

    @abstractmethod
    def encode_comparison(self, op, rhs):
        """
        Encode a comparison over the variable: self <op> rhs.

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
        Encode w*self as a weighted sum of Boolean variables.

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
        """Create direct encoding of integer variable `v`."""
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
        """Return encoding variables."""
        return self.bvars

    def decode(self):
        """Decode integer assignment from its encoding's assignment."""
        for i, v in enumerate([var.value() for var in self.vars()]):
            if v is None:
                return None
            elif v is True:
                return self.offset + i
        raise ValueError(f"The direct encoding was assigned all-false: {self.vars()}")

    def encode_domain_constraint(self):
        """
        Return consistency constraints.

        Variable x has exactly one value from domain,
        so only one of the Boolean variables can be True
        """
        return [cp.sum(self.bvars) == 1]

    def eq(self, d):
        """Return a literal whether x==d."""
        i = d - self.offset
        if i in range(len(self.bvars)):
            return self.bvars[i]
        else:  # don't use try IndexError since negative values wrap
            return cp.BoolVal(False)

    def encode_comparison(self, op, rhs):
        """Encode a comparison over the variable: self <op> rhs."""
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
        """Rewrite term w*self to terms [w1, w2 ,...]*[bv1, bv2, ...]."""
        return [w * (self.offset + i) for i in range(len(self.bvars))], self.bvars


# TODO: class IntVarEncOrder(IntVarEnc)
# TODO: class IntVarEncLog(IntVarEnc)
