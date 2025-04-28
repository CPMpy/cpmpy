"""Convert integer linear constraints to pseudo-boolean constraints."""

from typing import List
import itertools
import math
import cpmpy as cp
from abc import ABC, abstractmethod
from ..expressions.variables import _BoolVarImpl, _IntVarImpl
from ..expressions.core import Comparison, Operator, BoolVal
from ..transformations.get_variables import get_variables
from ..expressions.core import Expression

UNKNOWN_COMPARATOR_ERROR = ValueError(
    "Comparator is not known or should have been simplified by linearize."
)
EMPTY_DOMAIN_ERROR = ValueError(
    "Attempted to encode variable with empty domain (which is unsat)"
)


def int2bool(cpm_lst: List[Expression], ivarmap, encoding="auto"):
    """Convert integer linear constraints to pseudo-boolean constraints."""
    assert encoding in (
        "auto",
        "direct",
        "order",
        "binary",
    ), "Only auto, direct, order, and binary encoding are supported"

    cpm_out = []
    for expr in cpm_lst:
        constraints, domain_constraints = _encode_expr(ivarmap, expr, encoding)
        cpm_out += domain_constraints + constraints
    return cpm_out


def _encode_expr(ivarmap, expr, encoding):
    """Return encoded constraints and root-level constraints (e.g. domain constraints exactly-one, ..)."""
    constraints = []
    domain_constraints = []

    xs = get_variables(expr)
    # skip all Boolean expressions
    if all(isinstance(x, _BoolVarImpl) for x in xs):
        return ([expr], [])
    elif expr.name == "->":
        # Encode implication recursively
        p, consequent = expr.args
        constraints, domain_constraints = _encode_expr(ivarmap, consequent, encoding)
        return (
            [p.implies(constraint) for constraint in constraints],
            domain_constraints + [p == p],  # keep `p` in model
        )
    elif isinstance(expr, Comparison):
        lhs, rhs = expr.args
        # Encode linears with single left-hand term using more efficient `_encode_comparison`
        if type(lhs) is _IntVarImpl:
            return _encode_comparison(ivarmap, lhs, expr.name, rhs, encoding)
        elif lhs.name == "sum":
            if len(lhs.args) == 1:
                return _encode_comparison(
                    ivarmap, lhs.args[0], expr.name, rhs, encoding
                )
            else:
                return _encode_linear(ivarmap, lhs.args, expr.name, rhs, encoding)
        elif lhs.name == "wsum":
            return _encode_linear(
                ivarmap,
                lhs.args[1],
                expr.name,
                rhs,
                encoding,
                weights=lhs.args[0],
            )
        else:
            raise NotImplementedError(
                f"int2bool: comparison with lhs {lhs} not (yet?) supported"
            )

    else:
        raise NotImplementedError(
            f"int2bool: non-comparison {expr} not (yet?) supported"
        )


def _encode_int_var(ivarmap, x, encoding):
    """Return encoding of `x` and its domain constraints (if newly encoded)."""
    if isinstance(x, BoolVal):
        return x, []
    elif isinstance(x, _BoolVarImpl):
        return x, []
    elif x.name in ivarmap:
        return ivarmap[x.name], []
    else:
        if encoding == "direct":
            x_enc = IntVarEncDirect(x)
        elif encoding == "order":
            x_enc = IntVarEncOrder(x)
        elif encoding == "binary":
            x_enc = IntVarEncBinary(x)
        else:
            raise NotImplementedError(encoding)
        ivarmap[x.name] = x_enc
        return (x_enc, x_enc.encode_domain_constraint())


def _encode_linear(ivarmap, xs, cmp, rhs, encoding, weights=None, check_bounds=True):
    """
    Convert a weighted sum to a pseudo-boolean constraint.

    Accepts only bool/int/sum/wsum expressions

    Returns (newexpr, newcons)
    """
    if weights is None:
        weights = len(xs) * [1]

    # Check for trivial sat/unsat, since pysat does not handle those
    if check_bounds is True:
        # if the bounds are trivially sat/unsat, the value will be set to True/False
        value = None
        lb = sum(w * x.lb if w >= 0 else w * x.ub for w, x in zip(weights, xs))
        ub = sum(w * x.ub if w >= 0 else w * x.lb for w, x in zip(weights, xs))

        if cmp in ("==", "!="):
            if lb == rhs == ub:
                value = BoolVal(True)
            elif rhs not in range(lb, ub + 1):
                value = BoolVal(False)
            if cmp == "!=" and value is not None:
                value = ~value
        elif cmp == "<=":
            if ub <= rhs:
                value = BoolVal(True)
            elif lb > rhs:
                value = BoolVal(False)
        elif cmp == ">=":
            if lb >= rhs:
                value = BoolVal(True)
            elif ub < rhs:
                value = BoolVal(False)
        else:
            raise NotImplementedError

        # if trivial sat/unsat, ensure variables are encoded and return
        if value is not None:
            # TODO realy necessary to add encoding?
            domain_constraints = []
            for x in xs:
                _, x_dom_cons = _encode_int_var(
                    ivarmap, x, _decide_encoding(x, cmp, encoding)
                )
                domain_constraints += x_dom_cons
            return [value], domain_constraints

    terms = []
    domain_constraints = []
    for w, x in zip(weights, xs):
        if isinstance(x, _BoolVarImpl):
            terms += [(w, x)]
        else:
            x_enc, x_cons = _encode_int_var(
                ivarmap, x, _decide_encoding(x, cmp, encoding)
            )
            domain_constraints += x_cons
            new_terms, k = x_enc.encode_term(w)
            terms += new_terms
            rhs -= k

    if len(terms) == 0:
        lhs = 0
    else:
        pb_weights, pb_literals = _unzip(terms)

        # Revert back to sum if we happen to have constructed one
        if all(w == 1 for w in pb_weights):
            lhs = Operator("sum", pb_literals)
        else:
            lhs = Operator("wsum", (pb_weights, pb_literals))

    value = Comparison(cmp, lhs, rhs)

    return [value], domain_constraints


def _encode_comparison(ivarmap, lhs, cmp, rhs, encoding):
    encoding = _decide_encoding(lhs, cmp, encoding)
    lhs_enc, domain_constraints = _encode_int_var(ivarmap, lhs, encoding)
    constraints = lhs_enc.encode_comparison(cmp, rhs)
    return constraints, domain_constraints


def _decide_encoding(x, cmp=None, encoding="auto"):
    """Decide encoding of `x` via on simple heuristic based on linear comparator and domain size."""
    if encoding != "auto":
        return encoding
    elif _dom_size(x) >= 100:
        return "binary"
    elif cmp in (None, "==", "!="):
        return "direct"
    else:  # inequality
        return "order"


class IntVarEnc(ABC):
    """Abstract base class for integer variable encodings."""

    def __init__(self, x, xs):
        """Create encoding of integer variable `x`."""
        if _dom_size(x) == 0:
            raise ValueError("empty domain is unsat")
        self._x = x  # the encoded integer variable
        self._xs = xs  # its encoding variables

    def vars(self):
        """Return the Boolean variables in the encoding."""
        return self._xs

    def _offset(self, d):
        """ "Map domain value `d` to the encoding range (e.g. for the direct encoding to the encoding variable index x==i)"""
        return d - self._x.lb

    def decode(self):
        """Return the integer value of the encoding."""
        terms, k = self.encode_term()
        for weight, lit in terms:
            if lit is None:
                return None
            elif lit.value() is True:
                k += weight
        return k

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
        """Encode `w*self` as a weighted sum of Boolean variables plus a constant.

        Args:
            w: The weight to multiply the variable by

        Returns:
            tuple: (terms, k)  # with `terms` a list of `(weight, literal)` tuples and `k` an integer constant
        """
        pass


class IntVarEncDirect(IntVarEnc):
    """
    Direct (or sparse or one-hot) encoding of an integer variable.

    Uses a Boolean 'equality' variable for each value in the domain.
    """

    def __init__(self, x):
        """Create direct encoding of integer variable `x`."""
        dom_size = _dom_size(x)
        if dom_size == 0:
            raise EMPTY_DOMAIN_ERROR
        elif dom_size == 1:
            xs = cp.cpm_array([cp.boolvar(name=f"EncDir({x.name})")])
        else:
            xs = cp.boolvar(shape=dom_size, name=f"EncDir({x.name})")
        super().__init__(x, xs)

    def encode_domain_constraint(self):
        """
        Return consistency constraints.

        Variable x has exactly one value from domain,
        so only one of the Boolean variables can be True
        """
        return [cp.sum(self._xs) == 1]

    def eq(self, d):
        """Return a literal whether x==d."""
        if self._x.lb <= d <= self._x.ub:
            return self._xs[self._offset(d)]
        else:  # don't use `try .. except IndexError` since negative values wrap!
            return BoolVal(False)

    def encode_comparison(self, op, d):
        if op == "==":
            # one yes, hence also rest no, if rhs is not in domain will set all to no
            return [self.eq(d)]
        elif op == "!=":
            return [~self.eq(d)]
        elif op == "<=":
            # all higher values are False
            return list(~self._xs[self._offset(d + 1) :])
        elif op == ">=":
            # all lower-or-equal values are False
            return list(~self._xs[: self._offset(d)])
        else:
            raise UNKNOWN_COMPARATOR_ERROR

    def encode_term(self, w=1):
        return [(w * i, b) for i, b in enumerate(self._xs)], self._x.lb * w


class IntVarEncOrder(IntVarEnc):
    """
    Order (or thermometer) encoding of an integer variable.

    Uses a Boolean 'inequality' variable for each value in the domain.
    """

    def __init__(self, x):
        """Create order encoding of integer variable `x`."""
        dom_size = _dom_size(x)
        if dom_size == 0:
            raise EMPTY_DOMAIN_ERROR
        elif dom_size == 1:
            xs = cp.cpm_array([])
        elif dom_size == 2:
            xs = cp.cpm_array([cp.boolvar(name=f"EncOrd({x.name})")])
        else:
            # note that the order encoding requires one less variable than the direct encoding
            xs = cp.boolvar(shape=dom_size - 1, name=f"EncOrd({x.name})")
        super().__init__(x, xs)

    def encode_domain_constraint(self):
        """Return order encoding domain constraint (i.e. encoding variables are sorted in descending order)."""
        if len(self._xs) <= 1:
            return []
        # Encode implication chain `x>=d -> x>=d-1` (using sliding window trick)
        return [curr.implies(prev) for prev, curr in zip(self._xs, self._xs[1:])]

    def _offset(self, d):
        return d - self._x.lb - 1

    def geq(self, d):
        """Return a literal whether x>=d."""
        if d <= self._x.lb:
            return BoolVal(True)  # d < lb
        elif d > self._x.ub:
            return BoolVal(False)  # d > ub
        else:
            return self._xs[self._offset(d)]

    def encode_comparison(self, op, rhs):
        if op == "==":  # x>=d and x<d+1
            return [self.geq(rhs), ~self.geq(rhs + 1)]
        elif op == "!=":  # x<d or x>=d+1
            return [cp.any([~self.geq(rhs), self.geq(rhs + 1)])]
        elif op == ">=":
            return [self.geq(rhs)]
        elif op == "<=":
            return [~self.geq(rhs + 1)]
        else:
            raise UNKNOWN_COMPARATOR_ERROR

    def encode_term(self, w=1):
        """Rewrite term w*self to terms [w1, w2 ,...]*[bv1, bv2, ...]."""
        return [(w, b) for b in self._xs], w * self._x.lb


class IntVarEncBinary(IntVarEnc):
    """
    Binary (or "log") encoding of an integer variable.

    Uses a Boolean 'bit' variable to represent `x` using the unsigned binary represenntation offset by its lower bound (e.g. for x in 5..8, the assignment 00 maps to x=5)
    """

    def __init__(self, x):
        """Create binary encoding of integer variable `x`."""
        dom_size = _dom_size(x)
        if dom_size == 0:
            raise EMPTY_DOMAIN_ERROR

        bits = math.ceil(math.log2(dom_size))
        if bits == 0:
            xs = cp.cpm_array([])
        elif bits == 1:
            xs = cp.cpm_array([cp.boolvar(name=f"EncBin({x.name})")])
        else:
            xs = cp.boolvar(shape=bits, name=f"EncBin({x.name})")
        super().__init__(x, xs)

    def encode_domain_constraint(self):
        """Return binary encoding domain constraint (i.e. upper bound is respected with `self._x<=self._x.ub`. The lower bound is automatically enforced by offset binary which maps `000.. = self._x.lb`)."""
        # encode directly to avoid bounds check for this seemingly tautological constraint
        return self.encode_comparison("<=", self._x.ub, check_bounds=False)

    def _bitstring(self, d):
        """Return offset binary representation of `d` as Booleans in order of increasing signicance (e.g. `4` return `001`)."""
        # more efficient implementation probably not necessary
        i = self._offset(d)
        if (
            i == 0
        ):  # otherwise the bitstring will formatted to '0' (which is inconsistent with empty encodings of constants)
            return []
        else:
            return (
                True if bit == "1" else False for bit in reversed("{0:b}".format(i))
            )

    def eq(self, d):
        """Returns a list of literals which in conjunction enforced x==d."""
        if self._x.lb == d == self._x.ub:
            return [BoolVal(True)]
        elif self._x.lb <= d <= self._x.ub:
            # x_i = bit_i for every bit in the representation
            return [
                x if bit else (~x)
                for bit, x in itertools.zip_longest(self._bitstring(d), self._xs)
            ]
        else:  # don't use try IndexError since negative values wrap
            return [BoolVal(False)]

    def encode_comparison(self, cmp, d, check_bounds=True):
        if cmp == "==":  # x>=d and x<d+1
            return self.eq(d)
        elif cmp == "!=":  # x<d or x>=d+1
            return [cp.any(~x for x in self.eq(d))]
        elif cmp in (">=", "<="):
            # TODO lexicographic encoding might be more efficitive, but currently we just use the PB encoding
            constraint, domain_constraints = _encode_linear(
                {self._x.name: self}, [self._x], cmp, d, None, check_bounds=check_bounds
            )
            assert (
                domain_constraints == []
            ), f"{self._x} should have already been encoded, so no domain constraints should be returned"
            return constraint
        else:
            raise UNKNOWN_COMPARATOR_ERROR

    def encode_term(self, w=1):
        return [(w * (2**i), b) for i, b in enumerate(self._xs)], w * self._x.lb


def _unzip(iterable):
    return zip(*iterable)


def _dom_size(x):
    return x.ub + 1 - x.lb
