"""Convert integer linear constraints to pseudo-boolean constraints."""

import itertools
import math
from abc import ABC, abstractmethod
from typing import List

import cpmpy as cp

from ..expressions.core import BoolVal, Comparison, Expression, Operator
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.variables import _BoolVarImpl, _IntVarImpl, boolvar
from ..expressions.utils import is_int

UNKNOWN_COMPARATOR_ERROR = ValueError("Comparator is not known or should have been simplified by linearize.")
EMPTY_DOMAIN_ERROR = ValueError("Attempted to encode variable with empty domain (which is unsat)")


def int2bool(cpm_lst: List[Expression], ivarmap, encoding="auto"):
    """Convert integer linear constraints to pseudo-boolean constraints. Requires `linearize` transformation."""
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

    # skip all Boolean expressions
    if isinstance(expr, (BoolVal, _BoolVarImpl, DirectConstraint)) or expr.name == "or":
        return [expr], []
    elif expr.name == "->":
        # Encode implication recursively
        p, consequent = expr.args
        constraints, domain_constraints = _encode_expr(ivarmap, consequent, encoding)
        return (
            [p.implies(constraint) for constraint in constraints],
            domain_constraints,
        )
    elif isinstance(expr, Comparison):
        lhs, rhs = expr.args
        # Encode linears with single left-hand term using more efficient `_encode_comparison`
        if type(lhs) is _BoolVarImpl:
            return [expr], []
        elif type(lhs) is _IntVarImpl:
            return _encode_comparison(ivarmap, lhs, expr.name, rhs, encoding)
        elif lhs.name == "sum":
            if len(lhs.args) == 1:
                # even though it seems trivial (to call `_encode_comparison`), using recursion avoids bugs
                return _encode_expr(ivarmap, Comparison(expr.name, lhs.args[0], rhs), encoding)
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
            raise NotImplementedError(f"int2bool: comparison with lhs {lhs} not (yet?) supported")

    else:
        raise NotImplementedError(f"int2bool: non-comparison {expr} not (yet?) supported")


def _encode_int_var(ivarmap, x, encoding):
    """Return encoding of integer variable `x` and its domain constraints (if newly encoded)."""
    if isinstance(x, (BoolVal, _BoolVarImpl)):
        raise TypeError
    elif x.name in ivarmap:  # already encoded
        return ivarmap[x.name], []
    else:
        if encoding == "direct":
            ivarmap[x.name] = IntVarEncDirect(x)
        elif encoding == "order":
            ivarmap[x.name] = IntVarEncOrder(x)
        elif encoding == "binary":
            ivarmap[x.name] = IntVarEncLog(x)
        else:
            raise NotImplementedError(encoding)

        return (ivarmap[x.name], ivarmap[x.name].encode_domain_constraint())


def _encode_lin_expr(ivarmap, xs, weights, encoding, cmp=None):
    """Return encoding of the linear expression with variables `xs` and coefficients `weights`, using `encoding`. If the linear expression occus as part of an `Comparison` (e.g. linear constraint), and the `encoding` is `auto`, then comparator of the Comparison `cmp` can be given to guide the encoding selection."""
    terms = []
    domain_constraints = []
    k = 0
    for w, x in zip(weights, xs):
        # the linear may contain Boolean as well as integer variables
        if is_int(x):
            k += w * x
        elif isinstance(x, _BoolVarImpl):
            terms += [(w, x)]
        elif isinstance(x, _IntVarImpl):
            x_enc, x_cons = _encode_int_var(ivarmap, x, _decide_encoding(x, cmp, encoding))
            domain_constraints += x_cons
            # Encode the value of the integer variable as PB expression `(b_1*c_1) + ... + k`
            new_terms, k_ = x_enc.encode_term(w)
            terms += new_terms
            k += k_
        else:
            raise TypeError(f"Term {w} * {x} for {type(x)}")
    return terms, domain_constraints, k


def _encode_linear(ivarmap, xs, cmp, rhs, encoding, weights=None, check_bounds=True):
    """
    Convert a linear constraint to a pseudo-boolean constraint.

    Returns (newexpr, newcons)
    """
    if weights is None:
        weights = len(xs) * [1]

    # Check for trivial sat/unsat, since pysat does not handle those, and we avoid encoding them
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

        # if trivial sat/unsat, return early
        if value is not None:
            return [value], []

    terms, domain_constraints, k = _encode_lin_expr(ivarmap, xs, weights, encoding, cmp=cmp)
    rhs -= k

    if len(terms) == 0:
        # the unzip trick does not allow default for 0 length iterables
        lhs = 0
    else:
        # term tuples to two separate lists
        pb_weights, pb_literals = zip(*terms)

        # Revert back to `sum` if we happen to have constructed one
        if all(w == 1 for w in pb_weights):
            lhs = Operator("sum", pb_literals)
        else:
            lhs = Operator("wsum", (pb_weights, pb_literals))

    return [Comparison(cmp, lhs, rhs)], domain_constraints


def _encode_comparison(ivarmap, lhs, cmp, rhs, encoding):
    """Encode integer comparison to PB."""
    # TODO encode_expr should only use encode linear and check for "comparison" there
    encoding = _decide_encoding(lhs, cmp, encoding)
    lhs_enc, domain_constraints = _encode_int_var(ivarmap, lhs, encoding)
    constraints = lhs_enc.encode_comparison(cmp, rhs)
    return constraints, domain_constraints


def _decide_encoding(x, cmp=None, encoding="auto"):
    """Decide encoding of `x` via a simple heuristic based on linear comparator and domain size."""
    if encoding != "auto":
        return encoding
    elif _dom_size(x) >= 100:
        # This heuristic is chosen to be small to favour the binary encoding. This is because the PB encoding (e.g. generalized totalizer, ...) of a direct/order encoded PB constraints is quite inefficient unless the AMO/IC side-constraint is taken into account (which is not the case for pysat/pblib/pysdd).
        return "binary"
    elif cmp in (None, "==", "!="):
        return "direct"  # equalities suit the direct encoding
    else:  # inequalities suit the order encoding
        return "order"


class IntVarEnc(ABC):
    """Abstract base class for integer variable encodings."""

    def __init__(self, x, n, name):
        """Create encoding of integer variable `x` with `n` Boolean variables named by `name`."""
        if _dom_size(x) == 0:
            raise EMPTY_DOMAIN_ERROR

        self._x = x  # the encoded integer variable

        if n == 0:
            # `shape=(0,)` raises exception
            self._xs = cp.cpm_array([])
        else:
            # `x`'s encoding variables
            self._xs = boolvar(shape=(n,), name=name)

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
        # Requires |dom(x)| Boolean equality variables
        super().__init__(x, _dom_size(x), f"EncDir({x.name})")

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
        super().__init__(x, _dom_size(x) - 1, f"EncOrd({x.name})")

    def encode_domain_constraint(self):
        """Return order encoding domain constraint (i.e. encoding variables are sorted in descending order e.g. `111000`)."""
        if len(self._xs) <= 1:
            return []
        # Encode implication chain `x>=d -> x>=d-1` (using `zip` to create a sliding window)
        return [curr.implies(prev) for prev, curr in zip(self._xs, self._xs[1:])]

    def _offset(self, d):
        return d - self._x.lb - 1

    def eq(self, d):
        """Return a conjunction whether x==d."""
        if self._x.lb <= d <= self._x.ub:
            return [self.geq(d), ~self.geq(d + 1)]
        else:
            return [BoolVal(False)]

    def geq(self, d):
        """Return a literal whether x>=d."""
        if d <= self._x.lb:
            return BoolVal(True)
        elif d > self._x.ub:
            return BoolVal(False)
        else:
            return self._xs[self._offset(d)]

    def encode_comparison(self, cmp, d):
        if cmp == "==":  # x>=d and x<d+1
            return self.eq(d)
        elif cmp == "!=":  # x<d or x>=d+1
            return [cp.any(~lit for lit in self.eq(d))]
            # TODO we could just use ~cp.all and ~cp.any, but we need to make pysat more robust
            # return [~self.eq(d)]
        elif cmp == ">=":
            return [self.geq(d)]
        elif cmp == "<=":
            return [~self.geq(d + 1)]
        else:
            raise UNKNOWN_COMPARATOR_ERROR

    def encode_term(self, w=1):
        return [(w, b) for b in self._xs], w * self._x.lb


class IntVarEncLog(IntVarEnc):
    """
    Log (or "binary") encoding of an integer variable.

    Uses a Boolean 'bit' variable to represent `x` using the unsigned binary representation offset by its lower bound (e.g. for `x in 5..8`, the assignment `00` maps to `x=5`, and `11` to `x=8`). In other words, it is `k`-offset binary encoding where `k=x.lb`.
    """

    def __init__(self, x):
        """Create binary encoding of integer variable `x`."""
        bits = math.ceil(math.log2(_dom_size(x)))
        super().__init__(x, bits, f"EncBin({x.name})")

    def encode_domain_constraint(self):
        """Return binary encoding domain constraint (i.e. upper bound is respected with `self._x<=self._x.ub`. The lower bound is automatically enforced by offset binary which maps `000.. = self._x.lb`)."""
        # encode directly to avoid bounds check for this seemingly tautological constraint
        return self.encode_comparison("<=", self._x.ub, check_bounds=False)

    def _to_little_endian_offset_binary(self, d):
        """Return offset binary representation of `d` as Booleans in order of increasing significance ("little-endian").

        For more details on offset binary, see the docstring of this class. Note that if e.g. the offset (equal to `self.x.lb`) is 0, then for `d=4` we return `001`. If the offset (i.e. `self.x.lb`) is 1, then it returns `11`, representing 3 in binary, as binary value of 3 + offset of 1 = 4. Note that in this second case, one less bit is returned as we require only 2."""
        # more efficient implementation probably not necessary
        i = self._offset(d)
        if i == 0:
            # otherwise the bitstring will formatted to '0' (which is inconsistent with empty encodings of constants)
            return []
        else:
            # generate bitstring from least to most significant bit
            bitstring = reversed("{0:b}".format(i))
            return (True if bit == "1" else False for bit in bitstring)

    def eq(self, d):
        """Returns a list of literals which in conjunction enforced x==d."""
        if self._x.lb == d == self._x.ub:
            return [BoolVal(True)]
        elif self._x.lb <= d <= self._x.ub:
            # x_i = bit_i for every bit in the representation
            return [
                x if bit else (~x)
                for bit, x in itertools.zip_longest(self._to_little_endian_offset_binary(d), self._xs)
            ]
        else:  # don't use try IndexError since negative values wrap
            return [BoolVal(False)]

    def encode_comparison(self, cmp, d, check_bounds=True):
        if cmp == "==":  # x>=d and x<d+1
            return self.eq(d)
        elif cmp == "!=":  # x<d or x>=d+1
            return [cp.any(~lit for lit in self.eq(d))]
        elif cmp in (">=", "<="):
            # TODO lexicographic encoding might be more effective, but currently we just use the PB encoding
            constraint, domain_constraints = _encode_linear(
                {self._x.name: self}, [self._x], cmp, d, None, check_bounds=check_bounds
            )
            assert domain_constraints == [], (
                f"{self._x} should have already been encoded, so no domain constraints should be returned"
            )
            return constraint
        else:
            raise UNKNOWN_COMPARATOR_ERROR

    def encode_term(self, w=1):
        return [(w * (2**i), b) for i, b in enumerate(self._xs)], w * self._x.lb


def _dom_size(x):
    return x.ub + 1 - x.lb


def replace_int_user_vars(user_vars, ivarmap):
    """Replace integer user vars by the corresponding Booleans from ivarmap.
    Note that it returns a new set, it does not modify the user_vars set."""
    bool_user_vars = set()
    for x in user_vars:
        if isinstance(x, _BoolVarImpl):
            bool_user_vars.add(x)
        else:
            # extends set with encoding variables of `x`
            bool_user_vars.update(ivarmap[x.name].vars())
    return bool_user_vars
