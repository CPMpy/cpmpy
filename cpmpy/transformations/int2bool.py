"""
Convert integer linear constraints to pseudo-boolean constraints
"""

from typing import List
import cpmpy as cp
from abc import ABC, abstractmethod
from ..expressions.variables import _BoolVarImpl, _IntVarImpl
from ..expressions.core import Comparison, Operator
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

        # check if all variables are in the ivarmap
        for v in vs:
            if type(v) is _IntVarImpl and v.name not in ivarmap: 
                cons = int2bool_make(ivarmap, v, encoding, cpm_out)
                cpm_out.extend(cons)
        
        # we also need to support b -> subexpr
        # where subexpr's transformation is identical to non-reified expr
        # we do this with a special flag
        is_halfreif = False
        if expr.name == "->":
            is_halfreif = True
            b = expr.args[0]  # PAY ATTENTION: we will overwrite expr by the rhs of the ->
            expr = expr.args[1]
            cpm_out.append(b == b) # ensure b is in the model
        
        # now replace intvars with their encoding
        if isinstance(expr, Comparison):
            # special case: lhs is a single intvar
            lhs,rhs = expr.args
            if type(lhs) is _IntVarImpl:
                cons = ivarmap[lhs.name].encode_comparison(expr.name, rhs)
                if is_halfreif:
                    cpm_out.extend([b.implies(c) for c in cons])
                else:
                    cpm_out.extend(cons)
            elif lhs.name == "wsum":
                # if its a wsum, insert encoding of terms
                newweights = []
                newvars = []
                for w,v in zip(*lhs.args):
                    if type(v) is _IntVarImpl:
                        # get list of weights/vars to add
                        ws,vs = ivarmap[v.name].encode_term(w)
                        newweights.extend(ws)
                        newvars.extend(vs)
                    else:
                        newweights.append(w)
                        newvars.append(v)
                # make the new comparison over the new wsum
                expr = Comparison(expr.name, Operator("wsum", (newweights, newvars)), rhs)
                if is_halfreif:
                    cpm_out.append(b.implies(expr))
                else:
                    cpm_out.append(expr)
            elif lhs.name == "sum":
                if len(lhs.args) == 1:
                    assert type(lhs.args[0]) is _IntVarImpl, "Expected single intvar in sum"
                    cons = ivarmap[lhs.args[0].name].encode_comparison(expr.name, rhs)
                    if is_halfreif:
                        cpm_out.extend([b.implies(c) for c in cons])
                    else:
                        cpm_out.extend(cons)
                else:
                    # need to translate to wsum and insert encoding of terms
                    newweights = []
                    newvars = []
                    for v in lhs.args:
                        if type(v) is _IntVarImpl:
                            ws,vs = ivarmap[v.name].encode_term()
                            newweights.extend(ws)
                            newvars.extend(vs)
                        else:
                            newweights.append(1)
                            newvars.append(v)
                    # make the new comparison over the new wsum
                    expr = Comparison(expr.name, Operator("wsum", (newweights, newvars)), rhs)
                    if is_halfreif:
                        cpm_out.append(b.implies(expr))
                    else:
                        cpm_out.append(expr)
            else:
                raise NotImplementedError(f"int2bool: comparison with lhs {lhs} not (yet?) supported")
        else:
            raise NotImplementedError(f"int2bool: non-comparison {expr} not (yet?) supported")

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
            cons = int2bool_make(ivarmap, v, encoding)
            newcons.extend(cons)

    if isinstance(expr, _IntVarImpl):
        ws,vs = ivarmap[expr.name].encode_term()
        return Operator("wsum", (ws, vs)), newcons

    # rest: sum or wsum
    if expr.name == "sum":
        w = [1]*len(expr.args)
        v = expr.args
    elif expr.name == "wsum":
        w,v = expr.args
    else:
        raise NotImplementedError(f"int2bool_wsum: non-sum/wsum expression {expr} not supported")

    new_w, new_v = [], []
    for wi,vi in zip(w,v):
        if type(vi) is _IntVarImpl:
            # get list of weights/vars to add
            ws,vs = ivarmap[vi.name].encode_term(wi)
            new_w.extend(ws)
            new_v.extend(vs)
        else:
            new_w.append(wi)
            new_v.append(vi)

    return Operator("wsum", (new_w, new_v)), newcons


def int2bool_make(ivarmap, v, encoding="auto", expr=None):
    """
    Make the encoding for an integer variable
    """
    # for now, the only encoding is 'direct', so we dont inspect 'expr' at all
    enc = IntVarEncDirect(v)
    ivarmap[v.name] = enc
    return enc.encode_self()

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
        n = v.ub+1-v.lb  # number of Boolean variables
        self.bvars = cp.boolvar(shape=n, name=f"EncDir({v.name})")
    
    def vars(self):
        return self.bvars
    
    def decode(self, vals):
        """
        Decode the Boolean values to the integer value.
        """
        assert sum(vals) == 1, f"Expected exactly one True value in vals: {vals}"
        return sum(i for i,v in enumerate(vals) if v) + self.offset

    def encode_self(self):
        """
        Return consistency constraints

        Variable x has exactly one value from domain,
        so only one of the Boolean variables can be True
        """
        return [cp.sum(self.bvars) == 1]
    
    def encode_comparison(self, op, rhs):
        """
        Encode a comparison over the variable: self <op> rhs
        """
        if op == "==":
            # one yes, hence also rest no
            return [b if i==(rhs-self.offset) else ~b for i,b in enumerate(self.bvars)]
        elif op == "!=":
            return [~self.bvars[rhs - self.offset]]
        elif op == "<":
            # all higher-or-equal values are False
            return list(~self.bvars[rhs-self.offset:])
        elif op == "<=":
            # all higher values are False
            return list(~self.bvars[rhs-self.offset+1:])
        elif op == ">":
            # all lower values are False
            return list(~self.bvars[:rhs-self.offset+1])
        elif op == ">=":
            # all lower-or-equal values are False
            return list(~self.bvars[:rhs-self.offset])
        else:
            raise NotImplementedError(f"int2bool: comparison with op {op} unknown")
        
    def encode_term(self, w=1):
        """
        Rewrite term w*self to terms [w1, w2 ,...]*[bv1, bv2, ...]
        """
        o = self.offset
        return [w*(o+i) for i in range(len(self.bvars))], self.bvars

# TODO: class IntVarEncOrder(IntVarEnc)
# TODO: class IntVarEncLog(IntVarEnc)
