#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## expressions.py
##
"""
    The `Expression` superclass and common subclasses `Expression` and `Operator`.
    
    None of these objects should be directly created, they are automatically created through operator
    overloading on variables and expressions.

    Here is a list of standard python operators and what object (with what expr.name) it creates:

    Comparisons:

    - x == y        Comparison("==", x, y)
    - x != y        Comparison("!=", x, y)
    - x < y         Comparison("<", x, y)
    - x <= y        Comparison("<=", x, y)
    - x > y         Comparison(">", x, y)
    - x >= y        Comparison(">=", x, y)

    Mathematical operators:

    - -x            Operator("-", [x])
    - x + y         Operator("sum", [x,y])
    - sum([x,y,z])  Operator("sum", [x,y,z])
    - sum([c0*x, c1*y, c2*z])  Operator("wsum", [[c0,c1,c2],[x,y,z]])
    - x - y         Operator("sum", [x,-y])
    - x * y         Operator("mul", [x,y])
    - x / y         Operator("div", [x,y])
    - x % y         Operator("mod", [x,y])
    - x ** y        Operator("pow", [x,y])

    Logical operators:

    - x & y         Operator("and", [x,y])
    - x | y         Operator("or", [x,y])
    - ~x            Operator("not", [x])
                    or NegBoolView(x) in case x is a Boolean variable
    - x ^ y         Xor([x,y])  # a global constraint

    Python has no built-in operator for __implication__ that can be overloaded.
    CPMpy hence has a function 'implies()' that can be called:

    - x.implies(y)  Operator("->", [x,y])

    Apart from operator overloading, expressions implement two important functions:

    - `is_bool()`   which returns whether the __return type__ of the expression is Boolean.
                    If it does, the expression can be used as top-level constraint
                    or in logical operators.

    - `value()`     computes the value of this expression, by calling .value() on its
                    subexpressions and doing the appropriate computation
                    this is used to conveniently print variable values, objective values
                    and any other expression value (e.g. during debugging).
    
    ===============
    List of classes
    ===============
    .. autosummary::
        :nosignatures:

        Expression
        Comparison
        Operator
"""
import copy
import warnings
from types import GeneratorType
import numpy as np


from .utils import is_num, is_any_list, flatlist, argval, get_bounds, is_boolexpr, is_true_cst, is_false_cst
from ..exceptions import IncompleteFunctionError, TypeError


class Expression(object):
    """
    An Expression represents a symbolic function with a self.name and self.args (arguments)

    Each Expression is considered to be a function whose value can be used
      in other expressions

    Expressions may implement:
    - is_bool():    whether its return type is Boolean
    - value():      the value of the expression, default None
    - implies(x):   logical implication of this expression towards x
    - __repr__():   for pretty printing the expression
    - any __op__ python operator overloading
    """

    def __init__(self, name, arg_list):
        self.name = name

        if isinstance(arg_list, (tuple, GeneratorType)):
            arg_list = list(arg_list)
        elif isinstance(arg_list, np.ndarray):
            # must flatten
            arg_list = arg_list.reshape(-1)
        for i in range(len(arg_list)):
            if isinstance(arg_list[i], np.ndarray):
                # must flatten
                arg_list[i] = arg_list[i].reshape(-1)

        assert (is_any_list(arg_list)), "_list_ of arguments required, even if of length one e.g. [arg]"
        self.args = arg_list

    def __repr__(self):
        strargs = []
        for arg in self.args:
            if isinstance(arg, np.ndarray):
                # flatten
                strarg = ",".join(map(str, arg.flat))
                strargs.append(f"[{strarg}]")
            else:
                strargs.append(f"{arg}")
        return "{}({})".format(self.name, ",".join(strargs))

    def __hash__(self):
        return hash(self.__str__())

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
            Default: yes
        """
        return True

    def value(self):
        return None # default

    def get_bounds(self):
        if self.is_bool():
            return 0, 1 #default for boolean expressions
        raise NotImplementedError(f"`get_bounds` is not implemented for type {self}")

    # keep for backwards compatibility
    def deepcopy(self, memodict={}):
        warnings.warn("Deprecated, use copy.deepcopy() instead, will be removed in stable version", DeprecationWarning)
        return copy.deepcopy(self, memodict)

    # implication constraint: self -> other
    # Python does not offer relevant syntax...
    # for double implication, use equivalence self == other
    def implies(self, other):
        # other constant
        if is_true_cst(other):
            return BoolVal(True)
        if is_false_cst(other):
            return ~self
        return Operator('->', [self, other])

    # Comparisons
    def __eq__(self, other):
        # BoolExpr == 1|true|0|false, common case, simply BoolExpr
        if self.is_bool() and is_num(other):
            if other is True or other == 1:
                return self
            if other is False or other == 0:
                return ~self
        return Comparison("==", self, other)

    def __ne__(self, other):
        return Comparison("!=", self, other)

    def __lt__(self, other):
        return Comparison("<", self, other)

    def __le__(self, other):
        return Comparison("<=", self, other)

    def __gt__(self, other):
        return Comparison(">", self, other)

    def __ge__(self, other):
        return Comparison(">=", self, other)

    # Boolean Operators
    # Implements bitwise operations & | ^ and ~ (and, or, xor, not)
    def __and__(self, other):
        # some simple constant removal
        if is_true_cst(other):
            return self
        if is_false_cst(other):
            return BoolVal(False)
        # catch beginner mistake
        if is_num(other):
            raise TypeError(f"{self}&{other} is not valid because {other} is a number, did you forgot to put brackets? E.g. always write (x==2)&(y<5).")
        return Operator("and", [self, other])

    def __rand__(self, other):
        # some simple constant removal
        if is_true_cst(other):
            return self
        if is_false_cst(other):
            return BoolVal(False)
        # catch beginner mistake
        if is_num(other):
            raise TypeError(f"{other}&{self} is not valid because {other} is a number, did you forgot to put brackets? E.g. always write (x==2)&(y<5).")
        return Operator("and", [other, self])

    def __or__(self, other):
        # some simple constant removal
        if is_true_cst(other):
            return BoolVal(True)
        if is_false_cst(other):
            return self
        # catch beginner mistake
        if is_num(other):
            raise TypeError(f"{self}|{other} is not valid because {other} is a number, did you forgot to put brackets? E.g. always write (x==2)|(y<5).")
        return Operator("or", [self, other])

    def __ror__(self, other):
        # some simple constant removal
        if is_true_cst(other):
            return BoolVal(True)
        if is_false_cst(other):
            return self
        # catch beginner mistake
        if is_num(other):
            raise TypeError(f"{other}|{self} is not valid because {other} is a number, did you forgot to put brackets? E.g. always write (x==2)|(y<5).")
        return Operator("or", [other, self])

    def __xor__(self, other):
        # some simple constant removal
        if is_true_cst(other):
            return ~self
        if is_false_cst(other):
            return self
        # avoid cyclic import
        from .globalconstraints import Xor
        return Xor([self, other])

    def __rxor__(self, other):
        # some simple constant removal
        if is_true_cst(other):
            return ~self
        if is_false_cst(other):
            return self
        # avoid cyclic import
        from .globalconstraints import Xor
        return Xor([other, self])

    # Mathematical Operators, including 'r'everse if it exists
    # Addition
    def __add__(self, other):
        if is_num(other) and other == 0:
            return self
        return Operator("sum", [self, other])

    def __radd__(self, other):
        if is_num(other) and other == 0:
            return self
        return Operator("sum", [other, self])

    # substraction
    def __sub__(self, other):
        # if is_num(other) and other == 0:
        #     return self
        # return Operator("sub", [self, other])
        return self.__add__(-other)

    def __rsub__(self, other):
        # if is_num(other) and other == 0:
        #     return -self
        # return Operator("sub", [other, self])
        return (-self).__radd__(other)
    
    # multiplication, puts the 'constant' (other) first
    def __mul__(self, other):
        if is_num(other) and other == 1:
            return self
        # this unnecessarily complicates wsum creation
        #if is_num(other) and other == 0:
        #    return other
        return Operator("mul", [self, other])

    def __rmul__(self, other):
        if is_num(other) and other == 1:
            return self
        # this unnecessarily complicates wsum creation
        #if is_num(other) and other == 0:
        #    return other
        return Operator("mul", [other, self])

    # matrix multipliciation TODO?
    #object.__matmul__(self, other)

    # other mathematical ones
    def __truediv__(self, other):
        warnings.warn("We only support floordivision, use // in stead of /", SyntaxWarning)
        return self.__floordiv__(other)

    def __rtruediv__(self, other):
        warnings.warn("We only support floordivision, use // in stead of /", SyntaxWarning)
        return self.__rfloordiv__(other)

    def __floordiv__(self, other):
        if is_num(other) and other == 1:
            return self
        return Operator("div", [self, other])

    def __rfloordiv__(self, other):
        return Operator("div", [other, self])

    def __mod__(self, other):
        return Operator("mod", [self, other])

    def __rmod__(self, other):
        return Operator("mod", [other, self])

    def __pow__(self, other, modulo=None):
        assert (modulo is None), "Power operator: modulo not supported"
        if other == 0:
            return 1
        elif other == 1:
            return self
        return Operator("pow", [self, other])

    def __rpow__(self, other, modulo=None):
        assert (modulo is None), "Power operator: modulo not supported"
        return Operator("pow", [other, self])

    # Not implemented: (yet?)
    #object.__divmod__(self, other)

    # unary mathematical operators
    def __neg__(self):
        # special case, -(w*x) -> -w*x
        if self.name == 'mul' and is_num(self.args[0]):
            return Operator(self.name, [-self.args[0], self.args[1]])
        elif self.name == 'wsum':
            # negate the constant weights
            return Operator(self.name, [[-a for a in self.args[0]], self.args[1]])
        return Operator("-", [self])

    def __pos__(self):
        return self

    def __abs__(self):
        from .globalfunctions import Abs
        return Abs(self)

    def __invert__(self):
        if not (is_boolexpr(self)):
            raise TypeError("Not operator is only allowed on boolean expressions: {0}".format(self))
        return Operator("not", [self])


class BoolVal(Expression):
    """
        Wrapper for python or numpy BoolVals
    """

    def __init__(self, arg):
        assert is_true_cst(arg) or is_false_cst(arg)
        super(BoolVal, self).__init__("boolval", [bool(arg)])

    def value(self):
        return self.args[0]

    def __invert__(self):
        return BoolVal(not self.args[0])

    def __bool__(self):
        """Called to implement truth value testing and the built-in operation bool(), return stored value"""
        return self.args[0]


class Comparison(Expression):
    """Represents a comparison between two sub-expressions
    """
    allowed = {'==', '!=', '<=', '<', '>=', '>'}

    def __init__(self, name, left, right):
        assert (name in Comparison.allowed), f"Symbol {name} not allowed"
        super().__init__(name, [left, right])

    def __repr__(self):
        if all(isinstance(x, Expression) for x in self.args):
            return "({}) {} ({})".format(self.args[0], self.name, self.args[1]) 
        # if not: prettier printing without braces
        return "{} {} {}".format(self.args[0], self.name, self.args[1]) 

    # return the value of the expression
    # optional, default: None
    def value(self):
        arg_vals = [argval(a) for a in self.args]
        if any(a is None for a in arg_vals): return None
        if   self.name == "==": return arg_vals[0] == arg_vals[1]
        elif self.name == "!=": return arg_vals[0] != arg_vals[1]
        elif self.name == "<":  return arg_vals[0] < arg_vals[1]
        elif self.name == "<=": return arg_vals[0] <= arg_vals[1]
        elif self.name == ">":  return arg_vals[0] > arg_vals[1]
        elif self.name == ">=": return arg_vals[0] >= arg_vals[1]
        return None # default


class Operator(Expression):
    """
    All kinds of mathematical and logical operators on expressions

    Convention for 2-ary operators: if one of the two is a constant,
    it is stored first (as expr[0]), this eases weighted sum detection
    """
    allowed = {
        #name: (arity, is_bool)       arity 0 = n-ary, min 2
        'and': (0, True),
        'or':  (0, True),
        '->':  (2, True),
        'not': (1, True),
        'sum': (0, False),
        'wsum': (2, False),
        'sub': (2, False), # x - y
        'mul': (2, False),
        'div': (2, False),
        'mod': (2, False),
        'pow': (2, False),
        '-':   (1, False), # -x
    }
    printmap = {'sum': '+', 'sub': '-', 'mul': '*', 'div': '//'}

    def __init__(self, name, arg_list):
        # sanity checks
        assert (name in Operator.allowed), "Operator {} not allowed".format(name)
        arity, is_bool_op = Operator.allowed[name]
        if is_bool_op:
            #only boolean arguments allowed
            for arg in arg_list:
                if not is_boolexpr(arg):
                    raise TypeError("{}-operator only accepts boolean arguments, not {}".format(name,arg))
        if arity == 0:
            arg_list = flatlist(arg_list)
            assert (len(arg_list) >= 1), "Operator: n-ary operators require at least one argument"
        else:
            assert (len(arg_list) == arity), "Operator: {}, number of arguments must be {}".format(name, arity)

        # automatic weighted sum (wsum) creation:
        # if all args are an expression (not a constant)
        #    and one of the args is a wsum,
        #                    or a product of a constant and an expression,
        # then create a wsum of weights,expressions over all
        if name == 'sum' and \
                all(not is_num(a) for a in arg_list) and \
                any(_wsum_should(a) for a in arg_list):
            we = [_wsum_make(a) for a in arg_list]
            w = [wi for w, _ in we for wi in w]
            e = [ei for _, e in we for ei in e]
            name = 'wsum'
            arg_list = [w, e]

        # we have the requirement that weighted sums are [weights, expressions]
        if name == 'wsum':
            assert all(is_num(a) for a in arg_list[0]), "wsum: arg0 has to be all constants but is: "+str(arg_list[0])

        # small cleanup: nested n-ary operators are merged into the toplevel
        # (this is actually against our design principle of creating
        #  expressions the way the user wrote them)
        if arity == 0:
            i = 0 # length can change
            while i < len(arg_list):
                if isinstance(arg_list[i], Operator) and arg_list[i].name == name:
                    # merge args in at this position
                    l = len(arg_list[i].args)
                    arg_list[i:i+1] = arg_list[i].args
                    i += l
                i += 1

        # another cleanup, translate -(v*c) to v*-c
        if hasattr(arg_list[0], 'name'):
            if name == '-' and arg_list[0].name == 'mul' and len(arg_list[0].args) == 2:
                mul_args = arg_list[0].args
                if is_num(mul_args[0]):
                    name = 'mul'
                    arg_list = (-mul_args[0], mul_args[1])
                elif is_num(mul_args[1]):
                    name = 'mul'
                    arg_list = (mul_args[0], -mul_args[1])

        super().__init__(name, arg_list)

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return Operator.allowed[self.name][1]
    
    def __repr__(self):
        printname = self.name
        if printname in Operator.printmap:
            printname = Operator.printmap[printname]

        # special cases
        if self.name == '-': # unary -
            return "-({})".format(self.args[0])

        # weighted sum
        if self.name == 'wsum':
            return f"sum({self.args[0]} * {self.args[1]})"

        # infix printing of two arguments
        if len(self.args) == 2:
            # bracketed printing of non-constants
            def wrap_bracket(arg):
                if isinstance(arg, Expression):
                    return f"({arg})"
                return arg
            return "{} {} {}".format(wrap_bracket(self.args[0]),
                                     printname,
                                     wrap_bracket(self.args[1]))

        return "{}({})".format(self.name, self.args)

    def value(self):
        if self.name == "wsum":
            # wsum: arg0 is list of constants, no .value() use as is
            arg_vals = [self.args[0], [argval(arg) for arg in self.args[1]]]
        else:
            arg_vals = [argval(arg) for arg in self.args]


        if any(a is None for a in arg_vals): return None
        # non-boolean
        elif self.name == "sum": return sum(arg_vals)
        elif self.name == "wsum": return sum(arg_vals[0]*np.array(arg_vals[1]))
        elif self.name == "mul": return arg_vals[0] * arg_vals[1]
        elif self.name == "sub": return arg_vals[0] - arg_vals[1]
        elif self.name == "mod": return arg_vals[0] % arg_vals[1]
        elif self.name == "pow": return arg_vals[0] ** arg_vals[1]
        elif self.name == "-":   return -arg_vals[0]
        elif self.name == "div":
            try:
                return arg_vals[0] // arg_vals[1]
            except ZeroDivisionError:
                raise IncompleteFunctionError(f"Division by zero during value computation for expression {self}")

        # boolean
        elif self.name == "and": return all(arg_vals)
        elif self.name == "or" : return any(arg_vals)
        elif self.name == "->": return (not arg_vals[0]) or arg_vals[1]
        elif self.name == "not": return not arg_vals[0]

        return None # default

    def get_bounds(self):
        """
        Returns an estimate of lower and upper bound of the expression.
        These bounds are safe: all possible values for the expression agree with the bounds.
        These bounds are not tight: it may be possible that the bound itself is not a possible value for the expression.
        """
        if self.is_bool():
            return 0, 1 #boolean
        elif self.name == 'mul':
            lb1, ub1 = get_bounds(self.args[0])
            lb2, ub2 = get_bounds(self.args[1])
            bounds = [lb1 * lb2, lb1 * ub2, ub1 * lb2, ub1 * ub2]
            return min(bounds), max(bounds)
        elif self.name == 'sum':
            lbs, ubs = zip(*[get_bounds(x) for x in self.args])
            return sum(lbs), sum(ubs)
        elif self.name == 'wsum':
            weights, vars = self.args
            var_bounds = np.array([get_bounds(arg) for arg in vars]).T
            bounds = var_bounds * weights
            return bounds.min(axis=0).sum(), bounds.max(axis=0).sum()  # for every column is axis=0...
        elif self.name == 'sub':
            lb1, ub1 = get_bounds(self.args[0])
            lb2, ub2 = get_bounds(self.args[1])
            return lb1-ub2, ub1-lb2
        elif self.name == 'div':
            lb1, ub1 = get_bounds(self.args[0])
            lb2, ub2 = get_bounds(self.args[1])
            if lb2 <= 0 <= ub2:
                raise ZeroDivisionError("division by domain containing 0 is not supported")
            bounds = [lb1 // lb2, lb1 // ub2, ub1 // lb2, ub1 // ub2]
            return min(bounds), max(bounds)
        elif self.name == 'mod':
            lb1, ub1 = get_bounds(self.args[0])
            lb2, ub2 = get_bounds(self.args[1])
            if lb2 <= 0 <= ub2:
                raise ZeroDivisionError("% by domain containing 0 is not supported")
            elif ub2 < 0:
                return lb2 + 1, 0
            elif lb2 > 0:
                return 0, ub2 - 1
        elif self.name == 'pow':
            lb1, ub1 = get_bounds(self.args[0])
            lb2, ub2 = get_bounds(self.args[1])
            if lb2 < 0:
                raise NotImplementedError("Power operator: For integer values, exponent must be non-negative")
            bounds = [lb1**lb2, lb1**ub2, ub1**lb2, ub1**ub2]
            if lb1 < 0 and 0 < ub2:  
                # The lower and upper bounds depend on either the largest or the second largest exponent 
                # value when the base term can be negative. 
                # E.g., (-2)^2 is positive, but (-2)^1 is negative, so for (-2)^[0,2] we also need to add (-2)^1.
                bounds += [lb1 ** (ub2 - 1), ub1 ** (ub2 - 1)] 
                # This approach is safe but not tight (e.g., [-2,-1]^2 will give (-2,4) as range instead of [1,4]).
            return min(bounds), max(bounds)

        elif self.name == '-':
            lb1, ub1 = get_bounds(self.args[0])
            return -ub1, -lb1
        
        raise ValueError(f"Bound requested for unknown expression {self}, please report bug on github")
        
def _wsum_should(arg):
    """ Internal helper: should the arg be in a wsum instead of sum

    True if the arg is already a wsum,
    or if it is a product of a constant and an expression 
    (negation '-' does not mean it SHOULD be a wsum, because then
     all substractions are transformed into less readable wsums)
    """
    return isinstance(arg, Operator) and \
           (arg.name == 'wsum' or \
            (arg.name == 'mul' and len(arg.args) == 2 and \
             any(is_num(a) for a in arg.args)
            ) )

def _wsum_make(arg):
    """ Internal helper: prep the arg for wsum

    returns ([weights], [expressions]) where 'weights' are constants
    call only if arg is Operator
    """
    if arg.name == 'wsum':
        return arg.args
    elif arg.name == 'sum':
        return [1]*len(arg.args), arg.args
    elif arg.name == 'mul':
        if is_num(arg.args[0]):
            return [arg.args[0]], [arg.args[1]]
        elif is_num(arg.args[1]):
            return [arg.args[1]], [arg.args[0]]
        # else falls through to default below
    elif arg.name == '-':
        return [-1], [arg.args[0]]
    # default
    return [1], [arg]
