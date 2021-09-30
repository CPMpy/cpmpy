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

    - \-x            Operator("-", [x])
    - abs(x)        Operator("abs", [x])
    - x + y         Operator("sum", [x,y])
    - sum([x,y,z])  Operator("sum", [x,y,z])
    - x - y         Operator("sum", [x,-y])
    - x * y         Operator("mul", [x,y])
    - x / y         Operator("div", [x,y])
    - x % y         Operator("mod", [x,y])
    - x ** y        Operator("pow", [x,y])

    Logical operators:

    - x & y         Operator("and", [x,y])
    - x | y         Operator("or", [x,y])
    - x ^ y         Operator("xor", [x,y])

    Finally there are two special cases for logical operators 'implies' and '~/not'.
    
    Python has no built-in operator for __implication__ that can be overloaded.
    CPMpy hence has a function 'implies()' that can be called:

    - x.implies(y)  Operator("->", [x,y])

    For negation, we rewrite this to the more generic expression `x == 0`.
    (which in turn creates a `NegBoolView()` in case x is a Boolean variable)

    - ~x            x == 0


    Apart from operator overleading, expressions implement two important functions:

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
from types import GeneratorType
from collections.abc import Iterable
import numpy as np
from .utils import is_num, is_any_list, flatlist

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

        if isinstance(arg_list, (tuple,GeneratorType)):
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
                strarg = ",".join(map(str,arg.flat))
                strargs.append( f"[{strarg}]" )
            else:
                strargs.append( f"{arg}" )
        return "{}({})".format(self.name, ",".join(strargs))

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
            Default: yes
        """
        return True

    def value(self):
        return None # default

    # implication constraint: self -> other
    # Python does not offer relevant syntax...
    # for double implication, use equivalence self == other
    def implies(self, other):
        # other constant
        if other is True:
            return True
        if other is False:
            return ~self
        return Operator('->', [self, other])

    # Comparisons
    def __eq__(self, other):
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
        if other is True:
            return self
        if other is False:
            return False

        return Operator("and", [self, other])
    def __rand__(self, other):
        # some simple constant removal
        if other is True:
            return self
        if other is False:
            return False

        return Operator("and", [other, self])

    def __or__(self, other):
        # some simple constant removal
        if other is True:
            return True
        if other is False:
            return self

        return Operator("or", [self, other])
    def __ror__(self, other):
        # some simple constant removal
        if other is True:
            return True
        if other is False:
            return self

        return Operator("or", [other, self])

    def __xor__(self, other):
        # some simple constant removal
        if other is True:
            return ~self
        if other is False:
            return self

        return Operator("xor", [self, other])
    def __rxor__(self, other):
        # some simple constant removal
        if other is True:
            return ~self
        if other is False:
            return self

        return Operator("xor", [other, self])

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
        return Operator("mul", [self, other])
    def __rmul__(self, other):
        if is_num(other) and other == 1:
            return self
        return Operator("mul", [other, self])

    # matrix multipliciation TODO?
    #object.__matmul__(self, other)

    # other mathematical ones
    def __truediv__(self, other):
        if is_num(other) and other == 1:
            return self
        return Operator("div", [self, other])
    def __rtruediv__(self, other):
        return Operator("div", [other, self])

    def __mod__(self, other):
        return Operator("mod", [self, other])
    def __rmod__(self, other):
        return Operator("mod", [other, self])

    def __pow__(self, other, modulo=None):
        assert (modulo is None), "Power operator: modulo not supported"
        return Operator("pow", [self, other])
    def __rpow__(self, other, modulo=None):
        assert (modulo is None), "Power operator: modulo not supported"
        return Operator("pow", [other, self])

    # Not implemented: (yet?)
    #object.__floordiv__(self, other)
    #object.__divmod__(self, other)

    # unary mathematical operators
    def __neg__(self):
        return Operator("-", [self])
    def __pos__(self):
        return self
    def __abs__(self):
        return Operator("abs", [self])
    # 'not' for now, no unary constraint for it
    def __invert__(self):
        return (self == 0)


class Comparison(Expression):
    """Represents a comparison between two sub-expressions
    """
    allowed = {'==', '!=', '<=', '<', '>=', '>'}

    def __init__(self, name, left, right):
        assert (name in Comparison.allowed), "Symbol not allowed"
        super().__init__(name, [left, right])

    def __repr__(self):
        if all(isinstance(x, Expression) for x in self.args):
            return "({}) {} ({})".format(self.args[0], self.name, self.args[1]) 
        # if not: prettier printing without braces
        return "{} {} {}".format(self.args[0], self.name, self.args[1]) 

    # is bool, check special case
    def __eq__(self, other):
        if is_num(other) and other == 1:
            return self
        return super().__eq__(other)
        
    # return the value of the expression
    # optional, default: None
    def value(self):
        arg_vals = [arg.value() if isinstance(arg, Expression) else arg for arg in self.args]
        if any(a is None for a in arg_vals): return None
        if   self.name == "==": return (arg_vals[0] == arg_vals[1])
        elif self.name == "!=": return (arg_vals[0] != arg_vals[1])
        elif self.name == "<":  return (arg_vals[0] < arg_vals[1])
        elif self.name == "<=": return (arg_vals[0] <= arg_vals[1])
        elif self.name == ">":  return (arg_vals[0] > arg_vals[1])
        elif self.name == ">=": return (arg_vals[0] >= arg_vals[1])
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
        'xor': (0, True),
        '->':  (2, True),
        'sum': (0, False),
        'sub': (2, False), # x - y
        'mul': (2, False),
        'div': (2, False),
        'mod': (2, False),
        'pow': (2, False),
        '-':   (1, False), # -x
        'abs': (1, False),
    }
    printmap = {'sum': '+', 'sub': '-', 'mul': '*', 'div': '/'}

    def __init__(self, name, arg_list):
        # sanity checks
        assert (name in Operator.allowed), "Operator {} not allowed".format(name)
        arity, is_bool = Operator.allowed[name]
        if arity == 0:
            assert (len(arg_list) >= 2), "Operator: n-ary operators require at least two arguments"
            arg_list = flatlist(arg_list)
        else:
            assert (len(arg_list) == arity), "Operator: {}, number of arguments must be {}".format(name, arity)

        # convention for commutative binary operators:
        # swap if right is constant and left is not
        if len(arg_list) == 2 and is_num(arg_list[1]) and \
           name in {'sum', 'mul', 'and', 'or', 'xor'}:
            arg_list[0], arg_list[1] = arg_list[1], arg_list[0]

        # merge same operators for n-ary ones
        if arity == 0:
            i = 0 # length can change
            while i < len(arg_list):
                if isinstance(arg_list[i], Operator) and arg_list[i].name == name:
                    # merge args in at this position
                    l = len(arg_list[i].args)
                    arg_list[i:i+1] = arg_list[i].args
                    i += l
                i += 1

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

    # associative operations {'and', 'or', 'xor', 'sum', 'mul'} are chained
    # + special case for weighted sum (sum of mul)

    def __and__(self, other):
        if self.name == 'and':
            self.args.append(other)
            return self
        return super().__and__(other)
    def __rand__(self, other):
        if self.name == 'and':
            self.args.insert(0,other)
            return self
        return super().__rand__(other)

    def __or__(self, other):
        if self.name == 'or':
            self.args.append(other)
            return self
        return super().__or__(other)
    def __ror__(self, other):
        if self.name == 'or':
            self.args.insert(0,other)
            return self
        return super().__ror__(other)

    def __xor__(self, other):
        if self.name == 'xor':
            self.args.append(other)
            return self
        return super().__xor__(other)
    def __rxor__(self, other):
        if self.name == 'xor':
            self.args.insert(0,other)
            return self
        return super().__rxor__(other)

    def __add__(self, other):
        if is_num(other) and other == 0:
            return self

        if self.name == 'sum':
            if not isinstance(other, Iterable):
                self.args.append(other)
            else: # vector
                self.args.extend(other)
            return self
        return super().__add__(other)

    def __radd__(self, other):
        # only for constants
        if is_num(other) and other == 0:
            return self

        if self.name == 'sum':
            if not isinstance(other, Iterable):
                self.args.insert(0, other)
            else: # vector
                self.args[:0] = other # prepend an array
            return self
        return super().__radd__(other)

    # substraction; in case of sum or wsum, add
    def __sub__(self, other):
        if is_num(other) and other == 0:
            return self

        if self.name == 'sum':
            self.args.append(-other)
            return self
        return super().__sub__(other)
    def __rsub__(self, other):
        if is_num(other) and other == 0:
            return -self

        return (-self).__radd__(other)

    # is bool, check special case
    def __eq__(self, other):
        if is_num(other) and other == 1:
            # check if bool operator, do not add == 1
            arity, is_bool = Operator.allowed[self.name]
            if is_bool:
                return self
        return super().__eq__(other)

    def value(self):
        arg_vals = [arg.value() if isinstance(arg, Expression) else arg for arg in self.args]
        if any(a is None for a in arg_vals): return None
        if   self.name == "sum": return sum(arg_vals)
        elif self.name == "mul": return arg_vals[0] * arg_vals[1]
        elif self.name == "sub": return arg_vals[0] - arg_vals[1]
        elif self.name == "div": return arg_vals[0] / arg_vals[1]
        elif self.name == "mod": return arg_vals[0] % arg_vals[1]
        elif self.name == "pow": return arg_vals[0] ** arg_vals[1]
        elif self.name == "-":   return -arg_vals[0]
        elif self.name == "abs": return -arg_vals[0] if arg_vals[0] < 0 else arg_vals[0]
        return None # default

