#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## expressions.py
##
"""
    ===============
    List of classes
    ===============
    .. autosummary::
        :nosignatures:

        Expression
        Comparison
        Operator

"""
import numpy as np

# Helpers for type checking
def is_num(arg):
    return isinstance(arg, (int, np.integer, float, np.float64))
def is_any_list(arg):
    return isinstance(arg, (list, tuple, np.ndarray))
def is_pure_list(arg):
    return isinstance(arg, (list, tuple))

# Overwriting all/any python built-ins
# all: listwise 'and'
def all(iterable):
    collect = [] # logical expressions
    for elem in iterable:
        if elem is False:
            return False # no need to create constraint
        elif elem is True:
            pass
        elif isinstance(elem, Expression):
            collect.append( elem.boolexpr() )
        else:
            raise Exception("unknown argument '{}' to 'all'".format(elem))
    if len(collect) == 1:
        return collect[0]
    if len(collect) >= 2:
        return Operator("and", collect)
    return True

# any: listwise 'or'
def any(iterable):
    collect = [] # logical expressions
    for elem in iterable:
        if elem is True:
            return True # no need to create constraint
        elif elem is False:
            pass
        elif isinstance(elem, Expression):
            collect.append( elem.boolexpr() )
        else:
            raise Exception("unknown argument '{}' to 'any'".format(elem))
    if len(collect) == 1:
        return collect[0]
    if len(collect) >= 2:
        return Operator("or", collect)
    return False


class Expression(object):
    """
    each Expression is a function with a self.name and self.args (arguments)
    each Expression is considered to be a function whose value can be used
      in other expressions
    each Expression may implement:
    - boolexpr(): the Boolean form of the expression
        default: (expr == 1)
        override for Boolean expressions (preferably through __eq__, see Comparison)
    - value(): the value of the expression, default None
    """

    def __init__(self, name, arg_list):
        if isinstance(arg_list, np.ndarray):
            # must flatten
            arg_list = arg_list.reshape(-1)
        for i in range(len(arg_list)):
            if isinstance(arg_list[i], np.ndarray):
                # must flatten
                arg_list[i] = arg_list[i].reshape(-1)

        assert (is_any_list(arg_list)), "_list_ of arguments required, even if of length one e.g. [arg]"
        self.name = name
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

    # booleanised expression
    # optional, default: (self == 1)
    def boolexpr(self):
        return (self == 1)

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
        if is_num(other) and other == 1:
            return self
        return Operator("mod", [self, other])
    def __rmod__(self, other):
        return Operator("mod", [other, self])

    def __pow__(self, other, modulo=None):
        assert (modulo is None), "Power operator: module not supported"
        return Operator("pow", [self, other])
    def __rpow__(self, other, modulo=None):
        assert (modulo is None), "Power operator: module not supported"
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
    # 'not' for now, no unary constraint for it but like boolexpr()
    def __invert__(self):
        return (self == 0)


class Comparison(Expression):
    allowed = {'==', '!=', '<=', '<', '>=', '>'}

    def __init__(self, name, left, right):
        assert (name in Comparison.allowed), "Symbol not allowed"
        # if vectorized, must match
        if hasattr(left, '__len__') and hasattr(right, '__len__'): 
            assert (len(left) == len(right)), "Comparison: arguments must have equal length"
        super().__init__(name, [left, right])

    def __repr__(self):
        if all(isinstance(x, Expression) for x in self.args):
            return "({}) {} ({})".format(self.args[0], self.name, self.args[1]) 
        # if not: prettier printing without braces
        return "{} {} {}".format(self.args[0], self.name, self.args[1]) 

    # it could be a vectorized constraint
    def __iter__(self):
        return (Comparison(self.name,l,r) for l,r in zip(self.args[0], self.args[1]))

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
    All kinds of operators on expressions,
    including mathematical and logical
    # convention for 2-ary operators: if one of the two is a constant,
    # it is stored first (as expr[0]), this eases weighted sum detection
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
        else:
            assert (len(arg_list) == arity), "Operator: {}, number of arguments must be {}".format(name, arity)
        if arity == 2 and hasattr(arg_list[0], '__len__'):
            # special case: can be two lists
            assert (len(arg_list[0]) == len(arg_list[1])), "Operator: the two arguments must have equal size"

        # convention for commutative binary operators:
        # swap if right is constant and left is not
        if len(arg_list) == 2 and is_num(arg_list[1]) and \
           name in {'sum', 'mul', 'and', 'or', 'xor'}:
            arg_list[0], arg_list[1] = arg_list[1], arg_list[0]

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

    # it could be a vectorized constraint
    def __iter__(self):
        if len(self.args) == 2:
            return (Operator(self.name, list(args)) for args in zip(self.args[0], self.args[1]))
        return super().__iter__(self)

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
            self.args.append(other)
            return self
        return super().__add__(other)

    def __radd__(self, other):
        # only for constants
        if is_num(other) and other == 0:
            return self

        if self.name == 'sum':
            self.args.insert(0, other)
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

