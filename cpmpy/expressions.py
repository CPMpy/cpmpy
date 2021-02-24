import numpy as np

# Helpers for type checking
def is_num(arg):
    return isinstance(arg, (int, np.integer, float, np.float))
def is_any_list(arg):
    return isinstance(arg, (list, tuple, np.ndarray))
def is_pure_list(arg):
    return isinstance(arg, (list, tuple))

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

    # return the value of the expression
    # optional, default: None
    def value(self):
        return None

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
        if is_num(other) and other == 0:
            return self
        return Operator("sub", [self, other])
    def __rsub__(self, other):
        if is_num(other) and other == 0:
            return -self
        return Operator("sub", [other, self])
    
    # multiplication
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
        assert (module is None), "Power operator: module not supported"
        return Operator("pow", [self, other])
    def __rpow__(self, other, modulo=None):
        assert (module is None), "Power operator: module not supported"
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
        if hasattr(left, '__len__'): 
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
        if len(arg_list) == 2 and all(is_num(x) for x in arg_list) and \
           name in {'sum', 'mul', 'and', 'or', 'xor'}:
            arg_list[0], arg_list[1] = arg_list[1], arg_list[0]

        super().__init__(name, arg_list)
    
    def __repr__(self):
        printname = self.name
        printmap = {'sum': '+', 'sub': '-', 'mul': '*', 'div': '/'}
        if printname in printmap:
            printname = printmap[printname]

        # special cases
        if self.name == '-': # unary -
            return "-{}".format(expr[0])

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
            self.expr.insert(0, other)
            return self
        return super().__radd__(other)

    # substraction; in case of sum or wsum, add
    def __sub__(self, other):
        if is_num(other) and other == 0:
            return self

        if self.name == 'sum':
            self.expr.append(-other)
            return self
        return super().__sub__(other)
    def __rsub__(self, other):
        if is_num(other) and other == 0:
            return -self

        if self.name == 'sum':
            self.expr.insert(0,-other)
            return self
        return super().__sub__(other)

    # is bool, check special case
    def __eq__(self, other):
        if is_num(other) and other == 1:
            # check if bool operator, do not add == 1
            arity, is_bool = Operator.allowed[self.name]
            if is_bool:
                return self
        return super().__eq__(other)
    
    def value(self):
        if self.name == "pow":
            raise NotImplementedError()

        operator_obj = {
            "sum": sum(self.args),
            "mul": self.args[0] * self.args[1],
            "sub": self.args[0] - self.args[1],
            "div": self.args[0] / self.args[1],
            "mod": self.args[0] % self.args[1],
            # "pow": self.args[0] ** self.args[1],
            "mul": self.args[0] * self.args[1],
            "-": -self.args[0],
            "abs": -self.args[0] if self.args[0].value() < 0 else self.args[0]
        }
        if self.name not in operator_obj:
            return None

        return operator_obj[self.name]

class Element(Expression):
    """
        constraint Arr[X] = Y
        'Y' here is optional, can use as function: Arr[X] + 3 <= Y
    """

    def __init__(self, arg_list):
        assert (len(arg_list) >= 2 and len(arg_list) <= 3), "Element takes 2 or three arguments"
        super().__init__("element", arg_list)


    def __repr__(self):
        out = "{}[{}]".format(self.args[0], self.args[1])
        if len(self.args) == 2:
            return out
        return "{} == {}".format(out, self.args[2])

    def __eq__(self, other):
        if len(self.args) == 2:
            # add as third argument
            self.args.append(other)
            return self

        # else: 3 arguments, reified variant, is bool
        if is_num(other) and other == 1:
            return self
        


# see globalconstraints.py for concrete instantiations
class GlobalConstraint(Expression):
    # add_equality_as_arg: bool, whether to catch 'self == expr' cases,
    # and add them to the 'args' argument list (e.g. for element: X[var] == 1)
    def __init__(self, name, arg_list, add_equality_as_arg=False, is_bool=True):
        super().__init__(name, arg_list)
        self.add_equality_as_arg = add_equality_as_arg
        self.is_bool = is_bool

    def __eq__(self, other):
        if self.add_equality_as_arg:
            self.args.append(other)
            return self

        if self.is_bool and is_num(other) and other == 1:
            return self
        
        # default
        return super().__eq__(other)
