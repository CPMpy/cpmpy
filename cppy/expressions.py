import numpy as np

# Helper: check if numeric, for weighted sum
def is_num(arg):
    return isinstance(arg, (int, np.integer, float, np.float))

class Expression(object):
    # return the value of the expression
    # optional, default: None
    def value(self):
        return None

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

    # addition
    def __add__(self, other):
        if is_num(other) and other == 0:
            return self
        return Sum([self, other])
    def __radd__(self, other): # for sum(), chaining happens in Sum()
        if is_num(other) and other == 0:
            return self
        return Sum([other, self])

    # substraction
    def __sub__(self, other):
        if is_num(other) and other == 0:
            return self
        return WeightedSum([1,-1], [self,other])
    def __rsub__(self, other):
        if is_num(other) and other == 0:
            return self
        return WeightedSum([-1,1], [other,self])
    
    # multiplication
    def __mul__(self, other):
        return MathOperator("*", self, other)
    def __rmul__(self, other):
        return MathOperator("*", other, self)

    def __truediv__(self, other):
        return MathOperator("/", self, other)
    def __rtruediv__(self, other):
        return MathOperator("/", other, self)

    def __mod__(self, other):
        return MathOperator("mod", self, other)
    def __rmod__(self, other):
        return MathOperator("mod", other, self)

    def __pow__(self, other, modulo=None):
        assert (module is None), "Power operator: module not supported"
        return MathOperator("pow", self, other)
    def __rpow__(self, other, modulo=None):
        assert (module is None), "Power operator: module not supported"
        return MathOperator("pow", other, self)

    # matrix multipliciation TODO?
    #object.__matmul__(self, other)
    
    # Not implemented: (yet?)
    #object.__floordiv__(self, other)
    #object.__divmod__(self, other)

# TODO Old dummy
class NumericExpression(Expression):
    pass
    
# convention: if one of the two is a constant, it is stored in 'left'
# this eases weighted sum detection
# + NOT allowed, use Sum([left, right]) instead!
class MathOperator(NumericExpression):
    allowed = {'*', '/', 'mod', 'pow'} # + is special cased to Sum
    def __init__(self, name, left, right):
        if hasattr(left, '__len__'):
            assert (len(left) == len(right)), "MathOperator: left and right must have equal size"
        assert (name in self.allowed), "MathOperator {} not allowed".format(name)
        self.name = name
        self.left = left
        self.right = right
        # convention: swap if right is constant and left is not
        if is_num(self.right) and not is_num(self.left):
            self.left, self.right = self.right, self.left
    
    def __repr__(self):
        return "{} {} {}".format(self.left, self.name, self.right)
    
    # it could be a vectorized constraint
    def __iter__(self):
        return (MathOperator(self.name,l,r) for l,r in zip(self.left, self.right))
    
    # make sum or weighted sum
    def __add__(self, other):
        if is_num(other) and other == 0:
            return self

        return self._do_add(self, other)
    def __radd__(self, other):
        if is_num(other) and other == 0:
            return self
        return self._do_add(other, self)

    def _do_add(_, a, b):
        # weighted sum detection
        if isinstance(a, MathOperator) and a.name == "*" and is_num(a.left) and \
           isinstance(b, MathOperator) and b.name == "*" and is_num(b.left):
            return WeightedSum([a.left,b.left], [a.right,b.right])
        
        return Sum([a, b])
        

class Sum(NumericExpression):
    def __init__(self, elems):
        # vectorized
        self.elems = elems
    
    def __repr__(self):
        return " + ".join(str(e) for e in self.elems)
    
    # make sum or weighted sum
    def __add__(self, other):
        if is_num(other) and other == 0:
            return self

        if isinstance(other, MathOperator) and other.name=="*":
            # offload to that weighted sum check
            return other.__radd__(self)

        self.elems.append(other)
        return self
    def __radd__(self, other):
        if is_num(other) and other == 0:
            return self

        if isinstance(other, MathOperator) and other.name=="*":
            # offload to that weighted sum check
            return other.__add__(self)

        self.elems.insert(0, other)
        return self

    # substraction, turn into weighted sum
    def __sub__(self, other):
        w = [1]*len(self.elems)
        return WeightedSum(w,self.elems) - other
    def __rsub__(self, other):
        w = [1]*len(self.elems)
        return other - WeightedSum(w,self.elems)

class WeightedSum(Sum):
    def __init__(self, weights, elems):
        super().__init__(elems)
        self.weights = weights
    
    def __repr__(self):
        return "sum( {} * {} )".format(self.weights, self.elems)
    
    # chain into the weighted sum (should we?)
    def __add__(self, other):
        if is_num(other) and other == 0:
            return self

        # merge the two
        if isinstance(other, WeightedSum):
            self.weights.extend(other.weights)
            self.elems.extend(other.elems)
            return self

        w,e = 1,other
        # add a mult if possible
        if isinstance(other, MathOperator) and other.name=="*" and is_num(other.left):
            w = other.left
            e = other.right

        self.weights.append(w)
        self.elems.append(e)
        return self
    def __radd__(self, other):
        if is_num(other) and other == 0:
            return self

        w,e = 1,other
        # add a mult if possible
        if isinstance(other, MathOperator) and other.name=="*" and is_num(other.left):
            w = other.left
            e = other.right

        self.weights.insert(0, w)
        self.elems.insert(0, e)
        return self

    # substraction
    def __sub__(self, other):
        if is_num(other) and other == 0:
            return self

        w,e = -1,other
        # sub a mult if possible
        if isinstance(other, MathOperator) and other.name=="*" and is_num(other.left):
            w = -other.left
            e = other.right

        self.weights.append(w)
        self.elems.append(e)
        return self
    def __rsub__(self, other):
        if is_num(other) and other == 0:
            return self

        w,e = -1,other
        # sub a mult if possible
        if isinstance(other, MathOperator) and other.name=="*" and is_num(other.left):
            w = -other.left
            e = other.right

        self.weights.insert(0, w)
        self.elems.insert(0, e)
        return self
    

    
# Implements bitwise operations & | ^ and ~ (and, or, xor, not)
# Python's built-in 'and' 'or' and 'not' can not be overloaded
class LogicalExpression(NumericExpression):
    def __and__(self, other):
        return BoolOperator("and", [self, other])
    def __rand__(self, other):
        return BoolOperator("and", [other, self])

    def __or__(self, other):
        return BoolOperator("or", [self, other])
    def __ror__(self, other):
        return BoolOperator("or", [other, self])

    def __xor__(self, other):
        return BoolOperator("xor", [self, other])
    def __rxor__(self, other):
        return BoolOperator("xor", [other, self])

    def __invert__(self):
        return (self == 0)

class BoolOperator(LogicalExpression):
    allowed = {'and', 'or', 'xor', '->'}
    def __init__(self, name, elems):
        assert (name in self.allowed), "BoolOperator {} not allowed".format(name)
        if name == '->':
            assert (len(elems) == 2), "BoolOperator '->' requires exactly 2 arguments"
        self.name = name
        self.elems = elems
    
    def __repr__(self):
        if len(self.elems) == 2:
            # bracketed printing if both not constant
            if all(isinstance(x, Expression) for x in self.elems):
                return "({}) {} ({})".format(self.elems[0], self.name, self.elems[1]) 
            return "{} {} {}".format(self.elems[0], self.name, self.elems[1])
        return "{}({})".format(self.name, self.elems)

    def _compatible(self, other):
        return isinstance(other, BoolOperator) and other.name == self.name

    # override to check same operator
    def __and__(self, other):
        if self._compatible(other):
            self.elems.append(other)
            return self
        return super().__and__(other)
    def __rand__(self, other):
        if self._compatible(other):
            self.elems.insert(0, other)
            return self
        return super().__rand__(other)

    def __or__(self, other):
        if self._compatible(other):
            self.elems.append(other)
            return self
        return super().__or__(other)
    def __ror__(self, other):
        if self._compatible(other):
            self.elems.insert(0, other)
            return self
        return super().__ror__(other)

    def __xor__(self, other):
        if self._compatible(other):
            self.elems.append(other)
            return self
        return super().__xor__(other)
    def __rxor__(self, other):
        if self._compatible(other):
            self.elems.insert(0, other)
            return self
        return super().__rxor__(other)

class Comparison(LogicalExpression):
    allowed = {'<=', '<', '>=', '>', '==', '!='}
    def __init__(self, name, left, right):
        assert (name in self.allowed), "Symbol not allowed"
        if hasattr(left, '__len__'):
            assert(len(left) == len(right))
        self.name = name
        self.left = left
        self.right = right
    
    def __repr__(self):
        if isinstance(self.left, Expression) and isinstance(self.right, Expression):
            return "({}) {} ({})".format(self.left, self.name, self.right) 
        return "{} {} {}".format(self.left, self.name, self.right) 

    # it could be a vectorized constraint
    def __iter__(self):
        return (Comparison(self.name,l,r) for l,r in zip(self.left, self.right))
    

# see globalconstraints.py for concrete instantiations
class GlobalConstraint(LogicalExpression):
    # add_equality_as_arg: bool, whether to catch 'self == expr' cases,
    # and add them to the 'elems' argument list (e.g. for element: X[var] == 1)
    def __init__(self, name, arg_list, add_equality_as_arg=False):
        assert (isinstance(arg_list, list)), "GlobalConstraint requires list of arguments, even if of length one e.g. [arg]"
        self.name = name
        self.elems = arg_list
        self.add_equality_as_arg = add_equality_as_arg

    def __repr__(self):
        ret = ""
        if len(self.elems) == 1:
            ret = "{}({})".format(self.name, self.elems[0])
        else:
            ret = "{}({})".format(self.name, ",".join(map(str,self.elems)))
        return ret.replace("\n","") # numpy args add own linebreaks...

    def __eq__(self, other):
        if self.add_equality_as_arg:
            self.elems.append(other)
            return self
        
        # default
        return super().__eq__(other)
            


class Objective(Expression):
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr
    
    def __repr__(self):
        return "{} {}".format(self.name, self.expr)

def Maximise(expr):
    return Objective("Maximize", expr)
def Maximize(expr):
    return Objective("Maximize", expr)

def Minimise(expr):
    return Objective("Minimize", expr)
def Minimize(expr):
    return Objective("Minimize", expr)
