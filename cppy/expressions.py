import numpy as np

# Helper: check if numeric, for weighted sum
def is_num(arg):
    return isinstance(arg, (int, np.integer, float, np.float))


class Expression(object):
    # preliminary choice not to expose <,<=,>,>= to LogicalExpr 
    def __eq__(self, other):
        return Comparison("==", self, other)
    def __ne__(self, other):
        return Comparison("!=", self, other)


class NumericExpression(Expression):
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
        return Sum(self, other)
    def __radd__(self, other): # for sum(), chaining happens in Sum()
        if is_num(other) and other == 0:
            return self
        return Sum(other, self)

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
        return Mul(self, other)
    def __rmul__(self, other):
        return Mul(other, self)

    # matrix multipliciation TODO?
    #object.__matmul__(self, other)
    
    # Not implemented: (yet?)
    #object.__truediv__(self, other)
    #object.__floordiv__(self, other)
    #object.__mod__(self, other)
    #object.__divmod__(self, other)
    #object.__pow__(self, other[, modulo])

    
# convention: if one of the two is a constant, it is stored in 'left'
# this eases weighted sum detection
class Mul(NumericExpression):
    def __init__(self, left, right):
        if hasattr(left, '__len__'):
            assert(len(left) == len(right))
        self.left = left
        self.right = right
        # convention: swap if right is constant and left is not
        if is_num(self.right) and not is_num(self.left):
            self.left, self.right = self.right, self.left
    
    def __repr__(self):
        return "{} * {}".format(self.left, self.right)
    
    # it could be a vectorized constraint
    def __iter__(self):
        return (Mul(l,r) for l,r in zip(self.left, self.right))
    
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
        if isinstance(a, Mul) and is_num(a.left) and \
           isinstance(b, Mul) and is_num(b.left):
            return WeightedSum([a.left,b.left], [a.right,b.right])
        
        return Sum(a, b)
        

class Sum(NumericExpression):
    def __init__(self, left, right):
        # vectorizable
        self.elems = [left, right]
    
    def __repr__(self):
        return " + ".join(str(e) for e in self.elems)
    
    # make sum or weighted sum
    def __add__(self, other):
        if is_num(other) and other == 0:
            return self

        if isinstance(other, Mul):
            # offload to that weighted sum check
            return other.__radd__(self)

        self.elems.append(other)
        return self
    def __radd__(self, other):
        if is_num(other) and other == 0:
            return self

        if isinstance(other, Mul):
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

class WeightedSum(NumericExpression):
    def __init__(self, weights, elems):
        self.weights = weights
        self.elems = elems
    
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
        if isinstance(other, Mul) and is_num(other.left):
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
        if isinstance(other, Mul) and is_num(other.left):
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
        if isinstance(other, Mul) and is_num(other.left):
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
        if isinstance(other, Mul) and is_num(other.left):
            w = -other.left
            e = other.right

        self.weights.insert(0, w)
        self.elems.insert(0, e)
        return self
    

    
# Implements bitwise operations & | ^ and ~ (and, or, xor, not)
# Python's built-in 'and' 'or' and 'not' can not be overloaded
class LogicalExpression(Expression):
    def __and__(self, other):
        return BoolOperator("and", self, other)
    def __rand__(self, other):
        return BoolOperator("and", other, self)

    def __or__(self, other):
        return BoolOperator("or", self, other)
    def __ror__(self, other):
        return BoolOperator("or", other, self)

    def __xor__(self, other):
        return BoolOperator("xor", self, other)
    def __rxor__(self, other):
        return BoolOperator("xor", other, self)

    def __invert__(self):
        return (self == 0)

class BoolOperator(LogicalExpression):
    allowed = {'and', 'or', 'xor', '->'}
    def __init__(self, name, left, right):
        assert (name in self.allowed), "Operator not allowed"
        self.name = name
        self.elems = [left, right]
    
    def __repr__(self):
        if len(self.elems) == 2:
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
    def __init__(self, symbol, left, right):
        assert (symbol in self.allowed), "Symbol not allowed"
        if hasattr(left, '__len__'):
            assert(len(left) == len(right))
        self.symbol = symbol
        self.left = left
        self.right = right
    
    def __repr__(self):
        if isinstance(self.left, Expression) and isinstance(self.right, Expression):
            return "({}) {} ({})".format(self.left, self.symbol, self.right) 
        return "{} {} {}".format(self.left, self.symbol, self.right) 

# see globalconstraints.py for concrete instantiations
class GlobalConstraint(LogicalExpression):
    def __init__(self, name, *args):
        self.name = name
        self.args = args

    def __repr__(self):
        if len(self.args) == 1:
            return "{}({})".format(self.name, self.args[0])
        else:
            return "{}({})".format(self.name, ",".join(map(str,self.args)))


class Objective(Expression):
    def __init__(self, name, *args):
        self.name = name
        self.args = args
    
    def __repr__(self):
        return "{} {}".format(self.name, self.args)
    
    def __getattr__(self, name):
        if name == 'value':
            return None
        else:
            # Default behaviour, which failed otherwise we would not be here
            return super().__getattribute__(name)

def Maximise(expr):
    return Objective("Maximise", expr)
def Maximize(expr):
    return Objective("Maximise", expr)

def Minimise(expr):
    return Objective("Minimise", expr)
def Minimize(expr):
    return Objective("Minimise", expr)
