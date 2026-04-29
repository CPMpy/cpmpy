import cpmpy as cp
from cpmpy.transformations.negation import push_down_negation, recurse_negation


class TestTransfNegation:

    def setup_method(self):
        self.bv = cp.boolvar(shape=3, name=tuple("abc"))
        self.iv = cp.intvar(0,5,shape=3, name=tuple("xyz"))

    def test_disjunction(self):

        expr = cp.any(self.bv)
        assert str(push_down_negation([~expr])) == "[~a, ~b, ~c]"
        assert str(recurse_negation(expr)) == "and(~a, ~b, ~c)"
    
    def test_conjunction(self):
        expr = cp.all(self.bv)
        assert str(push_down_negation([~expr])) == "[or(~a, ~b, ~c)]"
        assert str(recurse_negation(expr)) == "or(~a, ~b, ~c)"

    def test_implication(self):
        expr = self.bv[0].implies(self.bv[1])
        assert str(push_down_negation([~expr])) == "[a, ~b]"
        assert str(recurse_negation(expr)) == "(a) and (~b)"

    def test_double_negation(self):
        expr = ~(self.iv[0] == self.iv[1])
        assert str(push_down_negation([~expr])) == "[(x) == (y)]"
        assert str(recurse_negation(expr)) == "(x) == (y)"

    def test_comparison(self):
        expr = self.iv[0] < 3
        assert str(push_down_negation([~expr])) == "[x >= 3]"
        assert str(recurse_negation(expr)) == "x >= 3"

        expr = self.iv[0] <= 3
        assert str(push_down_negation([~expr])) == "[x > 3]"
        assert str(recurse_negation(expr)) == "x > 3"

        expr = self.iv[0] > 3
        assert str(push_down_negation([~expr])) == "[x <= 3]"
        assert str(recurse_negation(expr)) == "x <= 3"

        expr = self.iv[0] >= 3
        assert str(push_down_negation([~expr])) == "[x < 3]"
        assert str(recurse_negation(expr)) == "x < 3"

        expr = self.iv[0] == 3
        assert str(push_down_negation([~expr])) == "[x != 3]"
        assert str(recurse_negation(expr)) == "x != 3"

        expr = self.iv[0] != 3
        assert str(push_down_negation([~expr])) == "[x == 3]"
        assert str(recurse_negation(expr)) == "x == 3"

    def test_bool_comparison(self):
        
        a,b = self.bv[:2]
        expr = a == b
        # assert str(push_down_negation([~expr])) == "[(a) == (~b)]" # TODO!
        # assert str(recurse_negation(~expr)) == "(a) == (~b)" # TODO!
       
        expr = a != b
        assert str(push_down_negation([~expr])) == "[(a) == (b)]"
        assert str(recurse_negation(expr)) == "(a) == (b)"

        expr = a < b
        assert str(push_down_negation([~expr])) == "[(a) >= (b)]"
        assert str(recurse_negation(expr)) == "(a) >= (b)"

        expr = a <= b
        assert str(push_down_negation([~expr])) == "[(a) > (b)]"
        assert str(recurse_negation(expr)) == "(a) > (b)"

        expr = a > b
        assert str(push_down_negation([~expr])) == "[(a) <= (b)]"
        assert str(recurse_negation(expr)) == "(a) <= (b)"

        expr = a >= b
        assert str(push_down_negation([~expr])) == "[(a) < (b)]"
        assert str(recurse_negation(expr)) == "(a) < (b)"

    def negation_in_subexpr(self):

        expr = self.iv[0] + ~(self.bv[0] | self.bv[1]) <= 1
        assert str(push_down_negation(expr)) == "[x + (~a and ~b) <= 1]"
        assert str(recurse_negation(expr)) == "x + (~a and ~b) > 1"

    def test_deeply_nested_negation(self): # (old flatten tests)

        a,b,c = self.bv

        assert  str(push_down_negation([~(a|b)] )) == "[~a, ~b]"
        assert  str(push_down_negation([c.implies(~(a|b))])) == "[(c) -> ((~a) and (~b))]"
        assert  str(push_down_negation([~(c.implies(~(a|b)))])) == "[c, (a) or (b)]"
        assert  str(push_down_negation([~(c.implies(~(a&b)))])) == "[c, a, b]"
        assert  str(push_down_negation([(~c).implies(~(a|b))])) == "[(~c) -> ((~a) and (~b))]"
        assert  str(push_down_negation([(~c|b).implies(~(a|b))])) == "[((~c) or (b)) -> ((~a) and (~b))]"

