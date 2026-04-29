import cpmpy as cp
from cpmpy.transformations.flatten_model import flatten_model, flatten_constraint, get_or_make_var, flatten_objective, normalized_boolexpr
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl
from cpmpy.expressions.core import Operator

class TestFlattenModel:
    def setup_method(self):
        self.ivars = cp.intvar(1, 10, (5,))
        self.bvars = cp.boolvar((2,))
        #self.constraints = [self.ivars != 3] # should work in future (broadcasting)
        self.constraints = [iv != 3 for iv in self.ivars]

    def test_constraints(self):
        model = cp.Model(self.constraints)
        model2 = flatten_model(model)
        assert isinstance(model2, cp.Model)
        assert hasattr(model2, 'constraints')
        assert len(model2.constraints) > 1

    def test_objective(self):
        obj = self.ivars.sum()
        model = cp.Model(self.constraints, maximize=obj)
        model2 = flatten_model(model)
        assert model2.objective_ is not None
        assert not model2.objective_is_min

    def test_abs(self):
        l = cp.intvar(0,9, shape=3)
        # bounds used to be computed wrong, making both unsat
        assert  cp.Model(abs(l[0]-l[1])- abs(l[2]-l[1]) < 0).solve()
        assert  cp.Model(abs(l[0]-l[1])- abs(l[2]-l[1]) > 0).solve()

    def test_mod(self):
        iv1 = cp.intvar(2,9)
        iv2 = cp.intvar(5,9)
        m = cp.Model([(iv1+iv2) % 2 >= 0, (iv1+iv2) % 2 <= 1])
        assert  m.solve()


class TestFlattenConstraint:
    def setup_method(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0
        self.ivars = cp.intvar(1, 10, shape=(5,))
        self.bvars = cp.boolvar((3,))

    def test_eq(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        e = (x == y) 
        assert str([e]) == str(flatten_constraint(e))
        e = (x == ~y) 
        assert str([e]) == str(flatten_constraint(e))
        e = (a == b) 
        assert str([e]) == str(flatten_constraint(e))

    def test_nq(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        e = (x != y) 
        # assert "[(~BV1) == (BV0)]" == str(flatten_constraint(e)) -> part of push_down_negation now
        e = (x != ~y) 
        # assert "[(~BV1) == (~BV0)]" == str(flatten_constraint(e)) -> part of push_down_negation now
        e = (a != b) 
        assert "[(IV0) != (IV1)]" == str(flatten_constraint(e))

    def test_eq_comp(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        e = ((a > 5) == x)
        assert "[(IV0 > 5) == (BV0)]" == str(flatten_constraint(e))
        e = (x == (b < 3))
        assert "[(IV1 < 3) == (BV0)]" == str(flatten_constraint(e))
        e = ((a > 5) == (b < 3))
        assert len(flatten_constraint(e)) == 2
    
class TestFlattenExpr:
    def setup_method(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0
        self.ivars = cp.intvar(1, 10, shape=(5,))
        self.bvars = cp.boolvar((3,))

    # not directly tested on its own, new functions 'normalized_boolexpr' and 'normalized_numexpr'

    def test_get_or_make_var__bool(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        v, cons = get_or_make_var(x)
        assert str(v) == "BV0"
        assert {str(c) for c in cons} == set()

        v, cons = get_or_make_var(~x)
        assert str(v) == "~BV0"
        assert {str(c) for c in cons} == set()

        v, cons = get_or_make_var(x == y)
        assert str(v) == "BV3"
        assert {str(c) for c in cons} == {"((BV0) == (BV1)) == (BV3)"}

        v, cons = get_or_make_var(x != y)
        assert str(v) == "BV4"
        assert {str(c) for c in cons} == {"((BV0) == (~BV1)) == (BV4)"}

        v, cons = get_or_make_var(x > y)
        assert str(v) == "BV5"
        assert {str(c) for c in cons} == {"((BV0) > (BV1)) == (BV5)"}

        v, cons = get_or_make_var(x <= y)
        assert str(v) == "BV6"
        assert {str(c) for c in cons} == {"((BV0) <= (BV1)) == (BV6)"}

        v, cons = get_or_make_var((a > 10) == x)
        assert str(v) == "BV8"
        assert {str(c) for c in cons} == {
            "((BV7) == (BV0)) == (BV8)",
            "(IV0 > 10) == (BV7)",
        }

        v, cons = get_or_make_var((a > 10) == (d > 5))
        assert str(v) == "BV11"
        assert {str(c) for c in cons} == {
            "((BV10) == (BV9)) == (BV11)",
            "(IV0 > 10) == (BV10)",
            "(IV3 > 5) == (BV9)",
        }

        v, cons = get_or_make_var(a > c)
        assert str(v) == "BV12"
        assert {str(c) for c in cons} == {"((IV0) > (IV2)) == (BV12)"}

        v, cons = get_or_make_var(a + b > c)
        assert str(v) == "BV13"
        assert {str(c) for c in cons} == {"(((IV0) + (IV1)) > (IV2)) == (BV13)"}
        cp.intvar(0,2) # increase counter

        v, cons = get_or_make_var((a > b).implies(x))
        assert str(v) == "BV15"
        assert {str(c) for c in cons} == {
            "((~BV14) or (BV0)) == (BV15)",
            "((IV0) > (IV1)) == (BV14)",
        }

        v, cons = get_or_make_var(x & y)
        assert str(v) == "BV16"
        assert {str(c) for c in cons} == {"((BV0) and (BV1)) == (BV16)"}

        v, cons = get_or_make_var(x | y)
        assert str(v) == "BV17"
        assert {str(c) for c in cons} == {"((BV0) or (BV1)) == (BV17)"}

        v, cons = get_or_make_var(x.implies(y))
        assert str(v) == "BV18"
        assert {str(c) for c in cons} == {"((~BV0) or (BV1)) == (BV18)"}

        v, cons = get_or_make_var(x.implies(y | z))
        assert str(v) == "BV20"
        assert {str(c) for c in cons} == {
            "((~BV0) or (BV19)) == (BV20)",
            "((BV1) or (BV2)) == (BV19)",
        }

        v, cons = get_or_make_var((x & y).implies(y & z))
        assert str(v) == "BV23"
        assert {str(c) for c in cons} == {
            "((~BV21) or (BV22)) == (BV23)",
            "((BV0) and (BV1)) == (BV21)",
            "((BV1) and (BV2)) == (BV22)",
        }

        v, cons = get_or_make_var(x.implies(y.implies(z)))
        assert str(v) == "BV25"
        assert {str(c) for c in cons} == {
            "((~BV0) or (BV24)) == (BV25)",
            "((~BV1) or (BV2)) == (BV24)",
        }

        v, cons = get_or_make_var(a > 10)
        assert str(v) == "BV26"
        assert {str(c) for c in cons} == {"(IV0 > 10) == (BV26)"}

        v, cons = get_or_make_var((a > 10) & x & y)
        assert str(v) == "BV28"
        assert {str(c) for c in cons} == {
            "(and(BV27, BV0, BV1)) == (BV28)",
            "(IV0 > 10) == (BV27)",
        }

        v, cons = get_or_make_var(Operator('not', [x]) == y)
        assert str(v) == "BV29"
        assert {str(c) for c in cons} == {"((~BV0) == (BV1)) == (BV29)"}

    def test_get_or_make_var__num(self):
        (a,b,c,d,e) = self.ivars[:5]

        v, cons = get_or_make_var(a + b)
        assert str(v) == "IV5"
        assert {str(c) for c in cons} == {"((IV0) + (IV1)) == (IV5)"}

        v, cons = get_or_make_var(a + b + c)
        assert str(v) == "IV6"
        assert {str(c) for c in cons} == {"(sum(IV0, IV1, IV2)) == (IV6)"}

        v, cons = get_or_make_var(2 * a)
        assert str(v) == "IV7"
        assert {str(c) for c in cons} == {"(sum([2] * [IV0])) == (IV7)"}

        v, cons = get_or_make_var(a * b)
        assert str(v) == "IV8"
        assert {str(c) for c in cons} == {"((IV0) * (IV1)) == (IV8)"}

        v, cons = get_or_make_var(a // b)
        assert str(v) == "IV9"
        assert {str(c) for c in cons} == {"((IV0) div (IV1)) == (IV9)"}

        v, cons = get_or_make_var(1 // b)
        assert str(v) == "IV10"
        assert {str(c) for c in cons} == {"(1 div (IV1)) == (IV10)"}

        v, cons = get_or_make_var(a // 1)
        assert str(v) == "IV0"
        assert {str(c) for c in cons} == set()

        v, cons = get_or_make_var(abs(cp.intvar(-5, 5, name="x")))
        assert str(v) == "IV11"
        assert {str(c) for c in cons} == {"(abs(x)) == (IV11)"}

        v, cons = get_or_make_var(1 * a + 2 * b + 3 * c)
        assert str(v) == "IV12"
        assert {str(c) for c in cons} == {"(sum([1, 2, 3] * [IV0, IV1, IV2])) == (IV12)"}

        v, cons = get_or_make_var(cp.cpm_array([1, 2, 3])[a])
        assert str(v) == "IV13"
        assert {str(c) for c in cons} == {"([1 2 3][IV0]) == (IV13)"}

        v, cons = get_or_make_var(cp.cpm_array([b + c, 2, 3])[a])
        assert str(v) == "IV15"
        assert {str(c) for c in cons} == {
            "((IV14, 2, 3)[IV0]) == (IV15)",
            "((IV1) + (IV2)) == (IV14)",
        }

        v, cons = get_or_make_var(a * 2)
        assert str(v) == "IV16"
        assert {str(c) for c in cons} == {"(sum([2] * [IV0])) == (IV16)"}

    def test_objective(self):
        (a,b,c,d,e) = self.ivars[:5]

        assert  str(flatten_objective( a )) == f"({str(a)}, [])"
        assert  str(flatten_objective( -a )) == '(sum([-1] * [IV0]), [])'
        assert  str(flatten_objective( -2*a )) == '(sum([-2] * [IV0]), [])'
        assert  str(flatten_objective( a+b )) == f"(({str(a)}) + ({str(b)}), [])"
        assert  str(flatten_objective( a-b )) == '(sum([1, -1] * [IV0, IV1]), [])'
        assert  str(flatten_objective( -a+b )) == '(sum([-1, 1] * [IV0, IV1]), [])'
        assert  str(flatten_objective( a+b-c )) == "(sum([1, 1, -1] * [IV0, IV1, IV2]), [])"
        assert  str(flatten_objective( 2*a+3*b )) == "(sum([2, 3] * [IV0, IV1]), [])"
        assert  str(flatten_objective( 2*a+b*3 )) == "(sum([2, 3] * [IV0, IV1]), [])"
        assert  str(flatten_objective( 2*a-b*3 )) == "(sum([2, -3] * [IV0, IV1]), [])"
        assert  str(flatten_objective( 2*a-3*b+4*c )) == "(sum([2, -3, 4] * [IV0, IV1, IV2]), [])"
        assert  str(flatten_objective( 2*a+3*(b + c) )) == "(sum([2, 3, 3] * [IV0, IV1, IV2]), [])"
        assert  str(flatten_objective( 2*a-3*(b + 2*c) )) == "(sum([2, -3, -6] * [IV0, IV1, IV2]), [])"
        assert  str(flatten_objective( 2*a-3*(b - c*2) )) == '(sum([2, -3, 6] * [IV0, IV1, IV2]), [])'
        cp.intvar(0,2) # increase counter
        assert  str(flatten_objective( a//b+c )) == f"((IV6) + ({str(c)}), [(({str(a)}) div ({str(b)})) == (IV6)])"
        assert  str(flatten_objective( cp.cpm_array([1,2,3])[a] )) == "(IV7, [([1 2 3][IV0]) == (IV7)])"
        assert  str(flatten_objective( cp.cpm_array([1,2,3])[a]+b )) == "((IV8) + (IV1), [([1 2 3][IV0]) == (IV8)])"


    def test_constraint(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        assert  str(flatten_constraint( x )) == "[BV0]"
        assert  str(flatten_constraint( ~x )) == "[~BV0]"
        assert  str(flatten_constraint( [x,y] )) == "[BV0, BV1]"
        assert  str(flatten_constraint( x&y )) == "[BV0, BV1]"
        assert  str(flatten_constraint( x&y&~z )) == "[BV0, BV1, ~BV2]"
        assert  str(flatten_constraint( x.implies(y) )) == "[(BV0) -> (BV1)]"
        assert  str(flatten_constraint( x|(y.implies(z)) )) == "[or(BV0, ~BV1, BV2)]"
        assert  str(flatten_constraint( (a > 10)&x )) == "[IV0 > 10, BV0]"
        cp.boolvar() # increase counter
        assert  str(flatten_constraint( (a > 10).implies(x) )) == "[(IV0 > 10) -> (BV0)]"
        cp.boolvar() # increase counter
        assert  str(flatten_constraint( (a > 10) )) == "[IV0 > 10]"
        assert  str(flatten_constraint( (a > 10) == 1 )) == "[IV0 > 10]"
        # assert  str(flatten_constraint( (a > 10) == 0 )) == "[IV0 <= 10]" -> part of push_down_negation now
        assert  str(flatten_constraint( (a > 10) == x )) == "[(IV0 > 10) == (BV0)]"
        #self.assertEqual( str(flatten_constraint( x == (a > 10) )), "[(IV0 > 10) == (BV0)]" ) # TODO, make it do the swap (again)
        assert  str(flatten_constraint( (a > 10) | (b + c > 2) )) == "[(BV5) or (BV6), (IV0 > 10) == (BV5), ((IV1) + (IV2) > 2) == (BV6)]"
        assert  str(flatten_constraint( a > 10 )) == "[IV0 > 10]"
        assert  str(flatten_constraint( 10 > a )) == "[IV0 < 10]"# surprising
        assert  str(flatten_constraint( a+b > c )) == "[((IV0) + (IV1)) > (IV2)]"
        #self.assertEqual( str(flatten_constraint( c < a+b )), "[((IV0) + (IV1)) > (IV2)]" ) # TODO, make it do the swap (again)
        assert  str(flatten_constraint( (a+b > c) == x|y )) == "[(((IV0) + (IV1)) > (IV2)) == (BV7), ((BV0) or (BV1)) == (BV7)]"

        assert  str(flatten_constraint( a + b == c )) == "[((IV0) + (IV1)) == (IV2)]"
        #self.assertEqual( str(flatten_constraint( c != a + b )), "[((IV0) + (IV1)) != (IV2)]" ) # TODO, make it do the swap (again)
        assert  str(flatten_constraint( ((a > 5) == (b < 3)) )) == "[(IV0 > 5) == (BV8), (IV1 < 3) == (BV8)]"

        assert  str(flatten_constraint( cp.cpm_array([1,2,3])[a] == b )) == "[([1 2 3][IV0]) == (IV1)]"
        assert  str(flatten_constraint( cp.cpm_array([1,2,3])[a] > b )) == "[([1 2 3][IV0]) > (IV1)]"
        cp.intvar(0,2, 4) # increase counter
        assert  str(flatten_constraint( cp.cpm_array([1,2,3])[a] <= b )) == "[([1 2 3][IV0]) <= (IV1)]"
        assert  str(flatten_constraint( cp.AllDifferent([a+b,b+c,c+3]) )) == "[alldifferent(IV9,IV10,IV11), ((IV0) + (IV1)) == (IV9), ((IV1) + (IV2)) == (IV10), ((IV2) + 3) == (IV11)]"

        # issue #27
        assert  str(flatten_constraint( (a == 10).implies(b == c+d) )) == "[(IV0 == 10) -> (BV9), (((IV2) + (IV3)) == (IV1)) == (BV9)]"
        # different order should not create more tempvars
        assert  str(flatten_constraint( (a == 10).implies(c+d == b) )) == "[(IV0 == 10) -> (BV10), (((IV2) + (IV3)) == (IV1)) == (BV10)]"
        assert  str(flatten_constraint( a // b == c )) == "[((IV0) div (IV1)) == (IV2)]"
        assert  str(flatten_constraint( c == a // b )) == "[((IV0) div (IV1)) == (IV2)]"

        assert  str(a % 1 == 0) == "(IV0) mod 1 == 0"

        # boolexpr as numexpr
        assert  str(flatten_constraint((a + b == 2) <= c)) == "[(BV11) <= (IV2), ((IV0) + (IV1) == 2) == (BV11)]"

        # != in boolexpr, bug #170
        assert  str(normalized_boolexpr(x != (a == 1))) == "((BV12) == (~BV0), [(IV0 == 1) == (BV12)])"
        #simplify output
        assert  str(normalized_boolexpr(Operator('not',[x]) == y)) == "((~BV0) == (BV1), [])"
