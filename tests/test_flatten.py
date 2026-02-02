import unittest
import cpmpy as cp
from cpmpy.transformations.flatten_model import *
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl

class TestFlattenModel(unittest.TestCase):
    def setUp(self):
        self.ivars = cp.intvar(1, 10, (5,))
        self.bvars = cp.boolvar((2,))
        #self.constraints = [self.ivars != 3] # should work in future (broadcasting)
        self.constraints = [iv != 3 for iv in self.ivars]

    def test_constraints(self):
        model = cp.Model(self.constraints)
        model2 = flatten_model(model)
        self.assertTrue(isinstance(model2, cp.Model))
        self.assertTrue(hasattr(model2, 'constraints'))
        self.assertTrue(len(model2.constraints) > 1)

    def test_objective(self):
        obj = self.ivars.sum()
        model = cp.Model(self.constraints, maximize=obj)
        model2 = flatten_model(model)
        self.assertTrue(model2.objective_ is not None)
        self.assertFalse(model2.objective_is_min)

    def test_abs(self):
        l = cp.intvar(0,9, shape=3)
        # bounds used to be computed wrong, making both unsat
        self.assertTrue( cp.Model(abs(l[0]-l[1])- abs(l[2]-l[1]) < 0).solve() )
        self.assertTrue( cp.Model(abs(l[0]-l[1])- abs(l[2]-l[1]) > 0).solve() )

    def test_mod(self):
        iv1 = cp.intvar(2,9)
        iv2 = cp.intvar(5,9)
        m = cp.Model([(iv1+iv2) % 2 >= 0, (iv1+iv2) % 2 <= 1])
        self.assertTrue( m.solve() )


class TestFlattenConstraint(unittest.TestCase):
    def setUp(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0
        self.ivars = cp.intvar(1, 10, shape=(5,))
        self.bvars = cp.boolvar((3,))

    def test_eq(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        e = (x == y) 
        self.assertEqual(str([e]), str(flatten_constraint(e)) )
        e = (x == ~y) 
        self.assertEqual(str([e]), str(flatten_constraint(e)) )
        e = (a == b) 
        self.assertEqual(str([e]), str(flatten_constraint(e)) )

    def test_nq(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        e = (x != y) 
        self.assertEqual("[(~BV1) == (BV0)]", str(flatten_constraint(e)) )
        e = (x != ~y) 
        self.assertEqual("[(~BV1) == (~BV0)]", str(flatten_constraint(e)) )
        e = (a != b) 
        self.assertEqual("[(IV0) != (IV1)]", str(flatten_constraint(e)) )

    def test_eq_comp(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        e = ((a > 5) == x)
        self.assertEqual("[(IV0 > 5) == (BV0)]", str(flatten_constraint(e)))
        e = (x == (b < 3))
        self.assertEqual("[(IV1 < 3) == (BV0)]", str(flatten_constraint(e)))
        e = ((a > 5) == (b < 3))
        self.assertEqual(len(flatten_constraint(e)), 2)
    
class TestFlattenExpr(unittest.TestCase):
    def setUp(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0
        self.ivars = cp.intvar(1, 10, shape=(5,))
        self.bvars = cp.boolvar((3,))

    # not directly tested on its own, new functions 'normalized_boolexpr' and 'normalized_numexpr'

    def test_get_or_make_var__bool(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        self.assertEqual( str(get_or_make_var(x)), "(BV0, [])" )
        self.assertEqual( str(get_or_make_var(~x)), "(~BV0, [])" )

        self.assertEqual( str(get_or_make_var(x == y)), "(BV3, [((BV0) == (BV1)) == (BV3)])" )
        self.assertEqual( str(get_or_make_var(x != y)), "(BV4, [((BV0) == (~BV1)) == (BV4)])" )
        self.assertEqual( str(get_or_make_var(x > y)), "(BV5, [((BV0) > (BV1)) == (BV5)])" )
        self.assertEqual( str(get_or_make_var(x <= y)), "(BV6, [((BV0) <= (BV1)) == (BV6)])" )

        self.assertEqual( str(get_or_make_var((a > 10) == x)), "(BV8, [((BV7) == (BV0)) == (BV8), (IV0 > 10) == (BV7)])" )
        self.assertEqual( str(get_or_make_var( (a > 10) == (d > 5) )), "(BV11, [((BV10) == (BV9)) == (BV11), (IV0 > 10) == (BV10), (IV3 > 5) == (BV9)])" )
        self.assertEqual( str(get_or_make_var( a > c )), "(BV12, [((IV0) > (IV2)) == (BV12)])" )
        self.assertEqual( str(get_or_make_var( a + b > c )), "(BV13, [(((IV0) + (IV1)) > (IV2)) == (BV13)])" )
        cp.intvar(0,2) # increase counter

        self.assertEqual( str(get_or_make_var( (a>b).implies(x) )), "(BV15, [((~BV14) or (BV0)) == (BV15), ((IV0) > (IV1)) == (BV14)])" )
        self.assertEqual( str(get_or_make_var( x&y )), "(BV16, [((BV0) and (BV1)) == (BV16)])" )
        self.assertEqual( str(get_or_make_var( x|y )), "(BV17, [((BV0) or (BV1)) == (BV17)])" )
        self.assertEqual( str(get_or_make_var( x.implies(y) )), "(BV18, [((~BV0) or (BV1)) == (BV18)])" )
        self.assertEqual( str(get_or_make_var( x.implies(y|z) )), "(BV20, [((~BV0) or (BV19)) == (BV20), ((BV1) or (BV2)) == (BV19)])" )
        self.assertEqual( str(get_or_make_var( (x&y).implies(y&z) )), "(BV23, [((~BV21) or (BV22)) == (BV23), ((BV0) and (BV1)) == (BV21), ((BV1) and (BV2)) == (BV22)])" )
        self.assertEqual( str(get_or_make_var( x.implies(y.implies(z)) )), "(BV25, [((~BV0) or (BV24)) == (BV25), ((~BV1) or (BV2)) == (BV24)])" )

        self.assertEqual( str(get_or_make_var( (a > 10) )), "(BV26, [(IV0 > 10) == (BV26)])" )
        self.assertEqual( str(get_or_make_var( (a > 10)&x&y )), "(BV28, [(and([BV27, BV0, BV1])) == (BV28), (IV0 > 10) == (BV27)])" )

        self.assertEqual( str(get_or_make_var(Operator('not', [x]) == y)), '(BV29, [((~BV0) == (BV1)) == (BV29)])' )

    def test_get_or_make_var__num(self):
        (a,b,c,d,e) = self.ivars[:5]

        self.assertEqual( str(get_or_make_var( a+b )), "(IV5, [((IV0) + (IV1)) == (IV5)])" )
        self.assertEqual( str(get_or_make_var( a+b+c )), "(IV6, [(sum([IV0, IV1, IV2])) == (IV6)])" )
        self.assertEqual( str(get_or_make_var( 2*a )), "(IV7, [(sum([2] * [IV0])) == (IV7)])" )
        self.assertEqual( str(get_or_make_var( a*b )), "(IV8, [((IV0) * (IV1)) == (IV8)])" )
        self.assertEqual( str(get_or_make_var( a//b )), "(IV9, [((IV0) div (IV1)) == (IV9)])" )
        self.assertEqual( str(get_or_make_var( 1//b )), "(IV10, [(1 div (IV1)) == (IV10)])" )
        self.assertEqual( str(get_or_make_var( a//1 )), "(IV0, [])" )
        self.assertEqual( str(get_or_make_var( abs(cp.intvar(-5,5, name="x")) )), "(IV11, [(abs(x)) == (IV11)])" )
        self.assertEqual( str(get_or_make_var( 1*a + 2*b + 3*c )), "(IV12, [(sum([1, 2, 3] * [IV0, IV1, IV2])) == (IV12)])")
        self.assertEqual( str(get_or_make_var( cp.cpm_array([1,2,3])[a] )), "(IV13, [([1 2 3][IV0]) == (IV13)])" )
        self.assertEqual( str(get_or_make_var( cp.cpm_array([b+c,2,3])[a] )), "(IV15, [((IV14, 2, 3)[IV0]) == (IV15), ((IV1) + (IV2)) == (IV14)])" )
        self.assertEqual( str(get_or_make_var( a*2 )), "(IV16, [(sum([2] * [IV0])) == (IV16)])" )

    def test_objective(self):
        (a,b,c,d,e) = self.ivars[:5]

        self.assertEqual( str(flatten_objective( a )), f"({str(a)}, [])" )
        self.assertEqual( str(flatten_objective( -a )), '(sum([-1] * [IV0]), [])' )
        self.assertEqual( str(flatten_objective( -2*a )), '(sum([-2] * [IV0]), [])' )
        self.assertEqual( str(flatten_objective( a+b )), f"(({str(a)}) + ({str(b)}), [])" )
        self.assertEqual( str(flatten_objective( a-b )), '(sum([1, -1] * [IV0, IV1]), [])' )
        self.assertEqual( str(flatten_objective( -a+b )), '(sum([-1, 1] * [IV0, IV1]), [])' )
        self.assertEqual( str(flatten_objective( a+b-c )), "(sum([1, 1, -1] * [IV0, IV1, IV2]), [])" )
        self.assertEqual( str(flatten_objective( 2*a+3*b )), "(sum([2, 3] * [IV0, IV1]), [])" )
        self.assertEqual( str(flatten_objective( 2*a+b*3 )), "(sum([2, 3] * [IV0, IV1]), [])" )
        self.assertEqual( str(flatten_objective( 2*a-b*3 )), "(sum([2, -3] * [IV0, IV1]), [])" )
        self.assertEqual( str(flatten_objective( 2*a-3*b+4*c )), "(sum([2, -3, 4] * [IV0, IV1, IV2]), [])" )
        self.assertEqual( str(flatten_objective( 2*a+3*(b + c) )), "(sum([2, 3, 3] * [IV0, IV1, IV2]), [])" )
        self.assertEqual( str(flatten_objective( 2*a-3*(b + 2*c) )), "(sum([2, -3, -6] * [IV0, IV1, IV2]), [])" )
        self.assertEqual( str(flatten_objective( 2*a-3*(b - c*2) )), '(sum([2, -3, 6] * [IV0, IV1, IV2]), [])' )
        cp.intvar(0,2) # increase counter
        self.assertEqual( str(flatten_objective( a//b+c )), f"((IV6) + ({str(c)}), [(({str(a)}) div ({str(b)})) == (IV6)])" )
        self.assertEqual( str(flatten_objective( cp.cpm_array([1,2,3])[a] )), "(IV7, [([1 2 3][IV0]) == (IV7)])" )
        self.assertEqual( str(flatten_objective( cp.cpm_array([1,2,3])[a]+b )), "((IV8) + (IV1), [([1 2 3][IV0]) == (IV8)])" )


    def test_constraint(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        self.assertEqual( str(flatten_constraint( x )), "[BV0]" )
        self.assertEqual( str(flatten_constraint( ~x )), "[~BV0]" )
        self.assertEqual( str(flatten_constraint( [x,y] )), "[BV0, BV1]" )
        self.assertEqual( str(flatten_constraint( x&y )), "[BV0, BV1]" )
        self.assertEqual( str(flatten_constraint( x&y&~z )), "[BV0, BV1, ~BV2]" )
        self.assertEqual( str(flatten_constraint( x.implies(y) )), "[(BV0) -> (BV1)]" )
        self.assertEqual( str(flatten_constraint( x|(y.implies(z)) )), "[or([BV0, ~BV1, BV2])]" )
        self.assertEqual( str(flatten_constraint( (a > 10)&x )), "[IV0 > 10, BV0]" )
        cp.boolvar() # increase counter
        self.assertEqual( str(flatten_constraint( (a > 10).implies(x) )), "[(IV0 > 10) -> (BV0)]" )
        cp.boolvar() # increase counter
        self.assertEqual( str(flatten_constraint( (a > 10) )), "[IV0 > 10]" )
        self.assertEqual( str(flatten_constraint( (a > 10) == 1 )), "[IV0 > 10]" )
        self.assertEqual( str(flatten_constraint( (a > 10) == 0 )), "[IV0 <= 10]" )
        self.assertEqual( str(flatten_constraint( (a > 10) == x )), "[(IV0 > 10) == (BV0)]" )
        #self.assertEqual( str(flatten_constraint( x == (a > 10) )), "[(IV0 > 10) == (BV0)]" ) # TODO, make it do the swap (again)
        self.assertEqual( str(flatten_constraint( (a > 10) | (b + c > 2) )), "[(BV5) or (BV6), (IV0 > 10) == (BV5), ((IV1) + (IV2) > 2) == (BV6)]" )
        self.assertEqual( str(flatten_constraint( a > 10 )), "[IV0 > 10]" )
        self.assertEqual( str(flatten_constraint( 10 > a )), "[IV0 < 10]" ) # surprising
        self.assertEqual( str(flatten_constraint( a+b > c )), "[((IV0) + (IV1)) > (IV2)]" )
        #self.assertEqual( str(flatten_constraint( c < a+b )), "[((IV0) + (IV1)) > (IV2)]" ) # TODO, make it do the swap (again)
        self.assertEqual( str(flatten_constraint( (a+b > c) == x|y )), "[(((IV0) + (IV1)) > (IV2)) == (BV7), ((BV0) or (BV1)) == (BV7)]" )

        self.assertEqual( str(flatten_constraint( a + b == c )), "[((IV0) + (IV1)) == (IV2)]" )
        #self.assertEqual( str(flatten_constraint( c != a + b )), "[((IV0) + (IV1)) != (IV2)]" ) # TODO, make it do the swap (again)
        self.assertEqual( str(flatten_constraint( ((a > 5) == (b < 3)) )), "[(IV0 > 5) == (BV8), (IV1 < 3) == (BV8)]" )

        self.assertEqual( str(flatten_constraint( cp.cpm_array([1,2,3])[a] == b )), "[([1 2 3][IV0]) == (IV1)]" )
        self.assertEqual( str(flatten_constraint( cp.cpm_array([1,2,3])[a] > b )), "[([1 2 3][IV0]) > (IV1)]" )
        cp.intvar(0,2, 4) # increase counter
        self.assertEqual( str(flatten_constraint( cp.cpm_array([1,2,3])[a] <= b )), "[([1 2 3][IV0]) <= (IV1)]" )
        self.assertEqual( str(flatten_constraint( cp.AllDifferent([a+b,b+c,c+3]) )), "[alldifferent(IV9,IV10,IV11), ((IV0) + (IV1)) == (IV9), ((IV1) + (IV2)) == (IV10), ((IV2) + 3) == (IV11)]" )

        # issue #27
        self.assertEqual( str(flatten_constraint( (a == 10).implies(b == c+d) )), "[(IV0 == 10) -> (BV9), (((IV2) + (IV3)) == (IV1)) == (BV9)]" )
        # different order should not create more tempvars
        self.assertEqual( str(flatten_constraint( (a == 10).implies(c+d == b) )), "[(IV0 == 10) -> (BV10), (((IV2) + (IV3)) == (IV1)) == (BV10)]" )
        self.assertEqual( str(flatten_constraint( a // b == c )), "[((IV0) div (IV1)) == (IV2)]" )
        self.assertEqual( str(flatten_constraint( c == a // b )), "[((IV0) div (IV1)) == (IV2)]" )

        # double negation #146
        self.assertEqual( str(flatten_constraint( ~(~(a == 7)) )), "[IV0 == 7]" )

        # negated normal form tests
        self.assertEqual( str(flatten_constraint( ~(x|y) )), "[~BV0, ~BV1]" )
        self.assertEqual( str(flatten_constraint( z.implies(~(x|y)) )), "[(BV2) -> (~BV0), (BV2) -> (~BV1)]" )
        self.assertEqual( str(flatten_constraint( ~(z.implies(~(x|y))) )), "[BV2, (BV0) or (BV1)]" )
        self.assertEqual( str(flatten_constraint(~(z.implies(~(x&y))))), "[BV2, BV0, BV1]" )
        self.assertEqual( str(flatten_constraint((~z).implies(~(x|y)))), "[(~BV2) -> (~BV0), (~BV2) -> (~BV1)]" )
        self.assertEqual( str(flatten_constraint((~z|y).implies(~(x|y)))), "[(BV0) -> (BV2), (BV0) -> (~BV1), (BV1) -> (BV2), (BV1) -> (~BV1)]" )
        self.assertEqual( str(a % 1 == 0), "(IV0) mod 1 == 0" )

        # boolexpr as numexpr
        self.assertEqual( str(flatten_constraint((a + b == 2) <= c)), "[(BV11) <= (IV2), ((IV0) + (IV1) == 2) == (BV11)]" )

        # != in boolexpr, bug #170
        self.assertEqual( str(normalized_boolexpr(x != (a == 1))), "((BV12) == (~BV0), [(IV0 == 1) == (BV12)])" )
        #simplify output
        self.assertEqual( str(normalized_boolexpr(Operator('not',[x]) == y)), "((~BV0) == (BV1), [])" )
