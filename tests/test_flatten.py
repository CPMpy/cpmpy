import unittest
import cpmpy as cp
from cpmpy.model_tools.flatten_model import *

class TestFlattenModel(unittest.TestCase):
    def setUp(self):
        self.ivars = cp.IntVar(1, 10, (5,))
        self.bvars = cp.BoolVar((2,))
        #self.constraints = [self.ivars != 3] # should work in future (broadcasting)
        self.constraints = [iv != 3 for iv in self.ivars]

    def test_constraints(self):
        model = cp.Model(self.constraints)
        model2 = flatten_model(model)
        self.assertTrue(isinstance(model2, cp.Model))
        self.assertTrue(hasattr(model2, 'constraints'))
        self.assertTrue(len(model2.constraints) > 1)

    def test_objective(self):
        #obj = self.ivars.sum() # should work?
        obj = np.sum(self.ivars)
        model = cp.Model(self.constraints, maximize=obj)
        model2 = flatten_model(model)
        self.assertTrue(model2.objective is not None)
        self.assertTrue(model2.objective_max)


class TestFlattenConstraint(unittest.TestCase):
    def setUp(self):
        IntVarImpl.counter = 0
        BoolVarImpl.counter = 0
        self.ivars = cp.IntVar(1, 10, shape=(5,))
        self.bvars = cp.BoolVar((3,))

    def test_eq(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        e = (x == y) 
        self.assertEqual( e, flatten_constraint(e) )
        e = (x == ~y) 
        self.assertEqual( e, flatten_constraint(e) )
        e = (a == b) 
        self.assertEqual( e, flatten_constraint(e) )

    def test_nq(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        e = (x != y) 
        self.assertEqual( e, flatten_constraint(e) )
        e = (x != ~y) 
        self.assertEqual( e, flatten_constraint(e) )
        e = (a != b) 
        self.assertEqual( e, flatten_constraint(e) )

    def test_eq_comp(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        e = ((a > 5) == x)
        self.assertEqual( e, flatten_constraint(e) )
        e = (x == (b < 3))
        self.assertEqual( e, flatten_constraint(e) )
        e = ((a > 5) == (b < 3))
        self.assertEqual(len(flatten_constraint(e)), 2)
    
class TestFlattenExpr(unittest.TestCase):
    def setUp(self):
        IntVarImpl.counter = 0
        BoolVarImpl.counter = 0
        self.ivars = cp.IntVar(1, 10, shape=(5,))
        self.bvars = cp.BoolVar((3,))

    # not directly tested on its on, new functions 'normalized_boolexpr' and 'normalized_numexpr'

    def test_get_or_make_var__bool(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        self.assertEqual( str(get_or_make_var(x)), "(BV0, [])" )
        self.assertEqual( str(get_or_make_var(~x)), "(~BV0, [])" )

        self.assertEqual( str(get_or_make_var(x == y)), "(BV3, [((BV0) == (BV1)) == (BV3)])" )
        self.assertEqual( str(get_or_make_var(x != y)), "(BV4, [((BV0) != (BV1)) == (BV4)])" )
        self.assertEqual( str(get_or_make_var(x > y)), "(BV5, [((BV0) > (BV1)) == (BV5)])" )
        self.assertEqual( str(get_or_make_var(x <= y)), "(BV6, [((BV0) <= (BV1)) == (BV6)])" )

        self.assertEqual( str(get_or_make_var((a > 10) == x)), "(BV7, [((IV0 > 10) == (BV0)) == (BV7)])" )
        BoolVar() # increase counter
        self.assertEqual( str(get_or_make_var( (a > 10) == (d > 5) )), "(BV10, [((IV0 > 10) == (BV9)) == (BV10), (IV3 > 5) == (BV9)])" )
        BoolVar() # increase counter
        self.assertEqual( str(get_or_make_var( a > c )), "(BV12, [((IV0) > (IV2)) == (BV12)])" )
        self.assertEqual( str(get_or_make_var( a + b > c )), "(BV13, [(((IV0) + (IV1)) > (IV2)) == (BV13)])" )
        IntVar(0,2) # increase counter

        self.assertEqual( str(get_or_make_var( (a>b).implies(x) )), "(BV15, [((~BV14) or (BV0)) == (BV15), ((IV0) > (IV1)) == (BV14)])" )
        self.assertEqual( str(get_or_make_var( x&y )), "(BV16, [((BV0) and (BV1)) == (BV16)])" )
        self.assertEqual( str(get_or_make_var( x|y )), "(BV17, [((BV0) or (BV1)) == (BV17)])" )
        self.assertEqual( str(get_or_make_var( x.implies(y) )), "(BV18, [((~BV0) or (BV1)) == (BV18)])" )
        self.assertEqual( str(get_or_make_var( x.implies(y|z) )), "(BV20, [((~BV0) or (BV19)) == (BV20), ((BV1) or (BV2)) == (BV19)])" )
        self.assertEqual( str(get_or_make_var( (x&y).implies(y&z) )), "(BV23, [((~BV21) or (BV22)) == (BV23), ((BV0) and (BV1)) == (BV21), ((BV1) and (BV2)) == (BV22)])" )
        self.assertEqual( str(get_or_make_var( x.implies(y.implies(z)) )), "(BV25, [((~BV0) or (BV24)) == (BV25), ((~BV1) or (BV2)) == (BV24)])" )

        self.assertEqual( str(get_or_make_var( (a > 10) )), "(BV26, [(IV0 > 10) == (BV26)])" )
        self.assertEqual( str(get_or_make_var( (a > 10)&x&y )), "(BV28, [(and((BV27, BV0, BV1))) == (BV28), (IV0 > 10) == (BV27)])" )

    def test_get_or_make_var__num(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        self.assertEqual( str(get_or_make_var( a+b )), "(IV5, [((IV0) + (IV1)) == (IV5)])" )
        self.assertEqual( str(get_or_make_var( a+b+c )), "(IV6, [(sum((IV0, IV1, IV2))) == (IV6)])" )
        self.assertEqual( str(get_or_make_var( 2*a )), "(IV7, [(2 * (IV0)) == (IV7)])" ) # TODO, suboptimal
        self.assertEqual( str(get_or_make_var( a*b )), "(IV8, [((IV0) * (IV1)) == (IV8)])" )
        self.assertEqual( str(get_or_make_var( a/b )), "(IV9, [((IV0) / (IV1)) == (IV9)])" )
        self.assertEqual( str(get_or_make_var( 1/b )), "(IV10, [(1 / (IV1)) == (IV10)])" )
        self.assertEqual( str(get_or_make_var( a/1 )), "(IV0, [])" )
        self.assertEqual( str(get_or_make_var( abs(a) )), "(IV11, [(abs((IV0,))) == (IV11)])" )
        self.assertEqual( str(get_or_make_var( 1*a + 2*b + 3*c )), "(IV14, [(sum((IV0, IV12, IV13))) == (IV14), (2 * (IV1)) == (IV12), (3 * (IV2)) == (IV13)])" ) # TODO, suboptimal
        self.assertEqual( str(get_or_make_var( cparray([1,2,3])[a] )), "(IV15, [([1 2 3][IV0]) == (IV15)])" )
        self.assertEqual( str(get_or_make_var( cparray([b+c,2,3])[a] )), "(IV17, [((IV16, 2, 3)[IV0]) == (IV17), ((IV1) + (IV2)) == (IV16)])" )

    def test_objective(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        self.assertEqual( str(flatten_objective( a )), "(IV0, [])" )
        self.assertEqual( str(flatten_objective( a+b )), "((IV0) + (IV1), [])" )
        self.assertEqual( str(flatten_objective( 2*a+3*b )), "((IV5) + (IV6), [(2 * (IV0)) == (IV5), (3 * (IV1)) == (IV6)])" ) # TODO, wsum
        self.assertEqual( str(flatten_objective( a/b+c )), "((IV7) + (IV2), [((IV0) / (IV1)) == (IV7)])" )
        self.assertEqual( str(flatten_objective( cparray([1,2,3])[a] )), "(IV8, [([1 2 3][IV0]) == (IV8)])" )
        self.assertEqual( str(flatten_objective( cparray([1,2,3])[a]+b )), "((IV9) + (IV1), [([1 2 3][IV0]) == (IV9)])" )
        self.assertEqual( str(flatten_objective( a+b-c )), "(sum((IV0, IV1, IV10)), [(-1 * (IV2)) == (IV10)])" )

    def test_constraint(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        self.assertEqual( str(flatten_constraint( x )), "[BV0]" )
        self.assertEqual( str(flatten_constraint( ~x )), "[~BV0]" )
        self.assertEqual( str(flatten_constraint( [x,y] )), "[BV0, BV1]" )
        self.assertEqual( str(flatten_constraint( x&y )), "[BV0, BV1]" )
        self.assertEqual( str(flatten_constraint( x&y&~z )), "[BV0, BV1, ~BV2]" )
        self.assertEqual( str(flatten_constraint( x.implies(y) )), "[(BV0) -> (BV1)]" )
        self.assertEqual( str(flatten_constraint( (a > 10)&x )), "[IV0 > 10, BV0]" )
        BoolVar() # increase counter
        self.assertEqual( str(flatten_constraint( (a > 10).implies(x) )), "[(IV0 > 10) -> (BV0)]" )
        BoolVar() # increase counter
        self.assertEqual( str(flatten_constraint( (a > 10) )), "[IV0 > 10]" )
        self.assertEqual( str(flatten_constraint( (a > 10) == 1 )), "[IV0 > 10]" )
        self.assertEqual( str(flatten_constraint( (a > 10) == 0 )), "[IV0 > 10 == 0]" )
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

        self.assertEqual( str(flatten_constraint( cparray([1,2,3])[a] == b )), "[([1 2 3][IV0]) == (IV1)]" )
        self.assertEqual( str(flatten_constraint( cparray([1,2,3])[a] > b )), "[([1 2 3][IV0]) > (IV1)]" )
        IntVar(0,2, 4) # increase counter
        self.assertEqual( str(flatten_constraint( cparray([1,2,3])[a] <= b )), "[([1 2 3][IV0]) <= (IV1)]" )
        self.assertEqual( str(flatten_constraint( cp.alldifferent([a+b,b+c,c+3]) )), "[alldifferent(IV9,IV10,IV11), ((IV0) + (IV1)) == (IV9), ((IV1) + (IV2)) == (IV10), (3 + (IV2)) == (IV11)]" )

        # issue #27
        self.assertEqual( str(flatten_constraint( (a == 10).implies(b == c+d) )), "[(IV0 == 10) -> (BV9), (((IV2) + (IV3)) == (IV1)) == (BV9)]" )
        # different order should not create more tempvars
        self.assertEqual( str(flatten_constraint( (a == 10).implies(c+d == b) )), "[(IV0 == 10) -> (BV10), (((IV2) + (IV3)) == (IV1)) == (BV10)]" )
        self.assertEqual( str(flatten_constraint( a / b == c )), "[((IV0) / (IV1)) == (IV2)]" )
        self.assertEqual( str(flatten_constraint( c == a / b )), "[((IV0) / (IV1)) == (IV2)]" )

