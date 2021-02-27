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

    def test_boolexpr(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        self.assertEqual( str(flatten_boolexpr(x)), "(BV0, [])" )
        self.assertEqual( str(flatten_boolexpr(~x)), "(~BV0, [])" )

        self.assertEqual( str(flatten_boolexpr(x == y)), "(BV3, [((BV0) == (BV1)) == (BV3)])" )
        self.assertEqual( str(flatten_boolexpr(x != y)), "(BV4, [((BV0) != (BV1)) == (BV4)])" )
        self.assertEqual( str(flatten_boolexpr(x > y)), "(BV5, [((BV0) > (BV1)) == (BV5)])" )
        self.assertEqual( str(flatten_boolexpr(x <= y)), "(BV6, [((BV0) <= (BV1)) == (BV6)])" )

        self.assertEqual( str(flatten_boolexpr((a > 10) == x)), "(BV7, [((BV8) == (BV0)) == (BV7), (IV0 > 10) == (BV8)])" )
        self.assertEqual( str(flatten_boolexpr( (a > 10) == (d > 5) )), "(BV9, [((BV10) == (BV11)) == (BV9), (IV0 > 10) == (BV10), (IV3 > 5) == (BV11)])" )
        self.assertEqual( str(flatten_boolexpr( a > c )), "(BV12, [((IV0) > (IV2)) == (BV12)])" )
        self.assertEqual( str(flatten_boolexpr( a + b > c )), "(BV13, [((IV5) > (IV2)) == (BV13), ((IV0) + (IV1)) == (IV5)])" ) # TODO, suboptimal

        self.assertEqual( str(flatten_boolexpr( (a>b).implies(x) )), "(BV14, [((BV15) -> (BV0)) == (BV14), ((IV0) > (IV1)) == (BV15)])" )
        self.assertEqual( str(flatten_boolexpr( x&y )), "(BV16, [((BV0) and (BV1)) == (BV16)])" )
        self.assertEqual( str(flatten_boolexpr( x|y )), "(BV17, [((BV0) or (BV1)) == (BV17)])" )
        self.assertEqual( str(flatten_boolexpr( x.implies(y) )), "(BV18, [((BV0) -> (BV1)) == (BV18)])" )
        self.assertEqual( str(flatten_boolexpr( x.implies(y|z) )), "(BV19, [((BV0) -> (BV20)) == (BV19), ((BV1) or (BV2)) == (BV20)])" )
        self.assertEqual( str(flatten_boolexpr( (x&y).implies(y&z) )), "(BV21, [((BV22) -> (BV23)) == (BV21), ((BV0) and (BV1)) == (BV22), ((BV1) and (BV2)) == (BV23)])" )
        self.assertEqual( str(flatten_boolexpr( x.implies(y.implies(z)) )), "(BV24, [((BV0) -> (BV25)) == (BV24), ((BV1) -> (BV2)) == (BV25)])" )

        self.assertEqual( str(flatten_boolexpr( (a > 10) )), "(BV26, [(IV0 > 10) == (BV26)])" )
        self.assertEqual( str(flatten_boolexpr( (a > 10)&x&y )), "(BV27, [(and((BV28, BV0, BV1))) == (BV27), (IV0 > 10) == (BV28)])" )

    def test_numexpr(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        self.assertEqual( str(flatten_numexpr( a+b )), "(IV5, [((IV0) + (IV1)) == (IV5)])" )
        self.assertEqual( str(flatten_numexpr( a+b+c )), "(IV6, [(sum((IV0, IV1, IV2))) == (IV6)])" )
        self.assertEqual( str(flatten_numexpr( 2*a )), "(IV7, [(2 * (IV0)) == (IV7)])" ) # TODO, suboptimal
        self.assertEqual( str(flatten_numexpr( a*b )), "(IV8, [((IV0) * (IV1)) == (IV8)])" )
        self.assertEqual( str(flatten_numexpr( a/b )), "(IV9, [((IV0) / (IV1)) == (IV9)])" )
        self.assertEqual( str(flatten_numexpr( 1/b )), "(IV10, [(1 / (IV1)) == (IV10)])" )
        self.assertEqual( str(flatten_numexpr( a/1 )), "(IV0, [])" )
        self.assertEqual( str(flatten_numexpr( abs(a) )), "(IV11, [(abs((IV0,))) == (IV11)])" )
        self.assertEqual( str(flatten_numexpr( 1*a + 2*b + 3*c )), "(IV14, [(sum((IV0, IV12, IV13))) == (IV14), (2 * (IV1)) == (IV12), (3 * (IV2)) == (IV13)])" ) # TODO, suboptimal
        self.assertEqual( str(flatten_numexpr( cparray([1,2,3])[a] )), "(IV15, [[1 2 3][IV0] == IV15])" )
        self.assertEqual( str(flatten_numexpr( cparray([b+c,2,3])[a] )), "(IV17, [(IV16, 2, 3)[IV0] == IV17, ((IV1) + (IV2)) == (IV16)])" )

    def test_objective(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        self.assertEqual( str(flatten_objective( a )), "(IV0, [])" )
        self.assertEqual( str(flatten_objective( a+b )), "((IV0) + (IV1), [])" )
        self.assertEqual( str(flatten_objective( 2*a+3*b )), "((IV5) + (IV6), [(2 * (IV0)) == (IV5), (3 * (IV1)) == (IV6)])" ) # TODO, wsum
        self.assertEqual( str(flatten_objective( a/b+c )), "((IV7) + (IV2), [((IV0) / (IV1)) == (IV7)])" )
        self.assertEqual( str(flatten_objective( cparray([1,2,3])[a] )), "(IV8, [[1 2 3][IV0] == IV8])" )
        self.assertEqual( str(flatten_objective( cparray([1,2,3])[a]+b )), "((IV9) + (IV1), [[1 2 3][IV0] == IV9])" )

    def test_constraint(self):
        (a,b,c,d,e) = self.ivars[:5]
        (x,y,z) = self.bvars[:3]

        self.assertEqual( str(flatten_constraint( x )), "[BV0]" )
        self.assertEqual( str(flatten_constraint( ~x )), "[~BV0]" )
        self.assertEqual( str(flatten_constraint( [x,y] )), "[BV0, BV1]" )
        self.assertEqual( str(flatten_constraint( x&y )), "(BV0) and (BV1)" )
        self.assertEqual( str(flatten_constraint( x&y&~z )), "and([BV0, BV1, ~BV2])" )
        self.assertEqual( str(flatten_constraint( x.implies(y) )), "(BV0) -> (BV1)" )
        self.assertEqual( str(flatten_constraint( (a > 10)&x )), "[(BV3) and (BV0), (IV0 > 10) == (BV3)]" )
        self.assertEqual( str(flatten_constraint( (a > 10).implies(x) )), "[(BV4) -> (BV0), (IV0 > 10) == (BV4)]" )
        self.assertEqual( str(flatten_constraint( (a > 10) )), "[IV0 > 10]" )
        self.assertEqual( str(flatten_constraint( (a > 10) == 1 )), "[IV0 > 10]" )
        self.assertEqual( str(flatten_constraint( (a > 10) == 0 )), "[IV0 > 10 == 0]" )
        self.assertEqual( str(flatten_constraint( (a > 10) == x )), "[(IV0 > 10) == (BV0)]" )
        self.assertEqual( str(flatten_constraint( x == (a > 10) )), "[(IV0 > 10) == (BV0)]" )
        self.assertEqual( str(flatten_constraint( (a > 10) | (b + c > 2) )), "[(BV5) or (BV6), (IV0 > 10) == (BV5), (IV5 > 2) == (BV6), ((IV1) + (IV2)) == (IV5)]" )
        self.assertEqual( str(flatten_constraint( a > 10 )), "[IV0 > 10]" )
        self.assertEqual( str(flatten_constraint( 10 > a )), "[IV0 < 10]" ) # surprising
        self.assertEqual( str(flatten_constraint( a+b > c )), "[((IV0) + (IV1)) > (IV2)]" )
        self.assertEqual( str(flatten_constraint( c < a+b )), "[((IV0) + (IV1)) > (IV2)]" )
        self.assertEqual( str(flatten_constraint( (a+b > c) == x|y )), "[((IV6) > (IV2)) == (BV7), ((IV0) + (IV1)) == (IV6), ((BV0) or (BV1)) == (BV7)]" )

        self.assertEqual( str(flatten_constraint( a + b == c )), "[((IV0) + (IV1)) == (IV2)]" )
        self.assertEqual( str(flatten_constraint( c != a + b )), "[((IV0) + (IV1)) != (IV2)]" )
        self.assertEqual( str(flatten_constraint( ((a > 5) == (b < 3)) )), "[(IV0 > 5) == (BV8), (IV1 < 3) == (BV8)]" )

        self.assertEqual( str(flatten_constraint( cparray([1,2,3])[a] == b )), "[1 2 3][IV0] == IV1" )
        self.assertEqual( str(flatten_constraint( cparray([1,2,3])[a] > b )), "[(IV7) > (IV1), [1 2 3][IV0] == IV7]" )
        self.assertEqual( str(flatten_constraint( cparray([1,2,3])[a] <= b )), "[(IV8) <= (IV1), [1 2 3][IV0] == IV8]" )
        self.assertEqual( str(flatten_constraint( cp.alldifferent([a+b,b+c,c+3]) )), "[alldifferent(IV9,IV10,IV11), ((IV0) + (IV1)) == (IV9), ((IV1) + (IV2)) == (IV10), (3 + (IV2)) == (IV11)]" )
