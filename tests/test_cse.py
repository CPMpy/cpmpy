import unittest
import cpmpy as cp

from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl
from cpmpy.transformations.linearize import linearize_constraint
from cpmpy.transformations.reification import only_bv_reifies


class TestCSE(unittest.TestCase):

    def setUp(self) -> None:
        # ensure reproducable variable names
        _IntVarImpl.counter = 0 
        _BoolVarImpl.counter = 0   


    def test_flatten(self):
        
        x,y,z = cp.intvar(0,10, shape=3, name=tuple("xyz"))
       
        nested_alldiff = cp.AllDifferent(x,y+y,z)      
        csemap = dict()

        flat_cons = flatten_constraint(nested_alldiff, csemap=csemap)

        self.assertEqual(len(flat_cons), 2)
        fc = flat_cons[0]
        self.assertEqual(str(fc), "alldifferent(x,IV0,z)")
        self.assertEqual(len(csemap), 1)

        self.assertEqual(str(next(iter(csemap.keys()))), "(y) + (y)")
        self.assertEqual(str(csemap[y + y]), "IV0")

        # next time we use y + y, it should replace it IV0
        nested_cons2 = (y + y) % 3 == 0
        flat_cons = flatten_constraint(nested_cons2, csemap=csemap)
        self.assertEqual(len(flat_cons), 1)
        self.assertEqual(str(flat_cons[0]), "(IV0) mod 3 == 0")
        self.assertEqual(len(csemap), 1)
        

        # should also work for Boolean variables (introduce reification)
        nested_cons = (x + y + z <= 10) | (cp.AllDifferent(x,y,z))
        flat_cons = flatten_constraint(nested_cons, csemap=csemap)
        
        self.assertEqual(len(flat_cons), 3)
        
        self.assertEqual(str(flat_cons[0]), "(BV0) or (BV1)")
        self.assertEqual(str(flat_cons[1]), "(sum([x, y, z]) <= 10) == (BV0)")
        self.assertEqual(str(flat_cons[2]), "(alldifferent(x,y,z)) == (BV1)")
        
        # next time we use x + y + z <= 10, it should replace it with BV0
        nested_cons2 = ((x + y + z <= 10) ^ cp.boolvar(name="a"))
        flat_cons = flatten_constraint(nested_cons2, csemap=csemap)

        self.assertEqual(len(flat_cons), 1)
        self.assertEqual(str(flat_cons[0]), "BV0 xor a")



    def test_decompose(self):
        x,y,z = cp.intvar(0,10, shape=3, name=tuple("xyz"))
        q = cp.intvar(0,2, name="q")

        b = cp.boolvar(name="b")
        nested_cons = b == ((cp.max([x,y,z]) + q) <= 10)
        csemap = dict()
        decomp = decompose_in_tree([nested_cons], csemap=csemap)
    
        self.assertEqual(len(decomp), 6)
        self.assertEqual(str(decomp[0]), "(b) == ((IV0) + (q) <= 10)")
        self.assertEqual(str(decomp[1]), "(IV1) == (IV0)") # TODO... this seems stupid, why do we need this (comes from _max in Maximul decomp)?
        self.assertEqual(str(decomp[2]), "or([(x) >= (IV1), (y) >= (IV1), (z) >= (IV1)])")
        self.assertEqual(str(decomp[3]), "(x) <= (IV1)")
        self.assertEqual(str(decomp[4]), "(y) <= (IV1)")
        self.assertEqual(str(decomp[5]), "(z) <= (IV1)")

        # next time we use max([x,y,z]) it should replace the max-constraint with IV0
        #  ... it seems like we should be able to do more here e.g., cp.max([x,y,z]) != 42 should be replaced with IV0 != 42
        #  ...      but the current code-flow of decompose_in_tree and .decompose_comparison does not allow this
        nested2 = (q + cp.max([x,y,z]) != 42)
        decomp = decompose_in_tree([nested2], csemap=csemap)

        self.assertEqual(len(decomp), 1)
        self.assertEqual(str(decomp[0]), "(q) + (IV0) != 42")

    def test_only_numexpr_equality(self):
        x,y,z = cp.intvar(0,10, shape=3, name=tuple("xyz"))

        cons = cp.max([x,y,z]) <= 42
        csemap = dict()
        eq_cons = only_numexpr_equality([cons], csemap=csemap)
        
        self.assertEqual(len(eq_cons), 2)
        self.assertEqual(str(eq_cons[0]), "(max(x,y,z)) == (IV0)")
        self.assertEqual(str(eq_cons[1]), "IV0 <= 42")
        self.assertEqual(len(csemap), 1)
        
        # next time we use max([x,y,z]) it should replace it with IV0
        non_eq_cons = cp.max([x,y,z]) != 1337
        eq_cons = only_numexpr_equality([non_eq_cons], csemap=csemap)
        self.assertEqual(len(eq_cons), 1)
        self.assertEqual(str(eq_cons[0]), "IV0 != 1337")
        self.assertEqual(len(csemap), 1)

    def test_linearize(self):
        x,y,z = cp.intvar(0,10, shape=3, name=tuple("xyz"))
        
        cons = cp.max(x,y) < z
        csemap = dict()
        lin_cons = linearize_constraint([cons], supported={"max"}, csemap=csemap)
        
        self.assertEqual(len(lin_cons), 2)
        self.assertEqual(str(lin_cons[0]), "(max(x,y)) <= (IV0)")
        self.assertEqual(str(lin_cons[1]), "sum([1, -1] * [z, IV0]) == 1")
        
        # next time we use z - 1 it should replace it with IV0
        # ... not sure how to find a test for this...        

    ### other transformations only use csemap as argument to flatten_constraint internally, not sure how to easily test them

