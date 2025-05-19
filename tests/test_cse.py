import unittest
import cpmpy as cp

from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl


class TestCSE(unittest.TestCase):

    def setUp(self) -> None:
        # ensure reproducable variable names
        _IntVarImpl.counter = 0 
        _BoolVarImpl.counter = 0   


    def test_flatten(self):
        
        x,y,z = cp.intvar(0,10, shape=3, name=tuple("xyz"))
       
        nested_alldiff = cp.AllDifferent(x,y+y,z)      
        expr_dict = dict()

        flat_cons = flatten_constraint(nested_alldiff, expr_dict=expr_dict)

        self.assertEqual(len(flat_cons), 2)
        fc = flat_cons[0]
        self.assertEqual(str(fc), "alldifferent(x,IV0,z)")
        self.assertEqual(len(expr_dict), 1)

        self.assertEqual(str(next(iter(expr_dict.keys()))), "(y) + (y)")
        self.assertEqual(str(expr_dict[y + y]), "IV0")

        # next time we use y + y, it should replace it IV0
        nested_cons2 = (y + y) % 3 == 0
        flat_cons = flatten_constraint(nested_cons2, expr_dict=expr_dict)
        self.assertEqual(len(flat_cons), 1)
        self.assertEqual(str(flat_cons[0]), "(IV0) mod 3 == 0")
        self.assertEqual(len(expr_dict), 1)
        

        # should also work for Boolean variables (introduce reification)
        nested_cons = (x + y + z <= 10) | (cp.AllDifferent(x,y,z))
        flat_cons = flatten_constraint(nested_cons, expr_dict=expr_dict)
        
        self.assertEqual(len(flat_cons), 3)
        
        self.assertEqual(str(flat_cons[0]), "(BV0) or (BV1)")
        self.assertEqual(str(flat_cons[1]), "(sum([x, y, z]) <= 10) == (BV0)")
        self.assertEqual(str(flat_cons[2]), "(alldifferent(x,y,z)) == (BV1)")
        
        # next time we use x + y + z <= 10, it should replace it with BV0
        nested_cons2 = ((x + y + z <= 10) ^ cp.boolvar(name="a"))
        flat_cons = flatten_constraint(nested_cons2, expr_dict=expr_dict)

        self.assertEqual(len(flat_cons), 1)
        self.assertEqual(str(flat_cons[0]), "BV0 xor a")



    def test_decompose(self):
        x,y,z = cp.intvar(0,10, shape=3, name=tuple("xyz"))
        q = cp.intvar(0,2, name="q")

        b = cp.boolvar(name="b")
        nested_cons = b == ((cp.max([x,y,z]) + q) <= 10)
        expr_dict = dict()
        decomp = decompose_in_tree([nested_cons], expr_dict=expr_dict)
        print(decomp)
        print(expr_dict)
    
        self.assertEqual(len(decomp), 6)
        self.assertEqual(str(decomp[0]), "(b) == ((IV0) + (q) <= 10)")
        self.assertEqual(str(decomp[1]), "(IV1) == (IV0)") # TODO... this seems stupid, why do we need this (comes from _max in Maximul decomp)?
        self.assertEqual(str(decomp[2]), "or([(x) >= (IV1), (y) >= (IV1), (z) >= (IV1)])")
        self.assertEqual(str(decomp[3]), "(x) <= (IV1)")
        self.assertEqual(str(decomp[4]), "(y) <= (IV1)")
        self.assertEqual(str(decomp[5]), "(z) <= (IV1)")

        # next time we use max([x,y,z]) it should replace the max-constraint with IV0
        #  ... it seems like we should be able to do more here e.g., cp.max([x,y,z]) != should be replaced with IV0 != 42
        #  ...      but the current code-flow of decompose_in_tree and .decompose_comparison does not allow this
        nested2 = (q + cp.max([x,y,z]) != 42)
        decomp = decompose_in_tree([nested2], expr_dict=expr_dict)

        self.assertEqual(len(decomp), 1)
        self.assertEqual(str(decomp[0]), "(q) + (IV0) != 42")

        

    def test_safen(self):
        pass

    def test_only_numexpr_equality(self):
        pass

    def test_linearize(self):
        pass

    def test_only_bv_reifies(self):
        pass

    def test_only_implies(self):
        pass

    def test_to_cnf(self):
        pass

