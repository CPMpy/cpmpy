import cpmpy as cp

from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.flatten_model import flatten_constraint, flatten_objective
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl
from cpmpy.transformations.linearize import linearize_constraint
from cpmpy.transformations.reification import only_bv_reifies


class TestCSE:

    def setup_method(self) -> None:
        # ensure reproducable variable names
        _IntVarImpl.counter = 0 
        _BoolVarImpl.counter = 0   


    def test_flatten(self):
        
        x,y,z = cp.intvar(0,10, shape=3, name=tuple("xyz"))
       
        nested_alldiff = cp.AllDifferent(x,y+y,z)      
        csemap = dict()

        flat_cons = flatten_constraint(nested_alldiff, csemap=csemap)

        assert len(flat_cons) == 2
        fc = flat_cons[0]
        assert str(fc) == "alldifferent(x,IV0,z)"
        assert len(csemap) == 1

        assert str(next(iter(csemap.keys()))) == "(y) + (y)"
        assert str(csemap[y + y]) == "IV0"

        # next time we use y + y, it should replace it IV0
        nested_cons2 = (y + y) % 3 == 0
        flat_cons = flatten_constraint(nested_cons2, csemap=csemap)
        assert len(flat_cons) == 1
        assert str(flat_cons[0]) == "(IV0) mod 3 == 0"
        assert len(csemap) == 1
        

        # should also work for Boolean variables (introduce reification)
        nested_cons = (x + y + z <= 10) | (cp.AllDifferent(x,y,z))
        flat_cons = flatten_constraint(nested_cons, csemap=csemap)
        
        assert len(flat_cons) == 3
        
        assert str(flat_cons[0]) == "(BV0) or (BV1)"
        assert str(flat_cons[1]) == "(sum([x, y, z]) <= 10) == (BV0)"
        assert str(flat_cons[2]) == "(alldifferent(x,y,z)) == (BV1)"
        
        # next time we use x + y + z <= 10, it should replace it with BV0
        nested_cons2 = ((x + y + z <= 10) ^ cp.boolvar(name="a"))
        flat_cons = flatten_constraint(nested_cons2, csemap=csemap)

        assert len(flat_cons) == 1
        assert str(flat_cons[0]) == "BV0 xor a"



    def test_decompose(self):
        x,y,z = cp.intvar(0,10, shape=3, name=tuple("xyz"))
        q = cp.intvar(0,2, name="q")

        b = cp.boolvar(name="b")
        nested_cons = b == ((cp.max([x,y,z]) + q) <= 10)
        csemap = dict()
        decomp = decompose_in_tree([nested_cons], csemap=csemap)
    
        assert len(decomp) == 5
        assert str(decomp[0]) == "(b) == ((IV0) + (q) <= 10)"
        assert str(decomp[1]) == "(IV0) >= (x)"
        assert str(decomp[2]) == "(IV0) >= (y)"
        assert str(decomp[3]) == "(IV0) >= (z)"
        assert str(decomp[4]) == "or([(IV0) <= (x), (IV0) <= (y), (IV0) <= (z)])"


        # next time we use max([x,y,z]) it should replace the max-constraint with IV0
        nested2 = (q + cp.max([x,y,z]) != 42)
        decomp = decompose_in_tree([nested2], csemap=csemap)

        assert len(decomp) == 1
        assert str(decomp[0]) == "(q) + (IV0) != 42"

        # also in non-nested cases
        nested3 = (cp.max([x, y, z]) == 42)
        decomp = decompose_in_tree([nested3], csemap=csemap)

        assert len(decomp) == 1
        assert str(decomp[0]) == "IV0 == 42"


    def test_only_numexpr_equality(self):
        x,y,z = cp.intvar(0,10, shape=3, name=tuple("xyz"))

        cons = cp.max([x,y,z]) <= 42
        csemap = dict()
        eq_cons = only_numexpr_equality([cons], csemap=csemap)
        
        assert set([str(c) for c in eq_cons]) == {"(max(x,y,z)) == (IV0)", "IV0 <= 42"}
        assert len(csemap) == 1
        
        # next time we use max([x,y,z]) it should replace it with IV0
        non_eq_cons = cp.max([x,y,z]) != 1337
        eq_cons = only_numexpr_equality([non_eq_cons], csemap=csemap)
        assert len(eq_cons) == 1
        assert str(eq_cons[0]) == "IV0 != 1337"
        assert len(csemap) == 1

    def test_linearize(self):
        x,y,z = cp.intvar(0,10, shape=3, name=tuple("xyz"))
        
        cons = cp.max(x,y) < z
        csemap = dict()
        lin_cons = linearize_constraint([cons], supported={"max"}, csemap=csemap)
        
        assert len(lin_cons) == 2
        assert str(lin_cons[0]) == "(max(x,y)) <= (IV0)"
        assert str(lin_cons[1]) == "sum([1, -1] * [z, IV0]) == 1"
        
        # next time we use z - 1 it should replace it with IV0
        # ... not sure how to find a test for this...

    def test_objective(self):

        x,y,z = cp.intvar(0,10, shape=3, name=tuple("xyz"))

        obj = cp.max(x+y,z) - cp.min(x+y,z)

        csemap = dict()
        flat_obj, cons = flatten_objective(obj, csemap=csemap)
        assert len(cons) == 3
        assert len(csemap) == 3
        assert set(csemap.keys()) == \
                            {cp.max(x+y,z), cp.min(x+y,z), x+y}

        # assume we did some transformations before
        csemap = {cp.max(x+y,z) : cp.intvar(0,20, name="aux")}
        flat_obj, cons = flatten_objective(obj, csemap=csemap)
        assert len(cons) == 2# just replaced max with aux var
        assert len(csemap) == 3
        assert set(csemap.keys()) == \
                            {cp.max(x + y, z), cp.min(x + y, z), x + y}


    ### other transformations only use csemap as argument to flatten_constraint internally, not sure how to easily test them
