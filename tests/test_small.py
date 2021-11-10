import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.transformations.flatten_model import *

class TestSmall(unittest.TestCase):



    def test_get_or_make_var(self):
        a, b, c, d, e = intvar(3, 8, shape=5)
        a.name, b.name, c.name, d.name, e.name = ["a", "b", "c", "d", "e"]
        
        # print("\n\nget_or_make_var( a + b + c )", get_or_make_var( a + (b + (c + d)) ))
        print("\n\nflatten_objective( a + b + c ))", flatten_objective( a + (b - c)) )
    
        # print("\n\nget_or_make_var( 1*a + 2*b + 3*c ))", get_or_make_var( 1*a + 2*b + 3*c ))
    
        # print("\n\nget_or_make_var( 1*a + 2*( b + c)", get_or_make_var( a + 2*( b + c)))
        # print("\n\nget_or_make_var( 1*a + 2*( b + c)", get_or_make_var( 1 * a + 2*( b + c)))
        # print("\n\nget_or_make_var( 1*a + 2*b + 3*(c + d) ))", get_or_make_var( 1*a + 2*b + 3*(c + d) ))
        # print("\n\nget_or_make_var( 1*a + 2*( 2 * b + c)", get_or_make_var( 1*a + 2*( 2 * b + c)))
        # print("\n\nget_or_make_var( 1*a + 2*( 2 * b + 3 * c) ))", get_or_make_var( 1*a + 2*( 2 * b + 3 * c)))

        print("\n\nflatten_objective(1*a + 2*b + 3*c ))", flatten_objective(1*a + 2*b + 3*c ))
        print("\nflatten_objective( 1*a + 2*( b + c)", flatten_objective( 1*a + 2*( b + c)))
        print("\n\flatten_objective( 1*a + 2*b + 3*(c + d) ))", flatten_objective( 1*a + 2*b + 3*(c + d) ))
        print("\n\flatten_objective( 1*a + 2*( 2 * b + c)", flatten_objective( 1*a + 2*( 2 * b + c)))
        print("\n\flatten_objective( 1*a + 2*( 2 * b + 3 * c) ))", flatten_objective( 1*a + 2*( 2 * b + 3 * c)))

if __name__ == '__main__':
    unittest.main()