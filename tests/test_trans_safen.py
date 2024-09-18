import unittest

import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.safening import no_partial_functions
from cpmpy.expressions.utils import argval


class TestTransLinearize(unittest.TestCase):

    def test_division_by_zero(self):
        a = cp.intvar(1, 10, name="a")
        b = cp.intvar(0, 10, name="b")
        expr = (a // b) == 3

        safe_expr = no_partial_functions([expr])
        self.assertTrue(cp.Model(safe_expr).solve())
        self.assertTrue(argval(safe_expr))

        safened = no_partial_functions([expr | ~expr])
        solcount = cp.Model(safened).solveAll()
        self.assertEqual(solcount, 110)

    def test_division_by_zero_proper_hole(self):
        a = cp.intvar(1, 10, name="a")
        b = cp.intvar(-1, 10, name="b")
        expr = (a // b) == 3

        safe_expr = no_partial_functions([expr])
        self.assertTrue(cp.Model(safe_expr).solve())
        self.assertTrue(argval(safe_expr))

        safened = no_partial_functions([expr | ~expr])
        solcount = cp.Model(safened).solveAll()
        self.assertEqual(solcount, 120)


    def test_element_out_of_bounds(self):
        arr = cp.intvar(1,3, shape=3, name="x")
        idx = cp.intvar(-1, 4, name="i")
        expr = arr[idx] == 2

        safe_expr = no_partial_functions([expr])
        self.assertTrue(cp.Model(safe_expr).solve())
        self.assertTrue(argval(safe_expr))

        safened = no_partial_functions([expr | ~expr])
        solcount = cp.Model(safened).solveAll()
        self.assertEqual(solcount, 162)
    
    def test_multiple_partial_functions(self):
        a = cp.intvar(1, 5)
        b = cp.intvar(0, 2)
        arr = cp.intvar(1, 3, shape=3, name="x")
        idx = cp.intvar(-1, 4, name="i")

        expr = (a / b + arr[idx]) == 2

        safe_expr = no_partial_functions([expr])
        self.assertTrue(cp.Model(safe_expr).solve())
        self.assertTrue(argval(safe_expr))

        safened = no_partial_functions([expr | ~expr])
        solcount = cp.Model(safened).solveAll()
        self.assertEqual(solcount, 15*162)

    def test_nested_partial_functions(self):
        a = cp.intvar(1, 10)
        arr = cp.intvar(0,3, shape=3, name="x")
        idx = cp.intvar(-1, 4, name="i")

        expr = (a / arr[idx]) == 2

        safe_expr = no_partial_functions([expr])
        self.assertTrue(cp.Model(safe_expr).solve())
        self.assertTrue(argval(safe_expr))

        safened = no_partial_functions([expr | ~expr])
        solcount = cp.Model(safened).solveAll()
        self.assertEqual(solcount, 10*(4**3)*6)


    # def test_division_by_constant_zero(self):
    #     a = cp.intvar(1, 10)
    #     expr = (a / cp.intvar(0,0)) == 2
    #     safened = safen([expr | ~expr])
    #     solcount = cp.Model(safened).solveAll()
    #     self.assertEqual(solcount, 10)
    