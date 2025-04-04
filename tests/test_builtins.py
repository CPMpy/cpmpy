import unittest

import cpmpy as cp
from cpmpy.expressions.python_builtins import all as cpm_all, any as cpm_any
from cpmpy.exceptions import CPMpyException

iv = cp.intvar(-8, 8, shape=5)


class TestBuiltin(unittest.TestCase):

    def test_max(self):
        constraints = [cp.max(iv) + 9 <= 8]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())
        self.assertTrue(cp.max(iv.value()) <= -1)

        model = cp.Model(cp.max(iv).decompose_comparison('!=', 4))
        self.assertTrue(model.solve())
        self.assertNotEqual(str(cp.max(iv.value())), '4')

    def test_min(self):
        constraints = [cp.min(iv) + 9 == 8]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())
        self.assertEqual(str(cp.min(iv.value())), '-1')

        model = cp.Model(cp.min(iv).decompose_comparison('==', 4))
        self.assertTrue(model.solve())
        self.assertEqual(str(cp.min(iv.value())), '4')

    def test_abs(self):
        constraints = [cp.abs(iv[0]) + 9 <= 8]
        model = cp.Model(constraints)
        self.assertFalse(model.solve())

        #with list
        constraints = [cp.abs(iv+2) <= 8, iv < 0]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())

        constraints = [cp.abs([iv[0], iv[2], iv[1], -8]) <= 8, iv < 0]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())

        model = cp.Model(cp.abs(iv[0]).decompose_comparison('!=', 4))
        self.assertTrue(model.solve())
        self.assertNotEqual(str(cp.abs(iv[0].value())), '4')

    # Boolean builtins

    def test_all(self):
        # edge-cases
        # CPMpy False -> should return CPMpy False
        x = [cp.boolvar(), cp.BoolVal(False), cp.boolvar()]
        self.assertEqual(str(cpm_all(x)), "boolval(False)")
        # python False, but also expression! -> should return CPMpy False
        x = [cp.boolvar(), False, cp.boolvar()]
        self.assertEqual(str(cpm_all(x)), "boolval(False)")
        x = [False, cp.BoolVal(False)]
        self.assertEqual(str(cpm_all(x)), "boolval(False)")
        x = [cp.BoolVal(False), False]
        self.assertEqual(str(cpm_all(x)), "boolval(False)")
        # only Python constants
        x = [False, True]
        self.assertEqual(str(cpm_all(x)), "False")
        # one CPMPy True constant
        x = [cp.BoolVal(True)]
        self.assertEqual(str(cpm_all(x)), "boolval(True)")
        # one python True constant
        x = [True]
        self.assertEqual(str(cpm_all(x)), "True")
        # Python and CPMpy True constant
        x = [True, cp.BoolVal(True)]
        self.assertEqual(str(cpm_all(x)), "boolval(True)")
        x = [cp.BoolVal(True), True]
        self.assertEqual(str(cpm_all(x)), "boolval(True)")

        # should also work with overloaded operators
        expr = cp.BoolVal(False) & cp.BoolVal(True)
        self.assertEqual(str(expr), "boolval(False)")
        expr = False & cp.BoolVal(True)
        self.assertEqual(str(expr), "boolval(False)")
        expr = cp.BoolVal(False) & True
        self.assertEqual(str(expr), "boolval(False)")


    def test_any(self):
        # edge-cases
        # CPMpy True -> should return CPMpy True
        x = [cp.boolvar(), cp.BoolVal(True), cp.boolvar()]
        self.assertEqual(str(cpm_any(x)), "boolval(True)")
        # python True, but also expression! -> should return CPMpy True
        x = [cp.boolvar(), True, cp.boolvar()]
        self.assertEqual(str(cpm_any(x)), "boolval(True)")
        x = [True, cp.BoolVal(True)]
        self.assertEqual(str(cpm_any(x)), "boolval(True)")
        x = [cp.BoolVal(True), True]
        self.assertEqual(str(cpm_any(x)), "boolval(True)")
        # only Python constants
        x = [False, True]
        self.assertEqual(str(cpm_any(x)), "True")
        # one CPMPy True constant
        x = [cp.BoolVal(True)]
        self.assertEqual(str(cpm_any(x)), "boolval(True)")
        # one python True constant
        x = [True]
        self.assertEqual(str(cpm_any(x)), "True")
        # Python and CPMpy True constant
        x = [True, cp.BoolVal(True)]
        self.assertEqual(str(cpm_any(x)), "boolval(True)")
        x = [cp.BoolVal(True), True]  
        self.assertEqual(str(cpm_any(x)), "boolval(True)")

        expr = cp.BoolVal(False) | cp.BoolVal(True)
        self.assertEqual(str(expr), "boolval(True)")
        expr = False | cp.BoolVal(True)
        self.assertEqual(str(expr), "boolval(True)")
        expr = cp.BoolVal(False) | True
        self.assertEqual(str(expr), "boolval(True)")