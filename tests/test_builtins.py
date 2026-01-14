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

        _max, define = cp.max(iv).decompose()
        model = cp.Model(_max != 4, define)

        self.assertTrue(model.solve())
        self.assertNotEqual(max(iv.value()), 4)
        self.assertNotEqual(cp.max(iv).value(), 4)

    def test_min(self):
        constraints = [cp.min(iv) + 9 == 8]
        model = cp.Model(constraints)
        self.assertTrue(model.solve())
        self.assertEqual(str(cp.min(iv.value())), '-1')

        _min, define = cp.max(iv).decompose()
        model = cp.Model(_min != 4, define)

        self.assertTrue(model.solve())
        self.assertNotEqual(min(iv.value()), 4)
        self.assertNotEqual(cp.min(iv).value(), 4)


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

        _abs, define = cp.abs(iv[0]).decompose()
        model = cp.Model(_abs != 4, define)

        self.assertTrue(model.solve())
        self.assertNotEqual(abs(iv[0].value()), 4)
        self.assertNotEqual(cp.abs(iv[0]).value(), 4)

    # Boolean builtins
    def test_all(self):
        # edge-cases
        # Only CPMpy expressions
        x = [cp.boolvar(), cp.BoolVal(False), cp.boolvar()]
        self.assertEqual(str(cpm_all(x)), "boolval(False)")        
        x = [cp.BoolVal(True)]
        self.assertEqual(str(cpm_all(x)), "boolval(True)")
        x = [cp.BoolVal(False)]
        self.assertEqual(str(cpm_all(x)), "boolval(False)")

        # mix of Python and CPMpy expressions
        x = [cp.boolvar(), False, cp.boolvar()]
        self.assertEqual(str(cpm_all(x)), "boolval(False)")
        x = [False, cp.BoolVal(False)]
        self.assertEqual(str(cpm_all(x)), "boolval(False)")
        x = [False, cp.BoolVal(True)]
        self.assertEqual(str(cpm_all(x)), "boolval(False)")
        x = [cp.BoolVal(False), False]
        self.assertEqual(str(cpm_all(x)), "boolval(False)")
        x = [cp.BoolVal(True), False]
        self.assertEqual(str(cpm_all(x)), "boolval(False)")

        # only Python constants, should override default
        x = [False, True]
        self.assertEqual(str(cpm_all(x)), "False")
        x = []
        self.assertEqual(str(cpm_all(x)), "True")

        # should also work with overloaded operators
        expr = cp.BoolVal(False) & cp.BoolVal(True)
        self.assertEqual(str(expr), "boolval(False)")
        expr = False & cp.BoolVal(True)
        self.assertEqual(str(expr), "boolval(False)")
        expr = cp.BoolVal(False) & True
        self.assertEqual(str(expr), "boolval(False)")

        # 1 and 0 are not Boolean
        self.assertRaises(ValueError, lambda : cp.BoolVal(False) & 1)
        self.assertRaises(ValueError, lambda : cp.BoolVal(False) & 0)

    def test_any(self):
        # edge-cases

        # Only CPMpy expressions
        x = [cp.boolvar(), cp.BoolVal(True), cp.boolvar()]
        self.assertEqual(str(cpm_any(x)), "boolval(True)")
        x = [cp.BoolVal(True)]
        self.assertEqual(str(cpm_any(x)), "boolval(True)")
        x = [cp.BoolVal(False)]
        self.assertEqual(str(cpm_any(x)), "boolval(False)")
        

        # mix of Python and CPMpy expressions
        x = [cp.boolvar(), True, cp.boolvar()]
        self.assertEqual(str(cpm_any(x)), "boolval(True)")
        x = [True, cp.BoolVal(True)]
        self.assertEqual(str(cpm_any(x)), "boolval(True)")
        x = [False, cp.BoolVal(False)]
        self.assertEqual(str(cpm_any(x)), "boolval(False)")
        x = [cp.BoolVal(True), True]
        self.assertEqual(str(cpm_any(x)), "boolval(True)")
        
        # only Python constants, should override default
        x = [False, True]
        self.assertEqual(str(cpm_any(x)), "True")
        x = []
        self.assertEqual(str(cpm_any(x)), "False")
        
        # should also work with overloaded operators
        expr = cp.BoolVal(False) | cp.BoolVal(True)
        self.assertEqual(str(expr), "boolval(True)")
        expr = False | cp.BoolVal(True)
        self.assertEqual(str(expr), "boolval(True)")
        expr = cp.BoolVal(False) | True
        self.assertEqual(str(expr), "boolval(True)")

        # 1 and 0 are not Boolean
        self.assertRaises(ValueError, lambda : cp.BoolVal(False) | 1)
        self.assertRaises(ValueError, lambda : cp.BoolVal(False) | 0)