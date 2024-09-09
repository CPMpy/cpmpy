import unittest

import cpmpy as cp
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

        model = cp.Model(cp.abs(iv[0]).decompose_comparison('!=', 4))
        self.assertTrue(model.solve())
        self.assertNotEqual(str(cp.abs(iv[0].value())), '4')