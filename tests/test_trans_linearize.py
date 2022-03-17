import unittest

from cpmpy.expressions import boolvar
from cpmpy.transformations.linearize import linearize_constraint


class TestTransLineariez(unittest.TestCase):

    def test_linearize(self):

        # Boolean
        a, b, c = [boolvar(name=var) for var in "abc"]

        # and
        cons = linearize_constraint(a & b)[0]
        self.assertEqual("(a) + (b) >= 2", str(cons))

        # or
        cons = linearize_constraint(a | b)[0]
        self.assertEqual("(a) + (b) >= 1", str(cons))

        # xor
        cons = linearize_constraint(a ^ b)[0]
        self.assertEqual("(a) + (b) == 1", str(cons))

        # implies
        cons = linearize_constraint(a.implies(b))[0]
        self.assertEqual("(a) -> (b >= 1)", str(cons))