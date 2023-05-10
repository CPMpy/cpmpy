import unittest

import cpmpy as cp
from cpmpy.expressions.core import Operator, BoolVal, Comparison
from cpmpy.transformations.normalize import simplify_boolean, toplevel_list


class TransSimplify(unittest.TestCase):

    def setUp(self) -> None:
        self.bvs = cp.boolvar(shape=3, name="bv")
        self.ivs = cp.intvar(0, 5, shape=3, name="iv")

        self.transform = lambda x: simplify_boolean(toplevel_list(x))

    def test_bool_ops(self):
        expr = Operator("or", self.bvs.tolist() + [False])
        self.assertEqual(str(self.transform(expr)), "[or([bv[0], bv[1], bv[2]])]")

        expr = Operator("->", [self.bvs[0], True])
        self.assertEqual(str(self.transform(expr)), "[boolval(True)]")
        expr = Operator("->", [self.bvs[0], False])
        self.assertEqual(str(self.transform(expr)), "[~bv[0]]")
        expr = Operator("->", [True, self.bvs[0]])
        self.assertEqual(str(self.transform(expr)), "[bv[0]]")
        expr = Operator("->", [False, self.bvs[0]])
        self.assertEqual(str(self.transform(expr)), "[boolval(True)]")

    def test_bool_in_comp(self):
        expr = self.ivs[0] >= False
        self.assertEqual(str(self.transform(expr)), '[iv[0] >= 0]')
        expr = self.ivs[0] >= True
        self.assertEqual(str(self.transform(expr)), '[iv[0] >= 1]')

        expr = (cp.sum(self.ivs) + True) >= 10
        self.assertEqual(str(self.transform(expr)), '[sum([iv[0], iv[1], iv[2], 1]) >= 10]')

        expr = True + self.ivs[0] >= False
        self.assertEqual(str(self.transform(expr)), '[1 + (iv[0]) >= 0]')

    def test_boolvar_comps(self):
        num_args = {"<0": -1, "0": 0, "1": 1, ">0": 2}
        # test table from github (#add url)
        bv = self.bvs[0]
        test_dict = {
            "==": {"<0": False,"0": ~bv,  "1": bv,   ">0": False},
            "!=": {"<0": True, "0": bv,   "1": ~bv,  ">0": True},
            ">":  {"<0": True, "0": bv,   "1": False,">0": False},
            "<":  {"<0": False,"0": False,"1": ~bv,  ">0": True},
            ">=": {"<0": True, "0": True, "1": bv,   ">0": False},
            "<=": {"<0": False,"0": ~bv,  "1": True, ">0": True}
        }

        for op in test_dict:
            for rhs, val_should in test_dict[op].items():
                expr = Comparison(op, bv, num_args[rhs])
                print(expr)
                expr_should = BoolVal(val_should) if isinstance(val_should, bool) else val_should
                self.assertEqual(str(self.transform(expr)), str([expr_should]))