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
        expr = Operator("or", self.bvs.tolist() + [True])
        self.assertEqual(str(self.transform(expr)), "[boolval(True)]")

        expr = Operator("and", self.bvs.tolist() + [False]) + self.ivs[0] >= 10
        self.assertEqual(str(self.transform(expr)), "[0 + (iv[0]) >= 10]")
        expr = Operator("and", self.bvs.tolist() + [True]) + self.ivs[0] >= 10
        self.assertEqual(str(self.transform(expr)), "[(and([bv[0], bv[1], bv[2]])) + (iv[0]) >= 10]")


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
        num_args = {"<0": -1, "0": 0, "]0..1[": 0.5, "1": 1, ">0": 2}
        # test table from github (#add url)
        bv = self.bvs[0]
        test_dict = {
            "==": {"<0": False,"0": ~bv,  "]0..1[":False,"1": bv,   ">0": False},
            "!=": {"<0": True, "0": bv,   "]0..1[":True, "1": ~bv,  ">0": True},
            ">":  {"<0": True, "0": bv,   "]0..1[":bv,   "1": False,">0": False},
            "<":  {"<0": False,"0": False,"]0..1[":~bv,  "1": ~bv,  ">0": True},
            ">=": {"<0": True, "0": True, "]0..1[":bv,   "1": bv,   ">0": False},
            "<=": {"<0": False,"0": ~bv,  "]0..1[":~bv,  "1": True, ">0": True}
        }

        for op in test_dict:
            for rhs, val_should in test_dict[op].items():
                expr = Comparison(op, bv, num_args[rhs])
                print(expr)
                expr_should = BoolVal(val_should) if isinstance(val_should, bool) else val_should
                self.assertEqual(str(self.transform(expr)), str([expr_should]))


    def test_simplify_expressions(self):
        # global constraints
        expr = cp.AllDifferent(self.ivs) == 0
        self.assertEqual(str(self.transform(expr)), '[not([alldifferent(iv[0],iv[1],iv[2])])]')
        expr = 0 == cp.AllDifferent(self.ivs)
        self.assertEqual(str(self.transform(expr)), '[not([alldifferent(iv[0],iv[1],iv[2])])]')
        # with constant, does not change (surprisingly? but we cannot check what the res type is...)
        expr = cp.AllDifferent(self.ivs.tolist() + [False]) == 0
        self.assertEqual(str(self.transform(expr)), '[not([alldifferent(iv[0],iv[1],iv[2],boolval(False))])]')
        expr = 0 == cp.AllDifferent(self.ivs.tolist() + [True])
        self.assertEqual(str(self.transform(expr)), '[not([alldifferent(iv[0],iv[1],iv[2],boolval(True))])]')

        # global functions
        expr = cp.max(self.ivs) == 0
        self.assertEqual(str(self.transform(expr)), '[max(iv[0],iv[1],iv[2]) == 0]')
        expr = 0 == cp.max(self.ivs)
        self.assertEqual(str(self.transform(expr)), '[max(iv[0],iv[1],iv[2]) == 0]')
        # with constant, does not change (surprisingly? but we cannot check what the res type is...)
        expr = cp.max(self.ivs.tolist() + [False]) == 0
        self.assertEqual(str(self.transform(expr)), '[max(iv[0],iv[1],iv[2],boolval(False)) == 0]')
        expr = 0 == cp.max(self.ivs.tolist() + [True])
        self.assertEqual(str(self.transform(expr)), '[max(iv[0],iv[1],iv[2],boolval(True)) == 0]')

        expr = (self.ivs[0] <= self.ivs[1]) == 0
        self.assertEqual(str(self.transform(expr)), '[not([(iv[0]) <= (iv[1])])]')

        expr = (self.ivs[0] == self.ivs[1]) == 1
        self.assertEqual(str(self.transform(expr)), '[(iv[0]) == (iv[1])]')

        # very nested one
        expr = Operator("and", self.bvs[:1].tolist() + [BoolVal(False)]) == Operator("or", self.bvs)
        self.assertEqual(str(self.transform(expr)), '[and([~bv[0], ~bv[1], ~bv[2]])]')

    # issue #322
    def test_with_floats(self):
        expr = self.bvs[0] == 1.0
        self.assertEqual(str(self.transform(expr)), "[bv[0]]")
        expr = cp.AllDifferent(self.ivs) == 4.2
        self.assertEqual(str(self.transform(expr)), '[boolval(False)]')
        expr = 0.0 == cp.AllDifferent(self.ivs)
        self.assertEqual(str(self.transform(expr)), '[not([alldifferent(iv[0],iv[1],iv[2])])]')

        expr = (self.ivs[0] <= self.ivs[1]) < 0.8
        self.assertEqual(str(self.transform(expr)), '[(iv[0]) > (iv[1])]')

        expr = (self.ivs[0] == self.ivs[1]) == 1.0
        self.assertEqual(str(self.transform(expr)), '[(iv[0]) == (iv[1])]')


    def test_nested_boolval(self):

        bv = cp.boolvar(name="bv")
        x = cp.intvar(0, 3, name="x")
        cons = (x == 2) == (bv == 4)
        self.assertEqual(str(self.transform(cons)), "[x != 2]")
        self.assertTrue(cp.Model(cons).solve())

        # Simplify boolean expressions nested within a weighted sum
        #   wsum([1, 2], [bv[0] != 0, bv[1] != 1]) ----> wsum([1, 2], [bv[0], ~bv[1]])
        bv = cp.boolvar(name="bv", shape=2)
        weights = cp.cpm_array([1, 2])
        bool_as_ints = cp.cpm_array([0, 1])
        cons = sum( weights * (bv != bool_as_ints) ) == 1
        self.assertEqual(str(self.transform(cons)), "[sum([1, 2] * [bv[0], ~bv[1]]) == 1]")
        self.assertTrue(cp.Model(cons).solve())

