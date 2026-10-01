import pytest

import cpmpy as cp
from cpmpy.expressions.core import Operator, BoolVal, Comparison
from cpmpy.transformations.negation import push_down_negation
from cpmpy.transformations.normalize import simplify_boolean, toplevel_list


class TestTransSimplify:

    def setup_method(self) -> None:
        self.bvs = cp.boolvar(shape=3, name="bv")
        self.ivs = cp.intvar(0, 5, shape=3, name="iv")

        self.transform = lambda x: simplify_boolean(push_down_negation(toplevel_list(x)))

    def test_bool_ops(self):
        expr = Operator("or", self.bvs.tolist() + [False])
        assert str(self.transform(expr)) == "[or(bv[0], bv[1], bv[2])]"
        expr = Operator("or", self.bvs.tolist() + [True])
        assert str(self.transform(expr)) == "[boolval(True)]"
        # False not at the end: remaining args after the constant must be kept
        expr = Operator("or", [self.bvs[0], False, self.bvs[1]])
        assert str(self.transform(expr)) == "[(bv[0]) or (bv[1])]"

        expr = Operator("and", self.bvs.tolist() + [False]) + self.ivs[0] >= 10
        assert str(self.transform(expr)) == "[0 + (iv[0]) >= 10]"
        expr = Operator("and", self.bvs.tolist() + [True]) + self.ivs[0] >= 10
        assert str(self.transform(expr)) == "[(and(bv[0], bv[1], bv[2])) + (iv[0]) >= 10]"
        # toplevel_list would split a toplevel and, so call simplify_boolean directly
        assert str(simplify_boolean([Operator("and", self.bvs.tolist() + [False])])) == "[boolval(False)]"
        assert str(simplify_boolean([Operator("and", [self.bvs[0], True, self.bvs[1]])])) == "[(bv[0]) and (bv[1])]"
        assert str(simplify_boolean([Operator("and", [self.bvs[0], self.bvs[1]])])) == "[(bv[0]) and (bv[1])]"

        expr = Operator("->", [self.bvs[0], True])
        assert str(self.transform(expr)) == "[boolval(True)]"
        expr = Operator("->", [self.bvs[0], BoolVal(True)])
        assert str(self.transform(expr)) == "[boolval(True)]"
        expr = Operator("->", [self.bvs[0], False])
        assert str(self.transform(expr)) == "[~bv[0]]"
        expr = Operator("->", [self.bvs[0], BoolVal(False)])
        assert str(self.transform(expr)) == "[~bv[0]]"
        expr = Operator("->", [True, self.bvs[0]])
        assert str(self.transform(expr)) == "[bv[0]]"
        expr = Operator("->", [False, self.bvs[0]])
        assert str(self.transform(expr)) == "[boolval(True)]"
        expr = Operator("->", [True, True])
        assert str(self.transform(expr)) == "[boolval(True)]"
        expr = Operator("->", [True, False])
        assert str(self.transform(expr)) == "[boolval(False)]"
        expr = Operator("->", [self.bvs[0], self.bvs[1]])
        assert str(self.transform(expr)) == "[(bv[0]) -> (bv[1])]"
        # cond simplified recursively, implication itself stays
        expr = Operator("->", [Operator("or", [self.bvs[0], False]), self.bvs[1]])
        assert str(self.transform(expr)) == "[(bv[0]) -> (bv[1])]"

        # Boolean constants that fold an or/and/-> become 0/1 in a numerical context
        iv = self.ivs[0]
        expr = iv + Operator("or", self.bvs.tolist() + [True]) >= 10
        assert str(self.transform(expr)) == "[(iv[0]) + 1 >= 10]"
        expr = iv + Operator("->", [False, self.bvs[0]]) >= 10
        assert str(self.transform(expr)) == "[(iv[0]) + 1 >= 10]"
        expr = iv + Operator("->", [self.bvs[0], True]) >= 10
        assert str(self.transform(expr)) == "[(iv[0]) + 1 >= 10]"
        expr = iv + Operator("->", [True, True]) >= 10
        assert str(self.transform(expr)) == "[(iv[0]) + 1 >= 10]"
        expr = iv + Operator("->", [True, False]) >= 10
        assert str(self.transform(expr)) == "[(iv[0]) + 0 >= 10]"
        expr = iv + Operator("->", [self.bvs[0], False]) >= 10
        assert str(self.transform(expr)) == "[(iv[0]) + (~bv[0]) >= 10]"

    def test_degenerate_bool_ops(self):
        bv = self.bvs

        # a single argument does not need the operator
        assert str(simplify_boolean([Operator("or", [bv[0]])])) == "[bv[0]]"
        assert str(simplify_boolean([Operator("and", [bv[0]])])) == "[bv[0]]"
        # also when the other arguments are constants that get removed
        assert str(simplify_boolean([Operator("or", [bv[0], False])])) == "[bv[0]]"
        assert str(simplify_boolean([Operator("and", [bv[0], True])])) == "[bv[0]]"
        assert str(simplify_boolean([Operator("or", [bv[0], BoolVal(False)])])) == "[bv[0]]"
        assert str(simplify_boolean([Operator("and", [bv[0], BoolVal(True)])])) == "[bv[0]]"
        # nothing left, so the identity element of the operator
        assert str(simplify_boolean([Operator("or", [False, False])])) == "[boolval(False)]"
        assert str(simplify_boolean([Operator("and", [True, True])])) == "[boolval(True)]"
        assert str(simplify_boolean([Operator("or", [BoolVal(False), BoolVal(False)])])) == "[boolval(False)]"
        assert str(simplify_boolean([Operator("and", [BoolVal(True), BoolVal(True)])])) == "[boolval(True)]"

        # also in a nested context
        assert str(self.transform(Operator("or", [bv[0]]) == bv[1])) == "[(bv[0]) == (bv[1])]"
        assert str(self.transform(Operator("and", [bv[0]]).implies(bv[1]))) == "[(bv[0]) -> (bv[1])]"
        assert str(self.transform(Operator("or", [Operator("and", [bv[0]]), bv[1]]))) == "[(bv[0]) or (bv[1])]"

        # empty and/or in a numerical context become 0/1, not BoolVal
        iv = self.ivs[0]
        assert str(self.transform(iv + Operator("or", [False, False]) >= 10)) == "[(iv[0]) + 0 >= 10]"
        assert str(self.transform(iv + Operator("and", [True, True]) >= 10)) == "[(iv[0]) + 1 >= 10]"
        assert str(self.transform(iv >= Operator("or", [False, False]))) == "[iv[0] >= 0]"
        assert str(self.transform(iv >= Operator("and", [True, True]))) == "[iv[0] >= 1]"

    def test_bool_in_comp(self):
        expr = self.ivs[0] >= False
        assert str(self.transform(expr)) == '[iv[0] >= 0]'
        expr = self.ivs[0] >= True
        assert str(self.transform(expr)) == '[iv[0] >= 1]'

        expr = (cp.sum(self.ivs) + True) >= 10
        assert str(self.transform(expr)) == '[sum(iv[0], iv[1], iv[2], 1) >= 10]'

        expr = True + self.ivs[0] >= False
        assert str(self.transform(expr)) == '[1 + (iv[0]) >= 0]'

        expr = Operator("sum", [True, False, self.ivs[0]]) >= 10
        assert str(self.transform(expr)) == '[sum(1, 0, iv[0]) >= 10]'
        expr = cp.sum(self.ivs) >= 10
        assert str(self.transform(expr)) == '[sum(iv[0], iv[1], iv[2]) >= 10]'

    def test_boolvar_comps(self):
        num_args = {"<0": -1, "0": 0, "1": 1, ">0": 2}
        # test table from github (#add url)
        bv = self.bvs[0]
        test_dict = {
            "==": {"<0": False, "0": ~bv, "1": bv, ">0": False},
            "!=": {"<0": True, "0": bv, "1": ~bv, ">0": True},
            ">":  {"<0": True, "0": bv, "1": False, ">0": False},
            "<":  {"<0": False, "0": False, "1": ~bv, ">0": True},
            ">=": {"<0": True, "0": True, "1": bv, ">0": False},
            "<=": {"<0": False, "0": ~bv, "1": True, ">0": True}
        }

        for op in test_dict:
            for rhs, val_should in test_dict[op].items():
                expr = Comparison(op, bv, num_args[rhs])
                print(expr)
                expr_should = BoolVal(val_should) if isinstance(val_should, bool) else val_should
                assert str(self.transform(expr)) == str([expr_should])

                # same comparison in a numerical context: True/False fold to 1/0
                num_expr = self.ivs[0] + Comparison(op, bv, num_args[rhs]) >= 10
                if isinstance(val_should, bool):
                    assert str(self.transform(num_expr)) == f"[(iv[0]) + {int(val_should)} >= 10]"
                else:
                    assert str(self.transform(num_expr)) == f"[(iv[0]) + ({val_should}) >= 10]"

        # comparisons written with the number on the left are flipped onto the table above
        assert str(self.transform(Comparison("<", 0, bv))) == "[bv[0]]"
        assert str(self.transform(Comparison(">", 0, bv))) == "[boolval(False)]"
        assert str(self.transform(Comparison("<=", 0, bv))) == "[boolval(True)]"
        assert str(self.transform(Comparison(">=", 0, bv))) == "[~bv[0]]"
        assert str(self.transform(Comparison("==", bv, BoolVal(True)))) == "[bv[0]]"
        assert str(self.transform(Comparison("==", bv, BoolVal(False)))) == "[~bv[0]]"

        # integer-vs-integer comparisons keep the expression on the lhs
        iv = self.ivs[0]
        assert str(self.transform(Comparison("<", 0, iv))) == "[iv[0] > 0]"
        assert str(self.transform(Comparison(">", 5, iv))) == "[iv[0] < 5]"
        assert str(self.transform(Comparison("<=", 0, iv))) == "[iv[0] >= 0]"
        assert str(self.transform(Comparison(">=", 5, iv))) == "[iv[0] <= 5]"
        assert str(self.transform(Comparison("==", 1, iv))) == "[iv[0] == 1]"

        # two numeric constants: eval_comparison folds to a Boolean, 0/1 in numerical context
        assert str(self.transform(Comparison(">=", 3, 5))) == "[boolval(False)]"
        assert str(self.transform(Comparison("==", 1, 1))) == "[boolval(True)]"
        assert str(self.transform(iv + Comparison(">=", 3, 5) >= 10)) == "[(iv[0]) + 0 >= 10]"
        assert str(self.transform(iv + Comparison("==", 1, 1) >= 10)) == "[(iv[0]) + 1 >= 10]"

        with pytest.raises(ValueError, match="floating point"):
            self.transform(Comparison("<", bv, 0.5))


    def test_simplify_expressions(self):
        # global constraints
        expr = cp.AllDifferent(self.ivs) == 0
        assert str(self.transform(expr)) == '[not(alldifferent(iv[0],iv[1],iv[2]))]'
        expr = 0 == cp.AllDifferent(self.ivs)
        assert str(self.transform(expr)) == '[not(alldifferent(iv[0],iv[1],iv[2]))]'
        # with constant, does not change (surprisingly? but we cannot check what the res type is...)
        expr = cp.AllDifferent(self.ivs.tolist() + [False]) == 0
        assert str(self.transform(expr)) == '[not(alldifferent(iv[0],iv[1],iv[2],boolval(False)))]'
        expr = 0 == cp.AllDifferent(self.ivs.tolist() + [True])
        assert str(self.transform(expr)) == '[not(alldifferent(iv[0],iv[1],iv[2],boolval(True)))]'

        # global functions
        expr = cp.max(self.ivs) == 0
        assert str(self.transform(expr)) == '[max(iv[0],iv[1],iv[2]) == 0]'
        expr = 0 == cp.max(self.ivs)
        assert str(self.transform(expr)) == '[max(iv[0],iv[1],iv[2]) == 0]'
        # with constant, does not change (surprisingly? but we cannot check what the res type is...)
        expr = cp.max(self.ivs.tolist() + [False]) == 0
        assert str(self.transform(expr)) == '[max(iv[0],iv[1],iv[2],boolval(False)) == 0]'
        expr = 0 == cp.max(self.ivs.tolist() + [True])
        assert str(self.transform(expr)) == '[max(iv[0],iv[1],iv[2],boolval(True)) == 0]'

        expr = ~cp.AllDifferent(self.ivs)
        assert str(self.transform(expr)) == '[not(alldifferent(iv[0],iv[1],iv[2]))]'
        expr = ~cp.AllDifferent(self.ivs.tolist() + [False])
        assert str(self.transform(expr)) == '[not(alldifferent(iv[0],iv[1],iv[2],boolval(False)))]'

        expr = (self.ivs[0] <= self.ivs[1]) == 0
        assert str(self.transform(expr)) == '[(iv[0]) > (iv[1])]'

        expr = (self.ivs[0] == self.ivs[1]) == 1
        assert str(self.transform(expr)) == '[(iv[0]) == (iv[1])]'

        # very nested one
        expr = Operator("and", self.bvs[:1].tolist() + [BoolVal(False)]) == Operator("or", self.bvs)
        assert str(self.transform(expr)) == '[and(~bv[0], ~bv[1], ~bv[2])]'

    def test_nested_boolval(self):

        bv = cp.boolvar(name="bv")
        x = cp.intvar(0, 3, name="x")
        cons = (x == 2) == (bv == 4)
        assert str(self.transform(cons)) == "[x != 2]"
        assert cp.Model(cons).solve()

        # Simplify boolean expressions nested within a weighted sum
        #   wsum([1, 2], [bv[0] != 0, bv[1] != 1]) ----> wsum([1, 2], [bv[0], ~bv[1]])
        bv = cp.boolvar(name="bv", shape=2)
        weights = cp.cpm_array([1, 2])
        bool_as_ints = cp.cpm_array([0, 1])
        cons = sum( weights * (bv != bool_as_ints) ) == 1
        assert str(self.transform(cons)) == "[sum([1, 2] * [bv[0], ~bv[1]]) == 1]"
        assert cp.Model(cons).solve()

        # Boolean constants in a wsum become integers; an already-integer wsum is unchanged
        iv = self.ivs
        cons = Operator("wsum", [[1, 2, 3], [iv[0], True, iv[1]]]) == 1
        assert str(self.transform(cons)) == "[sum([1, 2, 3] * [iv[0], 1, iv[1]]) == 1]"
        cons = Operator("wsum", [[1, 2, 3], [True, False, iv[0]]]) == 1
        assert str(self.transform(cons)) == "[sum([1, 2, 3] * [1, 0, iv[0]]) == 1]"
        cons = Operator("wsum", [[1, 2], [iv[0], iv[1]]]) == 1
        assert str(self.transform(cons)) == "[sum([1, 2] * [iv[0], iv[1]]) == 1]"
