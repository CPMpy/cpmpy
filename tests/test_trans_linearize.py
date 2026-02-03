import unittest

import cpmpy as cp
from cpmpy.expressions import boolvar, intvar
from cpmpy.expressions.core import Operator
from cpmpy.transformations.flatten_model import flatten_objective
from cpmpy.transformations.linearize import linearize_constraint, canonical_comparison, only_positive_bv, only_positive_coefficients, only_positive_bv_wsum_const, only_positive_bv_wsum
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl


class TestTransLinearize(unittest.TestCase):

    def setUp(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0
        self.ivars = cp.intvar(1, 10, shape=(5,))
        self.bvars = cp.boolvar((3,))

    def test_linearize(self):

        # Boolean
        a, b, c = [boolvar(name=var) for var in "abc"]

        # and
        cons = linearize_constraint([a & b])[0]
        self.assertEqual("(a) + (b) >= 2", str(cons))

        # or
        cons = linearize_constraint([a | b])[0]
        self.assertEqual("(a) + (b) >= 1", str(cons))

        # implies
        cons = linearize_constraint([a.implies(b)])[0]
        self.assertEqual("sum([1, -1] * [a, b]) <= 0", str(cons))
    
    def test_bug_168(self):
        from cpmpy.solvers import CPM_gurobi
        if CPM_gurobi.supported():
            bv = boolvar(shape=2)
            iv = intvar(1, 9)
            e1 = (bv[0] * bv[1] == iv)
            s1 = cp.Model(e1).solve("gurobi")
            self.assertTrue(s1)
            self.assertEqual([bv[0].value(), bv[1].value(), iv.value()],[True, True, 1])
            
    def test_bug_468(self):
        from cpmpy.solvers import CPM_exact, CPM_gurobi
        a, b, c = boolvar(shape=3)
        m = cp.Model(cp.any([a, b, c]))
        m.minimize(3*a + 4*~b + 3*~c)
        if CPM_gurobi.supported():
            m.solve("gurobi")
            self.assertEqual([a.value(), b.value(), c.value()], [False, True, True])
        if CPM_exact.supported():
            m.solve("exact")
            self.assertEqual([a.value(), b.value(), c.value()], [False, True, True])

    def test_constraint(self):
        x,y,z = [cp.intvar(0,5, name=n) for n in "xyz"]
        a,b,c = [cp.boolvar(name=n) for n in "abc"]

        # test and
        self.assertEqual(str(linearize_constraint([a & b & c])), "[sum([a, b, c]) >= 3]")
        self.assertEqual(str(linearize_constraint([a & b & (~c)])), "[sum([a, b, ~c]) >= 3]")
        # test or
        self.assertEqual(str(linearize_constraint([a | b | c])), "[sum([a, b, c]) >= 1]")
        self.assertEqual(str(linearize_constraint([a | b | (~c)])), "[sum([a, b, ~c]) >= 1]")
        # test implies
        self.assertEqual(str(linearize_constraint([a.implies(b)])), "[sum([1, -1] * [a, b]) <= 0]")
        self.assertEqual(str(linearize_constraint([a.implies(~b)])), "[sum([1, -1] * [a, ~b]) <= 0]")
        self.assertEqual(str(linearize_constraint([a.implies(x+y+z >= 0)])), str([]))
        self.assertEqual(str(linearize_constraint([a.implies(x+y+z >= 2)])), "[(a) -> (sum([x, y, z]) >= 2)]")
        self.assertEqual(str(linearize_constraint([a.implies(x+y+z > 0)])), "[(a) -> (sum([x, y, z]) >= 1)]")
        # test sub
        self.assertEqual(str(linearize_constraint([Operator("sub",[x,y]) >= z])), "[sum([1, -1, -1] * [x, y, z]) >= 0]")
        # test mul
        self.assertEqual(str(linearize_constraint([3 * x > 2])), "[sum([3] * [x]) >= 3]")
        # test <
        self.assertEqual((str(linearize_constraint([x + y  < z]))), "[sum([1, 1, -1] * [x, y, z]) <= -1]")
        # test >
        self.assertEqual((str(linearize_constraint([x + y  > z]))), "[sum([1, 1, -1] * [x, y, z]) >= 1]")
        # test !=
        c1,c2 = linearize_constraint([x + y  != z])
        self.assertEqual(str(c1), "(BV3) -> (sum([1, 1, -1] * [x, y, z]) <= -1)")
        self.assertEqual(str(c2), "(~BV3) -> (sum([1, 1, -1] * [x, y, z]) >= 1)")
        c1, c2, c3 = linearize_constraint([a.implies(x != y)])
        self.assertEqual(str(c1), "(a) -> (sum([1, -1, -6] * [x, y, BV4]) <= -1)")
        self.assertEqual(str(c2), "(a) -> (sum([1, -1, -6] * [x, y, BV4]) >= -5)")
        self.assertEqual(str(c3), "sum([1, -1] * [~a, ~BV4]) <= 0")


    def test_single_boolvar(self):
        """ Linearize should convert Boolean literals to constraints (either linear or clause) """
        p = cp.boolvar(name="p")
        self.assertEqual(str([p >= 1]), str(linearize_constraint([p])))
        self.assertEqual(str([p <= 0]), str(linearize_constraint([~p])))
        self.assertEqual(str([Operator("or", [p])]), str(linearize_constraint([p], supported={"or"})))
        self.assertEqual(str([Operator("or", [~p])]), str(linearize_constraint([~p], supported={"or"})))

    def test_neq(self):
        # not equals is a tricky constraint to linearize, do some extra tests on it here

        x, y, z = [cp.intvar(0, 5, name=n) for n in "xyz"]
        a, b, c = [cp.boolvar(name=n) for n in "abc"]

        cons = [2*x + 3*y + 4*z != 10]
        self.assertEqual(str(linearize_constraint(cons)),"[(BV3) -> (sum([2, 3, 4] * [x, y, z]) <= 9), (~BV3) -> (sum([2, 3, 4] * [x, y, z]) >= 11)]")

        cons = [a.implies(x != y)]
        lin_cons = linearize_constraint(cons)
        cons_vals = []
        cp.Model(lin_cons).solveAll(solver="ortools", display=lambda : cons_vals.append(cons[0].value()))
        print(len(cons_vals))
        self.assertTrue(all(cons_vals))
        # self.assertEqual(str(linearize_constraint(cons)), "[(a) -> (sum([1, -1, -6] * [x, y, BV4]) <= -1), (a) -> (sum([1, -1, -6] * [x, y, BV4]) >= -5)]")

    def test_alldiff(self):
        # alldiff has a specialized linearization

        x = cp.intvar(1, 5, shape=3, name="x")
        cons = cp.AllDifferent(x)
        lincons = linearize_constraint([cons])

        def cb():
            assert cons.value()

        n_sols = cp.Model(lincons).solveAll(display=cb)
        self.assertEqual(n_sols, 5 * 4 * 3)

        # should also work with constants in arguments
        x,y,z = x
        cons = cp.AllDifferent([x,3,y,True,z])
        lincons = linearize_constraint([cons])

        def cb():
            assert cons.value()

        n_sols = cp.Model(lincons).solveAll(display=cb)
        self.assertEqual(n_sols, 3 * 2 * 1) # 1 and 3 not allowed

    # def test_issue_580(self): -> Modulo is now a global constraint
    #     x = cp.intvar(1, 5, name='x')
    #     lin_mod = linearize_constraint([x % 2 == 0], supported={"mul","sum", "wsum"})
    #     self.assertTrue(cp.Model(lin_mod).solve())
    #     self.assertIn(x.value(),{2,4})
    #
    #     lin_mod = linearize_constraint([x % 2 <= 0], supported={"mul", "sum", "wsum"})
    #     self.assertEqual(str(lin_mod), '[IV7 <= 0, sum([2, -1] * [IV8, IV9]) == 0, sum([1, 1, -1] * [IV9, IV7, x]) == 0]')
    #     self.assertTrue(cp.Model(lin_mod).solve())
    #     self.assertIn(x.value(), {2, 4}) # can never be < 0
    #
    #     lin_mod = linearize_constraint([x % 2 == 1], supported={"mul", "sum", "wsum"})
    #     self.assertTrue(cp.Model(lin_mod).solve())
    #     self.assertIn(x.value(), {1,3,5})

    def test_issue_546(self):
        # https://github.com/CPMpy/cpmpy/issues/546
        arr = cp.cpm_array([cp.intvar(0, 5), cp.intvar(0, 5), 5, 4]) # combination of decision variables and constants
        c = cp.AllDifferent(arr)

        linear_c = linearize_constraint([c])
        # this triggers an error
        pos_c = only_positive_bv([c])

        # also test full transformation stack
        if "gurobi" in cp.SolverLookup.solvernames(): # otherwise, not supported
            model = cp.Model(c)
            model.solve(solver="gurobi")

        if "exact" in cp.SolverLookup.solvernames(): # otherwise, not supported
            model = cp.Model(c)
            model.solve(solver="exact")

    def test_sub(self):
        x = cp.intvar(0,10, name="x")
        y = cp.intvar(0,10, name="y")

        cons = Operator("sub", [3, x]) == y
        [lin_cons] = linearize_constraint([cons])
        self.assertEqual(str(lin_cons), "sum([-1, -1] * [x, y]) == -3")

        cons = Operator("sub", [x, 3]) == y
        [lin_cons] = linearize_constraint([cons])
        self.assertEqual(str(lin_cons), "sum([1, -1] * [x, y]) == 3")

        cons = Operator("sub", [x,y]) == 3
        [lin_cons] = linearize_constraint([cons])
        self.assertEqual(str(lin_cons), "sum([1, -1] * [x, y]) == 3")

    def test_bool_mult(self):

        x = cp.intvar(-5, 10, name="x")
        y = cp.intvar(-5, 10, name="y")
        a = cp.boolvar(name="a")
        b = cp.boolvar(name="b")

        def assert_cons_is_true(cons):
            return lambda : self.assertTrue(cons.value())

        cons = b * x == y
        bt,bf = linearize_constraint([cons])
        self.assertEqual(str(bt), "(b) -> (sum([1, -1] * [x, y]) == 0)")
        self.assertEqual(str(bf), "(~b) -> (sum([y]) == 0)")

        cp.Model([bt,bf]).solveAll(display=assert_cons_is_true(cons))

        cons = x * b == y
        bt,bf = linearize_constraint([cons])
        self.assertEqual(str(bt), "(b) -> (sum([1, -1] * [x, y]) == 0)")
        self.assertEqual(str(bf), "(~b) -> (sum([y]) == 0)")

        cp.Model([bt,bf]).solveAll(display=assert_cons_is_true(cons))

        cons = a.implies(b * x <= y)
        lin_cons = linearize_constraint([cons])
        self.assertEqual(str(lin_cons[0]), "(a) -> (sum([1, -1, -15] * [x, y, ~b]) <= 0)")
        self.assertEqual(str(lin_cons[1]), "(a) -> (sum([1, 5] * [y, b]) >= 0)")

        lin_cnt = cp.Model(lin_cons).solveAll(display=assert_cons_is_true(cons))
        cons_cnt = cp.Model(cons).solveAll(display=assert_cons_is_true(cp.all(lin_cons)))
        self.assertEqual(lin_cnt, cons_cnt)

        cons = a.implies(b * x >= y)
        lin_cons = linearize_constraint([cons])
        self.assertEqual(str(lin_cons[0]), "(a) -> (sum([1, -1, 15] * [x, y, ~b]) >= 0)")
        self.assertEqual(str(lin_cons[1]), "(a) -> (sum([1, -10] * [y, b]) <= 0)")

        lin_cnt = cp.Model(lin_cons).solveAll(display=assert_cons_is_true(cons))
        cons_cnt = cp.Model(cons).solveAll(display=assert_cons_is_true(cp.all(lin_cons)))
        self.assertEqual(lin_cnt, cons_cnt)


    def test_implies(self):

        x = cp.intvar(1, 10, name="x")
        y = cp.intvar(1, 10, name="y")
        b = cp.boolvar(name="b")

        cons = b.implies(x + y <= 5)
        [lin_cons] = linearize_constraint([cons], supported={"sum", "wsum"}) # no support for "->"
        self.assertEqual(str(lin_cons), "sum([1, 1, -15] * [x, y, ~b]) <= 5")

        cons = b.implies(x + y >= 5)
        [lin_cons] = linearize_constraint([cons], supported={"sum", "wsum"})  # no support for "->"
        self.assertEqual(str(lin_cons), "sum([1, 1, 3] * [x, y, ~b]) >= 5")

        cons = b.implies(x + y == 5)
        lin_cons = linearize_constraint([cons], supported={"sum", "wsum"})  # no support for "->"
        assert len(lin_cons) == 2
        self.assertEqual(str(lin_cons[0]), "sum([1, 1, -15] * [x, y, ~b]) <= 5")
        self.assertEqual(str(lin_cons[1]), "sum([1, 1, 3] * [x, y, ~b]) >= 5")






class TestConstRhs(unittest.TestCase):

    def test_numvar(self):
        a, b = [cp.intvar(0, 10, name=n) for n in "ab"]

        cons = linearize_constraint([a <= b])[0]
        self.assertEqual("sum([1, -1] * [a, b]) <= 0", str(cons))

    def test_sum(self):
        a,b,c = [cp.intvar(0,10,name=n) for n in "abc"]
        rhs = intvar(0,10,name="r")

        cons = linearize_constraint([cp.sum([a,b,c]) <= rhs])[0]
        self.assertEqual("sum([1, 1, 1, -1] * [a, b, c, r]) <= 0", str(cons))

    def test_wsum(self):
        a, b, c = [cp.intvar(0, 10,name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")

        cons = 1*a + 2*b + 3*c <= rhs
        cons = linearize_constraint([cons])[0]
        self.assertEqual("sum([1, 2, 3, -1] * [a, b, c, r]) <= 0", str(cons))

    def test_impl(self):
        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")
        cond = cp.boolvar(name="bv")

        cons = [cond.implies(1 * a + 2 * b + 3 * c <= rhs)]
        cons = linearize_constraint(cons)[0]
        self.assertEqual("(bv) -> (sum([1, 2, 3, -1] * [a, b, c, r]) <= 0)", str(cons))

        cons = [(~cond).implies(1 * a + 2 * b + 3 * c <= rhs)]
        cons = linearize_constraint(cons)[0]
        self.assertEqual("(~bv) -> (sum([1, 2, 3, -1] * [a, b, c, r]) <= 0)", str(cons))

    def test_others(self):

        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")

        cons = [cp.max([a,b,c]) <= rhs]
        print(linearize_constraint(cons, supported={"max"}))
        cons = linearize_constraint(cons, supported={"max"})[0]
        self.assertEqual("(max(a,b,c)) <= (r)", str(cons))

        cons = [cp.AllDifferent([a,b,c])]
        print(linearize_constraint(cons, supported={"alldifferent"}))
        cons = linearize_constraint(cons, supported={"alldifferent"})[0]
        self.assertEqual("alldifferent(a,b,c)", str(cons))


class TestVarsLhs(unittest.TestCase):

    def setUp(self): # reset counters
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0

    def test_trivial_unsat_sum(self):
        a,b,c = [cp.intvar(0,10,name=n) for n in "abc"]
        rhs = 5

        # trivial UNSAT
        cons = linearize_constraint([cp.sum([a,b,c,10]) <= rhs])[0]
        self.assertEqual(str(cp.BoolVal(False)), str(cons))

    def test_sum(self):
        a,b,c = [cp.intvar(0,10,name=n) for n in "abc"]
        rhs = 15

        cons = linearize_constraint([cp.sum([a,b,c,10]) <= rhs])[0]
        self.assertEqual(str(cp.sum([a,b,c]) <= 5), str(cons))

    def test_wsum(self):
        a, b, c = [cp.intvar(0, 10,name=n) for n in "abc"]
        rhs = 5

        cons = [Operator("wsum",[[1,2,3,-1],[a,b,c,10]]) <= rhs]
        cons = linearize_constraint(cons)[0]
        self.assertEqual("sum([1, 2, 3] * [a, b, c]) <= 15", str(cons))

    def test_impl(self):
        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = 5
        cond = cp.boolvar(name="bv")

        cons = [cond.implies(Operator("wsum",[[1,2,3,-1],[a,b,c,10]]) <= rhs)]
        cons = linearize_constraint(cons)[0]
        self.assertEqual("(bv) -> (sum([1, 2, 3] * [a, b, c]) <= 15)", str(cons))

        cons = [(~cond).implies(Operator("wsum",[[1,2,3,-1],[a,b,c,10]]) <= rhs)]
        cons = linearize_constraint(cons)[0]
        self.assertEqual("(~bv) -> (sum([1, 2, 3] * [a, b, c]) <= 15)", str(cons))

    # def test_pow(self): -> pow is a global constraint now
    #
    #     a,b = cp.intvar(0,32, name=tuple("ab"), shape=2)
    #
    #     cons = a ** 3 == b
    #     lin_cons = linearize_constraint([cons], supported={"sum", "wsum", "mul"})
    #
    #     self.assertEqual(str(lin_cons[0]), "((a) * (a)) == (IV0)")
    #     self.assertEqual(str(lin_cons[1]), "((a) * (IV0)) == (IV1)")
    #     self.assertEqual(str(lin_cons[2]), "sum([1, -1] * [IV1, b]) == 0")
    #
    #     cons = a ** 5 == b
    #     lin_cons = linearize_constraint([cons], supported={"sum", "wsum", "mul"})
    #     model = cp.Model(lin_cons + [a == 2])
    #     self.assertTrue(model.solve())
    #     self.assertEqual(b.value(), 32)  # 2^5 = 32
    #
    #     # Test x^0 == y (should equal 1)
    #     cons = a ** 0 == b
    #     lin_cons = linearize_constraint([cons], supported={"sum", "wsum", "mul"})
    #     model = cp.Model(lin_cons + [a == 3])
    #     self.assertTrue(model.solve())
    #     self.assertEqual(b.value(), 1)
    #
    #     # not supported pow with exponent being a variable
    #     cons = a ** b == 3
    #     self.assertRaises(NotImplementedError,
    #                       lambda :  linearize_constraint([cons], supported={"sum", "wsum", "mul"}))
    #
    #     # not supported pow with exponent being a float
    #     cons = a ** 3.5 == b
    #     self.assertRaises(NotImplementedError,
    #                       lambda :  linearize_constraint([cons], supported={"sum", "wsum", "mul"}))
    #
    #     # not supported pow with exponent being a negative integer
    #     cons = a ** -3 == b
    #     self.assertRaises(NotImplementedError,
    #                       lambda :  linearize_constraint([cons], supported={"sum", "wsum", "mul"}))
    #
    #     # not supported pow when mul is not supported
    #     cons = a ** 3 == b
    #     self.assertRaises(NotImplementedError,
    #                       lambda :  linearize_constraint([cons], supported={"sum", "wsum"}))


    def test_others(self):

        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")

        cons = [cp.max([a,b,c,5]) <= rhs]
        cons = linearize_constraint(cons, supported={"max"})[0]
        self.assertEqual("(max(a,b,c,5)) <= (r)", str(cons))

        cons = [cp.AllDifferent([a, b, c])]
        cons = linearize_constraint(cons, supported={"alldifferent"})[0]
        self.assertEqual("alldifferent(a,b,c)", str(cons))

class testCanonical_comparison(unittest.TestCase):
    def setUp(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0
        self.ivars = cp.intvar(1, 10, shape=(5,))
        self.bvars = cp.boolvar((3,))
    
    def test_sum(self):
        a,b,c = [cp.intvar(0,10,name=n) for n in "abc"]
        rhs = 5

        cons = canonical_comparison([cp.sum([a,b,c,10]) <= rhs])[0]
        self.assertEqual("sum([a, b, c]) <= -5", str(cons))

        rhs = cp.sum([b,c])
        cons = canonical_comparison([cp.sum([a, b]) <= rhs])[0]
        self.assertEqual("sum([1, 1, -1, -1] * [a, b, b, c]) <= 0", str(cons))

    def test_div(self):
        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = 5

        cons = canonical_comparison([ a / b <= rhs])[0]
        self.assertEqual("(a) div (b) <= 5", str(cons))

        #when adding division
        #cons = canonical_comparison([a / b <= c / rhs])[0]
        #cons = canonical_comparison([a + b <= c/rhs])[0]


    def test_wsum(self):
        a, b, c = [cp.intvar(0, 10,name=n) for n in "abc"]
        rhs = 5

        cons = [Operator("wsum",[[1,2,3,-1],[a,b,c,10]]) <= rhs]
        cons = canonical_comparison(cons)[0]
        self.assertEqual("sum([1, 2, 3] * [a, b, c]) <= 15", str(cons))

    def test_impl(self):
        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = 5
        cond = cp.boolvar(name="bv")

        cons = [cond.implies(Operator("wsum",[[1,2,3,-1],[a,b,c,10]]) <= rhs)]
        cons = canonical_comparison(cons)[0]
        self.assertEqual("(bv) -> (sum([1, 2, 3] * [a, b, c]) <= 15)", str(cons))


        cons = [(~cond).implies(Operator("wsum",[[1,2,3,-1],[a,b,c,10]]) <= rhs)]
        cons = canonical_comparison(cons)[0]
        self.assertEqual("(~bv) -> (sum([1, 2, 3] * [a, b, c]) <= 15)", str(cons))

    def test_others(self):

        a, b, c = [cp.intvar(0, 10, name=n) for n in "abc"]
        rhs = intvar(0, 10, name="r")

        cons = [cp.max([a,b,c,5]) <= rhs]
        cons = canonical_comparison(cons)[0]
        self.assertEqual("(max(a,b,c,5)) <= (r)", str(cons))

        cons = [cp.AllDifferent([a, b, c])]
        cons = canonical_comparison(cons)[0]
        self.assertEqual("alldifferent(a,b,c)", str(cons))

    def test_only_positive_coefficients(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        only_pos = only_positive_coefficients([Operator("wsum",[[1,1,-1],[a,b,c]]) > 0])
        self.assertEqual(str([Operator("sum",[a, b, ~c]) > 1]), str(only_pos))

    def test_only_positive_coefficients_implied(self):
        a, b, c, p = [cp.boolvar(name=n) for n in "abcp"]
        only_pos = only_positive_coefficients([p.implies(Operator("wsum",[[1,1,-1],[a,b,c]]) > 0)])
        self.assertEqual(str([p.implies(Operator("sum",[a, b, ~c]) > 1)]), str(only_pos))

    def test_only_positive_coefficients_pb_and_int(self):
        a, b, c, x, y = [cp.boolvar(name=n) for n in "abc"] + [cp.intvar(0, 3, name=n) for n in "xy"]
        only_pos = only_positive_coefficients([Operator("wsum",[[1,1,-1,1,-1],[a,b,c,x,y]]) > 0])
        self.assertEqual(str([Operator("wsum",[[1,1,1,1,-1],[a,b,~c,x,y]]) > 1]), str(only_pos))

    def test_only_positive_bv_implied_by_literal(self):
        p = cp.boolvar(name="p")
        self.assertEqual(str([p >= 1]), str(only_positive_bv(linearize_constraint([p]))))

    def test_only_positive_bv_implied_by_negated_literal(self):
        p = cp.boolvar(name="p")
        self.assertEqual(str([p <= 0]), str(only_positive_bv(linearize_constraint([~p]))))
        
        
class testOnlyPositiveBv(unittest.TestCase):
    def setUp(self):
        _IntVarImpl.counter = 0
        _BoolVarImpl.counter = 0
        self.ivars = cp.intvar(1, 10, shape=(5,))
        self.bvars = cp.boolvar((3,))

    def test_only_positive_bv_wsum_const_positive_literal(self):
        p = cp.boolvar(name="p")
        self.assertEqual(str((p, 0)), str(only_positive_bv_wsum_const(p)))
        
    def test_only_positive_bv_wsum_const_negated_litral(self):
        p = cp.boolvar(name="p")
        self.assertEqual(str((Operator("wsum",[[-1],[p]]), 1)), str(only_positive_bv_wsum_const(~p)))
        
    def test_only_positive_bv_wsum_const_sum_positive_input(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        self.assertEqual(str((Operator("sum",[a,b,c]), 0)), str(only_positive_bv_wsum_const(a+b+c)))
        
    def test_only_positive_bv_wsum_const_sum(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        self.assertEqual(str((Operator("wsum",[[-1, -1, 1],[a,b,c]]), 2)), str(only_positive_bv_wsum_const(~a+~b+c)))
        
    def test_only_positive_bv_wsum_const_wsum_positive_input(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        self.assertEqual(str((Operator("wsum",[[4, 5, 1],[a,b,c]]), 0)), str(only_positive_bv_wsum_const(4*a+5*b+c)))
        
    def test_only_positive_bv_wsum_const_wsum(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        self.assertEqual(str((Operator("wsum",[[-4, 5, 1],[a,b,c]]), 4)), str(only_positive_bv_wsum_const(4*~a+5*b+c)))
        
    def test_only_positive_bv_wsum_const_non_linear(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        with self.assertRaises(ValueError) as cm:
            only_positive_bv_wsum_const(~a * b * c)
            self.assertEqual(str(cm.exception), "unexpected expression, should be sum, wsum or var but got ((~a) * (b)) * (c)")
            
    def test_only_positive_bv_wsum_positive_literal(self):
        p = cp.boolvar(name="p")
        obj = only_positive_bv_wsum(p)
        self.assertEqual(str(obj), str(p))
            
    def test_only_positive_bv_wsum_negated_literal(self):
        p = cp.boolvar(name="p")
        obj = only_positive_bv_wsum(~p)
        self.assertEqual(str(obj), str(Operator("wsum",[[-1, 1],[p, 1]])))
        
    def test_only_positive_bv_wsum_sum_positive_input(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        obj = only_positive_bv_wsum(a + b + c)
        self.assertEqual(str(obj), str(Operator("sum",[a,b,c])))
        
    def test_only_positive_bv_wsum_sum(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        obj = only_positive_bv_wsum(~a + ~b + c)
        self.assertEqual(str(obj), str(Operator("wsum",[[-1, -1, 1, 1],[a,b,c,2]])))
        
    def test_only_positive_bv_wsum_wsum_positive_input(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        obj = only_positive_bv_wsum(4*a + 5*b + c)
        self.assertEqual(str(obj), str(Operator("wsum",[[4, 5, 1],[a,b,c]])))
        
    def test_only_positive_bv_wsum_wsum(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        obj = only_positive_bv_wsum(4*~a + 5*~b + c)
        self.assertEqual(str(obj), str(Operator("wsum",[[-4, -5, 1, 1],[a,b,c,9]])))
        
    def test_only_positive_bv_wsum_non_linear_positive_input(self): # TODO: make boolvars
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        flat_obj, flat_cons = flatten_objective(a * b * c)
        flat_obj = only_positive_bv_wsum(flat_obj)
        self.assertEqual(str((flat_obj, flat_cons)), "(IV6, [((IV5) * (c)) == (IV6), ((a) * (b)) == (IV5)])")
        
    def test_only_positive_bv_wsum_non_linear(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        flat_obj, flat_cons = flatten_objective(~a * b * c)
        flat_obj = only_positive_bv_wsum(flat_obj)
        self.assertEqual(str((flat_obj, flat_cons)), "(IV6, [((IV5) * (c)) == (IV6), ((~a) * (b)) == (IV5)])")
        
    def test_only_flat_positive_bv_wsum_max(self):
        a, b = [cp.boolvar(name=n) for n in "ab"]
        expr = cp.max((~a * b), (a * ~b ))
        flat_obj, flat_cons = flatten_objective(expr)
        # self.assertEqual(str(expr.args), "(max((~a) * (b), (a) * (~b)))")
        obj = only_positive_bv_wsum(flat_obj)
        self.assertEqual(str((obj, flat_cons)), "(IV7, [(max(IV5,IV6)) == (IV7), ((~a) * (b)) == (IV5), ((a) * (~b)) == (IV6)])")
        
    def test_only_flat_positive_bv_max(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        obj = only_positive_bv(linearize_constraint([cp.max(~a,b) >= c], supported={"sum", "wsum", "max"}))
        self.assertEqual(str(obj), "[(max(BV3,b)) >= (c), (BV3) + (a) == 1]")
        
