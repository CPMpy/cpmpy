import unittest

import cpmpy as cp
from cpmpy.expressions import boolvar, intvar
from cpmpy.expressions.core import Operator
from cpmpy.expressions.utils import argvals
from cpmpy.transformations.linearize import linearize_constraint, canonical_comparison, only_positive_bv, only_positive_coefficients, only_positive_bv_sub, linearize_objective
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
        if CPM_gurobi.supported():
            b = boolvar()
            m = cp.Model(cp.any([b]))
            m.minimize(~b)
            m.solve(solver="gurobi")
            self.assertEqual([b.value()], [True])
            a, b, c = boolvar(shape=3)
            m = cp.Model(cp.any([a, b, c]))
            m.minimize(a + ~b + ~c)
            m.solve("gurobi")
            self.assertEqual([a.value(), b.value(), c.value()], [False, True, True])
            m = cp.Model(cp.any([a, b, c]))
            m.minimize(3*a + 4*~b + 3*~c)
            m.solve("gurobi")
            self.assertEqual([a.value(), b.value(), c.value()], [False, True, True])
            ivs = cp.intvar(0, 5, shape=3)
            m.maximize(ivs[0] * ivs[1] * ivs[2])
            m.solve("gurobi")
            self.assertEqual([ivs[0].value(), ivs[1].value(), ivs[2].value()], [5, 5, 5])
        if CPM_exact.supported():
            b = boolvar()
            m = cp.Model(cp.any([b]))
            m.minimize(~b)
            m.solve(solver="exact")
            self.assertEqual([b.value()], [True])
            a, b, c = boolvar(shape=3)
            m = cp.Model(cp.any([a, b, c]))
            m.minimize(a + ~b + ~c)
            m.solve("exact")
            self.assertEqual([a.value(), b.value(), c.value()], [False, True, True])
            m = cp.Model(cp.any([a, b, c]))
            m.minimize(3*a + 4*~b + 3*~c)
            m.solve("exact")
            self.assertEqual([a.value(), b.value(), c.value()], [False, True, True])
            ivs = cp.intvar(0, 5, shape=3)
            m.maximize(ivs[0] * ivs[1] * ivs[2])
            m.solve("exact")
            self.assertEqual([ivs[0].value(), ivs[1].value(), ivs[2].value()], [5, 5, 5])

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


    def test_linearize_modulo(self):
        x, z = cp.intvar(-2,2, shape=2, name=["x","z"])
        y = cp.intvar(1,5, name="y")
        vars = [x,y,z]

        constraint = [x % y  == z]
        lin_cons = linearize_constraint(constraint, supported={'sum', 'wsum', 'mul'})

        all_sols = set()
        lin_all_sols = set()
        count = cp.Model(constraint).solveAll(solver="ortools", display=lambda: all_sols.add(tuple(argvals(vars))))
        lin_count = cp.Model(lin_cons).solveAll(solver="ortools", display=lambda: lin_all_sols.add(tuple(argvals(vars))))

        self.assertSetEqual(all_sols, lin_all_sols) # same on decision vars
        self.assertEqual(count,lin_count) # same on all vars

    def test_linearize_division(self):
        x, z = cp.intvar(-2, 2, shape=2, name=["x", "z"])
        y = cp.intvar(1, 5, name="y")
        vars = [x, y, z]

        constraint = [x // y == z]
        lin_cons = linearize_constraint(constraint, supported={'sum', 'wsum', 'mul'})

        all_sols = set()
        lin_all_sols = set()
        count = cp.Model(constraint).solveAll(solver="ortools", display=lambda: all_sols.add(tuple(argvals(vars))))
        lin_count = cp.Model(lin_cons).solveAll(solver="ortools",
                                                display=lambda: lin_all_sols.add(tuple(argvals(vars))))

        self.assertSetEqual(all_sols, lin_all_sols)  # same on decision vars
        self.assertEqual(count, lin_count)  # same on all vars

    def test_abs(self):

        pos = cp.intvar(0,5,name="pos")
        neg = cp.intvar(-5,0,name="neg")
        x = cp.intvar(-5,5,name="x")
        y = cp.intvar(-5,5)

        for lhs in (pos,neg,x):
            cons = cp.Abs(lhs) == y
            cnt = cp.Model(cons).solveAll()
            lcnt = cp.Model(linearize_constraint([cons])).solveAll(display=self.assertTrue(cons.value()))
            self.assertEqual(cnt, lcnt)


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

    def test_issue_580(self):
        x = cp.intvar(1, 5, name='x')
        lin_mod = linearize_constraint([x % 2 == 0], supported={"mul","sum", "wsum"})
        self.assertTrue(cp.Model(lin_mod).solve())
        self.assertIn(x.value(),{2,4})


        lin_mod = linearize_constraint([x % 2 <= 0], supported={"mul", "sum", "wsum"})
        self.assertEqual(str(lin_mod), '[IV8 <= 0, sum([2, -1] * [IV9, IV10]) == 0, sum([1, -1] * [IV8, IV11]) == 0, sum([1, 1, -1] * [IV10, IV8, x]) == 0]')
        self.assertTrue(cp.Model(lin_mod).solve())
        self.assertIn(x.value(), {2, 4}) # can never be < 0

        lin_mod = linearize_constraint([x % 2 == 1], supported={"mul", "sum", "wsum"})
        self.assertTrue(cp.Model(lin_mod).solve())
        self.assertIn(x.value(), {1,3,5})

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

    def test_pow(self):

        a,b = cp.intvar(0,10, name=tuple("ab"), shape=2)

        cons = a ** 3 == b
        lin_cons = linearize_constraint([cons], supported={"sum", "wsum", "mul"})

        self.assertEqual(lin_cons[0], "((a) * (a)) == (IV0)")
        self.assertEqual(lin_cons[1], "((a) * (IV0)) == (IV1)")
        self.assertEqual(lin_cons[2], "sum([1, -1] * [IV1, b]) == 0")

        # this is not supported
        cons = a ** b == 3
        self.assertRaises(NotImplementedError,
                          lambda :  linearize_constraint([cons], supported={"sum", "wsum", "mul"}))

    def test_mod_triv(self):
        x,y = cp.intvar(1,3, name="x"), cp.intvar(1,3,name="y")
        # x mod y <= 2 is trivially true for x,y in 1..3
        self.assertEqual(str([]), str(linearize_constraint([(x % y) <= 2], supported={"mod"})))

    def test_mod(self):

        x,y = cp.intvar(1,3, name="x"), cp.intvar(1,3,name="y")

        # disallows 2 mod 3 = 2
        cons = (x % y) <= 1
        sols = set()
        cp.Model(cons).solveAll(display=lambda : sols.add((x.value(), y.value())))
        lincons = linearize_constraint([cons], supported={"sum", "wsum", "mul"})

        linsols = set()
        cp.Model(lincons).solveAll(display=lambda : linsols.add((x.value(), y.value())))
        self.assertSetEqual(sols, linsols)

        # check special cases of supported sets
        self.assertRaises(NotImplementedError,
                          lambda : linearize_constraint([cons], supported={"sum", "wsum"}),
                          )

        same_cons = linearize_constraint([cons], supported={"mod"})
        self.assertEqual(str(same_cons[0]), str(cons))

        # what about half-reified?
        bv = cp.boolvar(name="bv")
        cons = bv.implies((x % y) <= 1)
        sols = set()
        cp.Model(cons).solveAll(display=lambda: sols.add((bv.value(), x.value(), y.value())))
        lincons = linearize_constraint([cons], supported={"sum", "wsum", "mod"})

        linsols = set()
        cp.Model(lincons).solveAll(display=lambda: linsols.add((bv.value(), x.value(), y.value())))
        self.assertSetEqual(sols, linsols)

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
        self.assertEqual("(a) // (b) <= 5", str(cons))

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
        
    def test_only_positive_bv_sub_implied_by_literal(self):
        p = cp.boolvar(name="p")
        self.assertEqual(str((p, 0)), str(only_positive_bv_sub(p)))
        
    def test_only_positive_bv_sub_implied_by_negated_literal(self):
        p = cp.boolvar(name="p")
        self.assertEqual(str((Operator("wsum",[[-1],[p]]), 1)), str(only_positive_bv_sub(~p)))
        
    def test_only_positive_bv_sub_multiple_literals(self):
        a, b, c = [cp.boolvar(name=n) for n in "abc"]
        self.assertEqual(str((Operator("wsum",[[-1, -1, 1],[a,b,c]]), 2)), str(only_positive_bv_sub(~a+~b+c)))
