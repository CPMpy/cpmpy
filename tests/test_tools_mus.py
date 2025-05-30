import unittest
from unittest import TestCase

import cpmpy as cp
from cpmpy.tools import mss_opt, marco
from cpmpy.tools.explain import mus, mus_naive, quickxplain, quickxplain_naive, optimal_mus, optimal_mus_naive, mss, mcs


class MusTests(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mus_func = mus
        self.naive_func = mus_naive

    def test_circular(self):
        x = cp.intvar(0, 3, shape=4, name="x")
        # circular "bigger then", UNSAT
        cons = [
            x[0] > x[1], 
            x[1] > x[2],
            x[2] > x[0],
    
            x[3] > x[0],
            (x[3] > x[1]).implies((x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2])))
        ]

        self.assertEqual(set(self.mus_func(cons)), set(cons[:3]))
        self.assertEqual(set(self.naive_func(cons)), set(cons[:3]))

    def test_bug_191(self):
        """
        Original Bug request: https://github.com/CPMpy/cpmpy/issues/191
        When assum is a single boolvar and candidates is a list (of length 1), it fails.
        """
        bv = cp.boolvar(name="x")
        hard = [~bv]
        soft = [bv]

        mus_cons = self.mus_func(soft=soft, hard=hard, solver="ortools") # crashes
        self.assertEqual(set(mus_cons), set(soft))
        mus_naive_cons = self.naive_func(soft=soft, hard=hard) # crashes
        self.assertEqual(set(mus_naive_cons), set(soft))

    def test_bug_191_many_soft(self):
        """
        Checking whether bugfix 191  doesn't break anything in the MUS tool chain,
        when the number of soft constraints > 1.
        """
        x = cp.intvar(-9, 9, name="x")
        y = cp.intvar(-9, 9, name="y")
        hard = [x > 2]
        soft = [
            x + y < 6,
            y == 4
        ]

        mus_cons = self.mus_func(soft=soft, hard=hard) # crashes
        self.assertEqual(set(mus_cons), set(soft))
        mus_naive_cons = self.naive_func(soft=soft, hard=hard) # crashes
        self.assertEqual(set(mus_naive_cons), set(soft))

    def test_wglobal(self):
        x = cp.intvar(-9, 9, name="x")
        y = cp.intvar(-9, 9, name="y")

        cons = [
            x < 0,
            x > 2,
            x < 1,
            y > 0,
            y == 4, 
            (x + y > 0) | (y < 0),
            (y >= 0) | (x >= 0),
            (y < 0) | (x < 0),
            (y > 0) | (x < 0),
            cp.AllDifferent(x,y)
        ]

        # non-determinstic
        #self.assertEqual(set(mus(cons)), set(cons[1:3]))
        ms = self.mus_func(cons)
        self.assertLess(len(ms), len(cons))
        self.assertFalse(cp.Model(ms).solve())
        ms = self.naive_func(cons)
        self.assertLess(len(ms), len(cons))
        self.assertFalse(cp.Model(ms).solve())
        # self.assertEqual(set(self.naive_func(cons)), set(cons[:2]))


class QuickXplainTests(MusTests):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mus_func = quickxplain
        self.naive_func = quickxplain_naive

    def test_prefered(self):

        a,b,c,d = [cp.boolvar(name=n) for n in "abcd"]

        mus1 = [b,d]
        mus2 = [a,b,c]

        hard = [~cp.all(mus1), ~cp.all(mus2)]
        subset = self.mus_func([a,b,c,d],hard)
        self.assertSetEqual(set(subset), {a,b,c})
        subset2 = self.mus_func([d,c,b,a], hard)
        self.assertSetEqual(set(subset2), {b,d})

        subset = self.naive_func([a, b, c, d], hard)
        self.assertSetEqual(set(subset), {a, b, c})
        subset2 = self.naive_func([d, c, b, a], hard)
        self.assertSetEqual(set(subset2), {b, d})

class OptimalMUSTests(MusTests):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mus_func = optimal_mus
        self.mus_func_naive = optimal_mus_naive

    def test_weighted(self):
        a, b, c, d = [cp.boolvar(name=n) for n in "abcd"]

        mus1 = [b, d]
        mus2 = [a, b, c]

        hard = [~cp.all(mus1), ~cp.all(mus2)]
        subset = self.mus_func([a, b, c, d], hard, weights = [1,1,2,4])
        self.assertSetEqual(set(subset), {a, b, c})
        subset2 = self.mus_func([a,b,c,d], hard, weights= [2,3,4,2])
        self.assertSetEqual(set(subset2), {b, d})
        subset3 = self.mus_func([a,b,c,d], hard)
        self.assertEqual(set(subset3), {b,d})

        subset = self.mus_func_naive([a, b, c, d], hard, weights=[1, 1, 2, 4])
        self.assertSetEqual(set(subset), {a, b, c})
        subset2 = self.mus_func_naive([a, b, c, d], hard, weights=[2, 3, 4, 2])
        self.assertSetEqual(set(subset2), {b, d})
        subset3 = self.mus_func_naive([a, b, c, d], hard)
        self.assertEqual(set(subset3), {b, d})

class MARCOMUSTests(MusTests):

    def test_php(self):
        x = cp.boolvar(shape=(5,3), name="x")
        model = cp.Model()
        model += cp.cpm_array(x.sum(axis=1)) >= 1
        model += cp.cpm_array(x.sum(axis=0)) <= 1

        subsets = list(marco(soft=model.constraints))
        musses = [ss for kind, ss in subsets if kind == "MUS"]
        mcses = [ss for kind, ss in subsets if kind == "MCS"]
        self.assertEqual(len(musses), 5)
        self.assertEqual(len(mcses), 13)

        # also works when only enumerating MUSes?
        musses = list(marco(soft=model.constraints, return_mcs=False))
        self.assertEqual(len(musses), 5)
        # or only MCSes?
        mcses = list(marco(soft=model.constraints, return_mus=False))
        self.assertEqual(len(mcses), 13) # any combination of 3 pigeon constraints + 3 mcses with the hole constraints




class MSSTests(unittest.TestCase):

    def test_circular(self):
        x = cp.intvar(0, 3, shape=4, name="x")
        # circular "bigger then", UNSAT
        cons = [
            x[0] > x[1],
            x[1] > x[2],
            x[2] > x[0],

            x[3] > x[0],
            (x[3] > x[1]).implies((x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2])))
        ]

        self.assertLess(len(mss(cons)), len(cons))
        self.assertIn(cons[4], set(mss_opt(cons, weights=[1,1,1,1,5]))) # weighted version


class MCSTests(unittest.TestCase):

    def test_circular(self):
        x = cp.intvar(0, 3, shape=4, name="x")
        # circular "bigger then", UNSAT
        cons = [
            x[0] > x[1],
            x[1] > x[2],
            x[2] > x[0],

            x[3] > x[0],
            (x[3] > x[1]).implies((x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2])))
        ]
        self.assertEqual(len(mcs(cons)), 1)



if __name__ == '__main__':
    unittest.main()