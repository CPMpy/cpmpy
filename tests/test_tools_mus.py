import unittest
from unittest import TestCase

from cpmpy import *
from cpmpy.tools.mus import mus, mus_naive

class MusTests(TestCase):

    def test_circular(self):
        x = intvar(0, 3, shape=4, name="x")
        # circular "bigger then", UNSAT
        cons = [
            x[0] > x[1], 
            x[1] > x[2],
            x[2] > x[0],
    
            x[3] > x[0],
            (x[3] > x[1]).implies((x[3] > x[2]) & ((x[3] == 3) | (x[1] == x[2])))
        ]

        self.assertEqual(mus(cons), cons[:3])
        self.assertEqual(mus_naive(cons), cons[:3])

    def test_bug_191(self):
        """
        Original Bug request: https://github.com/CPMpy/cpmpy/issues/191
        When assum is a single boolvar and candidates is a list (of length 1), it fails.
        """
        bv = boolvar(name="x")
        hard = [~bv]
        soft = [bv]

        mus_cons = mus(soft=soft, hard=hard) # crashes
        self.assertEqual(set(mus_cons), set(soft))
        mus_naive_cons = mus_naive(soft=soft, hard=hard) # crashes
        self.assertEqual(set(mus_naive_cons), set(soft))

    def test_bug_191_many_soft(self):
        """
        Checking whether bugfix 191  doesn't break anything in the MUS tool chain,
        when the number of soft constraints > 1.
        """
        x = intvar(-9, 9, name="x")
        y = intvar(-9, 9, name="y")
        hard = [x > 2]
        soft = [
            x + y < 6,
            y == 4
        ]

        mus_cons = mus(soft=soft, hard=hard) # crashes
        self.assertEqual(set(mus_cons), set(soft))
        mus_naive_cons = mus_naive(soft=soft, hard=hard) # crashes
        self.assertEqual(set(mus_naive_cons), set(soft))

    def test_wglobal(self):
        x = intvar(-9, 9, name="x")
        y = intvar(-9, 9, name="y")

        cons = [
            x < 0, 
            x < 1,
            x > 2,
            y > 0,
            y == 4, 
            (x + y > 0) | (y < 0),
            (y >= 0) | (x >= 0),
            (y < 0) | (x < 0),
            (y > 0) | (x < 0),
            AllDifferent(x,y) # invalid for musx_assum
        ]

        # non-determinstic
        #self.assertEqual(set(mus(cons)), set(cons[1:3]))
        ms = mus(cons)
        self.assertLess(len(ms), len(cons))
        self.assertFalse(Model(ms).solve())
        self.assertEqual(set(mus_naive(cons)), set(cons[1:3]))

if __name__ == '__main__':
    unittest.main()