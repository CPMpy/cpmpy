import time
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

        self.assertEqual(set(mus(cons)), set(cons[1:3]))
        self.assertEqual(set(mus_naive(cons)), set(cons[1:3]))
