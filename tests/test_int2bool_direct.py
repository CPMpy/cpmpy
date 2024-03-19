import unittest

import numpy as np
import pytest

from cpmpy import *
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.solvers.int2bool_direct import CPM_int2bool_direct


class TestInt2BoolDirect(unittest.TestCase):

    def test_i2bd_small(self):
        x = intvar(3,6, shape=3, name="x")
        c = (x > 3) & AllDifferent(x)

        s = CPM_int2bool_direct(subsolver="ortools")  # "pysat")
        s += c
        self.assertTrue(s.solve())
        for cc in toplevel_list(c):
            self.assertTrue(cc.value())

    def test_i2bd_sudoku(self):
        e = 0 # value for empty cells
        given = np.array([
            [e, e, 2,  4, 1, e,  e, e, 5],
            [1, e, 4,  3, e, e,  e, e, e],
            [e, 8, e,  2, 7, 5,  3, 4, 1],

            [e, e, e,  e, 3, 1,  e, e, e],
            [7, 9, e,  e, e, e,  e, 2, e],
            [e, e, e,  e, e, e,  e, e, e],

            [e, e, e,  e, e, 4,  e, 6, e],
            [5, e, e,  8, e, e,  4, e, 9],
            [e, 4, e,  1, e, 3,  5, 7, e]])

        model = Model()

        # Variables
        puzzle = intvar(1, 9, shape=given.shape, name="puzzle")

        # Constraints on rows and columns
        model += [AllDifferent(row) for row in puzzle]
        model += [AllDifferent(col) for col in puzzle.T]

        # Constraints on blocks
        for i in range(0,9, 3):
            for j in range(0,9, 3):
                model += AllDifferent(puzzle[i:i+3, j:j+3])

        # Constraints on values (cells that are not empty)
        model += (puzzle[given!=e] == given[given!=e])
        
        s = CPM_int2bool_direct(model, subsolver="ortools")  # pysat")
        self.assertTrue(s.solve())
        self.assertTrue(puzzle[0,0].value() == 3)
        for cc in toplevel_list(model.constraints):
            self.assertTrue(cc.value())
        
