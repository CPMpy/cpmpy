import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.model import Model
from cpmpy.solvers.pysat import CPM_pysat
import numpy as np

class TestInt2BoolPySAT(unittest.TestCase):
    def test_base_bool_model(self):
        iv = intvar(lb=3, ub=7)

        m = Model(
            iv > 4
        )
        s = CPM_pysat(m)
        s.solve()
        print(iv.value())
    
    def test_incremental_int2bool_model(self):
        iv1 = intvar(lb=3, ub=7)
        iv2 = intvar(lb=3, ub=5)

        m = Model(
            iv1 > 6
        )

        s = CPM_pysat(m)
        s.solve()
        print(iv1.value())

        s += iv2 < 4
        s.solve()
        print(iv1.value(), iv2.value())

class TestInt2boolPySATExamples(unittest.TestCase):
    def test_sudoku(self):

        e = 0 # value for empty cells
        given = np.array([
            [e, e, e,  2, e, 5,  e, e, e],
            [e, 9, e,  e, e, e,  7, 3, e],
            [e, e, 2,  e, e, 9,  e, 6, e],

            [2, e, e,  e, e, e,  4, e, 9],
            [e, e, e,  e, 7, e,  e, e, e],
            [6, e, 9,  e, e, e,  e, e, 1],

            [e, 8, e,  4, e, e,  1, e, e],
            [e, 6, 3,  e, e, e,  e, 8, e],
            [e, e, e,  6, e, 8,  e, e, e]])

        # Variables
        puzzle = intvar(1,9, shape=given.shape, name="puzzle")

        sudoku_iv_model = Model(
            # Constraints on values (cells that are not empty)
            puzzle[given!=e] == given[given!=e], # numpy's indexing, vectorized equality
            # Constraints on rows and columns
            [AllDifferent(row) for row in puzzle],
            [AllDifferent(col) for col in puzzle.T], # numpy's Transpose
        )

        # Constraints on blocks
        for i in range(0,9, 3):
            for j in range(0,9, 3):
                sudoku_iv_model += AllDifferent(puzzle[i:i+3, j:j+3]) # python's indexing

        CPM_pysat(sudoku_iv_model).solve()
        solution_pysat = puzzle.value()
        sudoku_iv_model.solve()
        solution_ortools = puzzle.value()

        for sol_pysat, sol_ortools in zip(solution_pysat.flat, solution_ortools.flat):
            self.assertEqual(sol_pysat, sol_ortools)

if __name__ == '__main__':
    unittest.main()


