import unittest
import numpy as np
import cpmpy as cp
from cpmpy.expressions import *
from cpmpy.model import Model
from examples.nqueens import nqueens

class TestInt2boolExamples(unittest.TestCase):
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

        sudoku_iv_model.solve()

        ivarmap, sudoku_bv_model = sudoku_iv_model.int2bool_onehot()
        sudoku_bv_model.solve()

        self.assertEqual(
            extract_solution(ivarmap),
            set((iv, iv.value() ) for iv in puzzle.flat )
        )

def extract_solution(ivarmap):

    sol = set()
    for iv, value_dict in ivarmap.items():
        n_val_assigned = sum(1 if bv.value() else 0 for iv_val, bv in value_dict.items())
        assert n_val_assigned == 1, f"Expected: 1, Got: {n_val_assigned} value can be assigned!"
        for iv_val, bv in value_dict.items():
            if bv.value():
                sol.add((iv, iv_val))

    return sol

if __name__ == '__main__':
    unittest.main()


