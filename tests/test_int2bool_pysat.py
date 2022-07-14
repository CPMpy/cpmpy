import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.expressions.globalconstraints import AllDifferent
from cpmpy.model import Model
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.int2bool_onehot import int2bool_model, intvar_to_boolvar
import numpy as np

class TestInt2BoolPySAT(unittest.TestCase):
    def setUp(self):
        self.iv = intvar(lb=2, ub=7,name="iv")
        self.bvs = boolvar(shape=5, name="bv")
        self.iv_vector = intvar(lb=2, ub=6, shape=5, name="iv_vec")
        self.iv_2dmatrix = intvar(lb=2, ub=4, shape=(5, 7), name="iv_2d")
        self.iv_3dmatrix = intvar(lb=2, ub=4, shape=(5, 6, 7), name="iv_3d")

    def test_base_bool_model(self):
        iv = intvar(lb=3, ub=7)

        m = Model(
            iv > 4
        )
        s = CPM_pysat(m)
        s.solve()
        self.assertTrue(iv.value() > 4)

    def test_simple_alldiff(self):
        iv1 = intvar(lb=3, ub=3)
        iv2 = intvar(lb=3, ub=4)

        m = Model(
            AllDifferent([iv1, iv2])
        )

        s = CPM_pysat(m)
        s.solve()
        print(iv1.value(), iv2.value())

    def test_incremental_int2bool_model(self):
        iv1 = intvar(lb=3, ub=7)
        iv2 = intvar(lb=3, ub=5)

        m = Model(
            iv1 > 6
        )

        s = CPM_pysat(m)
        s.solve()
        self.assertTrue(iv1.value() > 6)

        s += iv2 < 4
        s.solve()
        self.assertTrue(iv2.value() < 4)

    # # ## CONSTRAINTS:
    def test_equals_var(self):

        iv_model = Model(
            self.iv == 6
        )

        s = CPM_pysat(iv_model)
        s.solve()
        self.assertTrue(self.iv.value() == 6)


    def test_equals_vector(self):

        iv_model = Model(
            self.iv_vector[0] == 2,
            self.iv_vector[1] == 3,
            self.iv_vector[2] == 4,
            self.iv_vector[3] == 3,
            self.iv_vector[4] == 2,
        )

        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)
        iv_model.solve()
        bool_model.solve()

        for iv in self.iv_vector:
            self.assertTrue(ivarmap[iv][iv.value()])

        self.assertEqual(
            extract_solution(ivarmap),
            set((iv, iv.value()) for iv in self.iv_vector)
        )

    def test_equals_vector_assignment(self):

        iv_model = Model(
            self.iv_vector == [2, 3, 4, 3, 2]
        )
        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)
        bool_model.solve()

        iv_model_sol = extract_solution(ivarmap)

        iv_model2 = Model(
            self.iv_vector[0] == 2,
            self.iv_vector[1] == 3,
            self.iv_vector[2] == 4,
            self.iv_vector[3] == 3,
            self.iv_vector[4] == 2,
        )

        ivarmap2, bool_constraints2 = int2bool_model(iv_model2)
        bool_model2 = Model(bool_constraints2)
        bool_model2.solve()
        iv_model2_sol = extract_solution(ivarmap2)

        # both assignment should be valid
        self.assertEqual(iv_model_sol, iv_model2_sol)

    def test_different(self):
        iv_model = Model(
            self.iv != 2,
            self.iv != 3,
            self.iv != 4,
            self.iv != 5,
            self.iv != 6,
        )
        iv_model.solve()
        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)
        bool_model.solve()
        self.assertEqual(extract_solution(ivarmap), set([(self.iv, self.iv.value())]))

    def test_comparison_vars(self):
        iv_model = Model(
            self.iv_vector[0] < self.iv_vector[1],
            self.iv_vector[1] < self.iv_vector[2],
            self.iv_vector[2] < self.iv_vector[3],
            self.iv_vector[3] < self.iv_vector[4],
        )
        iv_model.solve()
        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)
        bool_model.solve()

        self.assertEqual(
            extract_solution(ivarmap),
            set((iv, iv.value()) for iv in self.iv_vector)
        )
    def test_comparison_vars2(self):

        iv_model2 = Model(
            self.iv_vector[0] > self.iv_vector[1],
            self.iv_vector[1] > self.iv_vector[2],
            self.iv_vector[2] > self.iv_vector[3],
            self.iv_vector[3] > self.iv_vector[4],
        )
        iv_model2.solve()
        ivarmap2, bool_constraints2 = int2bool_model(iv_model2)
        bool_model2 = Model(bool_constraints2)
        bool_model2.solve()

        self.assertEqual(
            extract_solution(ivarmap2),
            set((iv, iv.value()) for iv in self.iv_vector)
        )
    def test_comparison_vars3(self):
        iv_model3 = Model(
            self.iv_vector[0] > 3,
            self.iv_vector[0] < self.iv_vector[1],
            self.iv_vector[1] >= self.iv_vector[2],
            self.iv_vector[0] < self.iv_vector[2],
            self.iv_vector[0] < self.iv_vector[3],
            self.iv_vector[2] <= self.iv_vector[3],
            self.iv_vector[3] < self.iv_vector[4],
        )

        iv_model3.solve()

        ivarmap3, bool_constraints3 = int2bool_model(iv_model3)
        bool_model3 = Model(bool_constraints3)
        bool_model3.solve()
        self.assertEqual(
            extract_solution(ivarmap3),
            set((iv, iv.value()) for iv in self.iv_vector)
        )

    def test_comparison(self):
        iv_model = Model(
            self.iv > 5,
            self.iv < 7
        )
        iv_model.solve()
        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)
        bool_model.solve()
        self.assertEqual(extract_solution(ivarmap), set([(self.iv, self.iv.value())]))

        iv_model2 = Model(
            self.iv >= 6,
            self.iv <= 6
        )
        iv_model2.solve()
        ivarmap2, bool_constraints2 = int2bool_model(iv_model2)
        bool_model2 = Model(bool_constraints2)
        bool_model2.solve()
        self.assertEqual(extract_solution(ivarmap2), set([(self.iv, self.iv.value())]))

    def test_comparison_edge_cases(self):
        iv_model = Model(
            self.iv < 10,
            self.iv > -1,
            self.iv >= 7
        )

        iv_model.solve()
        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)
        bool_model.solve()
        self.assertEqual(extract_solution(ivarmap), set([(self.iv, self.iv.value())]))

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

        bv_solved = CPM_pysat(sudoku_iv_model).solve()
        bv_solution = puzzle.value()

        iv_solved = sudoku_iv_model.solve()
        iv_solution = puzzle.value()

        self.assertTrue(bv_solved & iv_solved)
        self.assertTrue(np.all(bv_solution == iv_solution))

class TestInt2BoolConstraints(unittest.TestCase):
    def setUp(self):
        self.iv = intvar(lb=2, ub=7,name="iv")
        self.bvs = boolvar(shape=5, name="bv")
        self.iv_vector = intvar(lb=2, ub=6, shape=5, name="iv_vec")
        self.iv_2dmatrix = intvar(lb=2, ub=4, shape=(5, 7), name="iv_2d")
        self.iv_3dmatrix = intvar(lb=2, ub=4, shape=(5, 6, 7), name="iv_3d")

    ## VARIABLES: intvar -> boolvar
    def test_intvar_to_boolvar(self):
        iv = intvar(0, 5, shape=1, name="x")
        ivarmap, _ = intvar_to_boolvar(iv)
        self.assertEqual(len(ivarmap[iv]), iv.ub - iv.lb + 1)

    def test_intvar_arr_to_boolvar_array(self):
        iv = intvar(0, 5, shape=10, name="x")
        ivarmap, _ = intvar_to_boolvar(iv)
        self.assertEqual(len(ivarmap), iv.shape[0])

        # handle more complex matrix data structures
        iv_matrix = intvar(0, 5, shape=(10, 10), name="x")
        ivarmap, _ = intvar_to_boolvar(iv_matrix)

        self.assertEqual(len(ivarmap), iv_matrix.shape[0] * iv_matrix.shape[1])

        for varmap in ivarmap.values():
            self.assertEqual(len(varmap), iv_matrix[0, 0].ub - iv_matrix[0, 0].lb + 1)

    # ## CONSTRAINTS:
    def test_equals_var(self):

        iv_model = Model(
            self.iv == 6
        )

        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)

        # solve both models and checking for result
        bool_model.solve()
        iv_model.solve()

        # return same value
        self.assertEqual(extract_solution(ivarmap), set([(self.iv, self.iv.value())]))

    def test_equals_vector(self):

        iv_model = Model(
            self.iv_vector[0] == 2,
            self.iv_vector[1] == 3,
            self.iv_vector[2] == 4,
            self.iv_vector[3] == 3,
            self.iv_vector[4] == 2,
        )

        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)
        iv_model.solve()
        bool_model.solve()

        for iv in self.iv_vector:
            self.assertTrue(ivarmap[iv][iv.value()])

        self.assertEqual(
            extract_solution(ivarmap),
            set((iv, iv.value()) for iv in self.iv_vector)
        )

    def test_equals_vector_assignment(self):

        iv_model = Model(
            self.iv_vector == [2, 3, 4, 3, 2]
        )
        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)
        bool_model.solve()

        iv_model_sol = extract_solution(ivarmap)

        iv_model2 = Model(
            self.iv_vector[0] == 2,
            self.iv_vector[1] == 3,
            self.iv_vector[2] == 4,
            self.iv_vector[3] == 3,
            self.iv_vector[4] == 2,
        )

        ivarmap2, bool_constraints2 = int2bool_model(iv_model2)
        bool_model2 = Model(bool_constraints2)
        bool_model2.solve()
        iv_model2_sol = extract_solution(ivarmap)

        # both assignment should be valid
        self.assertEqual(iv_model_sol, iv_model2_sol)

    def test_different(self):
        iv_model = Model(
            self.iv != 2,
            self.iv != 3,
            self.iv != 4,
            self.iv != 5,
            self.iv != 6,
        )
        iv_model.solve()
        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)
        bool_model.solve()
        self.assertEqual(extract_solution(ivarmap), set([(self.iv, self.iv.value())]))

    def test_comparison_vars(self):
        iv_model = Model(
            self.iv_vector[0] < self.iv_vector[1],
            self.iv_vector[1] < self.iv_vector[2],
            self.iv_vector[2] < self.iv_vector[3],
            self.iv_vector[3] < self.iv_vector[4],
        )
        iv_model.solve()
        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)
        bool_model.solve()

        self.assertEqual(
            extract_solution(ivarmap),
            set((iv, iv.value()) for iv in self.iv_vector)
        )
    def test_comparison_vars2(self):

        iv_model2 = Model(
            self.iv_vector[0] > self.iv_vector[1],
            self.iv_vector[1] > self.iv_vector[2],
            self.iv_vector[2] > self.iv_vector[3],
            self.iv_vector[3] > self.iv_vector[4],
        )
        iv_model2.solve()
        ivarmap2, bool_constraints2 = int2bool_model(iv_model2)
        bool_model2 = Model(bool_constraints2)
        bool_model2.solve()

        self.assertEqual(
            extract_solution(ivarmap2),
            set((iv, iv.value()) for iv in self.iv_vector)
        )
    def test_comparison_vars3(self):
        iv_model3 = Model(
            self.iv_vector[0] > 3,
            self.iv_vector[0] < self.iv_vector[1],
            self.iv_vector[1] >= self.iv_vector[2],
            self.iv_vector[0] < self.iv_vector[2],
            self.iv_vector[0] < self.iv_vector[3],
            self.iv_vector[2] <= self.iv_vector[3],
            self.iv_vector[3] < self.iv_vector[4],
        )

        iv_model3.solve()

        ivarmap3, bool_constraints3 = int2bool_model(iv_model3)
        bool_model3 = Model(bool_constraints3)
        bool_model3.solve()
        self.assertEqual(
            extract_solution(ivarmap3),
            set((iv, iv.value()) for iv in self.iv_vector)
        )

    def test_comparison(self):
        iv_model = Model(
            self.iv > 5,
            self.iv < 7
        )
        iv_model.solve()
        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)
        bool_model.solve()
        self.assertEqual(extract_solution(ivarmap), set([(self.iv, self.iv.value())]))

        iv_model2 = Model(
            self.iv >= 6,
            self.iv <= 6
        )
        iv_model2.solve()
        ivarmap2, bool_constraints2 = int2bool_model(iv_model2)
        bool_model2 = Model(bool_constraints2)
        bool_model2.solve()
        self.assertEqual(extract_solution(ivarmap2), set([(self.iv, self.iv.value())]))

    def test_comparison_edge_cases(self):
        iv_model = Model(
            self.iv < 10,
            self.iv > -1,
            self.iv >= 7
        )

        iv_model.solve()
        ivarmap, bool_constraints = int2bool_model(iv_model)
        bool_model = Model(bool_constraints)
        bool_model.solve()
        self.assertEqual(extract_solution(ivarmap), set([(self.iv, self.iv.value())]))


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
        ivarmap, sudoku_bool_constraints = int2bool_model(sudoku_iv_model)
        sudoku_bv_model = Model(sudoku_bool_constraints)
        sudoku_bv_model.solve()

        self.assertEqual(
            extract_solution(ivarmap),
            set((iv, iv.value() ) for iv in puzzle.flat )
        )

def extract_solution(ivarmap):

    sol = set()
    for iv, value_dict in ivarmap.items():
        n_val_assigned = sum(1 if bv.value() else 0 for bv in value_dict.values())
        assert n_val_assigned == 1, f"Expected: 1, Got: {n_val_assigned} value can be assigned!"
        for iv_val, bv in value_dict.items():
            if bv.value():
                sol.add((iv, iv_val))

    return sol


if __name__ == '__main__':
    unittest.main()


