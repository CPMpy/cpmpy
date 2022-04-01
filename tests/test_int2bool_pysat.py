import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.model import Model
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.int2bool_onehot import to_bool_constraint
import numpy as np

class TestInt2BoolPySAT(unittest.TestCase):
    # def setUp(self):
    #     self.iv = intvar(lb=2, ub=7,name="iv")
    #     self.bvs = boolvar(shape=5, name="bv")
    #     self.iv_vector = intvar(lb=2, ub=6, shape=5, name="iv_vec")
    #     self.iv_2dmatrix = intvar(lb=2, ub=4, shape=(5, 7), name="iv_2d")
    #     self.iv_3dmatrix = intvar(lb=2, ub=4, shape=(5, 6, 7), name="iv_3d")

    # def test_base_bool_model(self):
    #     iv = intvar(lb=3, ub=7)

    #     m = Model(
    #         iv > 4
    #     )
    #     s = CPM_pysat(m)
    #     s.solve()
    #     self.assertTrue(iv.value() > 4)
    
    def test_simple_alldiff(self):
        iv1 = intvar(lb=3, ub=3)
        iv2 = intvar(lb=3, ub=4)

        m = Model(
            alldifferent([iv1, iv2])
        )

        s = CPM_pysat(m)
        s.solve()
        print(iv1.value(), iv2.value())
    
    # def test_incremental_int2bool_model(self):
    #     iv1 = intvar(lb=3, ub=7)
    #     iv2 = intvar(lb=3, ub=5)

    #     m = Model(
    #         iv1 > 6
    #     )

    #     s = CPM_pysat(m)
    #     s.solve()
    #     self.assertTrue(iv1.value() > 6)

    #     s += iv2 < 4
    #     s.solve()
    #     self.assertTrue(iv2.value() < 4)

    # # ## CONSTRAINTS:
    # def test_equals_var(self):

    #     iv_model = Model(
    #         self.iv == 6
    #     )

    #     s = CPM_pysat(iv_model)
    #     s.solve()
    #     self.assertTrue(self.iv.value() == 6)


#     def test_equals_vector(self):

#         iv_model = Model(
#             self.iv_vector[0] == 2,
#             self.iv_vector[1] == 3,
#             self.iv_vector[2] == 4,
#             self.iv_vector[3] == 3,
#             self.iv_vector[4] == 2,
#         )

#         ivarmap, bool_constraints = int2bool_onehot(iv_model)
#         bool_model = Model(bool_constraints)
#         iv_model.solve()
#         bool_model.solve()

#         for iv in self.iv_vector:
#             self.assertTrue(ivarmap[iv][iv.value()])

#         self.assertEqual(
#             extract_solution(ivarmap),
#             set((iv, iv.value()) for iv in self.iv_vector)
#         )

#     def test_equals_vector_assignment(self):

#         iv_model = Model(
#             self.iv_vector == [2, 3, 4, 3, 2]
#         )
#         ivarmap, bool_constraints = int2bool_onehot(iv_model)
#         bool_model = Model(bool_constraints)
#         bool_model.solve()

#         iv_model_sol = extract_solution(ivarmap)

#         iv_model2 = Model(
#             self.iv_vector[0] == 2,
#             self.iv_vector[1] == 3,
#             self.iv_vector[2] == 4,
#             self.iv_vector[3] == 3,
#             self.iv_vector[4] == 2,
#         )

#         ivarmap2, bool_constraints2 = int2bool_onehot(iv_model2)
#         bool_model2 = Model(bool_constraints2)
#         bool_model2.solve()
#         iv_model2_sol = extract_solution(ivarmap)

#         # both assignment should be valid
#         self.assertEqual(iv_model_sol, iv_model2_sol)

#     def test_different(self):
#         iv_model = Model(
#             self.iv != 2,
#             self.iv != 3,
#             self.iv != 4,
#             self.iv != 5,
#             self.iv != 6,
#         )
#         iv_model.solve()
#         ivarmap, bool_constraints = int2bool_onehot(iv_model)
#         bool_model = Model(bool_constraints)
#         bool_model.solve()
#         self.assertEqual(extract_solution(ivarmap), set([(self.iv, self.iv.value())]))

#     def test_comparison_vars(self):
#         iv_model = Model(
#             self.iv_vector[0] < self.iv_vector[1],
#             self.iv_vector[1] < self.iv_vector[2],
#             self.iv_vector[2] < self.iv_vector[3],
#             self.iv_vector[3] < self.iv_vector[4],
#         )
#         iv_model.solve()
#         ivarmap, bool_constraints = int2bool_onehot(iv_model)
#         bool_model = Model(bool_constraints)
#         bool_model.solve()

#         self.assertEqual(
#             extract_solution(ivarmap),
#             set((iv, iv.value()) for iv in self.iv_vector)
#         )
#     def test_comparison_vars2(self):

#         iv_model2 = Model(
#             self.iv_vector[0] > self.iv_vector[1],
#             self.iv_vector[1] > self.iv_vector[2],
#             self.iv_vector[2] > self.iv_vector[3],
#             self.iv_vector[3] > self.iv_vector[4],
#         )
#         iv_model2.solve()
#         ivarmap2, bool_constraints2 = int2bool_onehot(iv_model2)
#         bool_model2 = Model(bool_constraints2)
#         bool_model2.solve()

#         self.assertEqual(
#             extract_solution(ivarmap2),
#             set((iv, iv.value()) for iv in self.iv_vector)
#         )
#     def test_comparison_vars3(self):
#         iv_model3 = Model(
#             self.iv_vector[0] > 3,
#             self.iv_vector[0] < self.iv_vector[1],
#             self.iv_vector[1] >= self.iv_vector[2],
#             self.iv_vector[0] < self.iv_vector[2],
#             self.iv_vector[0] < self.iv_vector[3],
#             self.iv_vector[2] <= self.iv_vector[3],
#             self.iv_vector[3] < self.iv_vector[4],
#         )

#         iv_model3.solve()

#         ivarmap3, bool_constraints3 = int2bool_onehot(iv_model3)
#         bool_model3 = Model(bool_constraints3)
#         bool_model3.solve()
#         self.assertEqual(
#             extract_solution(ivarmap3),
#             set((iv, iv.value()) for iv in self.iv_vector)
#         )

#     def test_comparison(self):
#         iv_model = Model(
#             self.iv > 5,
#             self.iv < 7
#         )
#         iv_model.solve()
#         ivarmap, bool_constraints = int2bool_onehot(iv_model)
#         bool_model = Model(bool_constraints)
#         bool_model.solve()
#         self.assertEqual(extract_solution(ivarmap), set([(self.iv, self.iv.value())]))

#         iv_model2 = Model(
#             self.iv >= 6,
#             self.iv <= 6
#         )
#         iv_model2.solve()
#         ivarmap2, bool_constraints2 = int2bool_onehot(iv_model2)
#         bool_model2 = Model(bool_constraints2)
#         bool_model2.solve()
#         self.assertEqual(extract_solution(ivarmap2), set([(self.iv, self.iv.value())]))

#     def test_comparison_edge_cases(self):
#         iv_model = Model(
#             self.iv < 10,
#             self.iv > -1,
#             self.iv >= 7
#         )

#         iv_model.solve()
#         ivarmap, bool_constraints = int2bool_onehot(iv_model)
#         bool_model = Model(bool_constraints)
#         bool_model.solve()
#         self.assertEqual(extract_solution(ivarmap), set([(self.iv, self.iv.value())]))


# class TestInt2boolExamples(unittest.TestCase):
#     def test_sudoku(self):

#         e = 0 # value for empty cells
#         given = np.array([
#             [e, e, e,  2, e, 5,  e, e, e],
#             [e, 9, e,  e, e, e,  7, 3, e],
#             [e, e, 2,  e, e, 9,  e, 6, e],

#             [2, e, e,  e, e, e,  4, e, 9],
#             [e, e, e,  e, 7, e,  e, e, e],
#             [6, e, 9,  e, e, e,  e, e, 1],

#             [e, 8, e,  4, e, e,  1, e, e],
#             [e, 6, 3,  e, e, e,  e, 8, e],
#             [e, e, e,  6, e, 8,  e, e, e]])

#         # Variables
#         puzzle = intvar(1,9, shape=given.shape, name="puzzle")

#         sudoku_iv_model = Model(
#             # Constraints on values (cells that are not empty)
#             puzzle[given!=e] == given[given!=e], # numpy's indexing, vectorized equality
#             # Constraints on rows and columns
#             [AllDifferent(row) for row in puzzle],
#             [AllDifferent(col) for col in puzzle.T], # numpy's Transpose
#         )

#         # Constraints on blocks
#         for i in range(0,9, 3):
#             for j in range(0,9, 3):
#                 sudoku_iv_model += AllDifferent(puzzle[i:i+3, j:j+3]) # python's indexing

#         sudoku_iv_model.solve()
#         ivarmap, sudoku_bool_constraints = int2bool_onehot(sudoku_iv_model)
#         sudoku_bv_model = Model(sudoku_bool_constraints)
#         sudoku_bv_model.solve()

#         self.assertEqual(
#             extract_solution(ivarmap),
#             set((iv, iv.value() ) for iv in puzzle.flat )
#         )

# def extract_solution(ivarmap):

#     sol = set()
#     for iv, value_dict in ivarmap.items():
#         n_val_assigned = sum(1 if bv.value() else 0 for iv_val, bv in value_dict.items())
#         assert n_val_assigned == 1, f"Expected: 1, Got: {n_val_assigned} value can be assigned!"
#         for iv_val, bv in value_dict.items():
#             if bv.value():
#                 sol.add((iv, iv_val))

#     return sol

    # abs() not implemented yet
    #def test_weird_expression(self):
    #    stage = 3
    #    wolf_pos = boolvar(stage)
    #    cabbage_pos = boolvar(stage)
    #    goat_pos = boolvar(stage)
    #    boat_pos = boolvar(stage)
    #    for i in range(stage-1):
    #        con = abs(wolf_pos[i] - wolf_pos[i+1]) + abs(goat_pos[i] - goat_pos[i+1]) + abs(cabbage_pos[i] - cabbage_pos[i+1]) <= 1
    #        # print(abs(wolf_pos[i] - wolf_pos[i+1]) + abs(goat_pos[i] - goat_pos[i+1]) + abs(cabbage_pos[i] - cabbage_pos[i+1]) <= 1)
    #        cnf_con = to_cnf(con)
    #        print(f"{con=}")
    #        print(f"{cnf_con=}")
    #        for ci in cnf_con:
    #            print(f"{ci=}")
    #            print(f"{to_bool_constraint(ci)}")

# class TestInt2boolPySATExamples(unittest.TestCase):
#     def test_sudoku(self):

#         e = 0 # value for empty cells
#         given = np.array([
#             [e, e, e,  2, e, 5,  e, e, e],
#             [e, 9, e,  e, e, e,  7, 3, e],
#             [e, e, 2,  e, e, 9,  e, 6, e],

#             [2, e, e,  e, e, e,  4, e, 9],
#             [e, e, e,  e, 7, e,  e, e, e],
#             [6, e, 9,  e, e, e,  e, e, 1],

#             [e, 8, e,  4, e, e,  1, e, e],
#             [e, 6, 3,  e, e, e,  e, 8, e],
#             [e, e, e,  6, e, 8,  e, e, e]])

#         # Variables
#         puzzle = intvar(1,9, shape=given.shape, name="puzzle")

#         sudoku_iv_model = Model(
#             # Constraints on values (cells that are not empty)
#             puzzle[given!=e] == given[given!=e], # numpy's indexing, vectorized equality
#             # Constraints on rows and columns
#             [AllDifferent(row) for row in puzzle],
#             [AllDifferent(col) for col in puzzle.T], # numpy's Transpose
#         )

#         # Constraints on blocks
#         for i in range(0,9, 3):
#             for j in range(0,9, 3):
#                 sudoku_iv_model += AllDifferent(puzzle[i:i+3, j:j+3]) # python's indexing

#         CPM_pysat(sudoku_iv_model).solve()
#         solution_pysat = puzzle.value()
#         sudoku_iv_model.solve()
#         solution_ortools = puzzle.value()

#         for sol_pysat, sol_ortools in zip(solution_pysat.flat, solution_ortools.flat):
#             self.assertEqual(sol_pysat, sol_ortools)

#     def test_wolf_goat_cabbage(self):
#         def model_wgc(stage):
#             wolf_pos = boolvar(stage)
#             cabbage_pos = boolvar(stage)
#             goat_pos = boolvar(stage)
#             boat_pos = boolvar(stage)

#             model = Model(
#                 # Initial situation
#                 (boat_pos[0] == 0),
#                 (wolf_pos[0] == 0),
#                 (goat_pos[0] == 0),
#                 (cabbage_pos[0] == 0),

#                 # Boat keeps moving between shores
#                 [boat_pos[i] != boat_pos[i-1] for i in range(1,stage)],

#                 # Final situation
#                 (boat_pos[-1] == 1),
#                 (wolf_pos[-1] == 1),
#                 (goat_pos[-1] == 1),
#                 (cabbage_pos[-1] == 1),

#                 # # Wolf and goat cannot be left alone
#                 [(goat_pos[i] != wolf_pos[i]) | (boat_pos[i] == wolf_pos[i]) for i in range(stage)],

#                 # # Goat and cabbage cannot be left alone
#                 [(goat_pos[i] != cabbage_pos[i]) | (boat_pos[i] == goat_pos[i]) for i in range(stage)],

#                 # # Only one animal/cabbage can move per turn
#                 [abs(wolf_pos[i] - wolf_pos[i+1]) + abs(goat_pos[i] - goat_pos[i+1]) + abs(cabbage_pos[i] - cabbage_pos[i+1]) <= 1 for i in range(stage-1)],
#             )

#             return (model, {"wolf_pos": wolf_pos, "goat_pos": goat_pos, "cabbage_pos": cabbage_pos, "boat_pos": boat_pos})

#         stage = 3
#         while True:
#             (model, vars) = model_wgc(stage)
#             s = CPM_pysat(model)
#             if s.solve():
#                 print("Found a solution for " + str(stage) + " stage!")
#                 for (name, var) in vars.items():
#                     print(f"{name}:\n{var.value()}")
#                 break
#             else:
#                 print("No solution for " + str(stage) + " stage")
#                 stage += 1


#     def test_zebra(self):

#         n_houses = 5

#         # colors[i] is the house of the ith color
#         yellow, green, red, white, blue = colors = intvar(0,n_houses-1, shape=n_houses)

#         # nations[i] is the house of the inhabitant with the ith nationality
#         italy, spain, japan, england, norway = nations = intvar(0,n_houses-1, shape=n_houses)

#         # jobs[i] is the house of the inhabitant with the ith job
#         painter, sculptor, diplomat, pianist, doctor = jobs = intvar(0,n_houses-1, shape=n_houses)

#         # pets[i] is the house of the inhabitant with the ith pet
#         cat, zebra, bear, snails, horse = pets = intvar(0,n_houses-1, shape=n_houses)

#         # drinks[i] is the house of the inhabitant with the ith preferred drink
#         milk, water, tea, coffee, juice = drinks = intvar(0,n_houses-1, shape=n_houses)

#         zebra_model = Model(
#             AllDifferent(colors),
#             AllDifferent(nations),
#             AllDifferent(jobs),
#             AllDifferent(pets),
#             AllDifferent(drinks),
#             painter == horse,
#             diplomat == coffee,
#             white == milk,
#             spain == painter,
#             england == red,
#             snails == sculptor,
#             1 == red - green,
#             1 == norway - blue,
#             doctor == milk,
#             japan == diplomat,
#             norway == zebra,
#             abs(green - white) == 1,
#             # # horse in {diplomat - 1, diplomat + 1},
#             1 == abs(diplomat-horse),
#             #italy in {red, white, green}
#             (italy == red)|(italy == white)|(italy == green),
#         )

#         zebra_pysat_solver = CPM_pysat(zebra_model)
#         zebra_pysat_solver.solve()

if __name__ == '__main__':
    unittest.main()


