import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.model import Model
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.int2bool_onehot import to_bool_constraint
import numpy as np

class TestInt2BoolPySAT(unittest.TestCase):
    def test_base_bool_model(self):
        iv = intvar(lb=3, ub=7)

        m = Model(
            iv > 4
        )
        s = CPM_pysat(m)
        s.solve()
    
    def test_incremental_int2bool_model(self):
        iv1 = intvar(lb=3, ub=7)
        iv2 = intvar(lb=3, ub=5)

        m = Model(
            iv1 > 6
        )

        s = CPM_pysat(m)
        s.solve()

        s += iv2 < 4
        s.solve()
    
    def test_weird_expression(self):
        stage = 3
        wolf_pos = boolvar(stage)
        cabbage_pos = boolvar(stage)
        goat_pos = boolvar(stage)
        boat_pos = boolvar(stage)
        for i in range(stage-1):
            con = abs(wolf_pos[i] - wolf_pos[i+1]) + abs(goat_pos[i] - goat_pos[i+1]) + abs(cabbage_pos[i] - cabbage_pos[i+1]) <= 1
            # print(abs(wolf_pos[i] - wolf_pos[i+1]) + abs(goat_pos[i] - goat_pos[i+1]) + abs(cabbage_pos[i] - cabbage_pos[i+1]) <= 1)
            cnf_con = to_cnf(con)
            print(f"{con=}")
            print(f"{cnf_con=}")
            for ci in cnf_con:
                print(f"{ci=}")
                print(f"{to_bool_constraint(ci)}")

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


