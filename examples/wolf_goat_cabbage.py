#!/usr/bin/python3
"""
Planning problem in CPMPy

Based on the famous planning puzzle.
https://en.wikipedia.org/wiki/Wolf,_goat_and_cabbage_problem
"""

from cpmpy import *

stages = 3

while stages < 20:

    wolf_pos = intvar(0, 1, stages)
    cabbage_pos = intvar(0, 1, stages)
    goat_pos = intvar(0, 1, stages)
    boat_pos = intvar(0, 1, stages)

    model = Model(
        # Initial situation
        (boat_pos[0] == 0),
        (wolf_pos[0] == 0),
        (goat_pos[0] == 0),
        (cabbage_pos[0] == 0),

        # Boat keeps moving between shores
        [boat_pos[i] != boat_pos[i-1] for i in range(1,stages)],   

        # Final situation
        (boat_pos[stages-1] == 1),
        (wolf_pos[stages-1] == 1),
        (goat_pos[stages-1] == 1),
        (cabbage_pos[stages-1] == 1),

        # # Wolf and goat cannot be left alone
        [(goat_pos[i] != wolf_pos[i]) | (boat_pos[i] == wolf_pos[i]) for i in range(stages)],

        # # Goat and cabbage cannot be left alone
        [(goat_pos[i] != cabbage_pos[i]) | (boat_pos[i] == goat_pos[i]) for i in range(stages)],

        # # Only one animal/cabbage can move per turn
        [abs(wolf_pos[i] - wolf_pos[i+1]) + abs(goat_pos[i] - goat_pos[i+1]) + abs(cabbage_pos[i] - cabbage_pos[i+1]) <= 1 for i in range(stages-1)],
    )

    if model.solve():
        print(boat_pos.value())
        print(wolf_pos.value())
        print(goat_pos.value())
        print(cabbage_pos.value())
        print("Found a solution for " + str(stages) + " stages!")
        break
    else:
        print("No solution for " + str(stages) + " stages")
        stages += 1