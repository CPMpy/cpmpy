#!/usr/bin/python3
"""
Planning problem in CPMPy

Based on the famous planning puzzle.
https://en.wikipedia.org/wiki/Wolf,_goat_and_cabbage_problem
The wolf, goat and cabbage have to be transported to the other side of a river by a boat.
At no time can the wolf and goat be left alone, as well as the goat and the cabbage.
"""

from cpmpy import *

def run():
    stage = 3
    while True:
        (model, vars) = model_wgc(stage)
        if model.solve():
            print("Found a solution for " + str(stage) + " stage!")
            for (name, var) in vars.items():
                print(f"{name}:\n{var.value()}")
            break
        else:
            print("No solution for " + str(stage) + " stage")
            stage += 1

def model_wgc(stage):
    wolf_pos = boolvar(stage)
    cabbage_pos = boolvar(stage)
    goat_pos = boolvar(stage)
    boat_pos = boolvar(stage)

    model = Model(
        # Initial situation
        (boat_pos[0] == 0),
        (wolf_pos[0] == 0),
        (goat_pos[0] == 0),
        (cabbage_pos[0] == 0),

        # Boat keeps moving between shores
        [boat_pos[i] != boat_pos[i-1] for i in range(1,stage)],   

        # Final situation
        (boat_pos[-1] == 1),
        (wolf_pos[-1] == 1),
        (goat_pos[-1] == 1),
        (cabbage_pos[-1] == 1),

        # # Wolf and goat cannot be left alone
        [(goat_pos[i] != wolf_pos[i]) | (boat_pos[i] == wolf_pos[i]) for i in range(stage)],

        # # Goat and cabbage cannot be left alone
        [(goat_pos[i] != cabbage_pos[i]) | (boat_pos[i] == goat_pos[i]) for i in range(stage)],

        # # Only one animal/cabbage can move per turn
        [abs(wolf_pos[i] - wolf_pos[i+1]) + abs(goat_pos[i] - goat_pos[i+1]) + abs(cabbage_pos[i] - cabbage_pos[i+1]) <= 1 for i in range(stage-1)],
    )

    return (model, {"wolf_pos": wolf_pos, "goat_pos": goat_pos, "cabbage_pos": cabbage_pos, "boat_pos": boat_pos})

if __name__ == "__main__":
    run()