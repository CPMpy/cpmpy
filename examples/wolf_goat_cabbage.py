#!/usr/bin/python3
"""
Planning problem in CPMPy

Based on the famous planning puzzle.
https://en.wikipedia.org/wiki/Wolf,_goat_and_cabbage_problem
"""

from cpmpy import *

def run():
    stage = 3
    while stage < 20:
        (model, vars) = model_wgc(stage)
        if model.solve():
            print("Found a solution for " + str(stage) + " stage!")
            for v in vars:
                print(v + ":\t" + str(vars[v].value()))
            break
        else:
            print("No solution for " + str(stage) + " stage")
            stage += 1

def model_wgc(stage):
    wolf_pos = intvar(0, 1, stage)
    cabbage_pos = intvar(0, 1, stage)
    goat_pos = intvar(0, 1, stage)
    boat_pos = intvar(0, 1, stage)

    model = Model(
        # Initial situation
        (boat_pos[0] == 0),
        (wolf_pos[0] == 0),
        (goat_pos[0] == 0),
        (cabbage_pos[0] == 0),

        # Boat keeps moving between shores
        [boat_pos[i] != boat_pos[i-1] for i in range(1,stage)],   

        # Final situation
        (boat_pos[stage-1] == 1),
        (wolf_pos[stage-1] == 1),
        (goat_pos[stage-1] == 1),
        (cabbage_pos[stage-1] == 1),

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