"""
Rehearsal problem in cpmpy.

Problem 039 on CSPlib
https://www.csplib.org/Problems/prob039/

A concert is to consist of a number of pieces of music of different durations,
each involving a different combination of the members of the orchestra. Players
can arrive at rehearsals immediately before the first piece in which they are
involved and depart immediately after the last piece in which they are involved.
The problem is to devise an order in which the pieces can be rehearsed so as to
minimize the total time that players are waiting to play.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_039_rehearsal/csplib_039_rehearsal.cpmpy.py)
"""

from cpmpy import *
import numpy as np


def rehearsal(num_pieces=9, num_players=5, duration=None, rehearsal_matrix=None):
    if duration is None:
        duration = [2, 4, 1, 3, 3, 2, 5, 7, 6]
    if rehearsal_matrix is None:
        rehearsal_matrix = [[1, 1, 0, 1, 0, 1, 1, 0, 1],
                            [1, 1, 0, 1, 1, 1, 0, 1, 0],
                            [1, 1, 0, 0, 0, 0, 1, 1, 0],
                            [1, 0, 0, 0, 1, 1, 0, 0, 1],
                            [0, 0, 1, 0, 1, 1, 1, 1, 0]]

    duration = cpm_array(duration)
    rehearsal = cpm_array(rehearsal_matrix)

    model = Model()

    # rehearsal_order[i] is the piece rehearsed in the i-th slot.
    rehearsal_order = intvar(0, num_pieces - 1, shape=num_pieces, name="rehearsal_order")
    # arrival[p] is the first slot where player p is present.
    arrival = intvar(0, num_pieces - 1, shape=num_players, name="arrival")
    # departure[p] is the last slot where player p is present.
    departure = intvar(0, num_pieces - 1, shape=num_players, name="departure")

    model += AllDifferent(rehearsal_order)

    # Link arrival and departure times to the rehearsal schedule.
    # A player must be present for all pieces they play in.
    for p in range(num_players):
        for i in range(num_pieces):
            # is_playing is an expression that is true if player p plays in the piece at slot i.
            is_playing = (rehearsal[p, rehearsal_order[i]] == 1)
            # If a player is playing, they must be present (between their arrival and departure slot).
            model += is_playing.implies((arrival[p] <= i) & (i <= departure[p]))

    # Objective: Minimize total waiting time
    # Waiting time for a player in a slot is the duration of the piece in that slot
    # if the player is present but not playing.
    waiting_times = []
    for p in range(num_players):
        for i in range(num_pieces):
            is_present = (arrival[p] <= i) & (i <= departure[p])
            is_not_playing = (rehearsal[p, rehearsal_order[i]] == 0)
            is_waiting = is_present & is_not_playing
            # Add the duration of the piece if the player is waiting.
            waiting_times.append(duration[rehearsal_order[i]] * is_waiting)

    model.minimize(sum(waiting_times))

    return model, (rehearsal_order,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    model, (rehearsal_order,) = rehearsal()

    if model.solve():
        print(f"Minimum total waiting time: {int(model.objective_value())}")
        print(f"Optimal rehearsal order: {rehearsal_order.value()}")
    else:
        raise ValueError("Model is unsatisfiable")
