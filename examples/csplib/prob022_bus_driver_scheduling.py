"""
Bus driver scheduling problem in cpmpy.

Problem 022 on CSPlib
https://www.csplib.org/Problems/prob022/

This problem requires finding an optimal schedule for bus drivers. Given a set of
tasks (pieces of work) to cover and a large set of possible shifts, where each shift
covers a subset of the tasks, we must select a subset of shifts that covers each
piece of work exactly once.

The goal is to cover all tasks while minimizing the total number of shifts used.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_022_bus_driver_scheduling/csplib_022_bus_driver_scheduling.cpmpy.py)
"""

import cpmpy as cp


def bus_driver_scheduling(num_tasks=12, num_shifts=14, shifts=None):
    if shifts is None:
        # shifts: a list of available shifts; each inner list contains the
        # indices of the pieces of work (tasks) that that shift covers. Tasks
        # are numbered 0..(num_tasks-1). The model will choose a subset of these
        # shifts so that every task is covered exactly once.
        shifts = [
            [0, 1, 2],
            [0, 1, 2, 3],
            [2, 3, 4, 5],
            [4, 5],
            [6, 7],
            [6, 7, 8],
            [8, 9],
            [8, 9, 10],
            [10, 11],
            [0, 4, 8],
            [0, 5, 10],
            [1, 6, 11],
            [2, 7, 9],
            [3, 6, 8]
        ]

    model = cp.Model()

    x = cp.boolvar(shape=num_shifts, name="x")

    for t in range(num_tasks):
        covering_shifts = [x[i] for i in range(num_shifts) if t in shifts[i]]
        model += (cp.sum(covering_shifts) == 1)

    model.minimize(cp.sum(x))

    return model, (x,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    model, (x,) = bus_driver_scheduling()

    if model.solve():
        selected = [i for i, v in enumerate(x.value()) if v]
        print(f"Number of shifts used: {len(selected)}")
        print(f"Selected shifts: {selected}")
    else:
        raise ValueError("Model is unsatisfiable")
