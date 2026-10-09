"""
Traveling Salesman Problem (TSP) in CPMpy.

Taken from the Google OR-Tools example:
https://developers.google.com/optimization/routing/tsp

Given a set of locations, find a closed path of minimal length that visits
each location exactly once. Modeled with successor variables and a Circuit
constraint. See also vrp.py for a multi-vehicle extension with dummy depots.
"""
import cpmpy as cp
import numpy as np


def compute_euclidean_distance_matrix(locations):
    """Computes distances between all points."""
    locs = np.asarray(locations)
    return np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=-1).astype(int)


def tsp(locations):
    """
    Build a Circuit-based TSP model over `locations`.

    Returns (model, (succ,)) with travel distance minimized.
    """
    n = len(locations)
    distance_matrix = cp.cpm_array(compute_euclidean_distance_matrix(locations))

    # succ[i]=j means j is visited immediately after i
    succ = cp.intvar(0, n - 1, shape=n)
    model = cp.Model(cp.Circuit(succ))
    model.minimize(cp.sum(distance_matrix[i, succ[i]] for i in range(n)))
    return model, (succ,)


if __name__ == "__main__":
    # data
    locations = [
        (288, 149), (288, 129), (270, 133), (256, 141), (256, 163), (246, 157),
        (236, 169), (228, 169), (228, 148), (220, 164), (212, 172), (204, 159),
    ]

    model, (succ,) = tsp(locations)

    if model.solve():
        print(model.status())
        print("Total Cost of solution", int(model.objective_value()))

        # display tour starting from location 0
        s = succ.value()
        tour = [0]
        i = 0
        while s[i] != 0:
            i = s[i]
            tour.append(i)
        tour.append(0)
        print(" --> ".join(map(str, tour)))
    else:
        raise ValueError("Model is unsatisfiable")
