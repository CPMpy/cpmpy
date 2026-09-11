"""
Maximum clique problem in cpmpy.

Problem 074 on CSPlib
https://www.csplib.org/Problems/prob074/

Given a simple undirected graph G = (V, E), a clique is a subset of vertices V
where every two distinct vertices in the subset are adjacent (connected by an edge).
The maximum clique problem is to find a clique of the largest possible size in a
given graph.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_074_maximum_clique/csplib_074_maximum_clique.cpmpy.py)
"""

import cpmpy as cp


def maximum_clique(n=5, adj=None):
    if adj is None:
        adj = [[0, 1, 0, 1, 0],
               [1, 0, 1, 0, 0],
               [0, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 0, 1, 1, 0]]

    c = cp.boolvar(shape=n, name="c")

    model = cp.Model()

    # Constraints
    # The clique property must hold: if two vertices i and j are not connected,
    # they cannot both be in the clique.
    for i in range(n):
        for j in range(i + 1, n):  # Iterate over unique pairs of vertices
            if adj[i][j] == 0:
                # At most one of the two non-adjacent vertices can be in the clique.
                model += c[i] + c[j] <= 1

    # Objective: Maximize the size of the clique.
    # The size is the total number of vertices selected.
    model.maximize(cp.sum(c))

    return model, (c,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    model, (c,) = maximum_clique()

    if model.solve():
        clique = [i for i, v in enumerate(c.value()) if v]
        print(f"Maximum clique size: {len(clique)}")
        print(f"Clique vertices: {clique}")
    else:
        raise ValueError("Model is unsatisfiable")
