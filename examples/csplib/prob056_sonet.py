"""
SONET ring loading problem in cpmpy.

Problem 056 on CSPlib
https://www.csplib.org/Problems/prob056/

In the SONET problem, we are given a set of nodes and the traffic demand for each
pair of nodes. The nodes are connected by a set of rings. A node is installed on a
ring using an add-drop multiplexer (ADM). For two nodes to communicate, they must be
on at least one common ring. Each ring has a capacity, limiting the number of nodes
it can host. The objective is to satisfy all communication demands while minimizing
the total number of ADMs used.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_056_sonet/csplib_056_sonet.cpmpy.py)
"""

import cpmpy as cp


def sonet(r=4, n=10, demand=None, capacity_nodes=None):
    if demand is None:
        demand = [[0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                  [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                  [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                  [1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
                  [1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]]
    if capacity_nodes is None:
        capacity_nodes = [3, 4, 5, 6]

    model = cp.Model()

    ring_config = cp.boolvar(shape=(r, n), name="ring_config")

    # Demand satisfaction
    for i in range(n):
        for j in range(i + 1, n):
            if demand[i][j] > 0:
                on_common_ring = ring_config[:, i] & ring_config[:, j]
                model += cp.sum(on_common_ring) >= 1

    # Ring capacity
    for k in range(r):
        model += cp.sum(ring_config[k, :]) <= capacity_nodes[k]

    model.minimize(cp.sum(ring_config))

    return model, (ring_config,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    model, (ring_config,) = sonet()

    if model.solve():
        print(f"Total ADMs used: {int(model.objective_value())}")
        for k, row in enumerate(ring_config.value()):
            nodes = [i for i, v in enumerate(row) if v]
            print(f"Ring {k}: nodes {nodes}")
    else:
        raise ValueError("Model is unsatisfiable")
