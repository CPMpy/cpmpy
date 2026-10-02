#!/usr/bin/env python
"""
Capacitated VRP with Large Neighborhood Search.

Successor + Circuit formulation. A greedy routing is the starting point, then
LNS removes *related* customer visits (Shaw, CP 1998, Fig. 1) and re-inserts
them with a CP solver.

Usage:
    python examples/advanced/vrp_lns.py
"""
import numpy as np
import cpmpy as cp
from cpmpy.tools.local_search import large_neighborhood_search

np.random.seed(0)

def get_vrp_model(n_vehicle, capacity, dist, demand, depot_dist):
    """Circuit model with dummy depot copies (one per vehicle)
    """
    n = len(demand)
    n_nodes = n + n_vehicle
    demand = np.concatenate([demand, np.zeros(n_vehicle, dtype=int)])
    dist_ext = np.zeros((n_nodes, n_nodes), dtype=int)
    dist_ext[:n, :n] = dist
    dist_ext[:n, n:] = np.asarray(depot_dist)[:, None]
    dist_ext[n:, :n] = np.asarray(depot_dist)[None, :]

    succ = cp.intvar(0, n_nodes - 1, shape=n_nodes, name="succ")
    load = cp.intvar(0, capacity, shape=n_nodes, name="load")
    dist_cpm = cp.cpm_array(dist_ext)
    demand_cpm = cp.cpm_array(demand)

    model = cp.Model(cp.Circuit(succ))
    model += [load[n + k] == 0 for k in range(n_vehicle)]
    for i in range(n_nodes):
        model += (succ[i] < n).implies(load[succ[i]] == load[i] + demand_cpm[succ[i]])

    # multi-objective: minimize traveled distance and number of vehicles used
    total_distance = cp.sum(dist_cpm[i, succ[i]] for i in range(n_nodes))
    model.minimize(total_distance)

    return model, succ, load


def get_destroy_function(succ, dist, load, num_to_remove, D):
    """
    Destroy arcs that are served by different vehicles and are closeby each other.
    Loosely based on 
        Shaw P. Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems. in CP 1998.
    """

    n = dist.shape[0]  # customers
    n_nodes = len(succ)

    def destroy():
        inplan = list(range(n))
        v = int(np.random.randint(n))
        inplan.remove(v)
        removed = [v]
        while len(removed) < num_to_remove:
            v = removed[int(np.random.randint(len(removed)))]
            inplan.sort(key=lambda j: dist[v, j])  # closest = most related
            chosen = inplan.pop(min(int(len(inplan) * np.random.random() ** D), len(inplan) - 1))
            removed.append(chosen)

        sv = [succ[i].value() for i in range(n_nodes)]
        for c in removed:
            pred = next(i for i in range(n_nodes) if sv[i] == c)
            succ[c].clear()
            succ[pred].clear()
        load.clear()

    return destroy

if __name__ == "__main__":
    n, capacity = 50, 30
    depot = np.array([50, 50])
    customers = np.random.randint(0, 100, size=(n * 2)).reshape(n, 2)
    demand = np.random.randint(1, 10, size=n)
    n_vehicle = int(np.ceil(demand.sum() / capacity)) + 2

    dist = np.sqrt(((customers[:, None] - customers) ** 2).sum(-1)).astype(int)
    np.fill_diagonal(dist, 0)
    depot_dist = np.sqrt(((customers - depot) ** 2).sum(-1)).astype(int)

    model, succ, load = get_vrp_model(n_vehicle, capacity, dist, demand, depot_dist)

    sat_model = model.copy()
    sat_model.objective_ = None
    assert sat_model.solve(solver="ortools", num_workers=1)

    print("Cost of initial solution:", int(model.objective_value()))

    destroy_operator = get_destroy_function(succ, dist, load, num_to_remove=int(n * 0.2), D=0.5)

    large_neighborhood_search(model, 
                              destroy_operator, 
                              total_time_limit=10,
                              solver = "ortools",
                              incremental = False, # ortools is not truely incremental anyway
                              num_workers=1, 
                              time_limit=1)
    
    print("LNS cost:", int(model.objective_value()))
