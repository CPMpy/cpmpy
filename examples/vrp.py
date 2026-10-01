"""
Vehicle Routing Problem (VRP) in CPMpy.

The goal is to find a set of routes of minimal total length for a fleet of
vehicles visiting a set of locations, starting at a cental depot.

Modeled with a Circuit constraint and dummy depots (one per vehicle).
Following the circuit, each edge between two depot nodes terminates a vehicle's route.
"""
import cpmpy as cp
import numpy as np


def compute_euclidean_distance_matrix(locations):
    """Computes distances between all points."""
    locs = np.asarray(locations)
    return np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=-1).astype(int)


def vrp(locations, depot, n_vehicles, demand=None, capacity=None):
    """
    Build a Circuit-based VRP by prepending `n_vehicles` copies of `depot`
    to `locations` (customers). Identical depot coords give distance 0, so
    unused vehicles can stay at the depot for free.

    Optional `demand` (one value per customer) and `capacity` add capacity
    constraints.

    Returns (model, (succ,)) or (model, (succ, load)) when capacity is set.
    """
    locs = [depot] * n_vehicles + list(locations)
    n = len(locs)
    depots = list(range(n_vehicles))
    customers = list(range(n_vehicles, n))

    distance_matrix = cp.cpm_array(compute_euclidean_distance_matrix(locs))

    # succ[i]=j means j is visited immediately after i
    succ = cp.intvar(0, n - 1, shape=n)
    model = cp.Model(cp.Circuit(succ))
    model.minimize(cp.sum(distance_matrix[i, succ[i]] for i in range(n)))

    if demand is not None and capacity is not None:
        demand_n = [0] * n_vehicles + list(demand)
        load = cp.intvar(0, capacity, shape=n)
        model += [load[d] == 0 for d in depots]  # leave depot empty
        model += [(succ[i] == j).implies(load[j] == load[i] + demand_n[j])
                  for i in range(n) for j in customers]
        return model, (succ, load)

    return model, (succ,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n_vehicles", type=int, default=3, help="Number of vehicles")
    parser.add_argument("-capacity", type=int, default=20, help="Vehicle capacity")
    args = parser.parse_args()

    # data
    depot = (288, 149)
    locations = [  # customers
        (288, 129), (270, 133), (256, 141), (256, 163),
        (236, 169), (228, 169), (228, 148), (220, 164),
    ]
    demand = [3, 4, 8, 8, 10, 5, 3, 9]  # per customer

    model, (succ, load) = vrp(locations, depot, n_vehicles=args.n_vehicles,
                              demand=demand, capacity=args.capacity)

    # number of vehicles used (depot-to-depot arc = unused)
    depots = list(range(args.n_vehicles))
    num_used = args.n_vehicles - cp.sum(cp.InDomain(succ[d], depots) for d in depots)

    # distances for printing (same expansion as in vrp)
    D = compute_euclidean_distance_matrix([depot] * args.n_vehicles + locations)

    if model.solve():
        print(model.status())
        print("Total Cost of solution", int(model.objective_value()))
        print("Vehicles used:", num_used.value())

        # display routes (customer indices 1..n_cust; depot as 0)
        s = succ.value()
        vehicle = 0
        for d in depots:
            if s[d] in depots:
                continue  # unused
            # display routes (customer indices 1..n_cust; depot as 0)
            stops = ["0 (load 0)"]
            dist = 0
            i = d
            while s[i] not in depots:
                j = s[i]
                dist += D[i, j]
                cust = j - args.n_vehicles + 1
                stops.append(f"{cust} (load {load[j].value()})")
                i = j
            dist += D[i, s[i]]
            stops.append("0")
            print(f"Vehicle {vehicle}: {' --> '.join(stops)}  (km={dist})")
            vehicle += 1
    else:
        raise ValueError("Model is unsatisfiable")
