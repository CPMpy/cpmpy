"""
Warehouse location problem in cpmpy.

Problem 034 on CSPlib
https://www.csplib.org/Problems/prob034/

In the Warehouse Location problem, a company must decide which warehouses to open
from a set of candidate locations to supply a set of stores. Each warehouse has a
fixed maintenance cost and a capacity limiting the number of stores it can supply.
Each store must be supplied by exactly one open warehouse, and there is an
associated supply cost that varies depending on the store and warehouse.

The objective is to minimize the total cost, which is the sum of the maintenance
costs for all opened warehouses and the supply costs for all stores.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_034_warehouse_location/csplib_034_warehouse_location.cpmpy.py)
"""

import cpmpy as cp


DEFAULT_CAPACITY = (1, 4, 2, 1, 3)

DEFAULT_COST_MATRIX = (
    (20, 24, 11, 25, 30),
    (28, 27, 82, 83, 74),
    (74, 97, 71, 96, 70),
    (2, 55, 73, 69, 61),
    (46, 96, 59, 83, 4),
    (42, 22, 29, 67, 59),
    (1, 5, 73, 59, 56),
    (10, 73, 13, 43, 96),
    (93, 35, 63, 85, 46),
    (47, 65, 55, 71, 95),
)

def warehouse_location(
    n_suppliers=5,
    n_stores=10,
    building_cost=30,
    capacity=DEFAULT_CAPACITY,
    cost_matrix=DEFAULT_COST_MATRIX,
):
    """Build a warehouse location model.

    Args:
        n_suppliers: Number of candidate warehouse locations.
        n_stores: Number of stores to supply.
        building_cost: Fixed cost for opening a warehouse.
        capacity: Capacity of each warehouse, length n_suppliers.
        cost_matrix: Supply cost matrix of shape n_stores x n_suppliers.

    Returns:
        (model, vars) where vars contains supplier assignments and open warehouses.
    """
    assert len(capacity) == n_suppliers
    assert len(cost_matrix) == n_stores
    assert all(len(row) == n_suppliers for row in cost_matrix)
    assert sum(capacity) >= n_stores

    model = cp.Model()

    supplier_assignment = cp.intvar(0, n_suppliers - 1, shape=n_stores, name="supplier_assignment")
    open_warehouses = cp.boolvar(shape=n_suppliers, name="open_warehouses")

    # Capacity constraints
    for w in range(n_suppliers):
        model += (cp.Count(supplier_assignment, w) <= capacity[w])

    # Channeling
    for w in range(n_suppliers):
        model += (open_warehouses[w] == (cp.Count(supplier_assignment, w) > 0))

    # Objective
    cost_matrix_cpm = cp.cpm_array(cost_matrix)
    supply_cost = cp.sum([cost_matrix_cpm[s, supplier_assignment[s]] for s in range(n_stores)])
    maintenance_cost = cp.sum(open_warehouses) * building_cost
    total_cost = supply_cost + maintenance_cost
    model.minimize(total_cost)

    return model, (supplier_assignment, open_warehouses)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    model, (supplier_assignment, open_warehouses) = warehouse_location()

    if model.solve():
        print(f"Minimum total cost: {int(model.objective_value())}")
        print(f"Open warehouses: {[int(v) for v in open_warehouses.value()]}")
        print(f"Store assignments: {supplier_assignment.value()}")
    else:
        raise ValueError("Model is unsatisfiable")
