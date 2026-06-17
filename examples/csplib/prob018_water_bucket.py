"""
Water bucket problem in cpmpy.

Problem 018 on CSPlib
https://www.csplib.org/Problems/prob018/

You are given an 8-pint bucket of water and two empty buckets which can contain
5 and 3 pints respectively. You are required to divide the water into two by
pouring water between buckets (that is, to end up with 4 pints in the 8-pint
bucket, and 4 pints in the 5-pint bucket).

Find the minimum number of transfers of water between buckets to reach the goal.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_018_water_bucket/csplib_018_water_bucket.cpmpy.py)
"""

import cpmpy as cp
import numpy as np


def water_bucket(capacities=None, initial_state=None, goal_state=None, max_steps=20):
    if capacities is None:
        capacities = [8, 5, 3]
    if initial_state is None:
        initial_state = [8, 0, 0]
    if goal_state is None:
        goal_state = [4, 4, 0]

    padding_value = -1
    total_water = max(initial_state)

    # Pre-compute all valid state transitions
    all_states = []
    for i in range(capacities[0] + 1):
        for j in range(capacities[1] + 1):
            if 0 <= capacities[0] - i - j <= capacities[2]:
                k = capacities[0] - i - j
                all_states.append((i, j, k))

    transitions = []
    for state in all_states:
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                current_amounts = list(state)
                pour_amount = min(current_amounts[i], capacities[j] - current_amounts[j])
                if pour_amount > 0:
                    next_state = list(current_amounts)
                    next_state[i] -= pour_amount
                    next_state[j] += pour_amount
                    transitions.append(tuple(state) + tuple(next_state))
    transitions = sorted(list(set(transitions)))

    model = cp.Model()

    sequence = cp.intvar(padding_value, total_water, shape=(max_steps, 3), name="states")

    # Sequence must start with initial state
    model += sequence[0, :] == initial_state

    # State properties
    for t in range(max_steps):
        is_padded = (sequence[t, 0] == padding_value)
        model += (~is_padded).implies(sum(sequence[t, :]) == total_water)
        for b in range(3):
            model += (~is_padded).implies((sequence[t, b] >= 0) & (sequence[t, b] <= capacities[b]))

    # Transition logic
    for t in range(max_steps - 1):
        is_padded_t = (sequence[t, 0] == padding_value)
        is_goal_t = (sequence[t, :] == goal_state).all()

        model += is_goal_t.implies((sequence[t + 1, :] == padding_value).all())
        model += is_padded_t.implies((sequence[t + 1, :] == padding_value).all())

        must_transfer = cp.Table(np.hstack([sequence[t, :], sequence[t + 1, :]]), transitions) & cp.any(sequence[t + 1, :] != sequence[t, :])
        model += (~is_goal_t & ~is_padded_t).implies(must_transfer)

    # Goal must be reached
    model += cp.any([(sequence[t, :] == goal_state).all() for t in range(max_steps)])

    cost = cp.sum([sequence[t, 0] != padding_value for t in range(max_steps)]) - 1
    model.minimize(cost)

    return model, (sequence,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-max_steps", type=int, default=20, help="Maximum number of steps")

    max_steps = parser.parse_args().max_steps

    model, (sequence,) = water_bucket(max_steps=max_steps)

    if model.solve():
        costs = model.objective_value()
        print(f"Minimum transfers: {costs}")
        print("Sequence of states:")
        for t in range(max_steps):
            state = sequence[t, :].value()
            if state[0] == -1:
                break
            print(f"  Step {t}: {state.tolist()}")
    else:
        raise ValueError("Model is unsatisfiable")
