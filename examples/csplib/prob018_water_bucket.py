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

import itertools
import cpmpy as cp
import numpy as np

def water_bucket(capacities=[8, 5, 3], initial_state=[8, 0, 0], goal_state=[4, 4, 0], max_steps=20):
    
    total_water = sum(initial_state)

    transitions = get_transitions(capacities, total_water)

    # once we reach the goal state, we can continue as is
    transitions.append(tuple(goal_state) + tuple(goal_state))
    transitions = sorted(list(set(transitions)))

    model = cp.Model()
    
    buckets = []
    for b, cap in enumerate(capacities):
        buckets.append(cp.intvar(0, cap, name=f"B{b}", shape=max_steps).T)        
    sequence = cp.cpm_array(buckets).T

    # Sequence must start with initial state
    model += cp.all(sequence[0] == initial_state)

    # Transition logic
    goal_states = []
    for t in range(max_steps - 1):
        
        is_goal = cp.all(sequence[t] == goal_state)
        goal_states.append(is_goal)
        model += is_goal.implies(cp.all(sequence[t+1] == sequence[t])) # no more pouring once the goal is reached

        # ensure valid transition at step t
        model += cp.Table(sequence[t].tolist() + sequence[t+1].tolist(), transitions)

    # minimize the number of non-goal states
    model.minimize(max_steps - cp.sum(goal_states))

    return model, (sequence,)


def get_transitions(capacities, total_water):
    """define valid transitions, brute force enumeration"""
    transitions = []
    cap_ranges = [range(cap + 1) for cap in capacities]
    for contents in itertools.product(*cap_ranges):
        if sum(contents) != total_water:
            continue # invalid state

        # all possible pourings, from one bucket to another
        for b_from in range(3):
            for b_to in range(3):
                if b_from == b_to:
                    continue # cannot pour to and from the same bucket

                pour_amount = min(contents[b_from], capacities[b_to] - contents[b_to])
                if pour_amount > 0:
                    new_contents = list(contents)
                    new_contents[b_from] -= pour_amount
                    new_contents[b_to] += pour_amount
                    transitions.append(tuple(contents) + tuple(new_contents))
    
    return transitions

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-max_steps", type=int, default=20, help="Maximum number of steps")

    max_steps = parser.parse_args().max_steps

    model, (sequence,) = water_bucket(max_steps=max_steps)

    # print(model)

    if model.solve():
        cost = model.objective_value()
        sequence = sequence.value()
        print(f"Minimum transfers: {cost-1}")
        print("Sequence of states:")
        for t in range(max_steps):
            state = sequence[t]
            print(f"  Step {t}: {state.tolist()}")
            if (state == sequence[t+1]).all(): 
                break # we reached the goal state
    else:
        raise ValueError("Model is unsatisfiable")
