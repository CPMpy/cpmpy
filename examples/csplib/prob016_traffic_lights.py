"""
Traffic lights problem in cpmpy.

Problem 016 on CSPlib
https://www.csplib.org/Problems/prob016/

Imagine a four-way traffic junction with eight traffic lights. Four lights
(V1 to V4) are for vehicles, and four (P1 to P4) are for pedestrians. The lights
have different states (e.g., red, green). The problem parameters describe
constraints on which combinations of light states are safe.

Vehicle light states: 0=red, 1=red-yellow, 2=green, 3=yellow
Pedestrian light states: 0=red, 2=green

Junction arms, numbered clockwise (each has vehicle and pedestrian lights):
             V1/P1
               |
    V4/P4 -----+----- V2/P2
               |
             V3/P3

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_016_traffic_lights/csplib_016_traffic_lights.cpmpy.py)
"""

import cpmpy as cp


RED, RED_YELLOW, GREEN, YELLOW = range(4)


def traffic_lights():
    vehicle_lights = cp.intvar(RED, YELLOW, shape=4, name=tuple(f"V{i}" for i in range(1, 5)))
    pedestrian_lights = cp.intvar(RED, GREEN, shape=4, name=tuple(f"P{i}" for i in range(1, 5)))

    # Allowed combinations for (V_i, P_i, V_{i+1}, P_{i+1})
    allowed_tuples = [
        [RED, RED, GREEN, GREEN],
        [RED_YELLOW, RED, YELLOW, RED],
        [GREEN, GREEN, RED, RED],
        [YELLOW, RED, RED_YELLOW, RED],
    ]

    model = cp.Model()

    for i in range(4):
        lights_i = [
            vehicle_lights[i],
            pedestrian_lights[i],
            vehicle_lights[(i + 1) % 4],
            pedestrian_lights[(i + 1) % 4],
        ]

        model += cp.Table(lights_i, allowed_tuples)

    return model, (vehicle_lights, pedestrian_lights)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    model, (vehicle_lights, pedestrian_lights) = traffic_lights()

    if model.solve():
        vehicle_state_names = {
            RED: "red",
            RED_YELLOW: "red-yellow",
            GREEN: "green",
            YELLOW: "yellow",
        }
        pedestrian_state_names = {
            RED: "red",
            GREEN: "green",
        }

        print(
            "Vehicle lights:",
            [(f"V{i + 1}", vehicle_state_names[v]) for i, v in enumerate(vehicle_lights.value())],
        )
        print(
            "Pedestrian lights:",
            [(f"P{i + 1}", pedestrian_state_names[v]) for i, v in enumerate(pedestrian_lights.value())],
        )
    else:
        raise ValueError("Model is unsatisfiable")