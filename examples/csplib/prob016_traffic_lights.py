"""
Traffic lights problem in cpmpy.

Problem 016 on CSPlib
https://www.csplib.org/Problems/prob016/

Imagine a four-way traffic junction with eight traffic lights. Four lights
(V1 to V4) are for vehicles, and four (P1 to P4) are for pedestrians. The lights
have different states (e.g., red, green). The problem parameters describe
constraints on which combinations of light states are safe.

Vehicle light states: 0=red, 1=red-yellow, 2=green, 3=yellow
Pedestrian light states: 0=red, 1=green

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_016_traffic_lights/csplib_016_traffic_lights.cpmpy.py)
"""

from cpmpy import *


def traffic_lights():
    V1 = intvar(0, 3, name="V1")
    V2 = intvar(0, 3, name="V2")
    V3 = intvar(0, 3, name="V3")
    V4 = intvar(0, 3, name="V4")
    P1 = intvar(0, 1, name="P1")
    P2 = intvar(0, 1, name="P2")
    P3 = intvar(0, 1, name="P3")
    P4 = intvar(0, 1, name="P4")

    vehicle_lights = [V1, V2, V3, V4]
    pedestrian_lights = [P1, P2, P3, P4]
    lights = cpm_array([V1, V2, V3, V4, P1, P2, P3, P4])

    # Allowed combinations for (V_i, P_i, V_{i+1}, P_{i+1})
    allowed_tuples = [[0, 0, 2, 1], [1, 0, 3, 0], [2, 1, 0, 0], [3, 0, 1, 0]]

    model = Model()

    for i in range(4):
        model += Table([vehicle_lights[i], pedestrian_lights[i],
                        vehicle_lights[(i + 1) % 4], pedestrian_lights[(i + 1) % 4]],
                       allowed_tuples)

    return model, (lights,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    model, (lights,) = traffic_lights()

    if model.solve():
        state_names = {0: "red", 1: "red-yellow", 2: "green", 3: "yellow"}
        vals = lights.value()
        print("Vehicle lights:", [(f"V{i+1}", state_names[v]) for i, v in enumerate(vals[:4])])
        print("Pedestrian lights:", [(f"P{i+1}", state_names.get(v, "green" if v == 1 else "red")) for i, v in enumerate(vals[4:])])
    else:
        raise ValueError("Model is unsatisfiable")
