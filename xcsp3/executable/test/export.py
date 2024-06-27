import json
import os
import pathlib
from conftest import get_all_instances, INSTANCES_DIR

instance_dir = os.path.join(INSTANCES_DIR, "XCSP23_V2")
all_instances = get_all_instances(INSTANCES_DIR, year="2023")
all_instances_2 = get_all_instances(INSTANCES_DIR, year="2023", relative=True)
for instance_type in all_instances.items():
    for instance in instance_type[1].items():
        with open(instance[1]) as f:
            instance_str = "\n".join(f.readlines())
            if "maximize" in instance_str:
                all_instances_2[instance_type[0]][instance[0]] = (instance[1], "max")
            elif "minimize" in instance_str:
                all_instances_2[instance_type[0]][instance[0]] = (instance[1], "min")

with open(os.path.join(pathlib.Path(__file__).parent.resolve(), 'instances23.json'), 'w') as f:
    json.dump(all_instances_2, f, indent=4)