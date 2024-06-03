from executable.test.conftest import INSTANCES_DIR, get_all_instances
from executable.main import run, Args
import json, sys, os, pathlib, tqdm

solver = "ortools"
subsolver = None
time_limit = 60*30
mem_limit = 10_000_000_000

def run_instance(instance_name, instance_path, solver, subsolver=None, outfile=None):
    args = Args(
        benchpath=instance_path,
        seed=42,
        solver=solver,
        subsolver=subsolver,
        profiler=True,
        time_limit=time_limit,
        mem_limit=mem_limit,
        intermediate=True
        )
    f = open(outfile, "w")
    with f as sys.stdout:
        run(args)
    f.close()
    sys.stdout = sys.__stdout__

def run_and_log_instance(instance_name, instance_path, solver, subsolver=None, outfile=None):
    res = run_instance(instance_name, instance_path, solver, subsolver, outfile)
    # pathlib.Path(os.path.join(pathlib.Path(__file__).parent.resolve(), "perf_stats")).mkdir(parents=True, exist_ok=True)
    # path = os.path.join(pathlib.Path(__file__).parent.resolve(), "perf_stats", instance_name)


instances_per_type = get_all_instances(INSTANCES_DIR)


# for type, instances in instances_per_type.items():
import itertools
for instance_type in ["CSP", "COP"]:
    pathlib.Path(os.path.join(pathlib.Path(__file__).parent.resolve(), "out", solver, instance_type,)).mkdir(parents=True, exist_ok=True)
    instances = instances_per_type[instance_type]
    # instances = dict(list(instances.items())[45:])
    for instance_name, instance_path in tqdm.tqdm(instances.items()):
        outfile = os.path.join(pathlib.Path(__file__).parent.resolve(), "out", solver, instance_type, instance_name + ".txt")
        run_and_log_instance(instance_name, instance_path, solver=solver, subsolver=subsolver, outfile=outfile)
        


# from perf_timer import PerfContext, TimerContext
# import time

# with PerfContext() as context:
#     with TimerContext("test"):
#         time.sleep(1)

# print(context.measurements)



