import os, sys
import glob
import time
import pytest
import pathlib
import subprocess
import signal
from contextlib import contextmanager
from pathlib import Path

from .conftest import TEST_OUTPUT_DIR, JAR


def run_instance(instance_name: str, instance_location: os.PathLike, solver: str, subsolver:str, verbose: bool = True, fresh: bool = False, time_limit:int=None, memory_limit:int=None, intermediate:bool=False, competition:bool=False):
    """
        Prepares the environment, runs the executable and checks the solution with SolutionChecker.
        Pipes all executable outputs to a file and adds additional, usefull data as comments 
        (e.g. the result of the SolutionChecker).
    """

    # Based on the location of the problem instance, detect the type of problem
    if "MiniCOP" in instance_location: instance_type = "MiniCOP"
    elif "MiniCSP" in instance_location: instance_type = "MiniCSP"
    elif "COP" in instance_location: instance_type = "COP"
    elif "CSP" in instance_location: instance_type = "CSP"
    
    # Configure files to pipe output to
    out_dir = os.path.join(os.getcwd(), TEST_OUTPUT_DIR, instance_type, solver + (f"-{subsolver}" if subsolver is not None else ""))
    out_file = os.path.join(out_dir, instance_name + ".txt")
    
    # Make directories if non-existant
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Check if model solving can be skipped (piped output already exist)
    already_tested = os.path.isfile(out_file) and (os.stat(out_file).st_size != 0)

    # Decide if we want to solve again
    if (not already_tested) or fresh:

        f = open(out_file, "w") # open file to pipe to

        # Run competition executable
        start = time.time()
        
        # If in competition model, call the executable through its cli instead of directly accessing its python interface
        if competition:

            # https://alexandra-zaharia.github.io/posts/kill-subprocess-and-its-children-on-timeout-python/
            try:
                # Create command
                cmd = ["python", os.path.join(pathlib.Path(__file__).parent.resolve(), "..", "main.py"), instance_location, f"--solver={solver}", f"--subsolver={subsolver}"]
                if time_limit is not None: cmd += [f"--time-limit={time_limit}"]
                if memory_limit is not None: cmd += [f"--mem-limit={memory_limit}"]
                if intermediate: cmd += [f"--intermediate"]

                # Run command
                p = subprocess.Popen(cmd, start_new_session=True, stdout=f)

                # Wait for result
                if time_limit is not None:
                    p.wait(timeout=time_limit)
                else:
                    p.wait()

            except subprocess.TimeoutExpired:
                # If executable has not returned before timeout, send SIGTERM
                print(f'Timeout for {solver}' + (f':{subsolver}' if subsolver is not None else "") + f':{instance_name} ({time_limit}s) expired', file=sys.stderr)
                print('Terminating the whole process group...', file=sys.stderr)
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)

        else:
            # Reset command line arguments, only then import (otherwise breaks pycsp3)
            sys.argv = [] 
            sys.path.insert(1, os.path.join(pathlib.Path(__file__).parent.resolve(), ".."))
            import main

            # Capture prints to file
            with f as sys.stdout:
                args = main.Args(
                    benchname=instance_location,
                    time_limit=time_limit,
                    mem_limit=memory_limit,
                    solver=solver,
                    subsolver=subsolver,
                    intermediate=intermediate
                )
                main.run(args)
            f.close()

            # Reset STDOUT
            sys.stdout = sys.__stdout__

        cpm_time = time.time() - start
    
    # Run SolutionChecker
    start = time.time()
    test_res_str = subprocess.check_output(["java", "-jar", JAR, instance_location, out_file, "-cm"], stderr=subprocess.STDOUT if verbose else None).decode("utf8")
    checker_time = time.time() - start

    if verbose: print(test_res_str)

    # Add SolutionChecker results in comments
    if (not already_tested) or fresh:
        f = open(out_file, "a")
        f.write("c" + chr(32) + test_res_str + "\n")
        f.write("c" + chr(32) + f"cpmpy time: {cpm_time}" + "\n")
        f.write("c" + chr(32) + f"validation time: {checker_time}" + "\n")
        f.write("c" + chr(32) + f"elapsed time: {cpm_time + checker_time}" + "\n")
        f.close()
    
    return test_res_str

# Pytest test function
# @pytest.mark.repeat(10)
def test_instance(pytestconfig, instance, solver, subsolver, fresh, time_limit, memory_limit, intermediate, verbose: bool = True, test=True, competition=False):
    """
        This is the actual function which gets called by pytest. All inputs are defined in `conftest.py`.
    """
    
    instance_name, instance_location = instance

    if verbose: print(f"Running instance {instance_name} on {solver}" + (f":{subsolver}" if subsolver is not None else ""))

    test_res_str = run_instance(instance_name, instance_location, solver=solver, subsolver=subsolver, verbose=verbose, fresh=fresh, time_limit=time_limit, memory_limit=memory_limit, intermediate=intermediate, competition=competition)

    # Assert that the result must be correct
    if test: assert(test_res_str[:2] == "OK")

if __name__ == "__main__":
    test_instance(None, "prof/test_instance\[instance5-ortools-True-None-None-True\].prof", "ortools", False, None, None, False)

