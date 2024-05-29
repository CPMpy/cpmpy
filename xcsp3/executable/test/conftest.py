import os, sys
import glob
import time
import pytest
import pathlib
import subprocess
import signal
from contextlib import contextmanager
from pathlib import Path

TEST_OUTPUT_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "..", "out") # where should the output of the executable be piped to (to later check its correctness using SolutionChecker)
JAR = os.path.join(pathlib.Path(__file__).parent.resolve(), "..", "..", "xcsp3-solutionChecker-2.5.jar") # jar file of the SolutionChecker
SUPPORTED_SOLVERS = ["choco", "ortools"] # Solvers supported by this testsuite (and the executable)
INSTANCES_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "..", "..", "instancesXCSP22")

def get_all_instances(instance_dir: os.PathLike):
    """
        Collects all problem instances in a dict:
        {
            <instance_type>: {
                <instance_name>: <instance_path> 
            }
        }
    """
    # Dicts to collect type specific instances
    COP = {}
    CSP = {}
    MiniCSP = {}
    MiniCOP = {}

    # Walk though filesystem and collect instances
    for root, dirs, files in os.walk(instance_dir):
        for file in files:
            if file.endswith(".xml"):
                name = file[:-4]
                location = os.path.join(root, file)

                if "MiniCOP" in location:
                    MiniCOP[name] = location
                elif "MiniCSP" in location:
                    MiniCSP[name] = location
                elif "COP" in location:
                    COP[name] = location
                elif "CSP" in location:
                    CSP[name] = location
                
    return {
        "COP": COP,
        "CSP": CSP,
        "MiniCOP": MiniCOP,
        "MiniCSP": MiniCSP
    }

# Instance type filters
main_instance_types = ["COP", "CSP"]
mini_instance_types = ["MiniCSP", "MiniCOP"]
cop_instance_types = ["COP", "MiniCOP"]
csp_instance_types = ["CSP", "MiniCSP"]

def instances(type) -> list:
    """
        Filters and aggregates problem instances based on the provided `type`.
    """
    if type is None:  
        instances = {instance[0]:instance[1] for instance_type in main_instance_types + mini_instance_types for instance in get_all_instances(INSTANCES_DIR)[instance_type].items()}
    elif type == "mini":
        instances = {instance[0]:instance[1] for instance_type in mini_instance_types for instance in get_all_instances(INSTANCES_DIR)[instance_type].items()}
    elif type == "main":
        instances = {instance[0]:instance[1] for instance_type in main_instance_types for instance in get_all_instances(INSTANCES_DIR)[instance_type].items()}
    elif type == "cop":
        instances = {instance[0]:instance[1] for instance_type in cop_instance_types for instance in get_all_instances(INSTANCES_DIR)[instance_type].items()}
    elif type == "csp":
        instances = {instance[0]:instance[1] for instance_type in csp_instance_types for instance in get_all_instances(INSTANCES_DIR)[instance_type].items()}
    elif type == "mini-csp":
        instances = get_all_instances(INSTANCES_DIR)["MiniCSP"]
    elif type == "mini-cop":
        instances = get_all_instances(INSTANCES_DIR)["MiniCOP"]
    elif type == "main-csp":
        instances = get_all_instances(INSTANCES_DIR)["CSP"]
    elif type == "main-cop":
        instances = get_all_instances(INSTANCES_DIR)["COP"]
    else:
        raise()
                
    # return instances.keys(), instances.values()
    return list(instances.items())

def pytest_addoption(parser):
    """
        Defines the cli arguments for pytest.
    """
    parser.addoption("--solver", action="store", default=None)
    parser.addoption("--subsolver", action="store", default=None)
    parser.addoption("--all", action="store_true", help="run all combinations")
    parser.addoption("--fresh", action="store_true", help="reset all stored results")
    parser.addoption("--time_limit", action="store", default=None, type=int)
    parser.addoption("--memory_limit", action="store", default=None, type=int)
    parser.addoption("--type", action="store", default=None)
    parser.addoption("--intermediate", action="store_true")
    parser.addoption("--competition", action="store_true")
    parser.addoption("--profiler", action="store_true")
    parser.addoption("--only_transform", action="store_true")


def pytest_generate_tests(metafunc):
    """
        Passes cli arguments and test instances to test suite functions.
    """

    # Get the test instances based on the provided filter
    instance = instances(type=metafunc.config.getoption("type"))

    # The test instances to solve
    if "instance" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            pass
        else:
            pass
        metafunc.parametrize("instance", instance)

    # The solver to use
    if "solver" in metafunc.fixturenames:
        if metafunc.config.getoption("solver"):
            solver = [metafunc.config.getoption("solver")]
        else:
            solver = SUPPORTED_SOLVERS
        metafunc.parametrize("solver", solver)

    # The subsolver to use
    if "subsolver" in metafunc.fixturenames:
        if metafunc.config.getoption("subsolver"):
            subsolver = [metafunc.config.getoption("subsolver")]
        else:
            subsolver = [None]
        metafunc.parametrize("subsolver", subsolver)

    # If the saved solve results should be reset
    if "fresh" in metafunc.fixturenames:
        if metafunc.config.getoption("fresh"):
            fresh = True
        else:
            fresh = False
        metafunc.parametrize("fresh", [fresh])

    # Time limit before SIGTERM
    if "time_limit" in metafunc.fixturenames:
        metafunc.parametrize("time_limit", [metafunc.config.getoption("time_limit")])

    # Memory limit
    if "memory_limit" in metafunc.fixturenames:
        metafunc.parametrize("memory_limit", [metafunc.config.getoption("memory_limit")])

    # If intermediate solutions should be shown
    if "intermediate" in metafunc.fixturenames:
        metafunc.parametrize("intermediate", [metafunc.config.getoption("intermediate")])

    # If intermediate solutions should be shown
    if "competition" in metafunc.fixturenames:
        metafunc.parametrize("competition", [metafunc.config.getoption("competition")])

    # If the executable should log performance profiles
    if "profiler" in metafunc.fixturenames:
        metafunc.parametrize("profiler", [metafunc.config.getoption("profiler")])

    # Only transform, don't solve
    if "only_transform" in metafunc.fixturenames:
        metafunc.parametrize("only_transform", [metafunc.config.getoption("only_transform")])