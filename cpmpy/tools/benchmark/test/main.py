import subprocess
import os
import sys
from concurrent.futures import ThreadPoolExecutor

def main():
    from cpmpy.tools.dataset.problem.psplib import PSPLibDataset
    from cpmpy.tools.rcpsp import read_rcpsp

    # dataset = XCSP3Dataset(root="./data", year=2025, track="CSP25", download=True)
    # dataset = OPBDataset(root="./data", year=2024, track="DEC-LIN", download=True)
    # dataset = JSPLibDataset(root="./data", download=True)
    dataset = PSPLibDataset(root="./data", download=True)

    time_limit = 10
    workers = 10

    with ThreadPoolExecutor(max_workers=workers) as executor:   
        futures = [executor.submit(run_instance, instance, metadata, time_limit) for instance, metadata in dataset]
        for future in futures:
            future.result()

def run_instance(instance, metadata, time_limit):
        this_file_path = os.path.dirname(os.path.abspath(__file__))
        this_python = sys.executable
        cmd_runexec = [
            "runexec", 
            "--walltimelimit", f"{time_limit}s", 
            "--no-container",
            "--"
        ]
        cmd = cmd_runexec + [
            this_python, os.path.join(this_file_path, "runner.py"), 
            instance, 
            "--solver", "ortools", 
            "--time_limit", str(time_limit), 
            "--seed", "1234567890", 
            "--intermediate", 
            "--cores", "1"
        ]
        print(" ".join(cmd))
        subprocess.run(cmd)
        

if __name__ == "__main__":
    main()