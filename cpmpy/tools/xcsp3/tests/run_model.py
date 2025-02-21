from os.path import join

import glob

import argparse
import math
import os
import sys
import traceback
import time
from pathlib import Path
from multiprocessing import Process,Lock, Manager, set_start_method,Pool, cpu_count
from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3

import cpmpy as cp
from cpmpy import SolverLookup
from cpmpy.tools.xcsp3.callbacks import CallbacksCPMPy


def check_positive(value):
    """
    Small helper function used in the argparser for checking if the input values are positive or not
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def run_model(lock, solver, xmodel):
    """
    Runs one XCSP3 instance
    """
    print("\nrunning model"+xmodel, flush=True, end="\n\n")
    parser = ParserXCSP3(xmodel)
    callbacks = CallbacksCPMPy()
    callbacks.force_exit = True
    callbacker = CallbackerXCSP3(parser, callbacks)
    try:
        callbacker.load_instance()
    except Exception as e:
        print("error parsing:")
        print(e)
    cb = callbacker.cb
    s = SolverLookup.get(solver, cb.cpm_model)
    try:
        if s.solve():
            print('sat')
        else:
            print('unsat')
    except Exception as e:
        print('error solving:')
        print(e)

if __name__ == '__main__':
    # get all the available solvers from cpympy
    available_solvers = cp.SolverLookup.solvernames()

    parser = argparse.ArgumentParser(description="A python application to fuzz_test your solver(s)")
    parser.add_argument("-s", "--solver", help="The Solver to use", required=False, type=str, choices=available_solvers,
                        default=available_solvers[0])
    parser.add_argument("-m", "--models", help="The path to load the models", required=False, type=str,
                        default="models")
    parser.add_argument("-p", "--amount-of-processes",
                        help="The amount of processes that will be used to run the tests", required=False,
                        default=cpu_count() - 1, type=check_positive)  # the -1 is for the main process
    args = parser.parse_args()
    start_time = time.time()

    print("\nUsing solver '"+args.solver+"' with models in '"+args.models+"'." , flush=True, end="\n\n")
    print("Will use "+str(args.amount_of_processes)+" parallel executions, starting...", flush=True, end="\n\n")

    # creating the vars for the multiprocessing
    set_start_method("spawn")  # TODO fork might be better here?
    lock = Lock()
    xmodels = []
    xmodels.extend(glob.glob(join(args.models, "*.xml")))

    if len(xmodels) == 0:
        print(f"no models in folder!")
        sys.exit(0)

    processes = []
    xmodel_iter = iter(xmodels)

    try:
        # Start initial batch of processes
        for _ in range(min(args.amount_of_processes, len(xmodels))):
            xmodel = next(xmodel_iter, None)
            if xmodel is not None:
                process_args = (lock, args.solver, xmodel)
                process = Process(target=run_model, args=process_args)
                processes.append(process)
                process.start()

        while processes:
            for process in processes:
                process.join(timeout=5)  # Check if process has finished
                if not process.is_alive():
                    processes.remove(process)
                    process.close()
                    xmodel = next(xmodel_iter, None)
                    if xmodel is not None:
                        process_args = (lock, args.solver, xmodel)
                        new_process = Process(target=run_model, args=process_args)
                        processes.append(new_process)
                        new_process.start()

    except KeyboardInterrupt as e:
        print("interrupting...", flush=True, end="\n")
    except Exception as e:
        print(f"An unexpected error occurred:\n{e} \nstacktrace:\n{traceback.format_exc()}", flush=True, end="\n")
    finally:
        print("\nRan models for " + str(math.floor((time.time() - start_time) / 60)) + " minutes", flush=True,
              end="\n")
        # terminate all the processes
        for process in processes:
            if process._popen is not None:
                process.terminate()
        print("Terminated run \n", flush=True, end="\n")


