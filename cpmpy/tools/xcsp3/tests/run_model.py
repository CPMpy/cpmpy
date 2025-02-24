import gc

import os
from os.path import join

import glob

import argparse
import math
import pandas as pd
import sys
import traceback
import time
from multiprocessing import Process,Lock, Manager, set_start_method,Pool, cpu_count
from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3

import cpmpy as cp
from cpmpy import SolverLookup
from cpmpy.tools.xcsp3.callbacks import CallbacksCPMPy
from cpmpy.tools.xcsp3.installer import install_xcsp3_instances_22, install_xcsp3_instances_23


def check_positive(value):
    """
    Small helper function used in the argparser for checking if the input values are positive or not
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def run_model(lock, solver, xmodel, df_path, solve=True):
    """
    Runs one XCSP3 instance
    """
    print("\nrunning model"+xmodel)
    parser = ParserXCSP3(xmodel)
    callbacks = CallbacksCPMPy()
    callbacks.force_exit = True
    callbacker = CallbackerXCSP3(parser, callbacks)
    try:
        start_time = time.time()
        callbacker.load_instance()
        end_time = time.time()
        t_parse = end_time - start_time
    except Exception as e:
        print("error parsing:")
        print(e)
    cb = callbacker.cb
    start_time = time.time()
    s_mock = SolverLookup.get(solver, model=None)  # mock solver object, to run transform without posting
    s_mock.transform(cb.cpm_model.constraints)
    end_time = time.time()
    t_transform = end_time - start_time  # only cpmpy transform without posting
    start_time = time.time()
    s = SolverLookup.get(solver, cb.cpm_model)
    end_time = time.time()
    t_add = end_time - start_time  # total transform + native solver interface
    try:
        if solve:
            solve_start_time = time.time()
            res = s.solve()
            solve_end_time = time.time()
            #print(f"Solving took {solve_end_time - solve_start_time} seconds")
            t_solve = solve_end_time - solve_start_time
        else:
            t_solve = 0
        write_to_dataframe(lock, xmodel, solver, t_parse, t_add, t_transform, t_solve, df_path)
    except Exception as e:
        print('error solving:')
        print(e)



def write_to_dataframe(lock, model_name, solver, t_parse, t_add, t_transform, t_solve, df_path):
    """
    Helper function to write model_name, t_solve, and t_transform to a dataframe.
    All subprocesses will write to this same dataframe.
    """
    lock.acquire()
    try:
        # Load existing dataframe or create a new one if it doesn't exist
        try:
            df = pd.read_csv(df_path)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["model_name", "solver", "t_parse", "t_add", "t_transform", "t_solve"])

        # Append new data
        new_data = {"model_name": model_name, "solver": solver, "t_parse": t_parse, "t_add": t_add, "t_transform": t_transform, "t_solve": t_solve}
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

        # Save dataframe back to CSV
        df.to_csv(df_path, index=False)
    finally:
        lock.release()

if __name__ == '__main__':
    gc.disable()  # more consistent timing without automatic garbage collection
    # get all the available solvers from cpympy
    available_solvers = cp.SolverLookup.solvernames()

    parser = argparse.ArgumentParser(description="A python application to fuzz_test your solver(s)")
    parser.add_argument("-s", "--solver", help="The Solver to use", required=False, type=str, choices=available_solvers,
                        default=available_solvers[0])
    parser.add_argument("-m", "--models", help="The path to load the models", required=False, type=str,
                        default="models")
    parser.add_argument("-o", "--output", help="The path to the output csv", required=False, type=str,
                        default="output.csv")
    parser.add_argument("-d", "--download", help="download xcsp3 competition instances (0 = false, 1 = true)", required=False, type=int,
                        default=0)
    parser.add_argument("-ns", "--noSolve", help="only run transform, don't solve models (0 = false, 1 = true)",
                        required=False, type=int,
                        default=0)
    parser.add_argument("-p", "--amount-of-processes",
                        help="The amount of processes that will be used to run the tests", required=False,
                        default=cpu_count() - 1, type=check_positive)  # the -1 is for the main process
    args = parser.parse_args()
    start_time = time.time()
    df_path = args.output
    # Empty the dataframe if it already exists
    if os.path.exists(df_path):
        pd.DataFrame(columns=["model_name", "solver", "t_parse", "t_add", "t_transform", "t_solve"]).to_csv(df_path, index=False)

    if args.download:
        install_xcsp3_instances_22()
        install_xcsp3_instances_23()

    if args.noSolve:
        only_transform = True
    else:
        only_transform = False

    lock = Lock()
    xmodels = []
    for root, dirs, files in os.walk(args.models):
        for file in files:
            if file.endswith(".xml"):
                xmodels.append(join(root, file))

    if len(xmodels) == 0:
        print(f"no models in folder!")
        sys.exit(0)

    print("\nUsing solver '"+args.solver+f" with {len(xmodels)} models in "+args.models+"'." , flush=True, end="\n\n")
    print("Will use "+str(args.amount_of_processes)+" parallel executions, starting...", flush=True, end="\n\n")

    processes = []
    xmodel_iter = iter(xmodels)

    try:
        # Start initial batch of processes
        for _ in range(min(args.amount_of_processes, len(xmodels))):
            xmodel = next(xmodel_iter, None)
            if xmodel is not None:
                process_args = (lock, args.solver, xmodel, df_path, not only_transform)
                process = Process(target=run_model, args=process_args)
                processes.append(process)
                process.start()

        while processes:
            for process in processes:
                process.join(timeout=5)  # Check if process has finished
                if not process.is_alive():
                    processes.remove(process)
                    process.close()
                    gc.collect()  # collect garbage after each instance. (is this overkill?)
                    xmodel = next(xmodel_iter, None)
                    if xmodel is not None:
                        process_args = (lock, args.solver, xmodel, df_path, not only_transform)
                        new_process = Process(target=run_model, args=process_args)
                        processes.append(new_process)
                        new_process.start()

    except KeyboardInterrupt as e:
        print("interrupting...", flush=True, end="\n")
    except Exception as e:
        print(f"An unexpected error occurred:\n{e} \nstacktrace:\n{traceback.format_exc()}", flush=True, end="\n")
    finally:
        gc.enable()
        print("\nRan models for " + str(math.floor((time.time() - start_time) / 60)) + " minutes", flush=True,
              end="\n")
        # terminate all the processes
        for process in processes:
            if process._popen is not None:
                process.terminate()
        print("Terminated run \n", flush=True, end="\n")


