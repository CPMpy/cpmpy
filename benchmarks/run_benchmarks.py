import timeit
import cpmpy as cp
import os
from os.path import join
import datetime
import glob
from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3
from callbackscpmpy import CallbacksCPMPy
from cpmpy.exceptions import TransformationNotImplementedError

# give this a meaningful name, so we know what branch was tested after the results are safed.
branch = 'main'
# set solver to test (suported: ortools)
solver = 'exact'
# solver timeout in seconds
time_limit = 60
# set true to only time transformations, and not call the solver
transonly = False

alltimes = {}
xmlmodels = []
cwd = os.getcwd()
if 'y' in cwd[-2:]:
    xmlmodels.extend(glob.glob(join("benchmarks", 'MiniCSP', "*.xml")))
    xmlmodels.extend(glob.glob(join("benchmarks", 'MiniCOP', "*.xml")))
else:
    xmlmodels.extend(glob.glob(join('MiniCSP', "*.xml")))
    xmlmodels.extend(glob.glob(join('MiniCOP', "*.xml")))

#for subdividing the models (use 'instances' directory for xmlmodels)
'''if 'y' in cwd[-2:]:
    xmlmodels.extend(glob.glob(join("benchmarks", 'instances', "*.xml")))
else:
    xmlmodels.extend(glob.glob(join('instances', "*.xml")))
for xmlmodel in xmlmodels:
    try:
        model = XCSPParser(xmlmodel)
        s = cp.SolverLookup.get('ortools', model)
        sat = s.solve(num_search_workers=1, time_limit=25)
        if (s.status().runtime > 22):
            #slow models
            name = os.path.basename(xmlmodel)
            os.rename(xmlmodel, xmlmodel[:len(xmlmodel) - len(name)] + 'slow\\' + name)
        elif sat:
            name = os.path.basename(xmlmodel)
            os.rename(xmlmodel, xmlmodel[:len(xmlmodel) - len(name)] + 'sat\\' + name)
        elif not sat:
            name = os.path.basename(xmlmodel)
            os.rename(xmlmodel, xmlmodel[:len(xmlmodel) - len(name)] + 'unsat\\' + name)
    except Exception:
        name = os.path.basename(xmlmodel)
        os.rename(xmlmodel, xmlmodel[:len(xmlmodel) - len(name)] + 'unsupported\\' + name)
'''
class Fakesolver():
    def __init__(self):
        self.timings = dict()

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limiter(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

print(xmlmodels)
for xmlmodel in xmlmodels[:10]:
    model = None
    def parse():
        parser = ParserXCSP3(xmlmodel)
        callbacks = CallbacksCPMPy()
        callbacks.force_exit = True
        # e.g., callbacks.recognize_unary_primitives = False
        callbacker = CallbackerXCSP3(parser, callbacks)
        callbacker.load_instance()
        cb = callbacker.cb
        global model
        model = cb.cpm_model
    # print(cb.cpm_variables)


    result = None
    t_parse = timeit.timeit(stmt=parse, number=1)
    try:
        s = cp.SolverLookup.get(solver, model)
    except TransformationNotImplementedError as e:
        s = Fakesolver()
    def solve_ortools():
        global result
        result = s.solve(num_search_workers=1, time_limit=time_limit)

    def solve_exact():
        global result
        result = s.solve(time_limit=time_limit)

    try:
        with time_limiter(time_limit + 30):
            if not transonly:
                print('solving')
                if solver == 'ortools':
                    t_solve = timeit.timeit(stmt=solve_ortools, number=1)
                else:
                    try:
                        t_solve = timeit.timeit(stmt=solve_exact, number=1)
                    except Exception as e:
                        print(e)
                        t_solve = 0
            else:
                t_solve = 0
    except TimeoutException as e:
        print('hard time out..')
        t_solve = time_limit

    print('solved in ', t_solve, 'seconds')
    timings = s.timings
    timings['solve'] = t_solve
    timings['parse'] = t_parse
    for opt in ['parse', 'decompose', 'flatten', 'reify', 'only_numexpr', 'only_bv', 'only_implies', 'linearize', 'only_pos_bv', 'get_vars', 'post_cons', 'solve']:
        if opt not in timings:
            # add unused transformations
            timings[opt] = 0
    alltimes[xmlmodel] = timings


import pandas as pd
# Maak een lege DataFrame aan
df = pd.DataFrame(columns=['parse', 'decompose', 'flatten', 'reify', 'only_numexpr', 'only_bv', 'only_implies', 'linearize', 'only_pos_bv', 'get_vars', 'post_cons', 'solve'])
dfs = []
for instance, values in alltimes.items():
    instance_df = pd.DataFrame({
        'parse': [values['parse']],
        'decompose': [values['decompose']],
        'flatten': [values['flatten']],
        'reify': [values['reify']],
        'only_numexpr': [values['only_numexpr']],
        'only_bv': [values['only_bv']],
        'only_implies': [values['only_implies']],
        'linearize': [values['linearize']],
        'only_pos_bv': [values['only_pos_bv']],
        'get_vars': [values['get_vars']],
        'post_cons': [values['post_cons']],
        'solve': [values['solve']]
    }, index = [instance])
    dfs.append(instance_df)

df = pd.concat(dfs, ignore_index=False)
total = pd.DataFrame({
    'parse': [df['parse'].sum()],
    'decompose': [df['decompose'].sum()],
    'flatten': [df['flatten'].sum()],
    'reify': [df['reify'].sum()],
    'only_numexpr': [df['only_numexpr'].sum()],
    'only_bv': [df['only_bv'].sum()],
    'only_implies': [df['only_implies'].sum()],
    'linearize': [df['linearize'].sum()],
    'only_pos_bv': [df['only_pos_bv'].sum()],
    'get_vars': [df['get_vars'].sum()],
    'post_cons': [df['post_cons'].sum()],
    'solve': [df['solve'].sum()]
}, index=['total'])

df = pd.concat([df, total], ignore_index=False)


now= str(datetime.datetime.now()).replace(':','.')
if 'y' in cwd[-2:]:
    filename = ((join("benchmarks", "results", branch + '_' + solver + '_' +  now + '.csv')))
else:
    filename = ((join("results", branch + '_' + solver + '_' + now + '.csv')))
df.to_csv(filename)
