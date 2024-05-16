import timeit
import cpmpy as cp
import os
from os.path import join
import datetime
import glob
from pycsp3.parser.xparser import CallbackerXCSP3, ParserXCSP3
from callbackscpmpy import CallbacksCPMPy
from cpmpy.tools.read_xcsp import XCSPParser

# give this a meaningful name, so we know what branch was tested after the results are safed.
branch = 'main'
# set solver to test (suported: ortools)
solver = 'ortools'
# solver timeout in seconds
time_limit = 20
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
    s = cp.SolverLookup.get(solver, model)
    def solv():
        global result
        result = s.solve(num_search_workers=1,time_limit=time_limit)

    if not transonly:
        print('solving')
        t_solve = timeit.timeit(stmt=solv, number=1)
    else:
        t_solve = 0
    timings = s.timings
    timings['solve'] = t_solve
    timings['parse'] = t_parse
    alltimes[xmlmodel] = timings


import pandas as pd
# Maak een lege DataFrame aan
df = pd.DataFrame(columns=['parse', 'decompose', 'flatten', 'reify', 'only_numexpr', 'only_bv', 'only_implies', 'get_vars', 'post_cons', 'solve'])
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
        'get_vars': [values['get_vars']],
        'post_cons': [values['post_cons']],
        'solve': [values['solve']]
    }, index = [instance])
    dfs.append(instance_df)

df = pd.concat(dfs, ignore_index=False)
total = pd.DataFrame({
    'parse': [df['parse']],
    'decompose': [df['decompose'].sum()],
    'flatten': [df['flatten'].sum()],
    'reify': [df['reify'].sum()],
    'only_numexpr': [df['only_numexpr'].sum()],
    'only_bv': [df['only_bv'].sum()],
    'only_implies': [df['only_implies'].sum()],
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