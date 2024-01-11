import timeit
import cpmpy as cp
import os
from os.path import join
import datetime
import glob
from cpmpy.tools.read_xcsp import XCSPParser

#give this a meaningful name, so we know what branch was tested after the results are safed.
branch = 'main'

alltimes = {}
xmlmodels = []
cwd = os.getcwd()
if 'y' in cwd[-2:]:
    xmlmodels.extend(glob.glob(join("benchmarks", 'instances', 'sat', "*.xml")))
    xmlmodels.extend(glob.glob(join("benchmarks", 'instances', 'unsat', "*.xml")))
else:
    xmlmodels.extend(glob.glob(join('instances', 'sat', "*.xml")))
    xmlmodels.extend(glob.glob(join('instances', 'unsat', "*.xml")))

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

for xmlmodel in xmlmodels:
    model = XCSPParser(xmlmodel)
    s = cp.SolverLookup.get('ortools',model)

    result = None
    def solv():
        global result
        result = s.solve(num_search_workers=1,time_limit=1800)

    t_solve = timeit.timeit(stmt=solv, number=1)
    timings = s.timings
    timings['solve'] = t_solve
    alltimes[xmlmodel] = timings
    if 'unsat' in xmlmodel:
        assert not result #should be unsat
    else:
        assert result #should be sat

    if result is None:
        assert False

import pandas as pd
# Maak een lege DataFrame aan
df = pd.DataFrame(columns=['decompose', 'flatten', 'reify', 'only_numexpr', 'only_bv', 'only_implies', 'get_vars', 'post_cons', 'solve'])
dfs = []
for instance, values in alltimes.items():
    instance_df = pd.DataFrame({
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
    filename = ((join("benchmarks", "results", branch + now + '.csv')))
else:
    filename = ((join("results", branch + now + '.csv')))
df.to_csv(filename)