import timeit
import cpmpy as cp
import os
from os.path import join
import glob
from cpmpy.tools.read_xcsp import XCSPParser

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

print(alltimes)