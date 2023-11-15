import pytest
import cpmpy as cp
import os
from os.path import join
import glob
from cpmpy.tools.read_xcsp import XCSPParser

MODELS = []
SOLVERNAMES = [name for name, solver in cp.SolverLookup.base_solvers() if solver.supported()]
xmlmodels = []
cwd = os.getcwd()
if 'y' in cwd[-2:]:
    xmlmodels.extend(glob.glob(join("benchmarks", 'instances', 'sat', "*.xml")))
    xmlmodels.extend(glob.glob(join("benchmarks", 'instances', 'unsat', "*.xml")))
else:
    xmlmodels.extend(glob.glob(join('instances', 'sat', "*.xml")))
    xmlmodels.extend(glob.glob(join('instances', 'unsat', "*.xml")))

#for subdividing the models (use 'instances' directory for xmlmodels above)
"""for xmlmodel in xmlmodels:
    try:
        model = XCSPParser(xmlmodel)
        s = cp.SolverLookup.get('ortools', model)
        sat = s.solve(num_search_workers=6, time_limit=300)
        if (s.status().runtime > 290):
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
        os.rename(xmlmodel, xmlmodel[:len(xmlmodel) - len(name)] + 'unsupported\\' + name)"""


@pytest.mark.parametrize('xmlmodel', xmlmodels, ids=str)
def test_solve_ortools(xmlmodel, benchmark):
    model = XCSPParser(xmlmodel)
    s = cp.SolverLookup.get('ortools',model)
    result = benchmark(s.solve,num_search_workers=6,time_limit=1800)


@pytest.mark.parametrize('xmlmodel', xmlmodels, ids=str)
def test_transform_ortools(xmlmodel, benchmark):
    model = XCSPParser(xmlmodel)
    s = benchmark(cp.SolverLookup.get,'ortools',model)