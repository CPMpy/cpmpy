#!/usr/bin/env python
import argparse
import cpmpy as cp
import random

def run_incr_xor(solvername, num_vars, xor_size, iter, use_solver=False, seed=42):
    """Test ISAT incrementality by adding XOR constraints incrementally."""
    if use_solver:
        m = cp.SolverLookup.get(solvername)
        solveargs = ()
    else:
        m = cp.Model()
        solveargs = (solvername,)
    random.seed(seed)
    total_time = 0.0

    bs = cp.boolvar(shape=num_vars)
    for _ in range(iter):
        m += cp.Xor([bs[i] for i in random.sample(range(num_vars), xor_size)])
        r = m.solve(*solveargs)
        if not r:
            print("Unsat after ", _+1, " iterations")
        total_time += m.status().runtime
    
    return total_time

def run_incr_obj(solvername, num_vars, num_xor, xor_size, num_obj, iter, use_solver=False, seed=42, perturb_range=2):
    """Test IOPT incrementality by perturbing weights over a random XOR model."""
    if use_solver:
        m = cp.SolverLookup.get(solvername)
        solveargs = ()
    else:
        m = cp.Model()
        solveargs = (solvername,)
    random.seed(seed)
    total_time = 0.0

    bs = cp.boolvar(shape=num_vars)
    for _ in range(num_xor):
        m += cp.Xor([bs[i] for i in random.sample(range(num_vars), xor_size)])
    
    weights = [random.randint(10, 50) for _ in range(num_obj)]
    m.maximize(cp.sum([weights[i] * bs[i] for i in range(num_obj)]))
    
    for _ in range(iter):
        weights = [w + random.randint(-perturb_range, perturb_range) for w in weights]
        m.maximize(cp.sum([weights[i] * bs[i] for i in range(num_obj)]))
        r = m.solve(*solveargs)
        if not r:
            print("Unsat after ", _+1, " iterations")
        total_time += m.status().runtime
    
    return total_time

def measure_incr(solvername, run_func, repeat=1, **kwargs):
    try:
        total_model = 0.0
        total_solver = 0.0
        for _ in range(repeat):
            time_model = run_func(solvername, use_solver=False, **kwargs)
            time_solver = run_func(solvername, use_solver=True, **kwargs)
            total_model += time_model
            total_solver += time_solver
        time_model = total_model / repeat
        time_solver = total_solver / repeat
        speedup = time_model / (time_solver + 1e-6)
        print(f"{solvername} {time_model:.4f} {time_solver:.4f} {speedup:.2f}x")
    except Exception as e:
        print(f"{solvername} crashed: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--solver", default=None, help="comma-separated names; default all except choco")
    p.add_argument("--repeat", type=int, default=1, help="average over this many runs")
    a = p.parse_args()
    if a.solver:
        solvernames = [s.strip() for s in a.solver.split(",") if s.strip()]
    else:
        solvernames = [s for s in sorted(cp.SolverLookup.solvernames()) if s != "choco"]

    print("\nTesting ISAT (expected speedup from 2x)")
    num_vars, xor_size = 100, 10
    iter_count = 50

    for solvername in solvernames:
        measure_incr(solvername, run_incr_xor, repeat=a.repeat, num_vars=num_vars, xor_size=xor_size, iter=iter_count)

    print("\nTesting IOPT (expected speedup from 2x)")
    num_xor, num_obj = 10, 20
    perturb_range = 2
    for solvername in solvernames:
        measure_incr(solvername, run_incr_obj, repeat=a.repeat, num_vars=num_vars, num_xor=num_xor, xor_size=xor_size, num_obj=num_obj, iter=iter_count, perturb_range=perturb_range)

