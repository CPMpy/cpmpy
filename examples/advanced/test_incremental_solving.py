#!/usr/bin/env python
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

def measure_incr(solvername, run_func, **kwargs):
    try:
        time_model = run_func(solvername, use_solver=False, **kwargs)
        time_solver = run_func(solvername, use_solver=True, **kwargs)
        speedup = time_model / (time_solver + 1e-6)
        print(f"{solvername} {time_model:.4f} {time_solver:.4f} {speedup:.2f}x")
    except Exception as e:
        print(f"{solvername} crashed: {e}")

if __name__ == "__main__":
    print("\nTesting ISAT (expected speedup from 2x)")
    num_vars, xor_size = 100, 10
    iter_count = 50
    
    for solvername in cp.SolverLookup.solvernames():
        if solvername == "choco":
            continue  # some painful crash that Python can not recover from
        measure_incr(solvername, run_incr_xor, num_vars=num_vars, xor_size=xor_size, iter=iter_count)

    print("\nTesting IOPT (expected speedup from 2x)")
    num_xor, num_obj = 10, 20
    perturb_range = 2
    for solvername in cp.SolverLookup.solvernames():
        if solvername == "choco":
            continue  # some painful crash that Python can not recover from
        measure_incr(solvername, run_incr_obj, num_vars=num_vars, num_xor=num_xor, xor_size=xor_size, num_obj=num_obj, iter=iter_count, perturb_range=perturb_range)

