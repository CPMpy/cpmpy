"""
Example showing direct solver access to the OR-tools solver object.

Uses the presolve feature of ortools and reads the protobuffer response directly.

Constructs new constraints for bounds found by the OR-tools presolve.
Consult https://developers.google.com/optimization/reference/sat/cp_model_presolve/CpModelPresolver for details about OR-tools presolver.
"""

from cpmpy import *


x,y,z = [intvar(lb=0,ub=5, name=n) for n in "xyz"]

model = Model([x < y, y != 3])
print("Propagating constraints in model:", model, sep="\n", end="\n\n")

s = SolverLookup.get("ortools", model)
# Do not actually solve, just run presolve procedure
s.solve(stop_after_presolve=True, fill_tightened_domains_in_response=True)

# Get bounds from response proto send to the native solver object
bounds = s.ort_solver.ResponseProto().tightened_variables
for cpm_var, ort_var in s._varmap.items():
    # Get bounds for variable
    bound = bounds[ort_var.Index()]
    """
        Structure of bounds object:
        ``` name: variable_name
            domain: lower_bound_1
            domain: upper_bound_1
            ....
            domain: lower_bound_n
            domain: upper_bound_n
    """
    lbs = [val for i,val in enumerate(bound.domain) if i % 2 == 0]
    ubs = [val for i,val in enumerate(bound.domain) if i % 2 == 1]
    # More elaborate constraints can be found here.
    # I.e. cpm_var != val if there is only one hole in the domain of the variable
    args = [(cpm_var >= lb) & (cpm_var <= ub) for lb, ub in zip(lbs, ubs)]

    print(f"Derived new constraints (bounds) for {cpm_var}:", any(args))