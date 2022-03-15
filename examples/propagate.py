from cpmpy import *


x,y,z = [intvar(lb=0,ub=5, name=n) for n in "xyz"]

model = Model([x < y, y != 3])
print("Propagating constraints in model:", model, sep="\n", end="\n\n")

s = SolverLookup.lookup("ortools")(model)
# Do not actually solve, just run presolve procedure
s.solve(stop_after_presolve=True, fill_tightened_domains_in_response=True)

# Get bounds from response proto send to the native solver object
bounds = s.ort_solver._CpSolver__solution.tightened_variables
for bound in bounds:
    # Linear scan over solver vars
    for cpm_var, ort_var in s._varmap.items():
        if bound.name == ort_var.Name():
            # Found the variable corresponding to the bound
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

            args = [(cpm_var >= lb) & (cpm_var <= ub) for lb, ub in zip(lbs, ubs)]

            print(f"Derived new constraints for {cpm_var}:", any(args))
            break