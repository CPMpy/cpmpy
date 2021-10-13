#!/usr/bin/python3
"""
Counterfactual explanation of a user query.

Based on application on the knapsack problem of Korikov, A., & Beck, J. C. Counterfactual Explanations via Inverse Constraint Programming.

Usecase:
 1) Some optimal solution x* is provided to the user by a constraint optimization solver.
 2) User enters a query to change solution variables x_i to a new value.
    This extra set of constraints is used to generate a new optimal solution x_d
 3) Using inverse optimization, we calculate the values of constraints the objective function which result in x_d being the optimal solution.
    Extra constraint: only values corresponding to foil items are allowed to change.
 4) The user is presented with the result.


Given:
    A constraint optimization problem (c, f, X) with an optimal solution x*
    A set of foil constraints assigning a value to variables in x*, resulting in x_d
Find:
    d such that min(f(d,X)) == x_d and ||c - d|| is minimal


The algorithm consists of 2 alternating problems: the master problem (MP), and the sub problem (SP)

Algorithm:
    1 S = {}
    2 Solve MP to obtain optimal d*
    3 Solve SP with d* as c to obtain x_0
    4 if objective(x_d) > objective(x_0):
        add x_0 to S
        go to 2
      else:
          return d*


Master problem:
    Constraints:
        - constraints of original forward problem
        - objective(d*, x) >= objective(d*, x_0) forall x_0 in S
        - foil constraints assigning values to x_i
        - foil contraints restricting the altering of c to match the foil constraint indices
    Objective:
        - Minimize || c - d* ||_1

Sub problem:
    The original forward problem


"""

from cpmpy import *
import numpy as np

INFINITY = np.iinfo(np.int32).max
verbose = False
np.random.seed(0)

def main():

    n = 10  #Number of items in the knapsack
    m = 5   #Number of items to change in the knapsack

    values, weights, capacity, x = generate_knapsack_model(n)
    #TODO: verbose statement?
    print_knapsack_model(values, weights, capacity, x)

    x_user, foil_items = generate_foil_knapsack(values, weights, capacity, x, m)
    
    # Pretty print user query
    #TODO: verbose statement?
    pp_uquery(x, foil_items)

    #Find the values vector such that x_foil is optimal
    d_star = master_problem(values, weights, capacity, x_user, foil_items)
    
    print(f"\n\nValues {d_star} results in the user query being optimal")
    print(f"Value of objective function using d* = {sum(d_star * x_user)}")


def generate_knapsack_model(n = 10):
    """
        Generation of a knapsack model
        Using same generation parameters as Korikov et al.
    """
    R = 1000
    values = np.random.randint(1,R,n)
    weights = np.random.randint(1,R,n)
    capacity = int(max([0.5 * sum(weights), R]))

    return values, weights, capacity, \
        solve_knapsack_problem(values, weights, capacity)


def generate_foil_knapsack(values, weights, capacity, x, m, tries=1):
    """
        Generate a set of foil constraints
        Pick m items and assign the opposite value.
        If this results in an unfeasable assignment (surpassing the capacity), try again
        @return A model
    """
    n = len(x)
    foil_idx = np.random.choice(n,m, replace=False)
    foil_vals = np.abs(1 - x[foil_idx])

    if sum(foil_vals * weights[foil_idx]) > capacity:
        if verbose:
            print(f"\rGenerated unfeasable user query, retrying...({tries})", end="")
        return generate_foil_knapsack(values, weights, capacity, x, m, tries+1)
    else:
        return extend_to_full_solution(values, 
                                       weights, 
                                       capacity, 
                                       foil_idx,
                                       foil_vals)\
                ,foil_idx

def extend_to_full_solution(values, weights, capacity, foil_idx, foil_vals):
    """
        Extend a given set of foil constraints to a full solution of the knapsack
        Formally:
            Given v and X, solve the COP 
    """
    xv = boolvar(shape=len(values), name="xv")
    constraints = [xv[foil_idx] == foil_vals]
    constraints += [sum(xv * weights) <= capacity]

    model = Model(
        constraints,
        maximize= sum(xv * values)
    )

    if model.solve() is not False:
        return xv.value()

def master_problem(values, weights, capacity, x_d, foil_idx):
    
    """
        Master problem: iteratively find better values for the c vector (new values are vector d)
        Mapping of variable names to names in the paper (TODO: link papper)
            - c = values
            - d = d
        @param x_d: A feasable solution found in the foil set X_v
        @param foil_idx: A vector containing the items on which the user asked an explanation
                         All other items must retain their original values
    """
    if verbose:
        print(f"\n\n{'='*10} Solving the master problem {'='*10}")
    known_solutions, i= [], 1
    while i:
        if verbose:
            print(f"\nStarting iteration {i}")
        d = intvar(0,INFINITY, values.shape, name="d")
        x = boolvar(shape=len(x_d), name="x")
        # The ususal knapsack constraint
        constraints = [sum(x * weights) <= capacity]

        # Extension of the knapsack problem to the master problem
        # Ensure the newly found values vector results in a better solution than any known solutions
        constraints += [sum(d * x) >= sum(d * known) for known in known_solutions]
        # Ensure values are only modified at foil indices 
        constraints += [d[i] == values[i] for i in range(len(values)) if i not in foil_idx]
        # Ensure the foil values assigned by the user remain the same
        constraints += [x[i] == bool(x_d[i]) for i in foil_idx]
       
        master_model = Model(
            constraints, 
            # Minimize the change to the values vector
            minimize = np.linalg.norm(values -  d, ord=1)
        )

        if master_model.solve() is not False:
            d_star = d.value()
            new_solution = solve_knapsack_problem(d_star, weights, capacity)
            if verbose:
                print(f"d* = {d_star}")
                print(f"d* * x_d = {sum(d_star * x_d)}")
                print(f"d* * x_0 = {sum(d_star * new_solution)}")
            if sum(d_star * x_d) >= sum(d_star * new_solution):
                return d_star
            else:
                known_solutions.append(new_solution)
            i += 1

        else:
            print("Model is UNSAT!")
            exit()

def solve_knapsack_problem(values, weights, capacity):
    """
        Ordinary 0-1 knapsack problem
        Solve for vector x ∈ {T,F}^n

        Based on the Numberjack model of Hakan Kjellerstrand
    """
    x = boolvar(len(values), name="x")
    model = Model(
                [sum(x * weights) <= capacity],
                maximize = sum(x * values)
            )
    if model.solve() is not False:
        return x.value()


############################################################################
# All functions below this line are purely used for pretty printing results#
############################################################################
def print_knapsack_model(values, weights, capacity, x):
    """
        Pretty prints a knapsack model.
    """
    print("Solution to the following knapsack problem")
    print(f"Capacity: {capacity}, used: {sum(x*weights)}")  
    print("Values =", values)
    print("Weights = ", weights)
    
    print("\nis:", x)
    print(f"Resulting in an objective value of {sum(x * values)}")

def pp_uquery(x, f_items):
    """
        Function to pretty print the user query to solver.
        Has no computational effects
    """

    include = sorted([str(a) for a in f_items[x[f_items] == 0]])
    exclude = sorted([str(a) for a in f_items[x[f_items] == 1]])
    print(f"\n\n{'='*10} User Query {'='*10}")

    print(f"I would like to change the following to the knapsack you provided:")
    if len(exclude) > 0:
        print(f"Leave out item{'s' if len(exclude) > 1 else ''} {','.join(exclude)}")
    if len(include) > 0:
        print(f"Put in item{'s' if len(include) > 1 else ''} {','.join(include)}")
    print("How should the values corresponding to these items change to make this assignment optimal?")

if __name__ == "__main__":
    main()