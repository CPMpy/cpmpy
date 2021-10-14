#!/usr/bin/python3
"""
Counterfactual explanation of a user query.

Based on application on the knapsack problem of Korikov, A., & Beck, J. C. mming, CP2021. Counterfactual Explanations via Inverse Constraint Programming.

Usecase:
 1) Some optimal solution x* is provided to the user by a constraint optimization solver.
 2) User enters a query to change solution variables x_i to a new value. 
    The assigment of a subset of variables are called a foil.
    This extra set of constraints is used to generate a new optimal solution x_d
 3) Using inverse optimization, we calculate the values of constraints in the objective function which result in x_d being the optimal solution.
    Extra constraint: only values corresponding to foil items are allowed to change.
 4) The user is presented with the constraints which would have resulted in an optimal solution given the foil constraints.

Intuition:
Iteratively find new constraints vector d with minimal change from c.
Improve the set of constraints in every iteration until x_d is optimal for the constraints given the objective function.

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
verbose = True
np.random.seed(0)


def main():

    n = 10  # Number of items in the knapsack
    m = 5  # Number of items to change in the knapsack

    values, weights, capacity = generate_knapsack_model(n)
    x = solve_knapsack_problem(values, weights, capacity)
    print_knapsack_model(values, weights, capacity, x)

    # We emulate a user query by selecting random items
    # These items are assigned the opposite value of the current solution
    # While generating this query, we ensure there exist a feasable solution.
    x_user, foil_items = generate_foil_knapsack(values, weights, capacity, x, m)

    # Pretty print user query
    pp_uquery(x, foil_items)

    # Find the values vector such that x_foil is optimal
    d_star = inverse_optimize(values, weights, capacity, x_user, foil_items)

    print(
        f"\n\nValues {d_star} results in a solution satisfying the user query being optimal"
    )
    print(f"Optimal knapsack satisfying user query: {x_user}")
    print(f"Value of objective function using d* = {sum(d_star * x_user)}")


def solve_knapsack_problem(values, weights, capacity):
    """
    Ordinary 0-1 knapsack problem
    Solve for vector x ∈ {T,F}^n

    Based on the Numberjack model of Hakan Kjellerstrand
    """
    x = boolvar(len(values), name="x")
    model = Model([sum(x * weights) <= capacity], maximize=sum(x * values))
    if model.solve() is not False:
        return x.value()
    else:
        raise ValueError("Model is UNSAT")


def generate_knapsack_model(n=10):
    """
    Generation of a knapsack model
    Using same generation parameters as Korikov et al.
    """
    R = 1000
    values = np.random.randint(1, R, n)
    weights = np.random.randint(1, R, n)
    capacity = int(max([0.5 * sum(weights), R]))

    return values, weights, capacity


def generate_foil_knapsack(values, weights, capacity, x, m, tries=1):
    """
    Generate a set of foil constraints
    Pick m items and assign the opposite value.
    If this results in an unfeasable assignment (surpassing the capacity), try again
    @return A model
    """
    n = len(x)
    foil_idx = np.random.choice(n, m, replace=False)
    foil_vals = np.abs(1 - x[foil_idx])

    if sum(foil_vals * weights[foil_idx]) > capacity:
        if verbose:
            print(f"\rGenerated unfeasable user query, retrying...({tries})", end="")
        return generate_foil_knapsack(values, weights, capacity, x, m, tries + 1)
    else:
        return (
            extend_to_full_solution(values, weights, capacity, foil_idx, foil_vals),
            foil_idx,
        )


def extend_to_full_solution(values, weights, capacity, foil_idx, foil_vals):
    """
    Extend a given set of foil constraints to a full solution of the knapsack
    Formally:
        Given v and X, solve the COP (c, v ∩ X)
    """
    xv = boolvar(shape=len(values), name="xv")
    constraints = [xv[foil_idx] == foil_vals]
    constraints += [sum(xv * weights) <= capacity]

    model = Model(constraints, maximize=sum(xv * values))

    if model.solve() is not False:
        return xv.value()


def make_master_problem(values, weights, capacity, x_d, foil_idx):
    """
    Creates the master problem.
    Returns both the model itself as well as the variables used in it.
    This way the variables can be used to add new constraints outside this building function.
    """

    d = intvar(0, INFINITY, values.shape, name="d")
    x = boolvar(shape=len(x_d), name="x")
    # Minimize the change to the values vector
    m = Model(minimize=np.linalg.norm(values - d, ord=1))

    # The ususal knapsack constraint
    m += [sum(x * weights) <= capacity]
    # Ensure values are only modified at foil indices
    m += [d[i] == values[i] for i in range(len(values)) if i not in foil_idx]
    # Ensure the foil values assigned by the user remain the same
    m += [x[i] == bool(x_d[i]) for i in foil_idx]

    return m, d, x


def make_sub_problem(values, weights, capacity):
    """
    Creates the sub problem
    Returns both the model itself as well as the variables in it.
    This way the variables can be used to add new constraints outside this building function.
    """
    x = boolvar(shape=len(values))
    return Model([sum(weights * x) <= capacity]), x


def inverse_optimize(values, weights, capacity, x_d, foil_idx):

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

    master_model, d, x = make_master_problem(values, weights, capacity, x_d, foil_idx)
    sub_model, x_0 = make_sub_problem(values, weights, capacity)

    i = 1
    while master_model.solve() is not False:
        d_star = d.value()
        sub_model.maximize(sum(x_0 * d.value()))
        sub_model.solve()
        if verbose:
            print(f"\nStarting iteration {i}")
            print(f"d* = {d_star}")
            print(f"d* * x_d = {sum(d_star * x_d)}")
            print(f"d* * x_0 = {sum(d_star * x_0.value())}")

        if sum(d_star * x_d) >= sum(d_star * x_0.value()):
            return d_star
        else:
            master_model += [sum(d * x) >= sum(d * x_0.value())]
        i += 1

    raise ValueError("Master model is UNSAT!")


############################################################################
# All functions below this line are purely used for pretty printing results#
############################################################################
def print_knapsack_model(values, weights, capacity, x):
    """
    Pretty prints a knapsack model.
    """
    print("Solution to the following knapsack problem")
    print("Values =", values)
    print("Weights = ", weights)
    print(f"Capacity: {capacity}")

    print("\nis:", x)
    print(f"Resulting in an objective value of {sum(x * values)}")
    print(f"Capacity used: {sum(x*weights)}")


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
    print(
        "How should the values corresponding to these items change to make this assignment optimal?"
    )


if __name__ == "__main__":
    main()
