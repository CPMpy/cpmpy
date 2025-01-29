"""
Finding counterfactual explanation through constraint relaxation.

Based on the "CounterFactualXplain"-algorithm of Dev Gupta, S., Genç, B., O'Sullivan, B. (2022, 7 april). Finding Counterfactual
 Explanations Through Constraint Relaxation. arXiv:2204.03429v1.

An adapted algorithm which finds counterfactual explanations of the laptop problem presented in the paper.

Use cases (changes possible with enable_iteration variable):
- Default: One explanation, the order in which the constraints are considered is the same order as in which the user_constraints are mentioned in the 'user_constraints'-variable
- An explanation for every possible order in which the constraints can be considered
"""
from cpmpy import *
import itertools

## This variable is responsible for the use cases
enable_iteration = False

"""
List of all the laptops.
Format: [Size, Memory, Life, Price]
All numbers must be multiplied by 100 since CPMpy can only work with Integer values. When creating the explanations, the values are automatically divided by 100.
"""
laptops = [[1540, 102400, 220, 149999],    # Unsatisfiable, relax life time to get a satisfiable result
            [1500, 51200, 1000, 261699],    
            [1500, 51200, 450, 189900],
            [1400, 51200, 1000, 189999]
            ]

size = intvar(0, 2000)
memory = intvar(0, 102500)
life = intvar(0, 1100)
price = intvar(0, 300000)

global size_cons
global memory_cons
global life_cons
# The foreground/user constraints:
size_cons = size >= 1500
memory_cons = memory >= 51200
life_cons = life >= 1000

user_constraints = [size_cons, memory_cons, life_cons]

# Information about the user constraints (usefull for creating explanations)
user_constraint_names = ["size", "memory", "lifetime"]
user_constraint_values = [1500, 51200, 1000]

global new_constraint_values
new_constraint_values = [0, 0, 0]

# The background constraints (must be fullfilled at all time)
price_cons = price <= 200000

background_constraints = [price_cons]

# The relaxation spaces of all the user constraints
size_relax = [1540,1500,1400,1110]
mem_relax = [102400,51200]
life_relax = [1100,1000,450,220]

relaxation_spaces = [size_relax, mem_relax, life_relax]

# A table constraint model with all the foreground and background constraints
model = Model(
    Table([size, memory, life, price], laptops),
    size_cons,
    memory_cons,
    life_cons,
    price_cons
)

def main():
    sat = model.solve()
    if sat:
        print(f"There exists a solution. The following laptop satisfies the constraints: size: {size.value()/100} inches, memory: {memory.value()/100} MB, life: {life.value()/100} hr, price: $ {price.value()/100}")
    elif no_sufficient_relax_space():
                print("The defined relaxation spaces are not large enough. It is not possible to relax a constraint")
    else:
        # For explanation about the enable_iteration: Check the intro and enable_iteration variable on line 17
        if enable_iteration:
            # Check the influence of a different order of constraint relaxations

            # save the old user constraint values and user constraints
            old_constraint_values = user_constraint_values.copy()
            old_user_constraints = user_constraints.copy()

            # Create permutations of the possible indices of the user constraints
            order_indices = list(range(0, len(user_constraints)))
            orders = list(itertools.permutations(order_indices))

            for nb_explanation, order in enumerate(orders):
                # Reset the constraints back to their original values
                reset_state(old_constraint_values, old_user_constraints)

                explanation = relax_problem(order)
                print(f"Explanation {nb_explanation}:")
                print("   " + explanation) 
        else:
            explanation = relax_problem(list(range(0, len(user_constraints))))
            print(explanation)


def no_sufficient_relax_space():
    for space in relaxation_spaces:
        if len(space) > 1:
            return False
    return True 


def reset_state(old_constraint_values, old_user_constraints):
    """
    Reset the initial states of the constraints.
    """
    user_constraint_values[:] = old_constraint_values
    user_constraints[:] = old_user_constraints


def relax_problem(order):
    # Create a new model step by step until we find a satisfiable result (bottom-up)
    new_model = Model(Table([size, memory, life, price], laptops))

    # Add the background constraints to the model, these constraints must be fullfilled at all cost
    for background_con in background_constraints:
        new_model += background_con

    # Save the old values of the constraints (necessary for the explanations)
    old_constraint_values = user_constraint_values.copy()

    indices_changed_constraints = list()
    i = 0
    while i < len(user_constraints):
        index = order[i] # We investigate different orders of constraints, so i can be different than order[i]

        user_con = user_constraints[index]

        test_model = new_model.copy() + user_con
        sat = test_model.solve()
        if sat:
            # The constraint don't have to be relaxed anymore, so add it to the new model
            new_model += user_con
            i += 1 # Check the next user constraint
        else:
            indices_changed_constraints.append(index)
            if (relaxation_spaces[index][relaxation_spaces[index].index(user_constraint_values[index])+1] != relaxation_spaces[index][-1]):
                # Check if the constraint is not the last element from the relaxation space
                # Otherwise you have to relax another element at the same time to find a solution
                relax_constraint(index)
            else:
                relax_constraint(index)
                new_model += user_con
                i += 1
        
    indices = remove_duplicates(indices_changed_constraints)

    if new_model.solve():
        return generate_explanation(old_constraint_values, indices)
    else:
        return "The model cannot be made feasible with these relaxation spaces"


def remove_duplicates(indices):
    """
    Removes the duplicates from the given list
    """
    result = []
    for index in indices:
        if index not in result:
            result.append(index)
    return result


def relax_constraint(index):
    """
    Generate the constraint to the next value in the relaxation space
    """
    new_constraint_values[index] = relaxation_spaces[index][relaxation_spaces[index].index(user_constraint_values[index])+1]
    user_constraint_values[index] = new_constraint_values[index]
    if user_constraints[index].name == ">=":
        user_constraints[index] = (user_constraints[index].args)[0] >= new_constraint_values[index]
    elif user_constraints[index].name == "<=":
        user_constraints[index] = (user_constraints[index].args)[0] <= new_constraint_values[index]
    else:
        user_constraints[index] = (user_constraints[index].args)[0] == new_constraint_values[index]


def generate_explanation(old_constraint_values, indices):
    explanation = f"There are {len(indices)} constraints that have to be relaxed to find at least one solution:\n"
    for index in indices:
        old_cons = old_constraint_values[index]/100
        new_cons = user_constraint_values[index]/100
        cons_name = user_constraint_names[index]
        explanation += f"       Change the {cons_name} user constraint from {old_cons} to {new_cons}\n"
    return explanation


if __name__ == "__main__":
    main()
