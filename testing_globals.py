import cpmpy as cp
from cpmpy.exceptions import CPMpyException, TypeError
import itertools
import traceback

solvers = ["ortools", "minizinc", "gurobi"] #, "choco"]

# Single list constraints (ordering and value constraints)
test_globals_single_list = [
    # Ordering constraints
    "Increasing", "Decreasing", "IncreasingStrict", "DecreasingStrict",
    # Value constraints
    "AllDifferent", "AllDifferentExcept0", "AllEqual", "Circuit"
]

# Two list constraints
test_globals_two_lists = [
    # Lexicographic constraints
    "LexLess", "LexLessEq",
    # Except-N constraints
    "AllDifferentExceptN", "AllEqualExceptN",
    # Other two-list constraints
    "Precedence"
]

# Table constraints
test_globals_table = [
    "Table", "ShortTable", "NegativeTable"
]

# Domain constraints
test_globals_domain = [
    "InDomain"
]

# Boolean constraints
test_globals_boolean = [
    "Xor", "IfThenElse"
]

# Multi-list constraints
test_globals_multi_list = [
    # Three list constraints
    "Inverse", "NoOverlap",
    # Four list constraints
    "GlobalCardinalityCount",
    # Five list constraints
    "Cumulative"
]

# 2D list constraints
test_globals_2d_list = [
    "LexChainLess", "LexChainLessEq"
]

# Create test variables
x = cp.intvar(-3, 5, shape=1)
y = cp.intvar(-3, 5, shape=1)
z = cp.intvar(-3, 5, shape=1)

# Create arrays of variables
iv_arr1 = cp.intvar(-8, 8, shape=5)
iv_arr2 = cp.intvar(-8, 8, shape=5)
iv_arr3 = cp.intvar(-8, 8, shape=5)
iv_arr4 = cp.intvar(-8, 8, shape=5)
iv_arr5 = cp.intvar(-8, 8, shape=5)

# Create boolean variables
b1 = cp.boolvar()
b2 = cp.boolvar()
b3 = cp.boolvar()

# Create test values
vals1 = [1, 5, 8, -4, 8]
vals2 = [2, 3, 1, 5, 4]
vals3 = [0, 1, 2, 3, 4]
vals4 = [4, 3, 2, 1, 0]

# Create mixed arrays (variables and values)
mixed_arr1 = [1, 5, iv_arr1[2], 8, iv_arr1[0]]
mixed_arr2 = [2, iv_arr2[1], 1, 5, 4]

# Create 2D arrays for testing
arr_2d1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
arr_2d2 = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

# Create table data
table_data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Create domain values
domain_values = [1, 2, 3, 4, 5]

# Statistics tracking
stats = {
    "success": 0,
    "cpmpy_exception": 0,
    "other_error": 0,
    "by_constraint": {},
    "by_solver": {}
}

def test_constraint(constraint_name, args, solver):
    """Test a constraint with given arguments and solver"""
    
    print(f"Testing constraint: {constraint_name}, Args: {args}, Solver: {solver}")
    
    # Initialize statistics for this constraint if not already done
    if constraint_name not in stats["by_constraint"]:
        stats["by_constraint"][constraint_name] = {
            "success": 0,
            "cpmpy_exception": 0,
            "other_error": 0
        }
    
    # Initialize statistics for this solver if not already done
    if solver not in stats["by_solver"]:
        stats["by_solver"][solver] = {
            "success": 0,
            "cpmpy_exception": 0,
            "other_error": 0
        }
    
    m = cp.Model()
    
    try:
        # Add the constraint to the model
        if constraint_name == "Increasing":
            m += cp.Increasing(*args)
        elif constraint_name == "Decreasing":
            m += cp.Decreasing(*args)
        elif constraint_name == "IncreasingStrict":
            m += cp.IncreasingStrict(*args)
        elif constraint_name == "DecreasingStrict":
            m += cp.DecreasingStrict(*args)
        elif constraint_name == "AllDifferent":
            m += cp.AllDifferent(*args)
        elif constraint_name == "AllDifferentExcept0":
            m += cp.AllDifferentExcept0(*args)
        elif constraint_name == "AllEqual":
            m += cp.AllEqual(*args)
        elif constraint_name == "Circuit":
            m += cp.Circuit(*args)
        elif constraint_name == "LexLess":
            m += cp.LexLess(*args)
        elif constraint_name == "LexLessEq":
            m += cp.LexLessEq(*args)
        elif constraint_name == "AllDifferentExceptN":
            m += cp.AllDifferentExceptN(*args)
        elif constraint_name == "AllEqualExceptN":
            m += cp.AllEqualExceptN(*args)
        elif constraint_name == "Precedence":
            m += cp.Precedence(*args)
        elif constraint_name == "Table":
            m += cp.Table(*args)
        elif constraint_name == "ShortTable":
            m += cp.ShortTable(*args)
        elif constraint_name == "NegativeTable":
            m += cp.NegativeTable(*args)
        elif constraint_name == "InDomain":
            m += cp.InDomain(*args)
        elif constraint_name == "Xor":
            m += cp.Xor(*args)
        elif constraint_name == "IfThenElse":
            m += cp.IfThenElse(*args)
        elif constraint_name == "Inverse":
            m += cp.Inverse(*args)
        elif constraint_name == "NoOverlap":
            m += cp.NoOverlap(*args)
        elif constraint_name == "GlobalCardinalityCount":
            m += cp.GlobalCardinalityCount(*args)
        elif constraint_name == "Cumulative":
            m += cp.Cumulative(*args)
        elif constraint_name == "LexChainLess":
            m += cp.LexChainLess(*args)
        elif constraint_name == "LexChainLessEq":
            m += cp.LexChainLessEq(*args)
        
        # Solve the model
        result = m.solve(solver=solver)
        
        # Print results
        print(f"Result: {result}")
        
        # Update statistics for successful run
        stats["success"] += 1
        stats["by_constraint"][constraint_name]["success"] += 1
        stats["by_solver"][solver]["success"] += 1
        
        return True
    
    except CPMpyException as e:
        print(f"Error with {constraint_name} using {solver}: {e}")
        
        # Update statistics for CPMpyException
        stats["cpmpy_exception"] += 1
        stats["by_constraint"][constraint_name]["cpmpy_exception"] += 1
        stats["by_solver"][solver]["cpmpy_exception"] += 1
        
        return False
    
    except Exception as e:
        print(f"Unexpected error with {constraint_name} using {solver}: {e}")
        traceback.print_exc()
        
        # Update statistics for other errors
        stats["other_error"] += 1
        stats["by_constraint"][constraint_name]["other_error"] += 1
        stats["by_solver"][solver]["other_error"] += 1
        
        return False

def test_single_list_constraints():
    """Test constraints that take a single list argument"""
    print("\n=== Testing Single List Constraints ===")
    
    # Test with different list arguments
    list_args_combinations = [
        iv_arr1,
        vals1,
        mixed_arr1
    ]
    
    for constraint in test_globals_single_list:
        for args in list_args_combinations:
            for solver in solvers:
                test_constraint(constraint, [args], solver)

def test_two_lists_constraints():
    """Test constraints that take two list arguments"""
    print("\n=== Testing Two Lists Constraints ===")
    
    # Test with different combinations of list arguments
    list_args_combinations = [
        (iv_arr1, vals1),
        (vals1, mixed_arr1),
        (mixed_arr1, mixed_arr2)
    ]
    
    for constraint in test_globals_two_lists:
        for args in list_args_combinations:
            for solver in solvers:
                test_constraint(constraint, args, solver)

def test_table_constraints():
    """Test table constraints"""
    print("\n=== Testing Table Constraints ===")
    
    # Table constraints typically take variables and a table of allowed tuples
    table_args_combinations = [
        ([iv_arr1[0], iv_arr1[1], iv_arr1[2]], table_data),
        ([x, y, z], table_data)
    ]
    
    for constraint in test_globals_table:
        for args in table_args_combinations:
            for solver in solvers:
                test_constraint(constraint, args, solver)

def test_domain_constraints():
    """Test domain constraints"""
    print("\n=== Testing Domain Constraints ===")
    
    # Domain constraints typically take a variable and a list of allowed values
    domain_args_combinations = [
        (x, domain_values),
        (iv_arr1[0], domain_values)
    ]
    
    for constraint in test_globals_domain:
        for args in domain_args_combinations:
            for solver in solvers:
                test_constraint(constraint, args, solver)

def test_boolean_constraints():
    """Test boolean constraints"""
    print("\n=== Testing Boolean Constraints ===")
    
    # Boolean constraints take boolean variables
    boolean_args_combinations = [
        (b1, b2, b3)  # For IfThenElse
    ]
    
    for constraint in test_globals_boolean:
        if constraint == "Xor":
            for solver in solvers:
                test_constraint(constraint, boolean_args_combinations, solver)
        elif constraint == "IfThenElse":
            for solver in solvers:
                test_constraint(constraint, boolean_args_combinations[0], solver)

def test_multi_list_constraints():
    """Test constraints that take multiple list arguments"""
    print("\n=== Testing Multi-List Constraints ===")
    
    # Different combinations for different multi-list constraints
    inverse_args = (iv_arr1, iv_arr2)
    no_overlap_args = (iv_arr1, iv_arr2, [10, 10, 10, 10, 10])  # Sizes
    gcc_args = (iv_arr1, vals1, iv_arr2)
    pos_int = cp.cpm_array([1, 2, 3, 4, 5])
    pos_vars = cp.intvar(1, 5, shape=5)
    cumulative_args = [(iv_arr1, pos_vars, iv_arr3, iv_arr4, 5), 
                       (iv_arr1, pos_int, iv_arr3, iv_arr4, 5), 
                       (pos_int, pos_vars, iv_arr3, iv_arr4, 5),
                       (iv_arr1, pos_vars, pos_int, iv_arr4, 5),
                       (iv_arr1, pos_vars, iv_arr3, pos_int, 5),
                       (iv_arr1, pos_vars, iv_arr3, iv_arr4, x)]
    
    for constraint in test_globals_multi_list:
        if constraint == "Inverse":
            for solver in solvers:
                test_constraint(constraint, inverse_args, solver)
        elif constraint == "NoOverlap":
            for solver in solvers:
                test_constraint(constraint, no_overlap_args, solver)
        elif constraint == "GlobalCardinalityCount":
            for solver in solvers:
                test_constraint(constraint, gcc_args, solver)
        elif constraint == "Cumulative":
            for args in cumulative_args:
                for solver in solvers:
                    test_constraint(constraint, args, solver)

def test_2d_list_constraints():
    """Test constraints that take 2D list arguments"""
    print("\n=== Testing 2D List Constraints ===")
    
    # 2D list constraints take 2D arrays
    arr_2d_args_combinations = [[arr_2d1], [arr_2d2]]

    
    for constraint in test_globals_2d_list:
        for args in arr_2d_args_combinations:
            for solver in solvers:
                test_constraint(constraint, args, solver)

def print_statistics():
    """Print statistics about the test runs"""
    print("\n=== TEST STATISTICS ===")
    print(f"Total runs: {stats['success'] + stats['cpmpy_exception'] + stats['other_error']}")
    print(f"Successful runs: {stats['success']}")
    print(f"CPMpyException errors: {stats['cpmpy_exception']}")
    print(f"Other errors: {stats['other_error']}")
    
    print("\n=== STATISTICS BY CONSTRAINT ===")
    for constraint, constraint_stats in sorted(stats["by_constraint"].items()):
        total = constraint_stats["success"] + constraint_stats["cpmpy_exception"] + constraint_stats["other_error"]
        success_rate = (constraint_stats["success"] / total * 100) if total > 0 else 0
        print(f"{constraint}: {constraint_stats['success']}/{total} successful ({success_rate:.1f}%)")
    
    print("\n=== STATISTICS BY SOLVER ===")
    for solver, solver_stats in sorted(stats["by_solver"].items()):
        total = solver_stats["success"] + solver_stats["cpmpy_exception"] + solver_stats["other_error"]
        success_rate = (solver_stats["success"] / total * 100) if total > 0 else 0
        print(f"{solver}: {solver_stats['success']}/{total} successful ({success_rate:.1f}%)")

# Run all tests
if __name__ == "__main__":
    test_single_list_constraints()
    test_two_lists_constraints()
    test_table_constraints()
    test_domain_constraints()
    test_boolean_constraints()
    test_multi_list_constraints()
    test_2d_list_constraints()
    
    # Print statistics at the end
    print_statistics()