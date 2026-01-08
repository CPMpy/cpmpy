"""
Example usage of the Nurserostering Dataset from schedulingbenchmarks.org

This example demonstrates how to use the dataset loader to parse and solve
nurserostering instances.

https://schedulingbenchmarks.org/nrp/
"""

from cpmpy.tools.dataset.problem.nurserostering import (
    NurseRosteringDataset,
    parse_scheduling_period,
    nurserostering_model,
    to_dataframes
)

try:
    from natsort import natsorted
    sort_key = natsorted
except ImportError:
    sort_key = None  # Use default sorted()

try:
    import pandas as pd
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 5000)
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

if __name__ == "__main__":
    # Example 1: Basic usage with native data structures
    dataset = NurseRosteringDataset(root=".", download=True, transform=parse_scheduling_period, sort_key=sort_key)
    print("Dataset size:", len(dataset))
    data, metadata = dataset[0]

    print(f"Instance: {metadata['name']}")
    print(f"Horizon: {data['horizon']} days")
    print(f"Number of nurses: {len(data['staff'])}")
    print(f"Number of shifts: {len(data['shifts'])}")

    # Solve the model
    model, nurse_view = nurserostering_model(**data)
    assert model.solve()

    print(f"\nFound optimal solution with penalty of {model.objective_value()}")
    assert model.objective_value() == 607  # optimal solution for the first instance

    # Pretty print solution (native Python, no pandas required)
    horizon = data['horizon']
    shift_ids = list(data['shifts'].keys())
    names = ["-"] + shift_ids
    sol = nurse_view.value()
    
    # Create table: rows are nurses + cover rows, columns are days
    table = []
    row_labels = []
    
    # Add nurse rows
    for i, nurse in enumerate(data['staff']):
        nurse_name = nurse.get('name', nurse.get('ID', f'Nurse_{i}'))
        row_labels.append(nurse_name)
        table.append([names[sol[i][d]] for d in range(horizon)])
    
    # Add cover rows (initialize with empty strings)
    for shift_id in shift_ids:
        row_labels.append(f'Cover {shift_id}')
        table.append([''] * horizon)
    
    # Fill in cover information
    for cover_request in data['cover']:
        shift = cover_request['ShiftID']
        day = cover_request['Day']
        requirement = cover_request['Requirement']
        # Count how many nurses are assigned to this shift on this day
        num_shifts = sum(1 for i in range(len(data['staff'])) 
                        if sol[i][day] == shift_ids.index(shift) + 1)  # +1 because 0 is FREE
        cover_row_idx = len(data['staff']) + shift_ids.index(shift)
        table[cover_row_idx][day] = f"{num_shifts}/{requirement}"
    
    # Print table
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_labels = [days[d % 7] for d in range(horizon)]
    
    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in table + [day_labels]) for i in range(horizon)]
    row_label_width = max(len(label) for label in row_labels)
    
    # Print header
    print(f"\n{'Schedule:':<{row_label_width}}", end="")
    for d, day_label in enumerate(day_labels):
        print(f" {day_label:>{col_widths[d]}}", end="")
    print()
    
    # Print separator
    print("-" * (row_label_width + 1 + sum(w + 1 for w in col_widths)))
    
    # Print rows
    for label, row in zip(row_labels, table):
        print(f"{label:<{row_label_width}}", end="")
        for d, val in enumerate(row):
            print(f" {str(val):>{col_widths[d]}}", end="")
        print()
    
    # Example 2: Using pandas DataFrames (optional)
    if HAS_PANDAS:
        print("\n" + "="*60)
        print("Example with pandas DataFrames:")
        print("="*60)
        
        def parse_with_dataframes(fname):
            return to_dataframes(parse_scheduling_period(fname))
        
        dataset_df = NurseRosteringDataset(root=".", download=False, transform=parse_with_dataframes, sort_key=sort_key)
        data_df, _ = dataset_df[0]
        
        print("\nStaff DataFrame:")
        print(data_df['staff'].head())
        
        print("\nShifts DataFrame:")
        print(data_df['shifts'])
        
        print("\nCover DataFrame:")
        print(data_df['cover'].head())
