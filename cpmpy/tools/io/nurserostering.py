#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## nurserostering.py
##
"""
Parser for the Nurse Rostering format.


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_nurserostering
"""


import os
import sys
import argparse
import tempfile
import cpmpy as cp
import re
from typing import Union, Callable, Optional, Any

from cpmpy.expressions.variables import NDVarArray

# Optional dependencies
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    from faker import Faker
    _HAS_FAKER = True
except ImportError:
    _HAS_FAKER = False

def _tag_to_data(
        string: str, 
        tag: str, 
        skip_lines: int=0, 
        datatype: Optional[type]=None, 
        names: Optional[list[str]]=None, 
        dtype: Optional[dict[str, type]]=None
    ) -> Union[int, float, str, list[dict[str, Any]], None]:
    """
    Extract data from a tagged section in the input string.
    
    Arguments:
        string (str): Input string containing tagged sections
        tag (str): Tag name to search for (e.g., "SECTION_SHIFTS")
        skip_lines (int): Number of lines to skip after the tag
        datatype (Optional[type]): Type hint for return value. If None, returns list of dicts (CSV rows).
                  If int, str, etc., returns that type parsed from first line.
        names (Optional[list[str]]): Optional list of column names to rename headers to. If provided, must match
               the number of columns or be shorter (extra columns will keep original names).
        dtype (Optional[dict[str, type]]): Optional dict mapping column names to data types for conversion.
               Example: {'Length': int, 'ShiftID': str}
    
    Returns:
        Optional[list[dict[str, Any]]]: 
            If datatype is None: list of dicts (CSV rows as dictionaries)
            If datatype is int, str, etc.: parsed value from first line
    """
    regex = rf'{tag}[\s\S]*?($|(?=\n\s*\n))'
    match = re.search(regex, string)
    
    if not match:
        return None
    
    lines = list(match.group().split("\n")[skip_lines+1:])
    if not lines:
        return None
    
    # If datatype is a simple type (int, str, etc.), parse accordingly
    if datatype is not None and datatype not in (list, dict):
        if datatype is int or datatype is float:
            # For numeric types, return first line
            first_line = lines[0].strip()
            return datatype(first_line) if first_line else None
        elif datatype is str:
            # For string type, return the whole data section
            return "\n".join(lines).strip()
            
    # Parse header
    headers = lines[0].split(",")
    # Clean headers: remove # and strip whitespace, but keep exact names
    headers = [h.replace("#", "").strip() for h in headers]
    
    # Rename columns if names provided
    if names is not None:
        for i, new_name in enumerate(names):
            if i < len(headers):
                headers[i] = new_name
    
    # Parse data rows
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split(",")
        # Pad values if needed
        while len(values) < len(headers):
            values.append("")
        row = {}
        for i in range(len(headers)):
            value = values[i].strip() if i < len(values) else ""
            col_name = headers[i]
            
            # Apply type conversion if dtype specified
            if dtype is not None and col_name in dtype:
                target_type = dtype[col_name]
                row[col_name] = target_type(value) if value else None
            else:
                row[col_name] = value
        rows.append(row)
    
    return rows

def parse_scheduling_period(filename: str):
    """
    Parse a nurserostering instance file.
    
    Arguments:
        filename (str): Path to the nurserostering instance file.
    
    Returns:
        dict: A dictionary with native Python data structures (lists of dicts).

    Raises:
        ValueError: If the file is not found.

    Note:
        - Use to_dataframes() transform to convert to pandas DataFrames if needed.
        - Use add_fake_names() transform to add randomly generated names to staff.
    """
    with open(filename, "r") as f:
        string = f.read()

    # Parse scheduling horizon
    horizon_val = _tag_to_data(string, "SECTION_HORIZON", skip_lines=2, datatype=int)
    if not isinstance(horizon_val, int):
        raise ValueError("Missing SECTION_HORIZON in nurserostering instance")
    horizon = horizon_val
    
    # Parse shifts - list of dicts with ShiftID as key
    shifts_rows = _tag_to_data(string, "SECTION_SHIFTS",
                               names=["ShiftID", "Length", "cannot follow"],
                               dtype={'ShiftID': str, 'Length': int, 'cannot follow': str})
    shifts = {}
    if isinstance(shifts_rows, list):
        for row in shifts_rows:
            cannot_follow_str = row.get("cannot follow") or ""
            shifts[row["ShiftID"]] = {
                "Length": row["Length"],
                "cannot follow": [v.strip() for v in cannot_follow_str.split("|") if v.strip()]
            }
    
    # Parse staff - list of dicts
    staff_rows = _tag_to_data(string, "SECTION_STAFF", 
                         names=["ID", "MaxShifts", "MaxTotalMinutes", "MinTotalMinutes", "MaxConsecutiveShifts", "MinConsecutiveShifts", "MinConsecutiveDaysOff", "MaxWeekends"],
                         dtype={'MaxShifts': str, 'MaxTotalMinutes': int, 'MinTotalMinutes': int, 'MaxConsecutiveShifts': int, 'MinConsecutiveShifts': int, 'MinConsecutiveDaysOff': int, 'MaxWeekends': int})
    staff: list[dict[str, Any]] = staff_rows if isinstance(staff_rows, list) else []
    
    # Process MaxShifts column - split by | and create max_shifts_* columns
    for idx, nurse in enumerate(staff):
        max_shifts_str = nurse.get("MaxShifts", "").strip()
        if max_shifts_str:
            max_shift_parts = max_shifts_str.split("|")
            for part in max_shift_parts:
                if "=" in part:
                    shift_id, max_val = part.split("=", 1)
                    shift_id = shift_id.strip()
                    max_val = max_val.strip()
                    if shift_id and max_val:
                        nurse[f"max_shifts_{shift_id}"] = int(max_val)
    
    # Parse days off - this section has variable columns (EmployeeID + N day indices)
    # Parse as raw string since column count varies per row
    days_off_raw = _tag_to_data(string, "SECTION_DAYS_OFF", datatype=str)
    days_off = []
    if isinstance(days_off_raw, str):
        for line in days_off_raw.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("employeeid"):
                continue
            # Parse CSV-style line (handles variable number of columns)
            parts = line.split(",")
            if len(parts) > 0:
                employee_id = parts[0].strip()
                # Remaining parts are day indices
                for day_str in parts[1:]:
                    day_str = day_str.strip()
                    if day_str and day_str.isdigit():
                        day_idx = int(day_str)
                        if 0 <= day_idx < horizon:
                            days_off.append({"EmployeeID": employee_id, "DayIndex": day_idx})
    
    # Parse shift requests
    shift_on_rows = _tag_to_data(string, "SECTION_SHIFT_ON_REQUESTS",
                            names=["EmployeeID", "Day", "ShiftID", "Weight"],
                            dtype={'Weight': int, "Day": int, "ShiftID": str})
    shift_off_rows = _tag_to_data(string, "SECTION_SHIFT_OFF_REQUESTS",
                            names=["EmployeeID", "Day", "ShiftID", "Weight"],
                            dtype={'Weight': int, "Day": int, "ShiftID": str})
    cover_rows = _tag_to_data(string, "SECTION_COVER",
                         names=["Day", "ShiftID", "Requirement", "Weight for under", "Weight for over"],
                         dtype={'Day': int, 'ShiftID': str, 'Requirement': int, 'Weight for under': int, 'Weight for over': int})

    return dict(horizon=horizon, shifts=shifts, staff=staff, days_off=days_off, 
                shift_on=shift_on_rows if isinstance(shift_on_rows, list) else [],
                shift_off=shift_off_rows if isinstance(shift_off_rows, list) else [],
                cover=cover_rows if isinstance(cover_rows, list) else [])


def add_fake_names(data: dict[str, Any], seed: int=0) -> dict[str, Any]:
    """
    Transform function to add randomly generated names to staff using Faker.
    
    This function can be used as a transform argument to NurseRosteringDataset
    to add fake names to the parsed data.
    
    Example:
        dataset = NurseRosteringDataset(
            root=".", 
            transform=lambda fname: add_fake_names(parse_scheduling_period(fname))
        )
    
    Or combine with other transforms:
        dataset = NurseRosteringDataset(
            root=".", 
            transform=lambda fname: to_dataframes(
                add_fake_names(parse_scheduling_period(fname))
            )
        )
    
    Arguments:
        data (dict[str, Any]): Dictionary returned by parse_scheduling_period()
        seed (int): Random seed for reproducible name generation (default: 0)
    
    Returns:
        dict[str, Any]: Dictionary with 'name' field added to each staff member
    
    Raises:
        ImportError: If Faker is not installed
    """
    if not _HAS_FAKER:
        raise ImportError("Faker is required for add_fake_names(). Install it with: pip install faker")
    
    fake = Faker()
    fake.seed_instance(seed)
    
    # Add names to staff
    for idx, nurse in enumerate(data["staff"]):
        nurse["name"] = fake.unique.first_name()
    
    return data


def to_dataframes(data: dict[str, Any]) -> dict[str, Any]:
    """
    Transform function to convert native data structures to pandas DataFrames.
    
    This function can be used as a transform argument to NurseRosteringDataset
    to convert the parsed data into pandas DataFrames for easier manipulation.
    
    Example:
        dataset = NurseRosteringDataset(
            root=".", 
            transform=lambda fname: to_dataframes(parse_scheduling_period(fname))
        )
    
    Arguments:
        data (dict[str, Any]): Dictionary returned by parse_scheduling_period()
    
    Returns:
        dict[str, Any]: Dictionary with pandas DataFrames instead of native structures
    
    Raises:
        ImportError: If pandas is not installed
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas is required for to_dataframes(). Install it with: pip install pandas")
    
    result = {"horizon": data["horizon"]}
    
    # Convert shifts dict to DataFrame
    shifts_rows = []
    for shift_id, shift_data in data["shifts"].items():
        row = {"ShiftID": shift_id, "Length": shift_data["Length"], 
               "cannot follow": "|".join(shift_data["cannot follow"])}
        shifts_rows.append(row)
    result["shifts"] = pd.DataFrame(shifts_rows).set_index("ShiftID")
    
    # Convert staff list to DataFrame
    result["staff"] = pd.DataFrame(data["staff"]).set_index("ID")
    
    # Convert days_off list to DataFrame
    result["days_off"] = pd.DataFrame(data["days_off"])
    
    # Convert shift_on, shift_off, cover lists to DataFrames
    result["shift_on"] = pd.DataFrame(data["shift_on"])
    result["shift_off"] = pd.DataFrame(data["shift_off"])
    result["cover"] = pd.DataFrame(data["cover"])
    
    return result


def model_nurserostering(
        horizon: int, 
        shifts: dict[str, dict[str, Any]], 
        staff: list[dict[str, Any]], 
        days_off: list[dict[str, Any]], 
        shift_on: list[dict[str, Any]], 
        shift_off: list[dict[str, Any]], 
        cover: list[dict[str, Any]]
    ) -> tuple[cp.Model, NDVarArray]:
    """
    Create a CPMpy model for nurserostering.
    
    Arguments:
        horizon (int): Number of days in the scheduling period
        shifts (dict[str, dict[str, Any]]): Dict mapping shift_id to dict with shift data
        staff (list[dict[str, Any]]): List of dicts, each representing a nurse with their constraints
        days_off (list[dict[str, Any]]): List of dicts with days off for each nurse
        shift_on (list[dict[str, Any]]): List of dicts with shift-on requests for each nurse
        shift_off (list[dict[str, Any]]): List of dicts with shift-off requests for each nurse
        cover (list[dict[str, Any]]): List of dicts with cover requirements for each day and shift

    Returns:
        tuple[cp.Model, NDVarArray]: A tuple containing the CPMpy model and the nurse view.
            - model (cp.Model): The CPMpy model for the nurserostering problem.
            - nurse_view (NDVarArray): The nurse view variable.
    """
    n_nurses = len(staff)

    FREE = 0
    shift_ids = list(shifts.keys())
    SHIFTS = ["F"] + shift_ids

    nurse_view = cp.intvar(0, len(shifts), shape=(n_nurses, horizon), name="nv")

    model = cp.Model()

    # Shifts which cannot follow the shift on the previous day.
    for shift_id, shift_data in shifts.items():
        for other_shift in shift_data['cannot follow']:
            model += (nurse_view[:,:-1] == SHIFTS.index(shift_id)).implies(
                nurse_view[:,1:] != SHIFTS.index(other_shift))

    # Maximum number of shifts of each type that can be assigned to each employee.
    for i, nurse in enumerate(staff):
        for shift_id in shift_ids:
            max_shifts = nurse[f"max_shifts_{shift_id}"]
            model += cp.Count(nurse_view[i], SHIFTS.index(shift_id)) <= max_shifts

    # Minimum and maximum amount of total time in minutes that can be assigned to each employee.
    shift_length = cp.cpm_array([0] + [shifts[sid]['Length'] for sid in shift_ids])  # FREE = length 0
    for i, nurse in enumerate(staff):
        time_worked = cp.sum(shift_length[nurse_view[i,d]] for d in range(horizon))
        model += time_worked <= nurse.get('MaxTotalMinutes')
        model += time_worked >= nurse.get('MinTotalMinutes')

    # Maximum number of consecutive shifts that can be worked before having a day off.
    for i, nurse in enumerate(staff):
        max_days = nurse.get('MaxConsecutiveShifts')
        if max_days is not None:
            for d in range(horizon - max_days):
                window = nurse_view[i,d:d+max_days+1]
                model += cp.Count(window, FREE) >= 1  # at least one holiday in this window

    # Minimum number of consecutive shifts that must be worked before having a day off.
    for i, nurse in enumerate(staff):
        min_days = nurse.get('MinConsecutiveShifts')
        if min_days is not None:
            for d in range(1, horizon):
                is_start_of_working_period = (nurse_view[i, d-1] == FREE) & (nurse_view[i, d] != FREE)
                model += is_start_of_working_period.implies(cp.all(nurse_view[i,d:d+min_days] != FREE))

    # Minimum number of consecutive days off.
    for i, nurse in enumerate(staff):
        min_days = nurse.get('MinConsecutiveDaysOff')
        if min_days is not None:
            for d in range(1, horizon):
                is_start_of_free_period = (nurse_view[i, d - 1] != FREE) & (nurse_view[i, d] == FREE)
                model += is_start_of_free_period.implies(cp.all(nurse_view[i, d:d + min_days] == FREE))

    # Max number of working weekends for each nurse
    weekends = [(i - 1, i) for i in range(1, horizon) if (i + 1) % 7 == 0]
    for i, nurse in enumerate(staff):
        n_weekends = cp.sum((nurse_view[i,sat] != FREE) | (nurse_view[i,sun] != FREE) for sat,sun in weekends)
        model += n_weekends <= nurse.get('MaxWeekends')

    # Days off
    for holiday in days_off:
        nurse_idx = next((idx for idx, nurse in enumerate(staff) if nurse['ID'] == holiday['EmployeeID']), None) # index of employee
        if nurse_idx is not None:
            model += nurse_view[nurse_idx,holiday['DayIndex']] == FREE

    # Shift requests, encode in linear objective
    objective = 0
    for request in shift_on or []:
        nurse_idx = next((idx for idx, nurse in enumerate(staff) if nurse['ID'] == request['EmployeeID']), None) # index of employee
        if nurse_idx is not None:
            cpm_request = nurse_view[nurse_idx, request['Day']] == SHIFTS.index(request['ShiftID'])
            objective += request['Weight'] * ~cpm_request

    # Shift off requests, encode in linear objective
    for request in shift_off or []:
        nurse_idx = next((idx for idx, nurse in enumerate(staff) if nurse['ID'] == request['EmployeeID']), None) # index of employee
        if nurse_idx is not None:
            cpm_request = nurse_view[nurse_idx, request['Day']] != SHIFTS.index(request['ShiftID'])
            objective += request['Weight'] * ~cpm_request

    # Cover constraints, encode in objective with slack variables
    for cover_request in cover:
        nb_nurses = cp.Count(nurse_view[:, cover_request['Day']], SHIFTS.index(cover_request['ShiftID']))
        slack_over, slack_under = cp.intvar(0, len(staff), shape=2)
        model += nb_nurses - slack_over + slack_under == cover_request["Requirement"]
        objective += cover_request["Weight for over"] * slack_over + cover_request["Weight for under"] * slack_under

    model.minimize(objective)

    return model, nurse_view


_std_open = open
def load_nurserostering(instance: Union[str, os.PathLike], open:Callable=open) -> cp.Model:
    """
    Loader for Nurse Rostering format. Loads an instance and returns its matching CPMpy model.

    Arguments: 
        instance (str or os.PathLike):
            - A file path to a Nurse Rostering file
            - OR a string containing the Nurse Rostering content directly
        open (Callable):
            If instance is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        cp.Model: The CPMpy model of the Nurse Rostering instance.
    """
    # If instance is a path to a file that exists -> use it directly
    if isinstance(instance, (str, os.PathLike)) and os.path.exists(instance):
        fname = os.fspath(instance)
    # If instance is a string containing file content -> write to temp file
    else:
        # Create a temporary file and write the content
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp.write(str(instance))
            fname = tmp.name

    try:
        # Use the existing parser from the dataset (expects a file path)
        data = parse_scheduling_period(fname)
        
        # Create the CPMpy model using the existing model builder
        model, _ = model_nurserostering(**data)
        
        return model
    finally:
        # Clean up temporary file if we created one
        if isinstance(instance, str) and not os.path.exists(instance) and os.path.exists(fname):
            os.unlink(fname)


def main():
    parser = argparse.ArgumentParser(description="Parse and solve a Nurse Rostering model using CPMpy")
    parser.add_argument("model", help="Path to a Nurse Rostering file (or raw content string if --string is given)")
    parser.add_argument("-s", "--solver", default=None, help="Solver name to use (default: CPMpy's default)")
    parser.add_argument("--string", action="store_true", help="Interpret the first argument (model) as a raw Nurse Rostering string instead of a file path")
    parser.add_argument("-t", "--time-limit", type=int, default=None, help="Time limit for the solver in seconds (default: no limit)")
    args = parser.parse_args()

    # Build the CPMpy model
    try:
        if args.string:
            model = load_nurserostering(args.model)
        else:
            model = load_nurserostering(os.path.expanduser(args.model))
    except Exception as e:
        sys.stderr.write(f"Error reading model: {e}\n")
        sys.exit(1)

    # Solve the model
    try:
        if args.solver:
            result = model.solve(solver=args.solver, time_limit=args.time_limit)
        else:
            result = model.solve(time_limit=args.time_limit)
    except Exception as e:
        sys.stderr.write(f"Error solving model: {e}\n")
        sys.exit(1)

    # Print results
    print("Status:", model.status())
    if result is not None:
        if model.has_objective():
            print("Objective:", model.objective_value())
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()

