"""
PyTorch-style Dataset for Nurserostering instances from schedulingbenchmarks.org

Simply create a dataset instance and start iterating over its contents:
The `metadata` contains usefull information about the current problem instance.

https://schedulingbenchmarks.org/nrp/
"""
import os
import pathlib
from typing import Tuple, Any
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import zipfile
import re

import cpmpy as cp

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


class NurseRosteringDataset(object):  # torch.utils.data.Dataset compatible

    """
    Nurserostering Dataset in a PyTorch compatible format.

    More information on nurserostering instances can be found here: https://schedulingbenchmarks.org/nrp/
    """

    def __init__(self, root: str = ".", transform=None, target_transform=None, download: bool = False, sort_key=None):
        """
        Initialize the Nurserostering Dataset.

        Arguments:
            root (str): Root directory containing the nurserostering instances (if 'download', instances will be downloaded to this location)
            transform (callable, optional): Optional transform to be applied on the instance data
            target_transform (callable, optional): Optional transform to be applied on the file path
            download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
            sort_key (callable, optional): Optional function to sort instance files. If None, uses Python's built-in sorted().
                                          For natural/numeric sorting, pass natsorted from natsort library.
                                          Example: from natsort import natsorted; dataset = NurseRosteringDataset(..., sort_key=natsorted)
        """

        self.root = pathlib.Path(root)
        self.instance_dir = pathlib.Path(os.path.join(self.root, "nurserostering"))
        self.transform = transform
        self.target_transform = target_transform
        self.sort_key = sorted if sort_key is None else sort_key

        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)

        if not self.instance_dir.exists():
            if not download:
                raise ValueError(f"Dataset not found in local file system. Please set download=True to download the dataset.")
            else:
                url = f"https://schedulingbenchmarks.org/nrp/data/instances1_24.zip" # download full repo...
                zip_path = pathlib.Path(os.path.join(root,"jsplib-master.zip"))

                print(f"Downloading Nurserostering instances from schedulingbenchmarks.org")

                try:
                    urlretrieve(url, str(zip_path))
                except (HTTPError, URLError) as e:
                    raise ValueError(f"No dataset available on {url}. Error: {str(e)}")

                # make directory and extract files
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    self.instance_dir.mkdir(parents=True, exist_ok=True)

                    # Extract files
                    for file_info in zip_ref.infolist():
                            filename = pathlib.Path(file_info.filename).name
                            with zip_ref.open(file_info) as source, open(self.instance_dir / filename, 'wb') as target:
                                target.write(source.read())

                 # Clean up the zip file
                zip_path.unlink()


    def __len__(self) -> int:
        """Return the total number of instances."""
        return len(list(self.instance_dir.glob("*.txt")))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a single Nurserostering instance filename and metadata.

        Args:
            index (int): Index of the instance to retrieve

        Returns:
            Tuple[Any, Any]: A tuple containing:
                - The filename of the instance
                - Metadata dictionary with file name, track, year etc.
        """
        if isinstance(index, int) and not (0 <= index < len(self)):
            raise IndexError("Index out of range")

        # Get all instance files and sort for deterministic behavior
        files = self.sort_key(list(self.instance_dir.glob("*.txt"))) # use .txt files instead of xml files
        file_path = files[index]

        filename = str(file_path)
        if self.transform:
            # user might want to process the filename to something else
            filename = self.transform(filename)

        metadata = dict(name=file_path.stem)

        if self.target_transform:
            metadata = self.target_transform(metadata)

        return filename, metadata

    def open(self, instance: os.PathLike) -> callable:
        return open(instance, "r")


def _tag_to_data(string, tag, skip_lines=0, datatype=None, names=None, dtype=None):
    """
    Extract data from a tagged section in the input string.
    
    Args:
        string: Input string containing tagged sections
        tag: Tag name to search for (e.g., "SECTION_SHIFTS")
        skip_lines: Number of lines to skip after the tag
        datatype: Type hint for return value. If None, returns list of dicts (CSV rows).
                  If int, str, etc., returns that type parsed from first line.
        names: Optional list of column names to rename headers to. If provided, must match
               the number of columns or be shorter (extra columns will keep original names).
        dtype: Optional dict mapping column names to data types for conversion.
               Example: {'Length': int, 'ShiftID': str}
    
    Returns:
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
    
    Args:
        filename: Path to the nurserostering instance file.
    
    Returns a dictionary with native Python data structures (lists of dicts).
    Use to_dataframes() transform to convert to pandas DataFrames if needed.
    Use add_fake_names() transform to add randomly generated names to staff.
    """
    with open(filename, "r") as f:
        string = f.read()

    # Parse scheduling horizon
    horizon = int(_tag_to_data(string, "SECTION_HORIZON", skip_lines=2, datatype=int))
    
    # Parse shifts - list of dicts with ShiftID as key
    shifts_rows = _tag_to_data(string, "SECTION_SHIFTS",
                               names=["ShiftID", "Length", "cannot follow"],
                               dtype={'ShiftID': str, 'Length': int, 'cannot follow': str})
    shifts = {}
    for row in shifts_rows:
        cannot_follow_str = row.get("cannot follow") or ""
        shifts[row["ShiftID"]] = {
            "Length": row["Length"],
            "cannot follow": [v.strip() for v in cannot_follow_str.split("|") if v.strip()]
        }
    
    # Parse staff - list of dicts
    staff = _tag_to_data(string, "SECTION_STAFF", 
                         names=["ID", "MaxShifts", "MaxTotalMinutes", "MinTotalMinutes", "MaxConsecutiveShifts", "MinConsecutiveShifts", "MinConsecutiveDaysOff", "MaxWeekends"],
                         dtype={'MaxShifts': str, 'MaxTotalMinutes': int, 'MinTotalMinutes': int, 'MaxConsecutiveShifts': int, 'MinConsecutiveShifts': int, 'MinConsecutiveDaysOff': int, 'MaxWeekends': int})
    
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
    if days_off_raw:
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
    shift_on = _tag_to_data(string, "SECTION_SHIFT_ON_REQUESTS",
                            names=["EmployeeID", "Day", "ShiftID", "Weight"],
                            dtype={'Weight': int, "Day": int, "ShiftID": str})
    shift_off = _tag_to_data(string, "SECTION_SHIFT_OFF_REQUESTS",
                            names=["EmployeeID", "Day", "ShiftID", "Weight"],
                            dtype={'Weight': int, "Day": int, "ShiftID": str})
    cover = _tag_to_data(string, "SECTION_COVER",
                         names=["Day", "ShiftID", "Requirement", "Weight for under", "Weight for over"],
                         dtype={'Day': int, 'ShiftID': str, 'Requirement': int, 'Weight for under': int, 'Weight for over': int})

    return dict(horizon=horizon, shifts=shifts, staff=staff, days_off=days_off, 
                shift_on=shift_on, shift_off=shift_off, cover=cover)


def add_fake_names(data, seed=0):
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
    
    Args:
        data: Dictionary returned by parse_scheduling_period()
        seed: Random seed for reproducible name generation (default: 0)
    
    Returns:
        Dictionary with 'name' field added to each staff member
    
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


def to_dataframes(data):
    """
    Transform function to convert native data structures to pandas DataFrames.
    
    This function can be used as a transform argument to NurseRosteringDataset
    to convert the parsed data into pandas DataFrames for easier manipulation.
    
    Example:
        dataset = NurseRosteringDataset(
            root=".", 
            transform=lambda fname: to_dataframes(parse_scheduling_period(fname))
        )
    
    Args:
        data: Dictionary returned by parse_scheduling_period()
    
    Returns:
        Dictionary with pandas DataFrames instead of native structures
    
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


def nurserostering_model(horizon, shifts, staff, days_off, shift_on, shift_off, cover):
    """
    Create a CPMpy model for nurserostering.
    
    Args:
        horizon: Number of days in the scheduling period
        shifts: Dict mapping shift_id to dict with shift data
        staff: List of dicts, each representing a nurse with their constraints
        days_off: List of dicts with days off for each nurse
        shift_on: List of dicts with shift-on requests for each nurse
        shift_off: List of dicts with shift-off requests for each nurse
        cover: List of dicts with cover requirements for each day and shift
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
        for d in range(horizon - max_days):
            window = nurse_view[i,d:d+max_days+1]
            model += cp.Count(window, FREE) >= 1  # at least one holiday in this window

    # Minimum number of consecutive shifts that must be worked before having a day off.
    for i, nurse in enumerate(staff):
        min_days = nurse.get('MinConsecutiveShifts')
        for d in range(1, horizon):
            is_start_of_working_period = (nurse_view[i, d-1] == FREE) & (nurse_view[i, d] != FREE)
            model += is_start_of_working_period.implies(cp.all(nurse_view[i,d:d+min_days] != FREE))

    # Minimum number of consecutive days off.
    for i, nurse in enumerate(staff):
        min_days = nurse.get('MinConsecutiveDaysOff')
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
        i = next((idx for idx, nurse in enumerate(staff) if nurse['ID'] == holiday['EmployeeID']), None) # index of employee
        model += nurse_view[i,holiday['DayIndex']] == FREE

    # Shift requests, encode in linear objective
    objective = 0
    for request in shift_on:
        i = next((idx for idx, nurse in enumerate(staff) if nurse['ID'] == request['EmployeeID']), None) # index of employee
        cpm_request = nurse_view[i, request['Day']] == SHIFTS.index(request['ShiftID'])
        objective += request['Weight'] * ~cpm_request

    # Shift off requests, encode in linear objective
    for request in shift_off:
        i = next((idx for idx, nurse in enumerate(staff) if nurse['ID'] == request['EmployeeID']), None) # index of employee
        cpm_request = nurse_view[i, request['Day']] != SHIFTS.index(request['ShiftID'])
        objective += request['Weight'] * ~cpm_request

    # Cover constraints, encode in objective with slack variables
    for cover_request in cover:
        nb_nurses = cp.Count(nurse_view[:, cover_request['Day']], SHIFTS.index(cover_request['ShiftID']))
        slack_over, slack_under = cp.intvar(0, len(staff), shape=2)
        model += nb_nurses - slack_over + slack_under == cover_request["Requirement"]
        objective += cover_request["Weight for over"] * slack_over + cover_request["Weight for under"] * slack_under

    model.minimize(objective)

    return model, nurse_view

if __name__ == "__main__":
    dataset = NurseRosteringDataset(root=".", download=True, transform=parse_scheduling_period)
    print("Dataset size:", len(dataset))
    
    data, metadata = dataset[0]
    print(data)

    model, nurse_view = nurserostering_model(**data)
    assert model.solve()

    print(f"Found optimal solution with penalty of {model.objective_value()}")
    assert model.objective_value() == 607 # optimal solution for the first instance

    # --- Pretty print solution without pandas ---
    
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
