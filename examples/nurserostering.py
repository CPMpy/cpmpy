"""
PyTorch-style Dataset for Nurserostering instances from schedulingbenchmarks.org

Simply create a dataset instance and start iterating over its contents:
The `metadata` contains usefull information about the current problem instance.
"""
import json
import pathlib
from io import StringIO
from os.path import join
from typing import Tuple, Any
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import zipfile

import faker
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from natsort import natsorted

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)




import cpmpy as cp

class NurseRosteringDataset(object):  # torch.utils.data.Dataset compatible

    """
    Nurserostering Dataset in a PyTorch compatible format.

    Arguments:
        root (str): Root directory containing the nurserostering instances (if 'download', instances will be downloaded to this location)
        transform (callable, optional): Optional transform to be applied on the instance data
        target_transform (callable, optional): Optional transform to be applied on the file path
        download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
    """

    def __init__(self, root: str = ".", transform=None, target_transform=None, download: bool = False):
        """
        Initialize the Nurserostering Dataset.
        """

        self.root = pathlib.Path(root)
        self.instance_dir = pathlib.Path(join(self.root, "nurserostering"))
        self.transform = transform
        self.target_transform = target_transform

        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)

        print(self.instance_dir, self.instance_dir.exists(), self.instance_dir.is_dir())
        if not self.instance_dir.exists():
            if not download:
                raise ValueError(f"Dataset not found in local file system. Please set download=True to download the dataset.")
            else:
                url = f"https://schedulingbenchmarks.org/nrp/data/instances1_24.zip" # download full repo...
                zip_path = pathlib.Path(join(root,"jsplib-master.zip"))

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
        return len(list(self.instance_dir.glob("*")))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a single RCPSP instance filename and metadata.

        Args:
            index (int or str): Index or name of the instance to retrieve

        Returns:
            Tuple[Any, Any]: A tuple containing:
                - The filename of the instance
                - Metadata dictionary with file name, track, year etc.
        """
        if isinstance(index, int) and (index < 0 or index >= len(self)):
            raise IndexError("Index out of range")

        # Get all instance files and sort for deterministic behavior # TODO: use natsort instead?
        files = natsorted(list(self.instance_dir.glob("*.txt"))) # use .txt files instead of xml files
        file_path = files[index]

        filename = str(file_path)
        if self.transform:
            # does not need to remain a filename...
            filename = self.transform(filename)

        metadata = dict(name=file_path.stem)

        if self.target_transform:
            metadata = self.target_transform(metadata)

        return filename, metadata


import re
def _tag_to_data(string, tag, skip_lines=0, datatype=pd.DataFrame, *args, **kwargs):

    regex = rf'{tag}[\s\S]*?($|(?=\n\s*\n))'
    match = re.search(regex, string)

    data = "\n".join(match.group().split("\n")[skip_lines+1:])
    if datatype == pd.DataFrame:
        kwargs = {"header":0, "index_col":0} | kwargs
        df = pd.read_csv(StringIO(data), *args, **kwargs)
        return  df.rename(columns=lambda x: x.strip())
    return datatype(data, *args, **kwargs)

def parse_scheduling_period(fname):
    from faker import Faker
    fake = Faker()
    fake.seed_instance(0)

    with open(fname, "r") as f:
        string = f.read()


    horizon = _tag_to_data(string, "SECTION_HORIZON", skip_lines=2, datatype=int)
    shifts = _tag_to_data(string, "SECTION_SHIFTS", names=["ShiftID", "Length", "cannot follow"],
                          dtype={'ShiftID':str, 'Length':int, 'cannot follow':str})
    shifts.fillna("", inplace=True)
    shifts["cannot follow"] = shifts["cannot follow"].apply(lambda val : val.split("|"))

    staff = _tag_to_data(string, "SECTION_STAFF", index_col=False)
    maxes = staff["MaxShifts"].str.split("|", expand=True)
    for col in maxes:
        shift_id = maxes[col].iloc[0].split("=")[0]
        column = maxes[col].apply(lambda x : x.split("=")[1])
        staff[f"max_shifts_{shift_id}"] = column.astype(int)

    staff["name"] = [fake.unique.first_name() for _ in staff.index]

    days_off = _tag_to_data(string, "SECTION_DAYS_OFF", datatype=str)
    # process string to be EmployeeID, Day off for each line
    rows = []
    for line in days_off.split("\n")[1:]:
        employee_id , *days = line.split(",")
        rows += [dict(EmployeeID=employee_id, DayIndex= int(d)) for d in days]
    days_off = pd.DataFrame(rows)


    shift_on = _tag_to_data(string, "SECTION_SHIFT_ON_REQUESTS", index_col=False)
    shift_off = _tag_to_data(string, "SECTION_SHIFT_OFF_REQUESTS", index_col=False)
    cover = _tag_to_data(string, "SECTION_COVER", index_col=False)

    return dict(horizon=horizon, shifts=shifts, staff=staff, days_off=days_off, shift_on=shift_on, shift_off=shift_off, cover=cover)


def nurserostering_model(horizon, shifts:pd.DataFrame, staff, days_off, shift_on, shift_off, cover):

    n_nurses = len(staff)

    FREE = 0
    SHIFTS = list(shifts.index)

    nurse_view = cp.intvar(0,len(shifts), shape=(n_nurses, horizon), name="nv")

    model = cp.Model()

    # Shifts which cannot follow the shift on the previous day.
    for id, shift in enumerate(shifts.iterrows()):
        for other_shift in shift['cannot follow']:
            model += (nurse_view[:,:-1] == id).implies(nurse_view[:,1:] != other_shift)

    # Maximum number of shifts of each type that can be assigned to each employee.
    for _, nurse in staff.iterrows():


        n = self.nurse_map.index(nurse['# ID'])
        for shift_id, shift in self.data.shifts.iterrows():
            n_shifts = cp.Count(self.nurse_view[n], self.shift_name_to_idx[shift_id])
            max_shifts = nurse[f"max_shifts_{shift_id}"]
            cons = n_shifts <= max_shifts
            cons.set_description(f"{nurse['name']} can work at most {max_shifts} {shift_id}-shifts")
            cons.visualize = get_visualizer(n, shift_id)
            constraints.append(cons)

    return constraints

if __name__ == "__main__":

    dataset = NurseRosteringDataset(root=".", download=True, transform=parse_scheduling_period)
    print("Dataset size:", len(dataset))
    print("Instance 0:")
    data, metadata = dataset[1]
    print(metadata)
    for key, df in data.items():
        print(key)
        print(df)