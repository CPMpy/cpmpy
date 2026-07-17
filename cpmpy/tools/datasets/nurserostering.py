"""
Nurse rostering is a staff scheduling problem; instances come from schedulingbenchmarks.org.
Origin: https://schedulingbenchmarks.org/nrp/
"""

from __future__ import annotations

import os
import pathlib
import zipfile
import io
from typing import Any, Optional, Callable

from cpmpy.tools.datasets.core import FileDataset
from cpmpy.tools.io.nurserostering import parse_scheduling_period, _model_nurserostering

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


class NurseRosteringDataset(FileDataset):  # torch.utils.data.Dataset compatible

    """
    Nurserostering Dataset in a PyTorch compatible format.

    - Origin: https://schedulingbenchmarks.org/nrp/
    - References: 

        - Strandmark, P., Qu, Y. and Curtois, T. First-order linear programming in a column generation-based heuristic approach to the nurse rostering problem. Computers & Operations Research, 2020. 120, p. 104945.
        - Demirovic, E., Musliu, N., and Winter, F. Modeling and solving staff scheduling with partial weighted maxSAT. Annals of Operations Research, 2019. 275(1): p. 79-99.
        - Smet P. Constraint reformulation for nurse rostering problems, in: PATAT 2018 twelfth international conference on the practice and theory of automated timetabling, Vienna, August, 2018, p. 69-80.
        - Rahimian, E., Akartunali, K., and Levine, J. A hybrid integer programming and variable neighbourhood search algorithm to solve nurse rostering problems. European Journal of Operational Research, 2017. 258(2): p. 411-423.

    To load an instance into a CPMpy model, use :func:`~cpmpy.tools.io.nurserostering.load_nurserostering`.
    For examples of using a loader as a dataset ``transform``, see the
    :ref:`modeling guide <modeling-datasets>`.

    Arguments:
        root (str): Root directory containing the nurserostering instances (if 'download', instances will be downloaded to this location)
        transform (callable, optional): Optional transform to be applied on the instance data (the file path of each problem instance)
        target_transform (callable, optional): Optional transform to be applied on the metadata (the metadata dictionary of each problem instance)
        download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
    """

    name = "nurserostering"
    description = "Nurse rostering benchmark instances from schedulingbenchmarks.org."
    homepage = "https://schedulingbenchmarks.org/nrp/"
    citation = [
        "Strandmark, P., Qu, Y. and Curtois, T. First-order linear programming in a column generation-based heuristic approach to the nurse rostering problem. Computers & Operations Research, 2020. 120, p. 104945.",
        "Demirovic, E., Musliu, N., and Winter, F. Modeling and solving staff scheduling with partial weighted maxSAT. Annals of Operations Research, 2019. 275(1): p. 79-99.",
        "Smet P. Constraint reformulation for nurse rostering problems, in: PATAT 2018 twelfth international conference on the practice and theory of automated timetabling, Vienna, August, 2018, p. 69-80.",
        "Rahimian, E., Akartunali, K., and Levine, J. A hybrid integer programming and variable neighbourhood search algorithm to solve nurse rostering problems. European Journal of Operational Research, 2017. 258(2): p. 411-423.",
    ]

    def __init__(self, root: str = ".", transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False,
                 **kwargs: Any):
        """
        Initialize the Nurserostering Dataset.
        """

        self.root = pathlib.Path(root)

        dataset_dir = self.root / self.name

        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform, target_transform=target_transform, 
            download=download, extension=".txt",
            **kwargs
        )

    @classmethod
    def parse(cls, instance: os.PathLike) -> dict[str, Any]:
        """
        Parse a nurse rostering instance into native Python data structures.
        """
        return parse_scheduling_period(instance, open=cls.open)

    def categories(self) -> dict[str, Any]:
        return {}  # no categories

    def collect_instance_metadata(self, file: pathlib.Path) -> dict[str, Any]:
        """
        Extract scheduling metadata from nurse rostering instance.
        """
        try:
            data = self.parse(file)
            return {
                "horizon": data["horizon"],
                "num_staff": len(data["staff"]),
                "num_shifts": len(data["shifts"]),
                "num_days_off": len(data.get("days_off", [])),
                "num_shift_on_requests": len(data.get("shift_on", []) or []),
                "num_shift_off_requests": len(data.get("shift_off", []) or []),
                "num_cover_requirements": len(data.get("cover", []) or []),
            }
        except Exception:
            pass
        return {}

    def download(self):
        
        url = "https://schedulingbenchmarks.org/nrp/data/"
        target = "instances1_24.zip" # download full repo...
        target_download_path = self.root / target

        print("Downloading Nurserostering instances from schedulingbenchmarks.org")

        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path))
        except ValueError as e:
            raise ValueError(f"No dataset available on {url}. Error: {str(e)}")

        # make directory and extract files
        with zipfile.ZipFile(target_download_path, 'r') as zip_ref:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)

            # Extract files
            for file_info in zip_ref.infolist():
                    filename = pathlib.Path(file_info.filename).name
                    with zip_ref.open(file_info) as source, open(self.dataset_dir / filename, 'wb') as target:
                        target.write(source.read())

        # Clean up the zip file
        target_download_path.unlink()

    @classmethod
    def open(cls, instance: os.PathLike) -> io.TextIOBase:
        return open(instance, "r")



if __name__ == "__main__":
    dataset = NurseRosteringDataset(root=".", download=True, transform=parse_scheduling_period)
    print("Dataset size:", len(dataset))
    
    data, metadata = dataset[0]
    print(data)

    model, nurse_view = _model_nurserostering(**data)
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