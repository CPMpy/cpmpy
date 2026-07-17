"""
PSPLib is a library of Project Scheduling Problem (RCPSP) benchmark instances.
Origin: https://www.om-db.wi.tum.de/psplib/main.html
"""

from __future__ import annotations

import os
import pathlib
import zipfile
from typing import Any, Optional, Callable, Union, Tuple
import builtins

from cpmpy.tools.datasets.core import FileDataset
from cpmpy.tools.io.rcpsp import parse_rcpsp

class PSPLibDataset(FileDataset):  # torch.utils.data.Dataset compatible
    """
    PSPlib Dataset in a PyTorch compatible format.

    - Origin: https://www.om-db.wi.tum.de/psplib/main.html
    - Reference: Kolisch, R., Sprecher, A. PSPLIB - A project scheduling problem library. European Journal of Operational Research, 96(1), 205-216, 1997.

    Arguments:
        root (str): Root directory containing the psplib instances (if 'download', instances will be downloaded to this location)
        variant (str): scheduling variant (only 'rcpsp' is supported for now)
        family (str): family name (e.g. j30, j60, etc...)
        transform (callable, optional): Optional transform to be applied on the instance data (the file path of each problem instance)
        target_transform (callable, optional): Optional transform to be applied on the metadata (the metadata dictionary of each problem instance)
        download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
    """

    name = "psplib"
    description = "Project Scheduling Problem Library (RCPSP) benchmark instances."
    homepage = "https://www.om-db.wi.tum.de/psplib/main.html"
    citation = [
        "Kolisch, R., Sprecher, A. PSPLIB - A project scheduling problem library. European Journal of Operational Research, 96(1), 205-216, 1997.",
    ]

    def __init__(self, root: str = ".", variant: str = "rcpsp", family: str = "j30",
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 download: bool = False, **kwargs: Any):
        """
        Constructor for a dataset object for PSPlib.

        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested variant/family combination is not available.
        """
        
        self.root = pathlib.Path(root)
        self.variant = variant
        self.family = family
        
        self.families = dict(
            rcpsp = ["j30", "j60", "j90", "j120"]
        )
        self.family_codes = dict(rcpsp="sm", mrcpsp="mm")

        if variant != "rcpsp":
            raise ValueError("Only 'rcpsp' variant is supported for now")
        if family not in self.families[variant]:
            raise ValueError(f"Unknown problem family. Must be any of {','.join(self.families[variant])}")
        
        dataset_dir = self.root / self.name / self.variant / self.family

        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform, target_transform=target_transform, 
            download=download, extension=f".{self.family_codes[self.variant]}",
            **kwargs
        )

    @classmethod
    def parse(cls, instance: os.PathLike) -> dict[str, Any]:
        """
        Parse a PSPLIB RCPSP instance into job data and capacities.
        """
        return parse_rcpsp(instance, open=cls.open)

    def categories(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "family": self.family
        }

    def collect_instance_metadata(self, file: pathlib.Path) -> dict[str, Any]:
        """
        Extract project metadata from SM file header.
        """
        import re
        result: dict[str, Any] = {}
        try:
            with self.open(file) as f:
                lines = f.readlines()

            in_project_info = False
            in_resource_avail = False
            for i, raw_line in enumerate(lines):
                line = raw_line.strip()
                if line.startswith("jobs"):
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        result["num_jobs"] = int(match.group(1))
                elif line.startswith("horizon"):
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        result["horizon"] = int(match.group(1))
                elif line.startswith("- renewable"):
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        result["num_renewable_resources"] = int(match.group(1))
                elif line.startswith("- nonrenewable"):
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        result["num_nonrenewable_resources"] = int(match.group(1))
                elif line.startswith("- doubly constrained"):
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        result["num_doubly_constrained_resources"] = int(match.group(1))
                elif line.startswith("PROJECT INFORMATION"):
                    in_project_info = True
                elif in_project_info and not line.startswith("*") and not line.startswith("pronr"):
                    # Data line: pronr #jobs rel.date duedate tardcost MPM-Time
                    parts = line.split()
                    if len(parts) >= 6:
                        result["duedate"] = int(parts[3])
                        result["tardcost"] = int(parts[4])
                        result["mpm_time"] = int(parts[5])
                    in_project_info = False
                elif line.startswith("RESOURCEAVAILABILITIES"):
                    in_resource_avail = True
                elif in_resource_avail and not line.startswith("*") and not line.startswith("R ") and not line.startswith("N "):
                    # Resource availability values line
                    parts = line.split()
                    if parts:
                        result["resource_availabilities"] = [int(x) for x in parts]
                    in_resource_avail = False
                elif line.startswith("PRECEDENCE RELATIONS") or line.startswith("REQUESTS/DURATIONS"):
                    in_project_info = False
        except Exception:
            pass
        return result

    def download(self):

        code = self.family_codes[self.variant]
        url = "https://www.om-db.wi.tum.de/psplib/"
        target = f"download_dataset.php?set={self.family}&mode={code}&format=zip"
        target_download_path = self.root / f"{self.family}.{code}.zip"

        print(f"Downloading PSPLib {self.variant} {self.family} instances from www.om-db.wi.tum.de")

        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path), desc=f"{self.family}.{code}.zip")
        except ValueError as e:
            raise ValueError(f"No dataset available for variant {self.variant} and family {self.family}. Error: {str(e)}")
        
        # make directory and extract files
        with zipfile.ZipFile(target_download_path, 'r') as zip_ref:                    
            # Create track folder in root directory, parents=True ensures recursive creation
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract files
            for file_info in zip_ref.infolist():
                # Extract file to family_dir, removing main_folder/track prefix
                filename = pathlib.Path(file_info.filename).name
                with zip_ref.open(file_info) as source, open(self.dataset_dir / filename, 'wb') as target:
                    target.write(source.read())
        # Clean up the zip file
        target_download_path.unlink()


if __name__ == "__main__":
    dataset = PSPLibDataset(variant="rcpsp", family="j30", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
    print("Instance 'j301_1':", dataset["j301_1"])
