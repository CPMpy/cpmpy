"""
PSPlib Dataset

https://www.om-db.wi.tum.de/psplib/getdata_sm.html
"""

import os
import pathlib
import io
import zipfile

from cpmpy.tools.dataset._base import _Dataset

class PSPLibDataset(_Dataset):  # torch.utils.data.Dataset compatible
    """
    PSPlib Dataset in a PyTorch compatible format.
    
    More information on PSPlib can be found here: https://www.om-db.wi.tum.de/psplib/main.html
    """
    
    name = "psplib"
    description = "Project Scheduling Problem Library (RCPSP) benchmark instances."
    url = "https://www.om-db.wi.tum.de/psplib/main.html"


    def __init__(self, root: str = ".", variant: str = "rcpsp", family: str = "j30", transform=None, target_transform=None, download: bool = False):
        """
        Constructor for a dataset object for PSPlib.

        Arguments:
            root (str): Root directory containing the psplib instances (if 'download', instances will be downloaded to this location)
            variant (str): scheduling variant (only 'rcpsp' is supported for now)
            family (str): family name (e.g. j30, j60, etc...)
            transform (callable, optional): Optional transform to be applied on the instance data
            target_transform (callable, optional): Optional transform to be applied on the file path
            download (bool): If True, downloads the dataset from the internet and puts it in `root` directory


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
            download=download, extension=f".{self.family_codes[self.variant]}"
        )

    @staticmethod
    def reader(file_path, open=open):
        """
        Reader for PSPLib dataset.
        Parses a file path directly into a CPMpy model.
        For backward compatibility. Consider using read() + load() instead.
        """
        from cpmpy.tools.io.rcpsp import load_rcpsp
        return load_rcpsp(file_path, open=open)

    @staticmethod
    def loader(content: str):
        """
        Loader for PSPLib dataset.
        Loads a CPMpy model from raw RCPSP content string.
        """
        from cpmpy.tools.io.rcpsp import load_rcpsp
        # load_rcpsp already supports raw strings
        return load_rcpsp(content)
        
    def category(self) -> dict:
        return {
            "variant": self.variant,
            "family": self.family
        }

    def collect_instance_metadata(self, file) -> dict:
        """Extract project metadata from SM file header."""
        import re
        result = {}
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

        url = "https://www.om-db.wi.tum.de/psplib/files/"
        target = f"{self.family}.{self.family_codes[self.variant]}.zip"
        target_download_path = self.root / target
        
        print(f"Downloading PSPLib {self.variant} {self.family} instances from www.om-db.wi.tum.de")

        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path), origins=self.origins)
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