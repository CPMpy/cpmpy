"""
MIPLib Dataset

https://maxsat-evaluations.github.io/
"""


import os
import gzip
import zipfile
import pathlib
import io

from cpmpy.tools.dataset._base import _Dataset
from cpmpy.tools.dataset.config import get_origins


class MIPLibDataset(_Dataset):  # torch.utils.data.Dataset compatible

    """
    MIPLib Dataset in a PyTorch compatible format.

    More information on MIPLib can be found here: https://miplib.zib.de/
    """
  
    name = "miplib"
    description = "Mixed Integer Programming Library benchmark instances."
    url = "https://miplib.zib.de/"
    license = ""
    citation = ""
    domain = "mixed integer programming"
    format = "MPS"
    origins = []  # Will be populated from config if available

    @staticmethod
    def _reader(file_path, open=open):
        from cpmpy.tools.io.scip import read_scip
        return read_scip(file_path, open=open)

    reader = _reader

    def collect_instance_metadata(self, file) -> dict:
        """Extract row/column counts from MPS file sections."""
        result = {}
        try:
            with self.open(file) as f:
                section = None
                num_rows = 0
                columns = set()
                has_objective = False
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("NAME"):
                        section = "NAME"
                    elif stripped == "ROWS":
                        section = "ROWS"
                    elif stripped == "COLUMNS":
                        section = "COLUMNS"
                    elif stripped in ("RHS", "RANGES", "BOUNDS", "ENDATA"):
                        section = stripped
                    elif section == "ROWS" and stripped:
                        parts = stripped.split()
                        if parts[0] == "N":
                            has_objective = True
                        else:
                            num_rows += 1
                    elif section == "COLUMNS" and stripped:
                        parts = stripped.split()
                        if parts:
                            columns.add(parts[0])
                    elif section in ("RHS", "RANGES", "BOUNDS", "ENDATA"):
                        pass  # skip to avoid parsing entire file
                        if section == "ENDATA":
                            break
                result["mps_num_rows"] = num_rows
                result["mps_num_columns"] = len(columns)
                result["mps_has_objective"] = has_objective
        except Exception:
            pass
        return result

    def __init__(
            self, 
            root: str = ".", 
            year: int = 2024, track: str = "exact-unweighted", 
            transform=None, target_transform=None, 
            download: bool = False
        ):
        """
        Constructor for a dataset object of the MIPLib competition.

        Arguments:
            root (str): Root directory where datasets are stored or will be downloaded to (default="."). 
            year (int): Year of the dataset to use (default=2024).
            track (str): Track name specifying which subset of the dataset instances to load (default="exact-unweighted").
            transform (callable, optional): Optional transform applied to the instance file path.
            target_transform (callable, optional): Optional transform applied to the metadata dictionary.
            download (bool): If True, downloads the dataset if it does not exist locally (default=False).

        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested year/track combination is not available.
        """

        self.root = pathlib.Path(root)
        self.year = year
        self.track = track

        dataset_dir = self.root / self.name / str(year) / track

        # Load origins from config
        if not self.origins:
            self.origins = get_origins(self.name)

        super().__init__(
            dataset_dir=dataset_dir, 
            transform=transform, target_transform=target_transform, 
            download=download, extension=".mps.gz"
        )

    def category(self) -> dict:
        return {
            "year": self.year,
            "track": self.track
        }
    
    def download(self):
        
        url = "https://miplib.zib.de/downloads/"
        target = "collection.zip"
        target_download_path = self.root / target

        print(f"Downloading MIPLib instances from miplib.zib.de")

        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path), origins=self.origins)
        except ValueError as e:
            raise ValueError(f"No dataset available on {url}. Error: {str(e)}")
        
        # Extract files
        with zipfile.ZipFile(target_download_path, 'r') as zip_ref:                    
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract files
            for file_info in zip_ref.infolist():
                filename = pathlib.Path(file_info.filename).name
                with zip_ref.open(file_info) as source, open(self.dataset_dir / filename, 'wb') as target:
                    target.write(source.read())

        # Clean up the zip file
        target_download_path.unlink()

    def open(self, instance: os.PathLike) -> io.TextIOBase:
        return gzip.open(instance, "rt") if str(instance).endswith(".gz") else open(instance)


if __name__ == "__main__":
    dataset = MIPLibDataset(download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
