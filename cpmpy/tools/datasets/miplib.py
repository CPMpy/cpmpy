"""
MIPLib is the Mixed Integer Programming Library of benchmark instances.
Origin: https://miplib.zib.de/
"""

from __future__ import annotations

import os
import gzip
import zipfile
import pathlib
import io
from typing import Any, Optional, Callable

from cpmpy.tools.datasets.core import FileDataset


class MIPLibDataset(FileDataset):  # torch.utils.data.Dataset compatible

    """
    MIPLib Dataset in a PyTorch compatible format.

    - Origin: https://miplib.zib.de/
    - Reference: Gleixner, A., et al. MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library. Mathematical Programming Computation, 2021.

    To load an instance into a CPMpy model, use :func:`~cpmpy.tools.io.scip_formats.load_scip_format`.
    For examples of using a loader as a dataset ``transform``, see the
    :ref:`modeling guide <modeling-datasets>`.

    Arguments:
        root (str): Root directory where datasets are stored or will be downloaded to (default=".").
        year (int): Year of the dataset to use (default=2024).
        track (str): Track name specifying which subset of the dataset instances to load (default="exact-unweighted").
        transform (callable, optional): Optional transform applied to the instance file path.
        target_transform (callable, optional): Optional transform applied to the metadata dictionary.
        download (bool): If True, downloads the dataset if it does not exist locally (default=False).
    """

    name = "miplib"
    description = "Mixed Integer Programming Library benchmark instances."
    homepage = "https://miplib.zib.de/"
    citation = [
        "Gleixner, A., Hendel, G., Gamrath, G., Achterberg, T., Bastubbe, M., Berthold, T., Christophel, P. M., Jarck, K., Koch, T., Linderoth, J., Lubbecke, M., Mittelmann, H. D., Ozyurt, D., Ralphs, T. K., Salvagnin, D., and Shinano, Y. MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library. Mathematical Programming Computation, 2021. https://doi.org/10.1007/s12532-020-00194-3.",
    ]

    def __init__(
            self,
            root: str = ".",
            year: int = 2024,
            transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
            download: bool = False,
            **kwargs: Any
        ):
        """
        Constructor for a dataset object of the MIPLib competition.

        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested year/track combination is not available.
        """

        self.root = pathlib.Path(root)
        self.year = year

        dataset_dir = self.root / self.name / str(year)

        super().__init__(
            dataset_dir=dataset_dir, 
            transform=transform, target_transform=target_transform, 
            download=download, extension=".mps.gz",
            **kwargs
        )

    def categories(self) -> dict[str, Any]:
        return {
            "year": self.year,
        }

    def download(self):
        
        url = "https://miplib.zib.de/downloads/"
        target = "collection.zip"
        target_download_path = self.root / target

        print("Downloading MIPLib instances from miplib.zib.de")

        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path))
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

    def collect_instance_metadata(self, file: pathlib.Path) -> dict[str, Any]:
        """
        Extract row/column counts from MPS file sections.
        """
        result: dict[str, Any] = {}
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

    @classmethod
    def open(cls, instance: os.PathLike) -> io.TextIOBase:
        return gzip.open(instance, "rt") if str(instance).endswith(".gz") else open(instance)


if __name__ == "__main__":
    dataset = MIPLibDataset(download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
