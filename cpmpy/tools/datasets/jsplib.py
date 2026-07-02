"""
JSPLib is a benchmark library of Job Shop Scheduling Problem (JSP) instances.
Origin: https://github.com/tamy0612/JSPLIB
"""

from __future__ import annotations

import io
import os
import json
import pathlib
import zipfile
from typing import Any, Optional, Dict, Callable
import numpy as np

import cpmpy as cp
from cpmpy.tools.datasets.core import FileDataset
from cpmpy.tools.io.jsplib import _parse_jsplib


class JSPLibDataset(FileDataset):  # torch.utils.data.Dataset compatible

    """
    JSP Dataset in a PyTorch compatible format.

    - Origin: https://github.com/tamy0612/JSPLIB
    - References: 

        - J. Adams, E. Balas, D. Zawack. 'The shifting bottleneck procedure for job shop scheduling.', Management Science, Vol. 34, Issue 3, pp. 391-401, 1988.
        - J.F. Muth, G.L. Thompson. 'Industrial scheduling.', Englewood Cliffs, NJ, Prentice-Hall, 1963.
        - S. Lawrence. 'Resource constrained project scheduling: an experimental investigation of heuristic scheduling techniques (Supplement).', Graduate School of Industrial Administration. Pittsburgh, Pennsylvania, Carnegie-Mellon University, 1984.
        - D. Applegate, W. Cook. 'A computational study of job-shop scheduling.', ORSA Journal on Computer, Vol. 3, Isuue 2, pp. 149-156, 1991.
        - R.H. Storer, S.D. Wu, R. Vaccari. 'New search spaces for sequencing problems with applications to job-shop scheduling.', Management Science Vol. 38, Issue 10, pp. 1495-1509, 1992.
        - T. Yamada, R. Nakano. 'A genetic algorithm applicable to large-scale job-shop problems.', Proceedings of the Second international workshop on parallel problem solving from Nature (PPSN'2). Brussels (Belgium), pp. 281-290, 1992.
        - E. Taillard. 'Benchmarks for basic scheduling problems', European Journal of Operational Research, Vol. 64, Issue 2, pp. 278-285, 1993.

    Arguments:
        root (str): Root directory containing the jsp instances (if 'download', instances will be downloaded to this location)
        transform (callable, optional): Optional transform to be applied on the instance data (the file path of each problem instance)
        target_transform (callable, optional): Optional transform to be applied on the metadata (the metadata dictionary of each problem instance)
        download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
    """

    name = "jsplib"
    description = "Job Shop Scheduling Problem benchmark library."
    homepage = "https://github.com/tamy0612/JSPLIB"
    citation = [
        "J. Adams, E. Balas, D. Zawack. 'The shifting bottleneck procedure for job shop scheduling.', Management Science, Vol. 34, Issue 3, pp. 391-401, 1988.",
        "J.F. Muth, G.L. Thompson. 'Industrial scheduling.', Englewood Cliffs, NJ, Prentice-Hall, 1963.",
        "S. Lawrence. 'Resource constrained project scheduling: an experimental investigation of heuristic scheduling techniques (Supplement).', Graduate School of Industrial Administration. Pittsburgh, Pennsylvania, Carnegie-Mellon University, 1984.",
        "D. Applegate, W. Cook. 'A computational study of job-shop scheduling.', ORSA Journal on Computer, Vol. 3, Isuue 2, pp. 149-156, 1991.",
        "R.H. Storer, S.D. Wu, R. Vaccari. 'New search spaces for sequencing problems with applications to job-shop scheduling.', Management Science Vol. 38, Issue 10, pp. 1495-1509, 1992.",
        "T. Yamada, R. Nakano. 'A genetic algorithm applicable to large-scale job-shop problems.', Proceedings of the Second international workshop on parallel problem solving from Nature (PPSN'2). Brussels (Belgium), pp. 281-290, 1992.",
        "E. Taillard. 'Benchmarks for basic scheduling problems', European Journal of Operational Research, Vol. 64, Issue 2, pp. 278-285, 1993.",
    ]

    def __init__(self, root: str = ".", 
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, 
                 download: bool = False, **kwargs: Any):
        """
        Initialize the JSPLib Dataset.
        """

        self.root = pathlib.Path(root)
        self._source_metadata_file = "instances.json"
        self._source_metadata = None  # Loaded lazily during metadata collection

        dataset_dir = self.root / self.name

        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform, target_transform=target_transform,
            download=download, extension="",
            **kwargs
        )

    @classmethod
    def parse(cls, instance: os.PathLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse a JSPLib instance into task routing and durations.
        """
        return _parse_jsplib(instance, open=cls.open)

    def categories(self) -> Dict[str, Any]:
        return {}  # no categories

    def collect_instance_metadata(self, file: pathlib.Path) -> Dict[str, Any]:
        """
        Extract metadata from instances.json and instance file header.
        """
        # Lazy load the source metadata
        if self._source_metadata is None:
            source_path = self.dataset_dir / self._source_metadata_file
            if source_path.exists():
                with open(source_path, "r") as f:
                    self._source_metadata = json.load(f)
            else:
                self._source_metadata = []

        result: Dict[str, Any] = {}

        # Extract description from file header comments
        try:
            with self.open(file) as f:
                desc_lines = []
                for line in f:
                    if not line.startswith("#"):
                        break
                    cleaned = line.strip().strip("#").strip()
                    # Skip separator lines and "instance <name>" lines
                    if cleaned and not cleaned.startswith("+++") and not cleaned.startswith("instance "):
                        desc_lines.append(cleaned)
                if desc_lines:
                    result["instance_description"] = " ".join(desc_lines)
        except Exception:
            pass

        # Merge data from instances.json
        stem = pathlib.Path(file).stem
        for entry in self._source_metadata:
            if entry.get("name") == stem:
                result["jobs"] = entry.get("jobs")
                result["machines"] = entry.get("machines")
                result["optimum"] = entry.get("optimum")
                if "bounds" in entry:
                    result["bounds"] = entry["bounds"]
                elif entry.get("optimum") is not None:
                    result["bounds"] = {
                        "upper": entry["optimum"],
                        "lower": entry["optimum"]
                    }
                break
        return result

    def download(self):

        url = "https://github.com/tamy0612/JSPLIB/archive/refs/heads/" # download full repo...
        target = "master.zip"
        target_download_path = self.root / target

        print("Downloading JSPLib instances from github.com/tamy0612/JSPLIB")

        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path))
        except ValueError as e:
            raise ValueError(f"No dataset available on {url}. Error: {str(e)}")

        # Extract files
        with zipfile.ZipFile(target_download_path, 'r') as zip_ref:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)

            # Extract files
            for file_info in zip_ref.infolist():
                if file_info.filename.startswith("JSPLIB-master/instances/") and file_info.file_size > 0:
                    filename = pathlib.Path(file_info.filename).name
                    with zip_ref.open(file_info) as source, open(self.dataset_dir / filename, 'wb') as target:
                        target.write(source.read())
            # extract source metadata file
            with zip_ref.open("JSPLIB-master/instances.json") as source, open(self.dataset_dir / self._source_metadata_file, 'wb') as target:
                target.write(source.read())

        # Clean up the zip file
        target_download_path.unlink()

    @classmethod
    def open(cls, instance: os.PathLike) -> io.TextIOBase:
        return open(instance, "r")

    def _list_instances(self):
        """
        List JSPLib instances, excluding metadata and JSON files.

        Special overwrite due to JSPLib not using file extensions for its instances.
        """
        return sorted([
            f for f in self.dataset_dir.rglob("*")
            if f.is_file()
            and not str(f).endswith(self.METADATA_EXTENSION)
            and not str(f).endswith(".json")
        ])


if __name__ == "__main__":
    dataset = JSPLibDataset(root=".", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:")
