"""
PyTorch-style Dataset for Jobshop instances from JSPLib

Simply create a dataset instance and start iterating over its contents:
The `metadata` contains usefull information about the current problem instance.

https://github.com/tamy0612/JSPLIB
"""

import io
import os
import json
import pathlib
from typing import Tuple, Any
import zipfile
import numpy as np

import cpmpy as cp
from cpmpy.tools.dataset._base import _Dataset
from cpmpy.tools.dataset.config import get_origins


class JSPLibDataset(_Dataset):  # torch.utils.data.Dataset compatible

    """
    JSP Dataset in a PyTorch compatible format.

    More information on JSPLib can be found here: https://github.com/tamy0612/JSPLIB
    """

    name = "jsplib"
    description = "Job Shop Scheduling Problem benchmark library."
    url = "https://github.com/tamy0612/JSPLIB"
    license = ""
    citation = ""
    domain = "scheduling"
    format = "JSPLib"
    origins = []  # Will be populated from config if available

    @staticmethod
    def _reader(file_path, open=open):
        from cpmpy.tools.io.jsplib import read_jsplib
        return read_jsplib(file_path, open=open)

    reader = _reader

    def __init__(self, root: str = ".", transform=None, target_transform=None, download: bool = False):
        """
        Initialize the JSPLib Dataset.

        Arguments:
            root (str): Root directory containing the jsp instances (if 'download', instances will be downloaded to this location)
            transform (callable, optional): Optional transform to be applied on the instance data
            target_transform (callable, optional): Optional transform to be applied on the file path
            download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
        """

        self.root = pathlib.Path(root)
        self._source_metadata_file = "instances.json"
        self._source_metadata = None  # Loaded lazily during metadata collection

        dataset_dir = self.root / self.name

        # Load origins from config
        if not self.origins:
            self.origins = get_origins(self.name)

        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform, target_transform=target_transform,
            download=download, extension=""
        )

    def category(self) -> dict:
        return {}  # no categories

    def _list_instances(self):
        """List JSPLib instances, excluding metadata and JSON files."""
        return sorted([
            f for f in self.dataset_dir.rglob("*")
            if f.is_file()
            and not str(f).endswith(self.METADATA_EXTENSION)
            and not str(f).endswith(".json")
        ])

    def collect_instance_metadata(self, file) -> dict:
        """Extract metadata from instances.json and instance file header."""
        # Lazy load the source metadata
        if self._source_metadata is None:
            source_path = self.dataset_dir / self._source_metadata_file
            if source_path.exists():
                with open(source_path, "r") as f:
                    self._source_metadata = json.load(f)
            else:
                self._source_metadata = []

        result = {}

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

    def __getitem__(self, index):
        """Supports both integer index and string name lookup."""
        if isinstance(index, str):
            files = self._list_instances()
            for file_path in files:
                if file_path.stem == index:
                    idx = files.index(file_path)
                    return super().__getitem__(idx)
            raise IndexError(f"Instance '{index}' not found in dataset")
        return super().__getitem__(index)

    def download(self):

        url = "https://github.com/tamy0612/JSPLIB/archive/refs/heads/" # download full repo...
        target = "master.zip"
        target_download_path = self.root / target

        print(f"Downloading JSPLib instances from github.com/tamy0612/JSPLIB")

        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path), origins=self.origins)
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

    def open(self, instance: os.PathLike) -> callable:
        return open(instance, "r")


def parse_jsp(filename: str):
    """
    Parse a JSPLib instance file
    Returns two matrices:
        - task to machines indicating on which machine to run which task
        - task durations: indicating the duration of each task
    """

    with open(filename, "r") as f:
        line = f.readline()
        while line.startswith("#"):
            line = f.readline()
        n_jobs, n_tasks = map(int, line.strip().split(" "))
        matrix = np.fromstring(f.read(), sep=" ", dtype=int).reshape((n_jobs, n_tasks*2))

        task_to_machines = np.empty(dtype=int, shape=(n_jobs, n_tasks))
        task_durations = np.empty(dtype=int, shape=(n_jobs, n_tasks))

        for t in range(n_tasks):
            task_to_machines[:, t] = matrix[:, t*2]
            task_durations[:, t] = matrix[:, t*2+1]

        return task_to_machines, task_durations


def jobshop_model(task_to_machines, task_durations):

    """
    Create a CPMpy model for the Jobshop problem.
    """

    task_to_machines = np.array(task_to_machines)
    dur = np.array(task_durations)

    assert task_to_machines.shape == task_durations.shape

    n_jobs, n_tasks = task_to_machines.shape

    start = cp.intvar(0, task_durations.sum(), name="start", shape=(n_jobs,n_tasks)) # extremely bad upperbound... TODO
    end = cp.intvar(0, task_durations.sum(), name="end", shape=(n_jobs,n_tasks)) # extremely bad upperbound... TODO
    makespan = cp.intvar(0, task_durations.sum(), name="makespan") # extremely bad upperbound... TODO

    model = cp.Model()
    model += start + dur == end
    model += end[:,:-1] <= start[:,1:] # precedences

    for machine in set(task_to_machines.flat):
        model += cp.NoOverlap(start[task_to_machines == machine],
                              dur[task_to_machines == machine],
                              end[task_to_machines == machine])

    model += end <= makespan
    model.minimize(makespan)

    return model, (start, makespan)


if __name__ == "__main__":
    dataset = JSPLibDataset(root=".", download=True, transform=parse_jsp)
    print("Dataset size:", len(dataset))
    print("Instance 0:")
