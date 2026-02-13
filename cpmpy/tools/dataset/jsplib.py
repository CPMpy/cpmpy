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


class JSPLibDataset(_Dataset):  # torch.utils.data.Dataset compatible

    """
    JSP Dataset in a PyTorch compatible format.
    
    More information on JSPLib can be found here: https://github.com/tamy0612/JSPLIB
    """

    name = "jsplib"
    
    def __init__(self, root: str = ".", transform=None, target_transform=None, download: bool = False):
        """
        Initialize the PSPLib Dataset.

        Arguments:
            root (str): Root directory containing the jsp instances (if 'download', instances will be downloaded to this location)
            transform (callable, optional): Optional transform to be applied on the instance data
            target_transform (callable, optional): Optional transform to be applied on the file path
            download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
        """
        
        self.root = pathlib.Path(root)
        self.metadata_file = "instances.json"

        dataset_dir = self.root / self.name

        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform, target_transform=target_transform, 
            download=download, extension=""
        )

    def category(self) -> dict:
        return {} # no categories
    
    def download(self):

        url = "https://github.com/tamy0612/JSPLIB/archive/refs/heads/" # download full repo...
        target = "master.zip"
        target_download_path = self.root / target

        print(f"Downloading JSPLib instances from github.com/tamy0612/JSPLIB")

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
            # extract metadata file
            with zip_ref.open("JSPLIB-master/instances.json") as source, open(self.dataset_dir / self.metadata_file, 'wb') as target:
                target.write(source.read())
        
        # Clean up the zip file
        target_download_path.unlink()

    
    def __getitem__(self, index: int|str) -> Tuple[Any, Any]:
        """
        Get a single JSPLib instance filename and metadata.

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
        files = sorted(list(self.dataset_dir.rglob("*[!.json]"))) # exclude metadata file
        if isinstance(index, int):
            file_path = files[index]
        elif isinstance(index, str):
            for file_path in files:
                if file_path.stem == index:
                    break
            else:
                raise IndexError(f"Instance {index} not found in dataset")

        filename = str(file_path)
        if self.transform:
            # does not need to remain a filename...
            filename = self.transform(filename)

        with open(self.dataset_dir / self.metadata_file, "r") as f:
            for entry in json.load(f):
                if entry["name"] == file_path.stem:
                    metadata = entry
                    if "bounds" not in metadata: 
                        metadata["bounds"] = {"upper": metadata["optimum"], "lower": metadata["optimum"]}
                    del metadata['path']
                    metadata['path'] = str(file_path)
                    break
            else:
                metadata = dict()
        
        if self.target_transform:
            metadata = self.target_transform(metadata)
            
        return filename, metadata
    
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
