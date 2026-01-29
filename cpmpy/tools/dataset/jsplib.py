"""
PyTorch-style Dataset for Jobshop instances from JSPLib

Simply create a dataset instance and start iterating over its contents:
The `metadata` contains usefull information about the current problem instance.

https://github.com/tamy0612/JSPLIB
"""
import os
import json
import pathlib
from os.path import join
from typing import Tuple, Any
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import zipfile
import numpy as np

import cpmpy as cp

class JSPLibDataset(object):  # torch.utils.data.Dataset compatible

    """
    JSP Dataset in a PyTorch compatible format.
    
    More information on JSPLib can be found here: https://github.com/tamy0612/JSPLIB
    """
    
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
        self.instance_dir = pathlib.Path(join(self.root, "jsplib"))
        self.metadata_file = "instances.json"
        self.transform = transform
        self.target_transform = target_transform

        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)

        if not self.instance_dir.exists():
            if not download:
                raise ValueError(f"Dataset not found in local file system. Please set download=True to download the dataset.")
            else:
                url = f"https://github.com/tamy0612/JSPLIB/archive/refs/heads/master.zip" # download full repo...
                url_path = url
                zip_path = pathlib.Path(join(root,"jsplib-master.zip"))

                print(f"Downloading JSPLib instances..")

                try:
                    urlretrieve(url_path, str(zip_path))
                except (HTTPError, URLError) as e:
                    raise ValueError(f"No dataset available on {url}. Error: {str(e)}")
                
                # make directory and extract files
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    self.instance_dir.mkdir(parents=True, exist_ok=True)

                    # Extract files
                    for file_info in zip_ref.infolist():
                        if file_info.filename.startswith("JSPLIB-master/instances/") and file_info.file_size > 0:
                            filename = pathlib.Path(file_info.filename).name
                            with zip_ref.open(file_info) as source, open(self.instance_dir / filename, 'wb') as target:
                                target.write(source.read())
                    # extract metadata file
                    with zip_ref.open("JSPLIB-master/instances.json") as source, open(self.instance_dir / self.metadata_file, 'wb') as target:
                        target.write(source.read())
                                 # Clean up the zip file
                zip_path.unlink()

        
    def __len__(self) -> int:
        """Return the total number of instances."""
        return len(list(self.instance_dir.glob("*")))
    
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
        files = sorted(list(self.instance_dir.glob("*[!.json]"))) # exclude metadata file
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

        with open(self.instance_dir / self.metadata_file, "r") as f:
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
    (machines, dur), metadata = dataset[0]
    print("Machines:", machines)
    print("Durations:", dur)
    print("Metadata:", metadata)

    print("Solving", metadata['name'])
    model, (start, makespan) = jobshop_model(task_to_machines=machines, task_durations=dur)
    assert model.solve(time_limit=10)

    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    pio.renderers.default = "browser" # ensure plotly opens figure in browser

    df = pd.DataFrame({"Start": start.value().flat, "Duration": dur.flat, "Machine": machines.flat})
    df["Job"] = [j for j in range(metadata['jobs']) for _ in range(metadata['machines']) ]
    df["Task"] = [j for _ in range(metadata['machines']) for j in range(metadata['jobs'])]
    df["Name"] = "T" + df["Job"].astype(str) + "-" + df["Task"].astype(str)
    print(df)
    ghant_fig = px.bar(df, orientation='h',
                       base="Start", x="Duration", y="Machine", color="Job", text="Name",
                       title=f"Jobshop instance {metadata['name']}, makespan: {makespan.value()}, status: {model.status()}"
                       )
    ghant_fig.show()