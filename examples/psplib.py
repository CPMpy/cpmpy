"""
PyTorch-style Dataset for RCPSP instances from psplib.com

Simply create a dataset instance (configured for the targeted family such as j30,j60 etc...) and start iterating over its contents:
The `metadata` contains usefull information about the current problem instance.
"""

import pathlib
from os.path import join
from typing import Tuple, Any
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import zipfile

import cpmpy as cp

class PSPLibDataset(object):  # torch.utils.data.Dataset compatible

    """
    PSPlib Dataset in a PyTorch compatible format.
    
    Arguments:
        root (str): Root directory containing the psplib instances (if 'download', instances will be downloaded to this location)
        variant (str): scheduling variant (only 'rcpsp' is supported for now)
        family (str): family name (e.g. j30, j60, etc...)
        transform (callable, optional): Optional transform to be applied on the instance data
        target_transform (callable, optional): Optional transform to be applied on the file path
        download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
    """
    
    def __init__(self, root: str = ".", variant: str = "rcpsp", family: str = "j30", transform=None, target_transform=None, download: bool = False):
        """
        Initialize the PSPLib Dataset.
        """
        
        self.root = pathlib.Path(root)
        self.variant = variant
        self.family = family
        self.transform = transform
        self.target_transform = target_transform
        self.family_dir = pathlib.Path(join(self.root, variant, family))
        
        self.families = dict(
            rcpsp = ["j30", "j60", "j90", "j120"]
        )
        self.family_codes = dict(rcpsp="sm", mrcpsp="mm")

        if variant != "rcpsp":
            raise ValueError("Only 'rcpsp' variant is supported for now")
        if family not in self.families[variant]:
            raise ValueError(f"Unknown problem family. Must be any of {','.join(self.families[variant])}")
        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)
        
        if not self.family_dir.exists():
            if not download:
                raise ValueError(f"Dataset for variant {variant} and family {family} not found. Please set download=True to download the dataset.")
            else:
                print(f"Downloading PSPLib {variant} {family} instances...")
                
                zip_name = f"{family}.{self.family_codes[variant]}.zip"
                url = f"https://www.om-db.wi.tum.de/psplib/files/"

                url_path = url + zip_name
                zip_path = self.root / zip_name
                
                try:
                    urlretrieve(url_path, str(zip_path))
                except (HTTPError, URLError) as e:
                    raise ValueError(f"No dataset available for variant {variant} and family {family}. Error: {str(e)}")
                
                # make directory and extract files
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:                    
                    # Create track folder in root directory, parents=True ensures recursive creation
                    self.family_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Extract files
                    for file_info in zip_ref.infolist():
                        # Extract file to family_dir, removing main_folder/track prefix
                        filename = pathlib.Path(file_info.filename).name
                        with zip_ref.open(file_info) as source, open(self.family_dir / filename, 'wb') as target:
                            target.write(source.read())
                # Clean up the zip file
                zip_path.unlink()

        
    def __len__(self) -> int:
        """Return the total number of instances."""
        return len(list(self.family_dir.glob(f"*.{self.family_codes[self.variant]}")))
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a single RCPSP instance filename and metadata.

        Args:
            index (int): Index of the instance to retrieve
            
        Returns:
            Tuple[Any, Any]: A tuple containing:
                - The filename of the instance
                - Metadata dictionary with file name, track, year etc.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        # Get all instance files and sort for deterministic behavior # TODO: use natsort instead?
        files = sorted(list(self.family_dir.glob(f"*.{self.family_codes[self.variant]}")))
        file_path = files[index]

        filename = str(file_path)
        if self.transform:
            # does not need to remain a filename...
            filename = self.transform(filename)
            
        # Basic metadata about the instance
        metadata = dict(
            variant = self.variant,
            family = self.family,
            name = file_path.stem
        )
        
        if self.target_transform:
            metadata = self.target_transform(metadata)
            
        return filename, metadata


def parse_rcpsp(filename: str) -> dict:
    import pandas as pd
    """
    Parse a RCPSP instance file
    Returns a Pandas DataFrame with the following columns:
    - jobnr (index)
    - mode (int)
    - duration (int)
    - successors (List[int])
    - R<machine_id> (int)
    """

    data = dict()
    with open(filename, "r") as f:
        line = f.readline()
        while not line.startswith("PRECEDENCE RELATIONS:"):
            line = f.readline()
        
        f.readline() # skip keyword line
        line = f.readline() # first line of table, skip
        while not line.startswith("*****"):
            jobnr, n_modes, n_succ, *succ = [int(x) for x in line.split(" ") if len(x.strip())]
            assert len(succ) == n_succ, "Expected %d successors for job %d, got %d" % (n_succ, jobnr, len(succ))
            data[jobnr] = dict(num_modes=n_modes, successors=succ)
            line = f.readline()

        # skip to job info
        while not line.startswith("REQUESTS/DURATIONS:"):
            line = f.readline()

        line = f.readline()
        _j, _m, _d, *_r = [x.strip() for x in line.split(" ") if len(x.strip())] # first line of table
        resource_names = [f"{_r[i]}{_r[i+1]}" for i in range(0,len(_r),2)]
        line = f.readline() # first line of table
        if line.startswith("----") or line.startswith("*****"): # intermediate line in table...
            line = f.readline() # skip

        while not line.startswith("*****"):
            jobnr, mode, duration, *resources = [int(x) for x in line.split(" ") if len(x.strip())]
            assert len(resources) == len(resource_names), "Expected %d resources for job %d, got %d" % (len(resource_names), jobnr, len(resources))
            data[jobnr].update(dict(mode=mode, duration=duration))
            data[jobnr].update({name : req for name, req in zip(resource_names, resources)})
            line = f.readline()
        
        # read resource availabilities
        while not line.startswith("RESOURCEAVAILABILITIES:"):
            line = f.readline()
        
        f.readline() # skip header
        capacities = [int(x) for x in f.readline().split(" ") if len(x)]

        df =pd.DataFrame([dict(jobnr=k ,**info) for k, info in data.items()], 
                          columns=["jobnr", "mode", "duration", "successors", *resource_names])
        df.set_index("jobnr", inplace=True)

        return df, dict(zip(resource_names, capacities))
    

def rcpsp_model(job_data, capacities):

    model = cp.Model()

    horizon = job_data.duration.sum() # worst case, all jobs sequential on a machine
    makespan = cp.intvar(0, horizon, name="makespan")

    start = cp.intvar(0, horizon, name="start", shape=len(job_data))
    end = cp.intvar(0, horizon, name="end", shape=len(job_data))

    # ensure capacity is not exceeded
    for rescource, capa in capacities.items():
        model += cp.Cumulative(
            start = start,
            duration = job_data['duration'].tolist(),
            end = end,
            demand = job_data[rescource].tolist(),
            capacity = capa
        )

    # enforce precedences
    for idx, (jobnr, info) in enumerate(job_data.iterrows()):
        for succ in info['successors']:
            model += end[idx] <= start[succ-1] # job ids start at idx 1

    model += end <= makespan
    model.minimize(makespan)

    return model, (start, end, makespan)


if __name__ == "__main__":
    dataset = PSPLibDataset(variant="rcpsp", family="j60", transform=parse_rcpsp, download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:")
    (table, capacities), metadata = dataset[0]
    print("Table:")
    print(table)
    print("Capacities:", capacities)
    print("Metadata:", metadata)

    model, (start, end, makespan) = rcpsp_model(job_data=table, capacities=capacities)
    assert model.solve()

    print("Found solution:", model.status())
    print("Makespan:", makespan.value())

    table["Start"] = start.value()
    table["Finish"] = end.value()
    table['Task'] = table.index
    
    import plotly.express as px
    import plotly.io as pio
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np
    pio.renderers.default = "browser" # ensure plotly opens figure in browser

    

    # Make the gantt chart
    ghant_fig = px.bar(table, orientation='h',
                       base="Start", x="duration", y="Task", color="Task", text="Task")   
    ghant_fig.show()

    # Make the resource usage plots (for each resource)
    fig = make_subplots(rows=len(capacities), cols=1, 
                        subplot_titles=[f"Resource {r}" for r in capacities.keys()])
    
    # For each resource
    for i, (resource, capacity) in enumerate(capacities.items(),1):
        # Create timeline of resource usage with step function
        timeline = np.zeros(int(makespan.value()) + 1)
        
        # Add resource usage for each job using step function
        for _, job in table.iterrows():
            start_time = int(job['Start'])
            end_time = int(job['Finish'])
            timeline[start_time:end_time] += job[resource]

        # Plot resource usage over time
        fig.add_trace(
            go.Scatter(x=list(range(len(timeline))), 
                      y=timeline,
                      line=dict(shape='hv'),
                      name=f"{resource} usage",
                      fill='tozeroy'),
            row=i, col=1
        )
        
        # Add capacity line
        fig.add_trace(
            go.Scatter(x=[0, len(timeline)],
                      y=[capacity, capacity],
                      name=f"{resource} capacity",
                      line=dict(color='red', dash='dash')),
            row=i, col=1
        )

    fig.show()
