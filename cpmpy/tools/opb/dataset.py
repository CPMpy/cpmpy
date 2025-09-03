"""
PyTorch-style Dataset for PB competition instances in restricted OPB PB24 format.

Simply create a dataset instance (configured for the targeted competition year/track) and start iterating over its contents:

.. code-block:: python

    from cpmpy.tools.opb import OPBDataset, read_opb

    for filename, metadata in OPBDataset(year=2016, track="DEC-LIN", download=True): # auto download dataset and iterate over its instances
        # Do whatever you want here, e.g. reading to a CPMpy model and solving it:
        model = read_opb(filename)
        model.solve()
        print(model.status())

The `metadata` contains usefull information about the current problem instance.

Since the dataset is PyTorch compatible, it can be used with a DataLoader:

.. code-block:: python

    from cpmpy.tools.opb import OPBDataset, read_opb

    # Initialize the dataset
    dataset = OPBDataset(year=2016, track="DEC-LIN", download=True):

    from torch.utils.data import DataLoader

    # Wrap dataset in a DataLoader
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Iterate over the dataset
    for batch in data_loader:
        # Your code here
"""

import os
import pathlib
from typing import Tuple, Any
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import tarfile

class OPBDataset(object):  # torch.utils.data.Dataset compatible

    """
    OPB PB24 Dataset in a PyTorch compatible format.
    
    Arguments:
        root (str): Root directory containing the OPB instances (if 'download', instances will be downloaded to this location)
        year (int): Competition year (2006, 2007, 2009, 2010, 2011, 2012, 2015, 2016 or 2024)
        track (str, optional): Filter instances by track type (e.g., "DEC-LIN", "DEC-NLC", "OPT-LIN", "OPT-NLC")
        transform (callable, optional): Optional transform to be applied on the instance data (the file path of each problem instance)
        target_transform (callable, optional): Optional transform to be applied on the metadata (the metadata dictionary of each problem instance)
        download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
    """
    
    def __init__(self, root: str = ".", year: int = 2023, track: str = None, transform=None, target_transform=None, download: bool = False):
        """
        Initialize the OPB Dataset.
        """
        self.root = pathlib.Path(root)
        self.year = year
        self.transform = transform
        self.target_transform = target_transform
        self.track = track
        self.dataset_dir = self.root / str(year) / track
        
        if not str(year).startswith('20'):
            raise ValueError("Year must start with '20'")

        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)
        
        if not self.dataset_dir.exists():
            if not download:
                raise ValueError(f"Dataset for year {year} and track {track} not found. Please set download=True to download the dataset.")
            else:
                print(f"Downloading OPB {year} instances...")
                url = f"https://www.cril.univ-artois.fr/PB24/benchs/"
                year_suffix = str(year)[2:]  # Drop the starting '20'
                url_path = url + f"normalized-PB{year_suffix}.tar"
                tar_path = self.root / f"normalized-extraPB{year_suffix}.tar"
                
                try:
                    urlretrieve(url_path, str(tar_path))
                except (HTTPError, URLError) as e:
                    raise ValueError(f"No dataset available for year {year}. Error: {str(e)}")
                
                # Extract only the specific track folder from the tar
                with tarfile.open(tar_path, "r:*") as tar_ref:  # r:* handles .tar, .tar.gz, .tar.bz2, etc.
                    # Get the main folder name
                    main_folder = None
                    for name in tar_ref.getnames():
                        if "/" in name:
                            main_folder = name.split("/")[0]
                            break

                    if main_folder is None:
                        raise ValueError(f"Could not find main folder in tar file")

                    # Extract only files from the specified track
                    # Get all unique track names from tar
                    tracks = set()
                    for member in tar_ref.getmembers():
                        parts = member.name.split("/")
                        if len(parts) > 2 and parts[0] == main_folder:
                            tracks.add(parts[1])

                    # Check if requested track exists
                    if track not in tracks:
                        raise ValueError(f"Track '{track}' not found in dataset. Available tracks: {sorted(tracks)}")

                    # Create track folder in root directory
                    self.dataset_dir.mkdir(parents=True, exist_ok=True)

                    # Extract files for the specified track
                    prefix = f"{main_folder}/{track}/"
                    for member in tar_ref.getmembers():
                        if member.name.startswith(prefix) and member.isfile():
                            # Path relative to main_folder/track
                            relative_path = member.name[len(prefix):]

                            # Flatten: replace "/" with "_" to encode subfolders (some instances have clashing names)
                            flat_name = relative_path.replace("/", "_")
                            target_path = self.dataset_dir / flat_name

                            with tar_ref.extractfile(member) as source, open(target_path, "wb") as target:
                                target.write(source.read())

                    # Clean up the tar file
                    tar_path.unlink()

        
    def __len__(self) -> int:
        """Return the total number of instances."""
        return len(list(self.dataset_dir.glob("*.opb.xz")))
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a single OPB instance filename and metadata.

        Args:
            index (int): Index of the instance to retrieve
            
        Returns:
            Tuple[Any, Any]: A tuple containing:
                - The filename of the instance
                - Metadata dictionary with file name, track, year etc.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        # Get all compressed XML files and sort for deterministic behavior
        files = sorted(list(self.dataset_dir.glob("*.opb.xz")))
        file_path = files[index]

        filename = str(file_path)
        if self.transform:
            # does not need to remain a filename...
            filename = self.transform(filename)
            
        # Basic metadata about the instance
        metadata = {
            'year': self.year,
            'track': self.track,
            'author': str(file_path).split(os.sep)[-1].split("_")[0],
            'name': file_path.stem.replace('.xml.lzma', ''),
            'path': filename,
        }
        if self.target_transform:
            metadata = self.target_transform(metadata)
            
        return filename, metadata

if __name__ == "__main__":
    dataset = OPBDataset(year=2024, track="DEC-LIN", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
