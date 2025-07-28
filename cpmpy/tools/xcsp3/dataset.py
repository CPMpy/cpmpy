"""
PyTorch-style Dataset for XCSP3 competition instances.

Simply create a dataset instance (configured for the targeted competition year/track) and start iterating over its contents:

.. code-block:: python

    from cpmpy.tools.xcsp3 import XCSP3Dataset, read_xcsp3

    for filename, metadata in XCSP3Dataset(year=2024, track="COP", download=True): # auto download dataset and iterate over its instances
        # Do whatever you want here, e.g. reading to a CPMpy model and solving it:
        model = read_xcsp3(filename)
        model.solve()
        print(model.status())

The `metadata` contains usefull information about the current problem instance.

Since the dataset is PyTorch compatible, it can be used with a DataLoader:

.. code-block:: python

    from cpmpy.tools.xcsp3 import XCSP3Dataset, read_xcsp3

    # Initialize the dataset
    dataset = XCSP3Dataset(year=2024, track="COP", download=True)

    from torch.utils.data import DataLoader

    # Wrap dataset in a DataLoader
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Iterate over the dataset
    for batch in data_loader:
        # Your code here
"""

import pathlib
from typing import Tuple, Any
import xml.etree.ElementTree as ET
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import zipfile

class XCSP3Dataset(object):  # torch.utils.data.Dataset compatible

    """
    XCSP3 Dataset in a PyTorch compatible format.
    
    Arguments:
        root (str): Root directory containing the XCSP3 instances (if 'download', instances will be downloaded to this location)
        year (int): Competition year (2022, 2023 or 2024)
        track (str, optional): Filter instances by track type (e.g., "COP", "CSP", "MiniCOP")
        transform (callable, optional): Optional transform to be applied on the instance data (the file path of each problem instance)
        target_transform (callable, optional): Optional transform to be applied on the metadata (the metadata dictionary of each problem instance)
        download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
    """
    
    def __init__(self, root: str = ".", year: int = 2023, track: str = None, transform=None, target_transform=None, download: bool = False):
        """
        Initialize the XCSP3 Dataset.
        """
        self.root = pathlib.Path(root)
        self.year = year
        self.transform = transform
        self.target_transform = target_transform
        self.track = track
        self.track_dir = self.root / str(year) / track
        
        if not str(year).startswith('20'):
            raise ValueError("Year must start with '20'")
        if not track:
            raise ValueError("Track must be specified, e.g. COP, CSP, MiniCOP, ...")
        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)
        
        if not self.track_dir.exists():
            if not download:
                raise ValueError(f"Dataset for year {year} and track {track} not found. Please set download=True to download the dataset.")
            else:
                print(f"Downloading XCSP3 {year} instances...")
                url = f"https://www.cril.univ-artois.fr/~lecoutre/compets/"
                year_suffix = str(year)[2:]  # Drop the starting '20'
                url_path = url + f"instancesXCSP{year_suffix}.zip"
                zip_path = self.root / f"instancesXCSP{year_suffix}.zip"
                
                try:
                    urlretrieve(url_path, str(zip_path))
                except (HTTPError, URLError) as e:
                    raise ValueError(f"No dataset available for year {year}. Error: {str(e)}")
                
                # Extract only the specific track folder from the zip
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Get the main folder name (e.g., "024_V3")
                    main_folder = None
                    for name in zip_ref.namelist():
                        if '/' in name:
                            main_folder = name.split('/')[0]
                            break
                    
                    if main_folder is None:
                        raise ValueError(f"Could not find main folder in zip file")

                    # Extract only files from the specified track
                    # Get all unique track names from zip
                    tracks = set()
                    for file_info in zip_ref.infolist():
                        parts = file_info.filename.split('/')
                        if len(parts) > 2 and parts[0] == main_folder:
                            tracks.add(parts[1])
                    
                    # Check if requested track exists
                    if track not in tracks:
                        raise ValueError(f"Track '{track}' not found in dataset. Available tracks: {sorted(tracks)}")
                    
                    # Create track folder in root directory, parents=True ensures recursive creation
                    self.track_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Extract files for the specified track
                    prefix = f"{main_folder}/{track}/"
                    for file_info in zip_ref.infolist():
                        if file_info.filename.startswith(prefix):
                            # Extract file to track_dir, removing main_folder/track prefix
                            filename = pathlib.Path(file_info.filename).name
                            with zip_ref.open(file_info) as source, open(self.track_dir / filename, 'wb') as target:
                                target.write(source.read())
                # Clean up the zip file
                zip_path.unlink()

        
    def __len__(self) -> int:
        """Return the total number of instances."""
        return len(list(self.track_dir.glob("*.xml.lzma")))
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a single XCSP3 instance filename and metadata.

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
        files = sorted(list(self.track_dir.glob("*.xml.lzma")))
        file_path = files[index]

        filename = str(file_path)
        if self.transform:
            # does not need to remain a filename...
            filename = self.transform(filename)
            
        # Basic metadata about the instance
        metadata = {
            'year': self.year,
            'track': self.track,
            'name': file_path.stem.replace('.xml.lzma', ''),
            'path': filename,
        }
        if self.target_transform:
            metadata = self.target_transform(metadata)
            
        return filename, metadata

if __name__ == "__main__":
    dataset = XCSP3Dataset(year=2024, track="MiniCOP", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
