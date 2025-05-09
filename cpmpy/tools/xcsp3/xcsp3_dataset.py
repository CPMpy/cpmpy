"""
PyTorch-style Dataset for XCSP3 competition instances.
"""

import os
import pathlib
from typing import List, Tuple, Dict, Any
import xml.etree.ElementTree as ET
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import zipfile
import lzma

class XCSP3Dataset(object):  # torch.utils.data.Dataset compatible
    def __init__(self, root: str = ".", year: int = 2023, track: str = None, transform=None, target_transform=None, download: bool = False):
        """
        Initialize the XCSP3 Dataset.
        
        Args:
            root (str): Root directory containing the XCSP3 instances
            year (int): Competition year (2022 or 2023)
            track (str, optional): Filter instances by track type (e.g., "COP", "CSP", "MiniCOP")
            transform (callable, optional): Optional transform to be applied on the instance data
            target_transform (callable, optional): Optional transform to be applied on the file path
            download (bool): If True, downloads the dataset from the internet and puts it in root directory
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
                    for file_info in zip_ref.infolist():
                        if file_info.filename.startswith(f"{main_folder}/{track}/"):
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
        Get a single XCSP3 instance.

        Unlike in the machine learning case, we expect that each instance is only
        needed once. Hence, we did not load all instances into memory in the constructor.
        Instead, we decompress and load here only this instance.
        
        Args:
            index (int): Index of the instance to retrieve
            
        Returns:
            Tuple[Any, Any]: A tuple containing:
                - The parsed XML ElementTree of the instance file content
                - Metadata dictionary with file name, track, year etc.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        # Get all compressed XML files and sort for deterministic behavior
        files = sorted(list(self.track_dir.glob("*.xml.lzma")))
        file_path = files[index]

        # Decompress the LZMA file and parse the XML
        with lzma.open(file_path, 'rb') as f:
            tree = ET.parse(f)
        root = tree.getroot()

        # Basic metadata about the instance
        metadata = {
            'year': self.year,
            'track': self.track,
            'name': file_path.stem.replace('.xml.lzma', ''),
            'path': str(file_path),
        }
        
        if self.transform:
            tree = self.transform(tree)
            
        if self.target_transform:
            metadata = self.target_transform(metadata)
            
        return tree, metadata

if __name__ == "__main__":
    dataset = XCSP3Dataset(year=2024, track="MiniCOP", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
