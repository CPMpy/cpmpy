"""
Dataset Base Class

This module defines the abstract `_Dataset` class, which serves as the foundation
for loading and managing benchmark instance collections in CPMpy-based experiments.  
It standardizes how datasets are stored, accessed, and optionally transformed.
"""

from abc import ABC, abstractmethod
import pathlib
from typing import Any, Tuple

class _Dataset(ABC):
    """
    Abstract base class for PyTorch-style datasets of benchmarking instances.

    The `_Dataset` class provides a standardized interface for downloading and
    accessing benchmark instances. This class should not be used on its own.
    """

    def __init__(
            self, 
            dataset_dir: str = ".",
            transform=None, target_transform=None, 
            download: bool = False,
            extension:str=".txt",
            **kwargs
        ):
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.extension = extension

        if not self.dataset_dir.exists():
            if not download:
                raise ValueError(f"Dataset not found. Please set download=True to download the dataset.")
            else:
                self.download()

        files = sorted(list(self.dataset_dir.glob(f"*{self.extension}")))
        if len(files) == 0:
            raise ValueError("Cannot find any instances inside dataset. Is it a valid dataset? If so, please report on GitHub.")
                
    @abstractmethod
    def category(self) -> dict:
        """
        Labels to distinguish instances into categories matching to those of the dataset.
        E.g. 
            - year
            - track
        """
        pass

    @abstractmethod
    def download(self, *args, **kwargs):
        """
        How the dataset should be downloaded.
        """
        pass

    @abstractmethod
    def open(self, instance) -> callable:
        """
        How an instance file from the dataset should be opened.
        Especially usefull when files come compressed and won't work with 
        python standard library's 'open', e.g. '.xz', '.lzma'.
        """
        pass

    def metadata(self, file) -> dict:
        metadata = self.category() | {
            'name': pathlib.Path(file).stem.replace(self.extension, ''),
            'path': file,
        }
        return metadata
    
    def __len__(self) -> int:
        """Return the total number of instances."""
        return len(list(self.dataset_dir.glob(f"*{self.extension}")))
    

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        # Get all compressed XML files and sort for deterministic behavior
        files = sorted(list(self.dataset_dir.glob(f"*{self.extension}")))
        file_path = files[index]

        filename = str(file_path)
        if self.transform:
            # does not need to remain a filename...
            filename = self.transform(filename)
            
        # Basic metadata about the instance
        metadata = self.metadata(file=filename, )
        if self.target_transform:
            metadata = self.target_transform(metadata)
            
        return filename, metadata
    
    




