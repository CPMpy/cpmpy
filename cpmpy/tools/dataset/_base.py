"""
Dataset Base Class

This module defines the abstract `_Dataset` class, which serves as the foundation
for loading and managing benchmark instance collections in CPMpy-based experiments.  
It standardizes how datasets are stored, accessed, and optionally transformed.
"""

from abc import ABC, abstractmethod
import os
import pathlib
import io
import tempfile
from typing import Any, Optional, Tuple
from urllib.error import URLError
from urllib.request import HTTPError, Request, urlopen

def format_bytes(bytes_num):
    """
    Format bytes into human-readable string (e.g., KB, MB, GB).
    """
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1024.0

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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
                files = sorted(list(self.dataset_dir.rglob(f"*{self.extension}")))
                print(f"Finished downloading {len(files)} instances")

        files = sorted(list(self.dataset_dir.rglob(f"*{self.extension}")))
        if len(files) == 0:
            raise ValueError(f"Cannot find any instances inside dataset {self.dataset_dir}. Is it a valid dataset? If so, please report on GitHub.")
                
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

    def open(self, instance) -> io.TextIOBase:
        """
        How an instance file from the dataset should be opened.
        Especially usefull when files come compressed and won't work with 
        python standard library's 'open', e.g. '.xz', '.lzma'.
        """
        return open(instance, "r")

    def metadata(self, file) -> dict:
        metadata = self.category() | {
            'dataset': self.name,
            'name': pathlib.Path(file).stem.replace(self.extension, ''),
            'path': file,
        }
        return metadata
    
    def __len__(self) -> int:
        """Return the total number of instances."""
        return len(list(self.dataset_dir.rglob(f"*{self.extension}")))
    

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        # Get all compressed XML files and sort for deterministic behavior
        files = sorted(list(self.dataset_dir.rglob(f"*{self.extension}")))
        file_path = files[index]
        filename = str(file_path)

        # Basic metadata about the instance
        metadata = self.metadata(file=filename)
        if self.target_transform:
            metadata = self.target_transform(metadata)

        if self.transform:
            # does not need to remain a filename...
            filename = self.transform(filename)
                        
        return filename, metadata

    @staticmethod
    def _download_file(url: str, target: str, destination: Optional[str] = None, 
                        desc: str = None, 
                        chunk_size: int = 1024 * 1024) -> os.PathLike:
        """
        Download a file from a URL with progress bar and speed information.
        
        This method provides a reusable download function with progress updates
        similar to pip and uv, showing download progress, speed, and ETA.
        
        Arguments:
            url (str): The URL to download from.
            target (str): The target filename to download.
            destination (str, optional): The destination path to save the file.
            desc (str, optional): Description to show in the progress bar. 
                                  If None, uses the filename.
            chunk_size (int): Size of each chunk for download in bytes (default=1MB).
        
        Returns:
            str: The destination path where the downloaded file is saved.
        """

        if desc is None:
            desc = target

        if destination is None:
            temp_destination = tempfile.NamedTemporaryFile(delete=False)
        else:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        try:
            req = Request(url + target)
            with urlopen(req) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                        
            _Dataset._download_sequential(url + target, destination if destination is not None else temp_destination.name, total_size, desc, chunk_size)
    
            if destination is None:
                temp_destination.close()

            return pathlib.Path(destination if destination is not None else temp_destination.name)

        except (HTTPError, URLError) as e:
            raise ValueError(f"Failed to download file from {url + target}. Error: {str(e)}")
    
    @staticmethod
    def _download_sequential(url: str, filepath: pathlib.Path, total_size: int, desc: str,
                             chunk_size: int = 1024 * 1024):
        """
        Download file sequentially with progress bar.
        """
        
        import sys
        
        req = Request(url)
        with urlopen(req) as response:
            if tqdm is not None:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             unit_divisor=1024, desc=desc, file=sys.stdout, 
                             miniters=1, dynamic_ncols=True, ascii=False) as pbar:
                        with open(filepath, 'wb') as f:
                            while True:
                                chunk = response.read(chunk_size) 
                                if not chunk:
                                    break
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # Unknown size
                    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, 
                             desc=desc, file=sys.stdout, miniters=1, 
                             dynamic_ncols=True, ascii=False) as pbar:
                        with open(filepath, 'wb') as f:
                            while True:
                                chunk = response.read(chunk_size)
                                if not chunk:
                                    break
                                f.write(chunk)
                                pbar.update(len(chunk))
            else:
                # Fallback to simple download if tqdm is not available
                downloaded = 0
                with open(filepath, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            sys.stdout.write(f"\r\033[KDownloading {desc}: {format_bytes(downloaded)}/{format_bytes(total_size)} ({percent:.1f}%)")
                        else:
                            sys.stdout.write(f"\r\033[KDownloading {desc}: {format_bytes(downloaded)}...")
                        sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()




