"""
Dataset Base Class

This module defines the abstract `_Dataset` class, which serves as the foundation
for loading and managing benchmark instance collections in CPMpy-based experiments.
It standardizes how datasets are downloaded, stored, accessed, and optionally transformed.

It provides a Pytorch compatible interface (constructor arguments like "transform" and the
methods __len__ and __getitem__ for iterating over the dataset).

Additionaly, it provides a collection of methods and helper functions to adapt the dataset
to the specific usecase requirements of constraint optimisation benchmarks.
"""

from abc import ABC, abstractmethod
import json
import os
import pathlib
import io
import tempfile
import warnings
from typing import Any, Iterator, Optional, Tuple, List, Union, Callable
from urllib.error import URLError
from urllib.request import HTTPError, Request, urlopen
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

from altair.utils.schemapi import _passthrough

# tqdm as an optional dependency, provides prettier progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import cpmpy as cp

# TODO: move elsewhere?
# Fields produced by extract_model_features() (after loading into a CPMpy model) 
#  - not portable across format translations
_MODEL_FEATURE_FIELDS = frozenset({
    "num_variables", "num_bool_variables", "num_int_variables",
    "num_constraints", "constraint_types", "has_objective",
    "objective_type", "domain_size_min", "domain_size_max", "domain_size_mean",
})

# TODO: move elsewhere?
# Prefixes for format-specific metadata fields (not portable across translations)
_FORMAT_SPECIFIC_PREFIXES = ("opb_", "wcnf_", "mps_", "xcsp_", "dimacs_")


def _format_bytes(bytes_num):
    """
    Format bytes into human-readable string (e.g., KB, MB, GB).

    Used to display download progress.
    """
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1024.0


def portable_instance_metadata(metadata: dict) -> dict:
    """
    Filter sidecar metadata to only portable, domain-specific fields.

    Strips model features (num_variables, constraint_types, ...),
    format-specific fields (opb_*, wcnf_*, mps_*, ...), and internal
    error fields (starting with ``_``).

    Keeps domain-specific metadata that is independent of the file format,
    such as ``jobs``, ``machines``, ``optimum``, ``horizon``, ``bounds``, etc.

    Arguments:
        metadata (dict): Full sidecar metadata dictionary.

    Returns:
        dict with only portable fields.
    """
    return {
        k: v for k, v in metadata.items()
        if not k.startswith("_")
        and k not in _MODEL_FEATURE_FIELDS
        and not any(k.startswith(p) for p in _FORMAT_SPECIFIC_PREFIXES)
    }


def _extract_model_features(model) -> dict:
    """
    Extract generic CP features from a CPMpy Model.

    Arguments:
        model: a cpmpy.Model instance

    Returns:
        dict with keys: num_variables, num_bool_variables, num_int_variables,
        num_constraints, constraint_types, has_objective, objective_type,
        domain_size_min, domain_size_max, domain_size_mean
    """
    from cpmpy.transformations.get_variables import get_variables_model
    from cpmpy.expressions.variables import _BoolVarImpl
    from cpmpy.expressions.core import Expression
    from cpmpy.expressions.utils import is_any_list

    variables = get_variables_model(model)

    num_bool = sum(1 for v in variables if isinstance(v, _BoolVarImpl))
    num_int = len(variables) - num_bool

    # Domain sizes (lb/ub available on all variable types)
    domain_sizes = [int(v.ub) - int(v.lb) + 1 for v in variables] if variables else []

    # Constraint types: collect .name from top-level constraints
    constraint_type_counts = {}

    def _count_constraints(c):
        if is_any_list(c):
            for sub in c:
                _count_constraints(sub)
        elif isinstance(c, Expression):
            name = c.name
            constraint_type_counts[name] = constraint_type_counts.get(name, 0) + 1

    for c in model.constraints:
        _count_constraints(c)

    num_constraints = sum(constraint_type_counts.values())

    # Objective
    has_obj = model.objective_ is not None
    obj_type = "none"
    if has_obj:
        obj_type = "min" if model.objective_is_min else "max"

    return {
        "num_variables": len(variables),
        "num_bool_variables": num_bool,
        "num_int_variables": num_int,
        "num_constraints": num_constraints,
        "constraint_types": constraint_type_counts,
        "has_objective": has_obj,
        "objective_type": obj_type,
        "domain_size_min": min(domain_sizes) if domain_sizes else None,
        "domain_size_max": max(domain_sizes) if domain_sizes else None,
        "domain_size_mean": round(sum(domain_sizes) / len(domain_sizes), 2) if domain_sizes else None,
    }


def extract_model_features(model) -> dict:
    """Public wrapper for extracting generic CPMpy model features."""
    return _extract_model_features(model)


# Global context for process-based metadata collection workers
_metadata_worker_context = {}


def _init_metadata_worker(context_dict, collect_metadata_func, reader_func, open_func):
    """Initialize worker process with dataset context."""
    global _metadata_worker_context
    _metadata_worker_context = context_dict.copy()
    _metadata_worker_context['collect_instance_metadata'] = collect_metadata_func
    _metadata_worker_context['reader'] = reader_func
    _metadata_worker_context['open_func'] = open_func


def _collect_one_metadata_worker(file_path_str):
    """Worker function for process-based metadata collection."""
    global _metadata_worker_context
    file_path = pathlib.Path(file_path_str)
    dataset_dir = pathlib.Path(_metadata_worker_context['dataset_dir'])
    meta_path = dataset_dir / (file_path.name + _metadata_worker_context['metadata_extension'])
    
    # Collect instance metadata using the provided function
    collect_metadata = _metadata_worker_context['collect_instance_metadata']
    try:
        instance_meta = collect_metadata(str(file_path))
    except Exception as e:
        instance_meta = {"_metadata_error": str(e)}

    # Separate portable from format-specific fields
    portable = portable_instance_metadata(instance_meta)
    format_specific = {
        k: v for k, v in instance_meta.items()
        if k not in portable and not k.startswith("_")
    }

    # Derive instance name
    stem = file_path.stem
    for ext in (".xml", ".wcnf", ".opb"):
        if stem.endswith(ext):
            stem = stem[:len(stem) - len(ext)]
            break

    # Build structured sidecar
    sidecar = {
        "dataset": _metadata_worker_context['dataset_metadata'],
        "instance_name": stem,
        "source_file": str(file_path.relative_to(dataset_dir)),
        "category": _metadata_worker_context['category'],
        "instance_metadata": portable,
        "format_metadata": format_specific,
    }

    if "_metadata_error" in instance_meta:
        sidecar["_metadata_error"] = instance_meta["_metadata_error"]

    # Preserve or compute model features
    model_features = None
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                existing = json.load(f)
            if "model_features" in existing:
                model_features = existing["model_features"]
        except (json.JSONDecodeError, IOError):
            pass

    if model_features is None:
        reader = _metadata_worker_context['reader']
        open_func = _metadata_worker_context['open_func']
        if not callable(reader):
            raise TypeError(
                f"Cannot extract model features for {file_path}: "
                "no dataset reader configured."
            )
        model = reader(str(file_path), open=open_func)
        model_features = extract_model_features(model)
    
    sidecar["model_features"] = model_features

    with open(meta_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    
    return str(file_path)


"""
dataset.map(transform)
dataset.filter(predicate)
dataset.shuffle(seed)
dataset.split(ratio)
"""

class Dataset(ABC):
    """
    Abstract base class for datasets.

    Each instance in a dataset is characterised by a (x, y) pair of:
        x: instance reference (e.g., file path, database key, generated seed, ...)
        y: metadata (solution, features, origin, etc.)
    """

    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        """
        Arguments:
            transform (callable, optional): Optional transform applied to the instance reference.
            target_transform (callable, optional): Optional transform applied to the metadata.
        """
        self.transform = transform
        self.target_transform = target_transform

class IndexedDataset(Dataset):
    """
    Abstract base class for indexed datasets.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of instances.
        """
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return the instance and metadata at the given index.

        Returns:
            x: instance reference (e.g., file path, database key, generated seed, ...)
            y: metadata (solution, features, origin, etc.)
        """
        pass

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class IterableDataset(Dataset):
    """
    Abstract base class for iterable datasets.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        """
        Return an iterator over the dataset.

        Returns:
            Iterator[Tuple[Any, Any]]: Iterator over the dataset, yielding (x, y) pairs of:
                x: instance reference (e.g., file path, database key, generated seed, ...)
                y: metadata (solution, features, origin, etc.)
        """
        pass

class FileDataset(IndexedDataset):
    """
    Abstract base class for PyTorch-style datasets of CO benchmarking instances.

    The `FileDataset` class provides a standardized interface for downloading and
    accessing file-backed benchmark instances. This class should not be used on its own.
    Instead have a look at one of the concrete subclasses, providing access to 
    well-known datasets from the community.
    """

    # Extension for metadata sidecar files
    METADATA_EXTENSION = ".meta.json"

    # -------------- Dataset-level metadata (override in subclasses) ------------- #
    
    @property
    @abstractmethod
    def name(self) -> str: pass

    @property
    @abstractmethod
    def description(self) -> str: pass

    @property
    @abstractmethod
    def url(self) -> str: pass

    @property
    def citation(self) -> List[str]: 
        return []

    # TODO: remove for now?
    # Multiple download origins (override in subclasses or via config)
    # Origins are tried in order, falling back to original url if all fail
    origins: List[str] = []  # List of URL bases to try before falling back to original url

    # ---------------------------------------------------------------------------- #


    def __init__(
            self,
            dataset_dir: str = ".",
            transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
            download: bool = False,
            extension: str = ".txt",
            metadata_workers: int = 1,
            **kwargs
        ):
        """
        Constructor for the _Dataset base class.

        Arguments:
            dataset_dir (str): Path to the dataset directory.
            transform (callable, optional): Optional transform applied to the instance file path.
            target_transform (callable, optional): Optional transform applied to the metadata dictionary.
            download (bool): If True, downloads the dataset if it does not exist locally (default=False).
            extension (str): Extension of the instance files. Used to filter instance files from the dataset directory.
            metadata_workers (int): Number of parallel workers for metadata collection during download (default: 1).

        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested year/track combination is not available.
            ValueError: If the dataset directory does not contain any instance files.
        """

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.extension = extension

        # TODO: remove for later?
        # if not self.origins:
        #     from cpmpy.tools.datasets.config import get_origins
        #     self.origins = get_origins(self.name)

        if not self._check_exists():
            if not download:
                raise ValueError("Dataset not found. Please set download=True to download the dataset.")
            else:
                self.download()
                self._collect_all_metadata(workers=metadata_workers)
                files = self._list_instances()
                print(f"Finished downloading {len(files)} instances")

        files = self._list_instances()
        if len(files) == 0:
            raise ValueError(f"Cannot find any instances inside dataset {self.dataset_dir}. Is it a valid dataset? If so, please report on GitHub.")

        super().__init__(transform=transform, target_transform=target_transform)

    def _check_exists(self) -> bool:
        """
        Check if the dataset exists (has been downloaded).
        """
        return self.dataset_dir.exists()

    # ---------------------------------------------------------------------------- #
    #                     Methods to implement in subclasses:                      #
    # ---------------------------------------------------------------------------- #

    @staticmethod
    @abstractmethod
    def _loader(content: str) -> cp.Model:
        """
        Loader for the dataset. Loads a CPMpy model from raw file content string.
        The content will be the raw text content of the file (already decompressed).
        
        Arguments:
            content (str): Raw file content string to load into a model.
            
        Returns:
            cp.Model: The loaded CPMpy model.
        """
        pass

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
        Download the dataset.
        """
        pass


    # ---------------------------------------------------------------------------- #
    #                        Methods to optionally overwrite                       #
    # ---------------------------------------------------------------------------- #

    def collect_instance_metadata(self, file: pathlib.Path) -> dict:
        """
        Provide domain-specific instance metadata.
        Called once after download for each instance.

        Arguments:
            file: path to the instance file

        Returns:
            dict with instance-specific metadata fields
        """
        return {}

    @classmethod
    def open(cls, instance: os.PathLike) -> io.TextIOBase:
        """
        How an instance file from the dataset should be opened.
        Especially usefull when files come compressed and won't work with
        python standard library's 'open', e.g. '.xz', '.lzma'.

        Arguments:
            instance (os.PathLike): File path to the instance file.

        Returns:
            io.TextIOBase: The opened file handle.
        """
        return open(instance, "r")

    def read(self, instance: os.PathLike) -> str:    
        """
        Read raw file contents from an instance file.
        Handles optional decompression automatically via dataset.open().
        
        Arguments:
            instance (os.PathLike): File path to the instance file.
        Returns:
            str: The raw file contents.
        """
        with self.open(instance) as f:
            return f.read()

    def load(self, instance: Union[str, os.PathLike]) -> cp.Model:
        """
        Load a CPMpy model from an instance file.
        
        Uses `.read()` to handle reading (decompressing + reading raw contents) and then turns 
        raw contents into a CPMpy model via `.loader()`.
        
        Arguments:
            instance (str or os.PathLike): 
                - File path to the instance file
                - OR a string containing the instance content directly
            
        Returns:
            cp.Model: The loaded CPMpy model.
        """

        # If instance is a path to a file -> open file
        if isinstance(instance, (str, os.PathLike)) and os.path.exists(instance):
           # Reading - use read() to decompress and read raw file contents
            content = self.read(instance)
        # If instance is a string containing a model -> use it directly
        else:
            content = instance

        # Loading - turn raw contents into CPMpy model
        return self.loader(content)



        


    # ---------------------------------------------------------------------------- #
    #                               Public interface                               #
    # ---------------------------------------------------------------------------- #

    def instance_metadata(self, file: pathlib.Path) -> dict:
        metadata = self.category() | {
            'dataset': self.name,
            'name': pathlib.Path(file).stem.replace(self.extension, ''),
            'path': file,
        }
        # Load sidecar metadata if it exists
        meta_path = self._metadata_path(file)
        if meta_path.exists():
            with open(meta_path, "r") as f:
                sidecar = json.load(f)
            # Structured: flatten instance_metadata, format_metadata, and model_features
            metadata.update(sidecar.get("instance_metadata", {}))
            metadata.update(sidecar.get("format_metadata", {}))
            metadata.update(sidecar.get("model_features", {}))
        return metadata

    @classmethod
    def dataset_metadata(cls) -> dict:
        """
        Return dataset-level metadata as a dictionary.
        """
        if isinstance(cls.citation, str):
            citations = [cls.citation] if cls.citation else []
        else:
            citations = list(cls.citation)

        return {
            "name": cls.name,
            "description": cls.description,
            "url": cls.url,
            "license": cls.license,
            "citation": citations,
        }


    # ---------------------------------------------------------------------------- #
    #                                   Internals                                  #
    # ---------------------------------------------------------------------------- #

    # ------------------------------ Instance access ----------------------------- #

    def _list_instances(self) -> list:
        """
        List all instance files, excluding metadata sidecar files.

        Returns a sorted list of `pathlib.Path` objects for all instance files
        matching the dataset's extension pattern.
        """
        return sorted([
            f for f in self.dataset_dir.rglob(f"*{self.extension}")
            if f.is_file() and not str(f).endswith(self.METADATA_EXTENSION)
        ])

    def __len__(self) -> int:
        """Return the total number of instances."""
        return len(self._list_instances())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        files = self._list_instances()
        file_path = files[index]
        filename = str(file_path)

        metadata = self.instance_metadata(file=filename)
        if self.target_transform:
            metadata = self.target_transform(metadata)

        if self.transform:
            filename = self.transform(filename)
            # Let transforms contribute to metadata (e.g. model verification info)
            if hasattr(self.transform, 'enrich_metadata'):
                metadata = self.transform.enrich_metadata(filename, metadata)

        return filename, metadata


    # ---------------------------- Metadata collection --------------------------- #

    def _metadata_path(self, instance_path: pathlib.Path) -> pathlib.Path:
        """
        Return the path to the `.meta.json` sidecar file for a given instance.

        Arguments:
            instance_path: path to the instance file

        Returns:
            path to the `.meta.json` sidecar file
        """
        return pathlib.Path(str(instance_path) + self.METADATA_EXTENSION)

    def _collect_all_metadata(self, force=False, workers=1):
        """
        Collect and store structured metadata sidecar files for all instances.

        Writes a structured `.meta.json` sidecar alongside each instance with:

        - `dataset`: dataset-level metadata (name, description, url, ...)
        - `instance_name`: logical instance name (filename stem)
        - `source_file`: path to the instance file
        - `category`: dataset category labels (year, track, variant, family)
        - `instance_metadata`: portable domain-specific metadata
        - `format_metadata`: format-specific metadata from the source format

        Arguments:
            force (bool): If True, re-collect instance metadata even if sidecar
                files already exist.
            workers (int): Number of parallel workers for metadata collection.
                Default is 1 (sequential). Use >1 for parallel processing.
        """
        files = self._list_instances()

        # Filter files that need processing
        files_to_process = []
        for file_path in files:
            meta_path = self._metadata_path(file_path)
            if force or not meta_path.exists():
                files_to_process.append(file_path)

        if not files_to_process:
            return

        # Process files sequentially or in parallel
        if workers <= 1:
            # Sequential processing
            if tqdm is not None:
                file_iter = tqdm(files_to_process, desc="Collecting metadata", unit="file")
            else:
                file_iter = files_to_process
                print(f"Collecting metadata for {len(files_to_process)} instances...")

            for file_path in file_iter:
                self._collect_one_metadata(file_path)
        else:
            # Parallel processing with ProcessPoolExecutor for CPU-bound work
            print(f"Collecting metadata for {len(files_to_process)} instances using {workers} workers...")
            
            # Use ProcessPoolExecutor with fork start method (Linux) to allow bound methods
            # On Linux, fork allows sharing the dataset instance, so bound methods work
            ctx = multiprocessing.get_context('fork')
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                futures = {executor.submit(self._collect_one_metadata, fp): fp for fp in files_to_process}
                
                if tqdm is not None:
                    iterator = tqdm(as_completed(futures), total=len(futures), desc="Collecting metadata", unit="file")
                else:
                    iterator = as_completed(futures)
                
                for future in iterator:
                    try:
                        future.result()
                    except Exception as e:
                        fp = futures[future]
                        print(f"Error collecting metadata for {fp.name}: {e}")

    def _collect_one_metadata(self, file_path):
        """Collect metadata for a single instance file."""
        meta_path = self._metadata_path(file_path)
        try:
            instance_meta = self.collect_instance_metadata(str(file_path))
        except Exception as e:
            instance_meta = {"_metadata_error": str(e)}

        # Separate portable from format-specific fields
        portable = portable_instance_metadata(instance_meta)
        format_specific = {
            k: v for k, v in instance_meta.items()
            if k not in portable and not k.startswith("_")
        }

        # Derive instance name (strip format-specific extensions)
        stem = file_path.stem
        for ext in (".xml", ".wcnf", ".opb"):
            if stem.endswith(ext):
                stem = stem[:len(stem) - len(ext)]
                break

        # Build structured sidecar
        sidecar = {
            "dataset": self.dataset_metadata(),
            "instance_name": stem,
            "source_file": str(file_path.relative_to(self.dataset_dir)),
            "category": self.category(),
            "instance_metadata": portable,
            "format_metadata": format_specific,
        }

        if "_metadata_error" in instance_meta:
            sidecar["_metadata_error"] = instance_meta["_metadata_error"]

        # Preserve previously extracted model features if present.
        # Otherwise, compute them from the parsed model when possible.
        model_features = None
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    existing = json.load(f)
                if "model_features" in existing:
                    model_features = existing["model_features"]
            except (json.JSONDecodeError, IOError):
                pass

        if model_features is None:
            if not callable(self.reader):
                raise TypeError(
                    f"Cannot extract model features for {file_path}: "
                    "no dataset reader configured. If unexpected, please open an issue on GitHub."
                )
            model = self.reader(str(file_path), open=self.open)
            model_features = extract_model_features(model)
    
        sidecar["model_features"] = model_features

        with open(meta_path, "w") as f:
            json.dump(sidecar, f, indent=2)

            
    # ----------------------------- Download methods ----------------------------- #

    @staticmethod
    def _try_origin(base_url: str, target: str, destination: str, desc: str, chunk_size: int) -> Optional[pathlib.Path]:
        """
        Try to download a file from a specific origin URL.
        
        Arguments:
            base_url (str): Base URL to try
            target (str): Target filename
            destination (str): Destination path
            desc (str): Description for progress bar
            chunk_size (int): Chunk size for download
            
        Returns:
            pathlib.Path if successful, None if failed
        """
        try:
            full_url = base_url.rstrip('/') + '/' + target.lstrip('/')
            req = Request(full_url)
            with urlopen(req) as response:
                total_size = int(response.headers.get('Content-Length', 0))
            
            FileDataset._download_sequential(full_url, destination, total_size, desc, chunk_size)
            return pathlib.Path(destination)
        except (HTTPError, URLError):
            return None

    @staticmethod
    def _download_file(url: str, target: str, destination: Optional[str] = None,
                        desc: str = None,
                        chunk_size: int = 1024 * 1024,
                        origins: Optional[List[str]] = None) -> os.PathLike:
        """
        Download a file from a URL with progress bar and speed information.
        Supports multiple origins with fallback.

        This method provides a reusable download function with progress updates
        similar to pip and uv, showing download progress, speed, and ETA.

        Arguments:
            url (str): The original URL to download from (used as fallback).
            target (str): The target filename to download.
            destination (str, optional): The destination path to save the file.
            desc (str, optional): Description to show in the progress bar.
                                  If None, uses the filename.
            chunk_size (int): Size of each chunk for download in bytes (default=1MB).
            origins (List[str], optional): List of alternative URL bases to try first.

        Returns:
            str: The destination path where the downloaded file is saved.
        """

        if desc is None:
            desc = target

        temp_destination = None
        if destination is None:
            temp_destination = tempfile.NamedTemporaryFile(delete=False)
            destination = temp_destination.name
        else:
            # Create parent directory if it doesn't exist and destination has a directory component
            dest_dir = os.path.dirname(destination)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)

        # Try custom origins first if provided
        if origins:
            for origin_url in origins:
                result = FileDataset._try_origin(origin_url, target, destination, desc, chunk_size)
                if result is not None:
                    return result

        # Fall back to original URL
        try:
            req = Request(url + target)
            with urlopen(req) as response:
                total_size = int(response.headers.get('Content-Length', 0))
            
            # Convert destination to Path for _download_sequential
            download_path = pathlib.Path(destination) if destination is not None else pathlib.Path(temp_destination.name)
            FileDataset._download_sequential(url + target, download_path, total_size, desc, chunk_size)
    
            if destination is None:
                temp_destination.close()

            return pathlib.Path(destination)

        except (HTTPError, URLError) as e:
            raise ValueError(f"Failed to download file from {url + target}. Error: {str(e)}")

    @staticmethod
    def _download_parallel(urls_and_targets: List[Tuple[str, str]], base_url: str, 
                           destination_dir: str, desc_prefix: str = "Downloading",
                           chunk_size: int = 1024 * 1024,
                           max_workers: Optional[int] = None,
                           origins: Optional[List[str]] = None) -> List[pathlib.Path]:
        """
        Download multiple files in parallel from a base URL.
        
        Arguments:
            urls_and_targets (List[Tuple[str, str]]): List of (url_suffix, target_filename) tuples
            base_url (str): Base URL for downloads (used as fallback)
            destination_dir (str): Directory to save files
            desc_prefix (str): Prefix for progress bar descriptions
            chunk_size (int): Chunk size for downloads
            max_workers (int, optional): Maximum number of parallel workers. Defaults to min(32, num_files)
            origins (List[str], optional): List of alternative URL bases to try first
            
        Returns:
            List[pathlib.Path]: List of downloaded file paths
        """
        os.makedirs(destination_dir, exist_ok=True)
        
        if max_workers is None:
            max_workers = min(32, len(urls_and_targets))
        
        downloaded_files = []
        errors = []
        
        def download_one(url_suffix: str, target: str) -> Tuple[Optional[pathlib.Path], Optional[str]]:
            dest_path = os.path.join(destination_dir, target)
            desc = f"{desc_prefix} {target}"
            
            # Try custom origins first
            if origins:
                for origin_url in origins:
                    result = FileDataset._try_origin(origin_url, url_suffix + target, dest_path, desc, chunk_size)
                    if result is not None:
                        return result, None
            
            # Fall back to original URL
            try:
                full_url = base_url.rstrip('/') + '/' + url_suffix.lstrip('/') + target
                req = Request(full_url)
                with urlopen(req) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                
                FileDataset._download_sequential(full_url, dest_path, total_size, desc, chunk_size)
                return pathlib.Path(dest_path), None
            except Exception as e:
                return None, str(e)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_one, url_suffix, target): (url_suffix, target)
                for url_suffix, target in urls_and_targets
            }
            
            for future in as_completed(futures):
                url_suffix, target = futures[future]
                result, error = future.result()
                if result is not None:
                    downloaded_files.append(result)
                else:
                    errors.append((target, error))
        
        if errors:
            error_msg = f"Failed to download {len(errors)}/{len(urls_and_targets)} files. "
            error_msg += f"First error: {errors[0][0]} - {errors[0][1]}"
            warnings.warn(error_msg)
        
        return downloaded_files

    @staticmethod
    def _download_sequential(url: str, filepath: pathlib.Path, total_size: int, desc: str,
                             chunk_size: int = 1024 * 1024):
        """Download file sequentially with progress bar."""
        import sys
        
        # Convert to Path if it's a string
        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        req = Request(url)
        with urlopen(req) as response:
            if tqdm is not None:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True,
                             unit_divisor=1024, desc=f"Downloading {desc}", file=sys.stdout,
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
                             desc=f"Downloading {desc}", file=sys.stdout, miniters=1,
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
                            sys.stdout.write(f"\r\033[KDownloading {desc}: {_format_bytes(downloaded)}/{_format_bytes(total_size)} ({percent:.1f}%)")
                        else:
                            sys.stdout.write(f"\r\033[KDownloading {desc}: {_format_bytes(downloaded)}...")
                        sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()



class URLDataset(IndexedDataset):
    """
    Abstract base class for URL-backed datasets.

    Each instance reference is a URL.
    """
    pass

class StreamingDataset(IterableDataset):
    """
    Abstract base class for streaming datasets.
    """
    pass

class GeneratedDataset(IterableDataset):
    """
    Abstract base class for generated datasets.
    """
    pass