"""
Dataset Base Classes

This module provides an abstract, PyTorch-style dataset interface for Constraint Optimisation (CO) benchmarks.
With a single line of code, classical benchmarks such as XCSP3, PSPLib, JSPLib, etc. can be downloaded and iterated over.

Whilst the class hierarchy put in place will support more exotic dataset types in the future, with a structure 
put in place that takes inspiration from conventions within the ML community, currently only file-based datasets 
are supported, i.e. datasets where the instances are stored as files on disk. 

The base classes standardize:

- download and local storage of benchmark instances (file-based datasets)
- instance access via ``__len__`` / ``__getitem__`` (PyTorch compatibility)
- optional ``parse``/``transform``/``target_transform`` arguments
- dataset metadata (with sidecar collection)

Main classes:

- :class:`Dataset`: minimal dataset base
- :class:`IndexedDataset`: indexable dataset base, instances are accessible by index
- :class:`FileDataset`: file-based dataset base with download + metadata support

Class hierarchy::

    Dataset (ABC)
    └── IndexedDataset (ABC)
        └── FileDataset (ABC)
            └── XCSP3Dataset
            └── (your dataset here)

To implement a new dataset, one needs to subclass one of the abstract dataset classes,
and provide implementation for the following methods:
- ``category``: return a dictionary of category labels, describing to which subset the dataset has been restricted (year, track, ...)
- ``download``: download the dataset (helper function :func:`_download_file` is provided)

Some optional methods to overwrite are:
- ``collect_instance_metadata``: collect metadata about individual instances (e.g. number of variables, constraints, ...), potentially domain specific 
- ``open``: how to open the instance file (e.g. for compressed files using .xz, .lzma, .gz, ...)

Datasets must also implement the following dataset metadata attributes:
- ``name``: the name of the dataset
- ``description``: a short description of the dataset
- ``homepage``: a URL to the homepage of the dataset
- ``citation``: a list of citations for the dataset

All parts for which an implementation must be provided are marked with an @abstractmethod decorator, 
raising a NotImplementedError if not overwritten.

Dataset files should be downloaded as-is, without any preprocessing or decompression. Upon initial download,
instance-level metadata gets auto collected and stored in a JSON sidecar file. All subsequent accesses to the dataset
will use the sidecar file to avoid re-collecting the metadata.

Iterating over the dataset is done in the same way as a PyTorch dataset. It returns 2-tuples (x,y) of:
- x: instance reference (a file path is the only supported instance reference type at the moment)
- y: instance metadata  (solution, features, origin, etc.)

Example:

.. code-block:: python

    dataset = MyDataset(download=True)
    for x, y in dataset:
        print(x, y)

The dataset also supports PyTorch-style transforms and target transforms.

.. code-block:: python
    dataset = MyDataset(download=True, transform=my_model_loader)
    for model, info in dataset:
        ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import pathlib
import io
import sys
import tempfile
from typing import Any, Optional, Tuple, List, Callable
from urllib.error import URLError
from urllib.request import HTTPError, Request, urlopen
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# tqdm as an optional dependency, provides prettier progress bars
tqdm: Any = None
try:
    from tqdm import tqdm as _tqdm
    tqdm = _tqdm
except ImportError:
    pass

import cpmpy as cp


def _format_bytes(bytes_num):
    """
    Format bytes into human-readable string (e.g., KB, MB, GB).

    Used to display download progress.
    """
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1024.0


class classproperty:
    """
    Descriptor that makes a method work as a class-level property (no () needed).
    Similar to @property, but for class methods.
    """

    def __init__(self, func):
        self.func = func
        self.__isabstractmethod__ = getattr(func, '__isabstractmethod__', False)

    def __get__(self, instance, owner):
        return self.func(owner)

class Dataset(ABC):
    """
    Abstract base class for CO datasets.

    Each instance in a dataset is characterised by a (x, y) pair of:
        x: instance reference (e.g., file path, database key, generated seed, ...)
        y: instance metadata  (solution, features, origin, etc.)
    """
    

    # -------------- Dataset-level metadata (override in subclasses) ------------- #

    @classproperty
    @abstractmethod
    def name(self) -> str: pass

    @classproperty
    @abstractmethod
    def description(self) -> str: pass

    @classproperty
    @abstractmethod
    def homepage(self) -> str: pass

    @classproperty
    def citation(self) -> List[str]:
        return []
    
    # ---------------------------------------------------------------------------- #

    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        """
        Arguments:
            transform (callable, optional):            Optional transform applied to the instance reference.
            target_transform (callable, optional):     Optional transform applied to the instance metadata.
        """
        self.transform = transform
        self.target_transform = target_transform

   
    # ---------------------------------------------------------------------------- #
    #                     Methods to implement in subclasses:                      #
    # ---------------------------------------------------------------------------- #

    @abstractmethod
    def instance_metadata(self, instance) -> dict:
        """
        Return the metadata for a given instance.
        """
        pass


    # ---------------------------------------------------------------------------- #
    #                               Public interface                               #
    # ---------------------------------------------------------------------------- #

    @classmethod
    def dataset_metadata(cls) -> dict:
        """
        Return dataset-level metadata as a dictionary.

        Returns:
            dict: The dataset-level metadata.
        """
        # Handle both single string and list of strings for citations
        if isinstance(cls.citation, str):
            citations = [cls.citation] if cls.citation else []
        else:
            citations = list(cls.citation)

        return {
            "name": cls.name,
            "description": cls.description,
            "homepage": cls.homepage,
            "citation": citations,
        }


class IndexedDataset(Dataset):
    """
    Abstract base class for indexed datasets.

    Indexed datasets are datasets where the instances are indexed by a unique identifier and 
    can be accessed by that identifier. For example its positional index within the dataset.

    Implementing this class requires implementing the following methods:
    - ``__len__``: return the total number of instances
    - ``__getitem__``: return the instance and metadata at the given index / identifier
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
        Return the instance and metadata at the given index / identifier.

        Returns:
            x: instance reference (e.g., file path, database key, generated seed, ...)
            y: instance metadata  (solution, features, origin, etc.)
        """
        pass

    def __iter__(self):
        """
        Iterate over the dataset.
        """
        for i in range(len(self)):
            yield self[i]


class FileDataset(IndexedDataset):
    """
    Abstract base class for PyTorch-style datasets of file-based CO benchmarking sets.

    The `FileDataset` class provides a standardized interface for downloading and
    accessing file-backed benchmark instances. This class should not be used on its own.
    Either have a look at one of the concrete subclasses, providing access to 
    well-known datasets from the community, or use this class as the base for your own dataset.

    Two dataset styles are supported:

    - Model-defined instances: files directly encode variables/constraints/objective
      (for example XCSP3, OPB, DIMACS, FlatZinc). In this case, users typically
      pass a loader as ``transform``, converting the raw file instance into a model.
    - Data-only instances: files encode problem data for a fixed family, but no
      model. In this case, subclasses should override ``parse()`` and users can
      enable ``parse=True`` to obtain parsed intermediate data structures
      (for example table/dict structures for RCPSP-style scheduling data), then
      build a model separately or via a transform.
    """ 
    # TODO documentation to add / improve
    # For a more detailed authoring guide (design patterns, metadata conventions,
    # and implementation checklist), see :ref:`datasets_advanced_authoring`.

    # Extension for metadata sidecar files
    METADATA_EXTENSION = ".meta.json"

    def __init__(
            self,
            dataset_dir: str | os.PathLike[str] = ".",
            transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
            download: bool = False,
            parse: bool = False,
            extension: str = ".txt",
            **kwargs
        ):
        """
        Constructor for the FileDataset base class.

        Arguments:
            dataset_dir (str): Path to the dataset directory.
            transform (callable, optional): Optional transform applied to the instance file path.
            target_transform (callable, optional): Optional transform applied to the metadata dictionary.
            download (bool): If True, downloads the dataset if it does not exist locally (default=False).
            parse (bool): If True, run ``self.parse(instance_path)`` before
                applying ``transform``. Intended for data-only datasets that do
                not directly encode a model in the source file.
            extension (str): Extension of the instance files. Used to filter instance files within the dataset directory.
            **kwargs: Advanced options. Currently supports:
                - ignore_sidecar (bool): If True, do not read/write metadata
                  sidecars and collect metadata on demand at iteration time
                  using ``collect_instance_metadata()`` (default=False).

        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested dataset variant (e.g. year/track) is not available.
            ValueError: If the dataset directory does not contain any instance files.
        """

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.extension = extension
        self._parse = parse

        # Advanced options
        self._ignore_sidecar = kwargs.pop("ignore_sidecar", False)

        if not self._check_exists():
            if not download:
                raise ValueError("Dataset not found. Please set download=True to download the dataset.")
            else:
                self.download()
                if not self._ignore_sidecar:
                    self._collect_all_metadata()
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


    @abstractmethod
    def categories(self) -> dict:
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
        Python standard library's 'open', e.g. '.xz', '.lzma'.

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

    def parse(self, instance: os.PathLike):
        """
        Parse an instance file into intermediate data structures.

        Override this for datasets whose files contain problem data but not an
        explicit model. Typical outputs are structures like tables, arrays, and
        dictionaries that can then be passed to a separate model-construction
        function.

        Default behavior is ``read(instance)``, i.e. return raw text content.

        Arguments:
            instance (os.PathLike): File path to the instance file.

        Returns:
            The parsed intermediate data structure(s).
        """
        return self.read(instance)


    # ---------------------------------------------------------------------------- #
    #                               Public interface                               #
    # ---------------------------------------------------------------------------- #


    def instance_metadata(self, instance: os.PathLike) -> dict:
        """
        Return the metadata for a given instance file.

        Arguments:
            file (os.PathLike): Path to the instance file.

        Returns:
            dict: The metadata for the instance.
        """
        metadata = {
            'id': str(instance),
            'dataset': self.name,
            'categories': self.categories(),
            'name': pathlib.Path(instance).name.replace(self.extension, ''),
            'path': instance,
        }

        # Advanced mode: bypass sidecars and collect metadata on demand.
        if self._ignore_sidecar:
            metadata.update(self.collect_instance_metadata(file=pathlib.Path(instance)))
            return metadata
        else:
            # Load sidecar metadata if it exists
            meta_path = self._metadata_path(instance)
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    sidecar = json.load(f)
                # Structured: flatten instance_metadata
                metadata.update(sidecar.get("instance_metadata", {}))
            return metadata


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
        """
        Return the total number of instances.
        """
        return len(self._list_instances())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return the instance and metadata at the given index.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        files = self._list_instances()
        file_path = files[index]

        metadata = self.instance_metadata(file_path)
        if self.target_transform:
            metadata = self.target_transform(metadata)

        # Instance reference is a string path (PyTorch-style); parse/transform may replace it.
        data: Any = str(file_path)

        # Built-in parse stage: parse the instance file into intermediate data structures.
        # Mostly meant for datasets where files represent data and modeling is separate.
        if self._parse:
            data = self.parse(file_path)

        if self.transform:
            # TODO maybe revisit this flow of execution once CPMpy model feature extraction has been added
            if isinstance(data, (str, os.PathLike)):
                data = self.transform(data, open=self.open)
            else:
                # Convenience for parse-first datasets where parse() returns
                # tuples and model builders take positional args.
                if isinstance(data, tuple):
                    data = self.transform(*data)
                else:
                    data = self.transform(data)
            # Let transforms contribute to metadata (e.g. model verification info)
            if hasattr(self.transform, 'enrich_metadata'):
                metadata = self.transform.enrich_metadata(data, metadata)
            elif isinstance(data, cp.Model):
                # Transform returned a CPMpy model; enrich metadata from model details.
                # metadata = _enrich_from_model(data, metadata) TODO for future metadata PR
                pass
        elif isinstance(data, cp.Model):
            # metadata = _enrich_from_model(data, metadata) TODO for future metadata PR
            pass

        return data, metadata


    # ---------------------------- Metadata collection --------------------------- #

    def _metadata_path(self, instance_path: os.PathLike) -> pathlib.Path:
        """
        Return the path to the `.meta.json` sidecar file for a given instance.

        Arguments:
            instance_path (os.PathLike): Path to the instance file.

        Returns:
            pathlib.Path: Path to the `.meta.json` sidecar file.
        """
        return pathlib.Path(str(instance_path) + self.METADATA_EXTENSION)

    def _collect_all_metadata(self, force: bool = False, workers: int = 1):
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
        if force:
            files_to_process = files
        else:
            for file_path in files:
                meta_path = self._metadata_path(file_path)
                if not meta_path.exists():
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
            
            # 'fork' is Linux-only; 'spawn' is the safe cross-platform default
            start_method = 'fork' if sys.platform == 'linux' else 'spawn'
            ctx = multiprocessing.get_context(start_method)
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
        """
        Collect metadata for a single instance file.

        Arguments:
            file_path (os.PathLike): Path to the instance file.
        """
        meta_path = self._metadata_path(file_path)
        metadata_error = None
        try:
            instance_meta = self.collect_instance_metadata(str(file_path))
        except Exception as e:
            instance_meta = {}
            metadata_error = str(e)

        # Derive logical instance name from dataset-specific extension (strict)
        filename = file_path.name
        suffix_len = len(self.extension)
        if suffix_len and not filename.endswith(self.extension):
            raise ValueError(
                f"Instance file '{filename}' does not end with dataset extension '{self.extension}'."
            )
        stem = filename[:-suffix_len] if suffix_len else filename

        # Build structured sidecar
        sidecar = {
            "dataset": self.dataset_metadata(),
            "instance_name": stem,
            "source_file": str(file_path.relative_to(self.dataset_dir)),
            "categories": self.categories(),
            "instance_metadata": instance_meta,
        }

        # Collect metadata collection error if any
        if metadata_error is not None:
            sidecar["_metadata_error"] = metadata_error

        with open(meta_path, "w") as f:
            json.dump(sidecar, f, indent=2)

            
    # ----------------------------- Download methods ----------------------------- #

    @staticmethod
    def _download_file(url: str, target: str, destination: Optional[str] = None,
                        desc: Optional[str] = None,
                        chunk_size: int = 1024 * 1024) -> os.PathLike:
        """
        Download a file from a URL with progress bar and speed information.

        This method provides a reusable download function with progress updates
        similar to pip and uv, showing download progress, speed, and ETA.

        Arguments:
            url (str): The original URL to download from (used as fallback).
            target (str): The target filename to download.
            destination (str, optional): The destination path to save the file.
            desc (str, optional): Description to show in the progress bar.
                                  If None, uses the filename.
            chunk_size (int): Size of each chunk for download in bytes (default=1MB).

        Returns:
            os.PathLike: The destination path where the downloaded file is saved.
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
    def _download_sequential(url: str, filepath: os.PathLike, total_size: int, desc: str,
                             chunk_size: int = 1024 * 1024):
        """
        Download file sequentially with progress bar.
        """
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


def from_files(dataset_dir: os.PathLike, extension: str = ".txt") -> FileDataset:
    """
    Create a FileDataset from a list of files.

    Example:

    .. code-block:: python

        dataset = from_files("path/to/dataset_files", extension=".txt")
        for x, y in dataset:
            print(x, y)
    """
    class FromFilesDataset(FileDataset):
        # Plain class attributes so that dataset_metadata() (a classmethod
        # that reads cls.name / cls.description / ...) works correctly.
        name = ""
        description = ""
        homepage = ""
        citation: List[str] = []

        def __init__(self, dataset_dir: os.PathLike, extension: str = ".txt"):
            # Set name from the directory so metadata contains something useful.
            self.name = pathlib.Path(dataset_dir).name
            super().__init__(dataset_dir=dataset_dir, extension=extension)

        def categories(self) -> dict:
            return {}

        def download(self) -> None:
            raise NotImplementedError("from_files() datasets are already local; downloading is not supported.")

    return FromFilesDataset(dataset_dir, extension)
