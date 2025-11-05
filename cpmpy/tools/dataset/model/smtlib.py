"""
PyTorch-style Dataset for SMT-LIB benchmark instances.

Simply create a dataset instance (configured for the targeted logic) and start iterating over its contents:

.. code-block:: python

    from cpmpy.tools.smtlib_dataset import SMTLibDataset
    from cpmpy.tools.smtlib import read_smtlib

    for filename, metadata in SMTLibDataset(year=2025, logic="QF_LIA", download=True): # auto download dataset and iterate over its instances
        # Do whatever you want here, e.g. reading to a CPMpy model and solving it:
        model = read_smtlib(filename)
        model.solve()
        print(model.status())

The `metadata` contains useful information about the current problem instance.

Since the dataset is PyTorch compatible, it can be used with a DataLoader:

.. code-block:: python

    from cpmpy.tools.smtlib_dataset import SMTLibDataset
    from cpmpy.tools.smtlib import read_smtlib

    # Initialize the dataset
    dataset = SMTLibDataset(year=2025, logic="QF_LIA", download=True)

    from torch.utils.data import DataLoader

    # Wrap dataset in a DataLoader
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Iterate over the dataset
    for batch in data_loader:
        # Your code here
"""

import pathlib
import json
import tarfile
from typing import Tuple, Any, Optional, List
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError

from .._base import _Dataset

class SMTLibDataset(_Dataset):  # torch.utils.data.Dataset compatible

    """
    SMT-LIB Dataset in a PyTorch compatible format.
    
    More information on the dataset can be found here: https://smt-lib.org/benchmarks.shtml
    """
    
    # Zenodo record ID for SMT-LIB 2025 release
    ZENODO_RECORD_ID = {
        2025: "16740866",
        2024: "11061097",
    }
    
    def __init__(
            self, 
            root: str = ".",
            year: int = 2025, logic: Optional[str] = "QF_LIA",
            transform=None, target_transform=None, 
            download: bool = False
        ):
        """
        Initialize the SMT-LIB Dataset.

        Arguments:
            root (str): Root directory containing the SMT-LIB instances (if 'download', instances will be downloaded to this location)
            year (int): SMT-LIB release year (default: 2025)
            logic (str): SMT-LIB logic name (e.g., "QF_LIA", "QF_UFLIA", "LIA", etc.) (default: "QF_LIA").
            transform (callable, optional): Optional transform to be applied on the instance data (the file path of each problem instance)
            target_transform (callable, optional): Optional transform to be applied on the metadata (the metadata dictionary of each problem instance)
            download (bool): If True, downloads the dataset from Zenodo and puts it in `root` directory

        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested year/logic combination is not available.
        """
        self.root = pathlib.Path(root)
        self.year = year
        self.logic = logic
        
        if year not in self.ZENODO_RECORD_ID:
            raise ValueError(f"Only years {list(self.ZENODO_RECORD_ID.keys())} are currently supported. Got year={year}")
        if not logic:
            raise ValueError("Logic must be specified, e.g. QF_LIA, QF_UFLIA, LIA, ...")

        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Get available logics from Zenodo
        self._available_logics = None
        
        dataset_dir = self.root / str(year) / logic
        
        super().__init__(
            dataset_dir=dataset_dir, 
            transform=transform, target_transform=target_transform, 
            download=download, extension=".smt2"
        )
    
    def _get_available_logics(self) -> List[str]:
        """
        Get list of available logics from Zenodo.
        """
        if self._available_logics is not None:
            return self._available_logics
        
        try:
            import urllib.request
            files_url = f"https://zenodo.org/api/records/{self.ZENODO_RECORD_ID[self.year]}/files"
            with urllib.request.urlopen(files_url) as response:
                data = json.loads(response.read())
                # Extract logic names from file keys (e.g., "QF_LIA.tar.zst" -> "QF_LIA")
                self._available_logics = [
                    f['key'].replace('.tar.zst', '') 
                    for f in data['entries'] 
                    if f['key'].endswith('.tar.zst')
                ]
            return self._available_logics
        except Exception as e:
            raise ValueError(f"Could not fetch available logics from Zenodo. Error: {str(e)}")

    def category(self) -> dict:
        return {
            "year": self.year,
            "logic": self.logic
        }
    
    def download(self):
        """
        Download and extract a specific logic archive.
        """
        print(f"Downloading SMT-LIB {self.year} instances for logic {self.logic}...")
        
        # Get available logics to validate
        available_logics = self._get_available_logics()
        if self.logic not in available_logics:
            raise ValueError(f"Logic '{self.logic}' not found. Available logics: {sorted(available_logics)}")
        
        # Check if archive already exists
        archive_name = f"{self.logic}.tar.zst"
        archive_path = self.root / archive_name
        
        if not archive_path.exists():
            # Download the archive
            archive_url = f"https://zenodo.org/api/records/{self.ZENODO_RECORD_ID[self.year]}/files/{archive_name}/content"
            
            try:
                print(f"Downloading {archive_name}...")
                urlretrieve(archive_url, str(archive_path))
            except (HTTPError, URLError) as e:
                raise ValueError(f"Could not download archive for logic {self.logic}. Error: {str(e)}")
        else:
            print(f"Archive {archive_name} already exists, skipping download...")
        
        # Extract the archive
        print(f"Extracting {archive_name}...")
        self._extract_tar_zst(archive_path, self.dataset_dir)
        
        # Clean up the archive file
        archive_path.unlink()
        print(f"Downloaded and extracted {self.logic} instances to {self.dataset_dir}")
        
    def _extract_tar_zst(self, archive_path: pathlib.Path, extract_to: pathlib.Path):
        """
        Extract a .tar.zst file to the specified directory using Python libraries.
        Handles nested directory structures by flattening .smt2 files into the target directory.
        """
        extract_to.mkdir(parents=True, exist_ok=True)
        
        try:
            import zstandard as zstd
        except ImportError:
            raise ImportError(
                "The 'zstandard' library is required to extract .tar.zst files.\n"
                "Please install it with: pip install zstandard"
            )
        
        try:
            # Decompress with zstd and extract tar archive
            dctx = zstd.ZstdDecompressor()
            with open(archive_path, 'rb') as compressed:
                with dctx.stream_reader(compressed) as reader:
                    # Use streaming mode ('r|') and iterate members directly
                    with tarfile.open(fileobj=reader, mode='r|') as tar:
                        # Extract only .smt2 files, flattening nested directory structure
                        for member in tar:
                            if member.isfile() and member.name.endswith('.smt2'):
                                # Get relative path and flatten nested directories
                                relative_path = member.name
                                # Flatten path by replacing / with _
                                if '/' in relative_path:
                                    flat_name = relative_path.replace('/', '_')
                                else:
                                    flat_name = relative_path
                                
                                target_path = extract_to / flat_name
                                
                                # Extract file
                                with tar.extractfile(member) as source:
                                    if source:
                                        with open(target_path, 'wb') as target:
                                            target.write(source.read())
        except Exception as e:
            raise ValueError(f"Could not extract archive {archive_path}. Error: {str(e)}")
    
    def __len__(self) -> int:
        """
        Return the total number of instances.
        """
        return len(list(self.dataset_dir.glob("*.smt2")))
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a single SMT-LIB instance filename and metadata.

        Args:
            index (int): Index of the instance to retrieve
            
        Returns:
            Tuple[Any, Any]: A tuple containing:
                - The filename of the instance
                - Metadata dictionary with file name, logic, year etc.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        # Get all SMT2 files
        files = sorted(list(self.dataset_dir.glob("*.smt2")))
        
        file_path = files[index]

        filename = str(file_path)
        if self.transform:
            # does not need to remain a filename...
            filename = self.transform(filename)
        
        # Extract logic from path
        logic_name = file_path.parts[-2]
        
        # Basic metadata about the instance
        metadata = {
            'year': self.year,
            'logic': logic_name,
            'name': file_path.stem,
            'path': filename,
        }
        if self.target_transform:
            metadata = self.target_transform(metadata)
            
        return filename, metadata
    
    def __iter__(self):
        """
        Iterate over the dataset.
        """
        for index in range(len(self)):
            yield self[index]
    
    @classmethod
    def list_available_logics(cls, year: int = 2025) -> List[str]:
        """
        List all available logics for a given year.
        
        Args:
            year (int): SMT-LIB release year (default: 2025)
            
        Returns:
            List of available logic names
        """
        if year not in cls.ZENODO_RECORD_ID:
            raise ValueError(f"Only years {list(cls.ZENODO_RECORD_ID.keys())} are currently supported. Got year={year}")
        
        # Create a temporary instance just to get available logics without downloading
        temp_instance = cls.__new__(cls)
        temp_instance.year = year
        temp_instance.ZENODO_RECORD_ID = cls.ZENODO_RECORD_ID
        temp_instance._available_logics = None
        return temp_instance._get_available_logics()

