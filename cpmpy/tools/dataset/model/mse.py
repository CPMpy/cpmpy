"""
MaxSAT Evaluation (MSE) Dataset

https://maxsat-evaluations.github.io/
"""


import os
import lzma
import zipfile
import pathlib
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError

from .._base import _Dataset

class MSEDataset(_Dataset):  # torch.utils.data.Dataset compatible
    """
    MaxSAT Evaluation (MSE) benchmark dataset.

    Provides access to benchmark instances from the MaxSAT Evaluation 
    competitions. Instances are grouped by `year` and `track` (e.g., 
    `"exact-unweighted"`, `"exact-weighted"`) and stored as `.wcnf.xz` files. 
    If the dataset is not available locally, it can be automatically 
    downloaded and extracted.

    More information on the competition can be found here: https://maxsat-evaluations.github.io/
    """

    def __init__(
            self, 
            root: str = ".", 
            year: int = 2024, track: str = "exact-unweighted", 
            transform=None, target_transform=None, 
            download: bool = False
        ):
        """
        Constructor for a dataset object of the MSE competition.

        Arguments:
            root (str): Root directory where datasets are stored or will be downloaded to (default="."). 
            year (int): Competition year of the dataset to use (default=2024).
            track (str): Track name specifying which subset of the competition instances to load (default="exact-unweighted").
            transform (callable, optional): Optional transform applied to the instance file path.
            target_transform (callable, optional): Optional transform applied to the metadata dictionary.
            download (bool): If True, downloads the dataset if it does not exist locally (default=False).


        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested year/track combination is not available.
        """

        self.root = pathlib.Path(root)
        self.year = year
        self.track = track

        # Check requested dataset
        if not str(year).startswith('20'):
            raise ValueError("Year must start with '20'")
        if not track:
            raise ValueError("Track must be specified, e.g. OPT-LIN, DEC-LIN, ...")

        dataset_dir = self.root / str(year) / track

        super().__init__(
            dataset_dir=dataset_dir, 
            transform=transform, target_transform=target_transform, 
            download=download, extension=".wcnf.xz"
        )


    def category(self) -> dict:
        return {
            "year": self.year,
            "track": self.track
        }
        
    
    def download(self):
        print(f"Downloading MaxSAT Eval {self.year} {self.track} instances...")
        
        zip_name = f"mse{str(self.year)[2:]}-{self.track}.zip"
        url = f"https://www.cs.helsinki.fi/group/coreo/MSE{self.year}-instances/"

        url_path = url + zip_name
        zip_path = self.root / zip_name
        
        try:
            urlretrieve(url_path, str(zip_path))
        except (HTTPError, URLError) as e:
            raise ValueError(f"No dataset available for year {self.year} and track {self.track}. Error: {str(e)}")
        
        # Extract only the specific track folder from the tar
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:                    
            # Create track folder in root directory, parents=True ensures recursive creation
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract files
            for file_info in zip_ref.infolist():
                # Extract file to family_dir, removing main_folder/track prefix
                filename = pathlib.Path(file_info.filename).name
                with zip_ref.open(file_info) as source, open(self.dataset_dir / filename, 'wb') as target:
                    target.write(source.read())
        # Clean up the zip file
        zip_path.unlink()

    def open(self, instance: os.PathLike) -> callable:
        return lzma.open(instance, "rt") if str(instance).endswith(".xz") else open(instance)

if __name__ == "__main__":
    dataset = MSEDataset(year=2024, track="exact-weighted", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
