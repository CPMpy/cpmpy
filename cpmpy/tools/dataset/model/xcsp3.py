"""
XCS3 Dataset

https://xcsp.org/instances/
"""

from functools import partial
import os
import lzma
import zipfile
import pathlib
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError

from cpmpy.tools.dataset._base import _Dataset


class XCSP3Dataset(_Dataset):
    """
    XCSP3 benchmark dataset.

    Provides access to benchmark instances from the XCSP3
    competitions. Instances are grouped by `year` and `track` (e.g., 
    `"CSP"`, `"eCOP"`) and stored as `.xml.lzma` files. 
    If the dataset is not available locally, it can be automatically 
    downloaded and extracted.

    More information on the competition can be found here: https://xcsp.org/competitions/
    """

    def __init__(
            self,
            root: str = ".", 
            year: int = 2023, track: str = "CSP", 
            transform=None, target_transform=None, 
            download: bool = False
        ):
        """
        Constructor for a dataset object of the XCP3 competition.

        Arguments:
            root (str): Root directory where datasets are stored or will be downloaded to (default="."). 
            year (int): Competition year of the dataset to use (default=2024).
            track (str): Track name specifying which subset of the competition instances to load (default="CSP").
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
            raise ValueError("Track must be specified, e.g. COP, CSP, ...")

        dataset_dir = self.root / str(year) / track

        super().__init__(
            dataset_dir=dataset_dir, 
            transform=transform, target_transform=target_transform, 
            download=download, extension=".xml.lzma"
        )


    def category(self) -> dict:
        return {
            "year": self.year,
            "track": self.track
        }

    def download(self):
        print(f"Downloading XCSP3 {self.year} instances...")

        url = f"https://www.cril.univ-artois.fr/~lecoutre/compets/"
        year_suffix = str(self.year)[2:]  # Drop the starting '20'
        url_path = url + f"instancesXCSP{year_suffix}.zip"
        zip_path = self.root / f"instancesXCSP{year_suffix}.zip"

        try:
            urlretrieve(url_path, str(zip_path))
        except (HTTPError, URLError) as e:
            raise ValueError(f"No dataset available for year {self.year}. Error: {str(e)}")
        
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
            if self.track not in tracks:
                raise ValueError(f"Track '{self.track}' not found in dataset. Available tracks: {sorted(tracks)}")
            
            # Create track folder in root directory, parents=True ensures recursive creation
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract files for the specified track
            prefix = f"{main_folder}/{self.track}/"
            for file_info in zip_ref.infolist():
                if file_info.filename.startswith(prefix):
                    # Extract file to track_dir, removing main_folder/track prefix
                    filename = pathlib.Path(file_info.filename).name
                    with zip_ref.open(file_info) as source, open(self.dataset_dir / filename, 'wb') as target:
                        target.write(source.read())
        # Clean up the zip file
        zip_path.unlink()

    def open(self, instance: os.PathLike) -> callable:
        return lzma.open(instance, mode='rt', encoding='utf-8') if str(instance).endswith(".lzma") else open(instance)


if __name__ == "__main__":
    dataset = XCSP3Dataset(year=2024, track="MiniCOP", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
