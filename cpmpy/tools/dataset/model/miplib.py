"""
MIPLib Dataset

https://maxsat-evaluations.github.io/
"""


import os
import gzip
import zipfile
import pathlib

from cpmpy.tools.dataset._base import _Dataset


class MIPLibDataset(_Dataset):  # torch.utils.data.Dataset compatible
  

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

        # # Check requested dataset
        # if not str(year).startswith('20'):
        #     raise ValueError("Year must start with '20'")
        # if not track:
        #     raise ValueError("Track must be specified, e.g. OPT-LIN, DEC-LIN, ...")

        dataset_dir = self.root / "miplib"

        super().__init__(
            dataset_dir=dataset_dir, 
            transform=transform, target_transform=target_transform, 
            download=download, extension=".mps.gz"
        )


    def category(self) -> dict:
        return {
            "year": self.year,
            "track": self.track
        }
        
    
    def download(self):
        print("Downloading MIPLib instances...")
        
        zip_name = "collection.zip"
        url = "https://miplib.zib.de/downloads/"

        dataset_dir = self.root / "miplib"

        if dataset_dir.exists():
            print(f"Using existing dataset directory: {dataset_dir}")
        else:
            print(f"Downloading {zip_name}...")
            try:
                cached_filepath = super().download_file(url, target=zip_name, desc=zip_name)
            except ValueError as e:
                raise ValueError(f"No dataset available. Error: {str(e)}")
        
        # Extract only the specific track folder from the tar
        with zipfile.ZipFile(cached_filepath, 'r') as zip_ref:                    
            # Create track folder in root directory, parents=True ensures recursive creation
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract files
            for file_info in zip_ref.infolist():
                # Extract file to family_dir, removing main_folder/track prefix
                filename = pathlib.Path(file_info.filename).name
                with zip_ref.open(file_info) as source, open(self.dataset_dir / filename, 'wb') as target:
                    target.write(source.read())
        # Do not cleanup cached file, as it is in the global cache directory
        # zip_path.unlink()

    def open(self, instance: os.PathLike) -> callable:
        return gzip.open(instance, "rt") if str(instance).endswith(".gz") else open(instance)

if __name__ == "__main__":
    dataset = MIPLibDataset(download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])

    from cpmpy.tools.mps import read_mps
    model = read_mps(dataset[0][0], open=dataset.open)
    print(model)
