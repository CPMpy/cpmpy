"""
MIPLib Dataset

https://maxsat-evaluations.github.io/
"""


import os
import gzip
import zipfile
import pathlib
import io

from cpmpy.tools.dataset._base import _Dataset


class MIPLibDataset(_Dataset):  # torch.utils.data.Dataset compatible

    """
    MIPLib Dataset in a PyTorch compatible format.

    More information on MIPLib can be found here: https://miplib.zib.de/
    """
  
    name = "miplib"

    def __init__(
            self, 
            root: str = ".", 
            year: int = 2024, track: str = "exact-unweighted", 
            transform=None, target_transform=None, 
            download: bool = False
        ):
        """
        Constructor for a dataset object of the MIPLib competition.

        Arguments:
            root (str): Root directory where datasets are stored or will be downloaded to (default="."). 
            year (int): Year of the dataset to use (default=2024).
            track (str): Track name specifying which subset of the dataset instances to load (default="exact-unweighted").
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

        dataset_dir = self.root / self.name / str(year) / track

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
        
        url = "https://miplib.zib.de/downloads/"
        target = "collection.zip"
        target_download_path = self.root / target

        print(f"Downloading MIPLib instances from miplib.zib.de")

        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path))
        except ValueError as e:
            raise ValueError(f"No dataset available on {url}. Error: {str(e)}")
        
        # Extract files
        with zipfile.ZipFile(target_download_path, 'r') as zip_ref:                    
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract files
            for file_info in zip_ref.infolist():
                filename = pathlib.Path(file_info.filename).name
                with zip_ref.open(file_info) as source, open(self.dataset_dir / filename, 'wb') as target:
                    target.write(source.read())

        # Clean up the zip file
        target_download_path.unlink()

    def open(self, instance: os.PathLike) -> io.TextIOBase:
        return gzip.open(instance, "rt") if str(instance).endswith(".gz") else open(instance)


if __name__ == "__main__":
    dataset = MIPLibDataset(download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])

    from cpmpy.tools.io.mps import read_mps
    model = read_mps(dataset[0][0], open=dataset.open)
    print(model)
