"""
PSPlib Dataset

https://www.om-db.wi.tum.de/psplib/getdata_sm.html
"""
import os
import pathlib
from typing import Tuple, Any
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import zipfile

class PSPLibDataset(object):  # torch.utils.data.Dataset compatible

    """
    PSPlib Dataset in a PyTorch compatible format.
    
    More information on PSPlib can be found here: https://www.om-db.wi.tum.de/psplib/main.html
    """
    
    def __init__(self, root: str = ".", variant: str = "rcpsp", family: str = "j30", transform=None, target_transform=None, download: bool = False):
        """
        Constructor for a dataset object for PSPlib.

        Arguments:
            root (str): Root directory containing the psplib instances (if 'download', instances will be downloaded to this location)
            variant (str): scheduling variant (only 'rcpsp' is supported for now)
            family (str): family name (e.g. j30, j60, etc...)
            transform (callable, optional): Optional transform to be applied on the instance data
            target_transform (callable, optional): Optional transform to be applied on the file path
            download (bool): If True, downloads the dataset from the internet and puts it in `root` directory


        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested variant/family combination is not available.
        """
        
        self.root = pathlib.Path(root)
        self.variant = variant
        self.family = family
        self.transform = transform
        self.target_transform = target_transform
        self.family_dir = pathlib.Path(os.path.join(self.root, variant, family))
        
        self.families = dict(
            rcpsp = ["j30", "j60", "j90", "j120"]
        )
        self.family_codes = dict(rcpsp="sm", mrcpsp="mm")

        if variant != "rcpsp":
            raise ValueError("Only 'rcpsp' variant is supported for now")
        if family not in self.families[variant]:
            raise ValueError(f"Unknown problem family. Must be any of {','.join(self.families[variant])}")
        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)
        
        if not self.family_dir.exists():
            if not download:
                raise ValueError(f"Dataset for variant {variant} and family {family} not found. Please set download=True to download the dataset.")
            else:
                print(f"Downloading PSPLib {variant} {family} instances...")
                
                zip_name = f"{family}.{self.family_codes[variant]}.zip"
                url = f"https://www.om-db.wi.tum.de/psplib/files/"

                url_path = url + zip_name
                zip_path = self.root / zip_name
                
                try:
                    urlretrieve(url_path, str(zip_path))
                except (HTTPError, URLError) as e:
                    raise ValueError(f"No dataset available for variant {variant} and family {family}. Error: {str(e)}")
                
                # make directory and extract files
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:                    
                    # Create track folder in root directory, parents=True ensures recursive creation
                    self.family_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Extract files
                    for file_info in zip_ref.infolist():
                        # Extract file to family_dir, removing main_folder/track prefix
                        filename = pathlib.Path(file_info.filename).name
                        with zip_ref.open(file_info) as source, open(self.family_dir / filename, 'wb') as target:
                            target.write(source.read())
                # Clean up the zip file
                zip_path.unlink()
            
    def open(self, instance: os.PathLike) -> callable:
        return open(instance, "r")

        
    def __len__(self) -> int:
        """Return the total number of instances."""
        return len(list(self.family_dir.glob(f"*.{self.family_codes[self.variant]}")))
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a single RCPSP instance filename and metadata.

        Args:
            index (int): Index of the instance to retrieve
            
        Returns:
            Tuple[Any, Any]: A tuple containing:
                - The filename of the instance
                - Metadata dictionary with file name, track, year etc.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        # Get all instance files and sort for deterministic behavior # TODO: use natsort instead?
        files = sorted(list(self.family_dir.glob(f"*.{self.family_codes[self.variant]}")))
        file_path = files[index]

        filename = str(file_path)
        if self.transform:
            # does not need to remain a filename...
            filename = self.transform(filename)
            
        # Basic metadata about the instance
        metadata = dict(
            variant = self.variant,
            family = self.family,
            name = file_path.stem
        )
        
        if self.target_transform:
            metadata = self.target_transform(metadata)
            
        return filename, metadata
    
if __name__ == "__main__":
    dataset = PSPLibDataset(variant="rcpsp", family="j30", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])