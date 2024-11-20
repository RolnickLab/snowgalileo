# TODO: delete this file later
from src.data.config import TIFS_FOLDER, DATA_FOLDER
from src.data.dataset import Dataset

dataset = Dataset(
    TIFS_FOLDER,
    download=True,
    h5py_folder=DATA_FOLDER / "h5pys",
    h5pys_only=False,
)