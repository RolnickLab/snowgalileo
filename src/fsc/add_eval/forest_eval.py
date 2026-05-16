from pathlib import Path
from typing import Union, cast

import numpy as np
import psutil
import rioxarray
import xarray as xr
from einops import rearrange

from src.config import DEFAULT_SEED
from src.fsc.landsat_eval import Landsa as BaseDataset
from src.data.earthengine.eo_eval import (
    EE_SPACE_BANDS,
)
from src.utils import seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()


class ForestMetaDataset(BaseDataset):
    """Dataset class for retrieving forest metadata from ESA WorldCover data"""

    def __init__(self, data_folder, download=False, h5pys_only=False, *args, **kwargs):
        super().__init__(
            data_folder=data_folder, download=download, h5pys_only=h5pys_only, *args, **kwargs
        )

    @staticmethod
    def retrieve_fractional_forest_cover(worldcover_map: np.ndarray) -> float:
        """Retrieve fractional forest cover from ESA WorldCover map
        Forest classes are 10 (Tree cover)
        """
        total_pixels = worldcover_map.size
        # TODO: remove check later
        assert total_pixels == 100 * 100, "Expected worldcover map to be 100x100 pixels"
        forest_pixels = np.sum(worldcover_map == 10)
        fractional_forest_cover = forest_pixels / total_pixels
        return fractional_forest_cover

    @classmethod
    def _get_ffc_and_location(cls, tif_path: Path):
        with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
            # [all_combined_bands, H, W]
            # all_combined_bands includes all dynamic-in-time bands
            # interleaved for all timesteps
            # followed by the static-in-time bands
            values = cast(np.ndarray, data.values)

            # extract lat, lon in EPSG:4326 from tif_path
            parts = tif_path.stem.split("_")
            lat = float(parts[3])
            lon = float(parts[4])

        space_x = rearrange(
            values[-len(EE_SPACE_BANDS) :],
            "c h w -> h w c",
        )
        space_x = cls._check_and_fillna(space_x, np.array(EE_SPACE_BANDS))

        worldcover_map = space_x[:, :, -1]
        ffc = cls.retrieve_fractional_forest_cover(worldcover_map)

        try:
            assert not np.isnan(space_x).any(), f"NaNs in space bands for {tif_path}"
            assert not np.isinf(space_x).any(), f"Infs in space bands for {tif_path}"
            return ffc, lat, lon
        except AssertionError as e:
            raise e

    def return_fractional_forest_cover_from_filename(self, filename: str):
        fractional_forest_cover_dict: dict[str, Union[int, str, float]] = {"filename": filename}

        tif_path = Path(self.data_folder / filename)
        assert tif_path.exists(), f"File {tif_path} does not exist"
        if tif_path.suffix == ".tif":
            try:
                ffc, lat, lon = self._get_ffc_and_location(tif_path)
                fractional_forest_cover_dict.update(
                    {
                        "fractional_forest_cover": ffc,
                        "lat": lat,
                        "lon": lon,
                    }
                )
                print(f"Processed {tif_path}")
            except Exception as e:
                print(f"Error processing {tif_path}: {e}")
                fractional_forest_cover_dict.update(
                    {
                        "fractional_forest_cover": -1,
                        "lat": "nan",
                        "lon": "nan",
                    }
                )
            return fractional_forest_cover_dict, filename
        else:
            raise ValueError(f"File {tif_path} is not a .tif file")
