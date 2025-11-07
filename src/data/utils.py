import shutil
import tempfile

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from src.data.config import NO_DATA_VALUE


def resample_resolution(tif_path):
    with rasterio.open(tif_path) as src:
        if src.crs.to_string() == "EPSG:4326":
            print(f"File '{tif_path}' is already in EPSG:4326. No reprojection needed.")
            return

        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        print(height, width)
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": "EPSG:4326", "transform": transform, "width": width, "height": height}
        )

        # Use a temporary file to avoid overwriting during processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmpfile:
            temp_path = tmpfile.name

        with rasterio.open(temp_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.nearest,
                )

        shutil.move(temp_path, tif_path)
        print(f"Reprojection complete. Input file '{tif_path}' has been updated.")


class RunningStats:
    """Inspired by: https://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation"""

    def __init__(self, shape):
        self.count = np.zeros(shape)
        self.mean = np.zeros(shape)
        self.M2 = np.zeros(shape)

    # For a new value new_value, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def update(self, new_data):
        assert new_data.ndim == 2, "new_data should be a 2D array (flattened pixels x channels)"
        assert new_data.shape[1] == self.mean.size, (
            "new_data should have the same number of channels as initialized"
        )
        for c in range(new_data.shape[-1]):
            # assert that there is no NO_DATA_VALUE in new_data[:, c]
            assert (new_data[:, c] != NO_DATA_VALUE).all(), f"NO_DATA_VALUE found in channel {c}"
            x = new_data[:, c]
            valid_mask = ~np.isnan(x)
            # if no valid data, skip
            if not valid_mask.any():
                continue
            x_valid = x[valid_mask]
            n = x_valid.size
            if n == 0:
                continue

            for value in x_valid:
                delta = value - self.mean[c]
                self.count[c] += 1
                self.mean[c] += delta / self.count[c]
                delta2 = value - self.mean[c]
                self.M2[c] += (delta * delta2)
                assert self.M2[c] is not np.nan, "M2 has become NaN, something went wrong."
                assert self.mean[c] is not np.nan, "Mean has become NaN, something went wrong."
                assert self.count[c] > 0, "Count is not positive, something went wrong."

    def finalize(self):
        assert (self.count > 1).all(), (
            "Not enough samples to compute standard deviation. Need at least two samples per channel."
        )
        assert (self.M2 >= 0).all(), (
            "M2 has negative values, something went wrong during the update steps."
        )
        # returns mean and standard deviation as per-channel arrays
        std = np.sqrt(self.M2 / (self.count - 1))
        assert not np.isnan(std).any(), (
            "Standard deviation has become NaN in finalize step, something went wrong."
        )
        assert not np.isnan(self.mean).any(), (
            "Mean has become NaN in finalize step, something went wrong."
        )
        return self.mean, std
