We will use the daily, gap-filled GSP product, as can be downloaded from https://download.geoservice.dlr.de/GSP/files/daily/ or used via STAC from . Bit positions 4-8 refer to the snow class. We want to binarize the product into value >= 64 == snow and value < 64 == no snow. If available, we want to use the ´NRT_SCE/´ product (re-processed), else ´SCE´. There might be unavailable data (after May 2025). 

The product is divided into daily, global composites, so we need to find a strategy to divide this into local patches. The projection of the data is WGS84, so we might need to convert our Landsat masks for comparison.

We need to create a GlobalSnowpack set that contains all data within our test sets. We will need to interpolate the Landsat masks to a resolution of 500m and binarize them.

[Plan]
1) We will have our test masks, and from the filenames of these test masks, we want to infer the date and the bounding box, to retrieve the corresponding data from GlobalSnowpack via STAC.
2) The eval filenames will be in format: `LC09_20230326_FSC92_77.31429713075256_-65.40066134098144.tif`.
3) Once we retrieved date and bounding box, we can retrieve the according file with `pystac`, and then crop the `.tif` file with `rasterio`.
4) We can then store the resulting files in a new folder.
5) Finally, we can create a evaluation setup for comparing the masks against the (binarized) labels.