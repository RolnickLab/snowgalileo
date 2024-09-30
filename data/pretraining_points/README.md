In addition to the two files stored in git, the GeoGLANCE dataset can be exported with the following GEE code:

```js
var glance_training = ee.FeatureCollection("projects/sat-io/open-datasets/GLANCE/GLANCE_TRAINING_DATA_V1")
var glance_locs = glance_training.select("Glance_ID")
Export.table.toDrive({
  collection: glance_locs,
  description: 'glance_locations_only',
  fileFormat: 'GeoJSON'
});

```
