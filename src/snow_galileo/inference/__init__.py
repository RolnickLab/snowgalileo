"""Stage-2 inference orchestration for the Bow Valley direct-source pipeline.

This package is **additive**: it adds a parallel, direct-source inference entry
point (:class:`~snow_galileo.inference.driver.InferenceGridDriver` +
:class:`~snow_galileo.inference.mosaic.DailyMosaicWriter`) alongside the existing GEE
``LandsatEval._predict_and_store_output`` path. It edits **no** downstream code —
the loader (``LandsatEvalDataset``), the model (``EncoderWithHead``), and the
GEE pipeline keep working unchanged. The one fragile coupling to the loader's
folder-driven ``__init__`` is isolated in :mod:`snow_galileo.inference._loader_bridge`.

Modules:
- :mod:`~snow_galileo.inference.windows` — sliding 8-day window + inference-day enumeration.
- :mod:`~snow_galileo.inference.mosaic` — :class:`DailyMosaicWriter` (per-day FSC COG, UTM 11N).
- :mod:`~snow_galileo.inference.driver` — :class:`InferenceGridDriver` (per-cell export → encoder → mosaic).
- :mod:`~snow_galileo.inference._loader_bridge` — read-only shim onto the unchanged loader.
"""
