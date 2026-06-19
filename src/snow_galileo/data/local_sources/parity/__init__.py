"""Value-domain parity de-risk logic for the local scene adapters (TASK-005/011).

Each module recreates the *value domain* a GEE collection produces for one
``(granule, cell)`` against the Phase-0 reference patch, so adapter drift can be
quantified independently of the production adapters:

- :mod:`.s1` — ``COPERNICUS/S1_GRD`` (calibrated, terrain-corrected σ⁰ dB via SNAP).
- :mod:`.s2` — ``COPERNICUS/S2_HARMONIZED`` (L1C DN with the −1000 harmonization offset).
- :mod:`.s3` — ``COPERNICUS/S3/OLCI`` ortho parity (SNAP ``Reproject orthorectify=true``).

These were originally one-shot spike scripts under ``scripts/spikes/``. The
importable ``run_*`` functions now live here so the parity test suite
(``tests/test_local_sources/test_s*_parity.py``) can import them from the shipped
package; the command-line entrypoints that drive them are thin wrappers under
``scripts/developer_scripts/bow_valley_inference_local/spikes/``. They remain
de-risk tools, **not** production adapters — those are TASK-012/013/014. See
``docs/agents/planning/bow_valley/020-data-ingestion/021-parity-spike-notes.md``.
"""
