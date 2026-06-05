# TASK-013b: Integrate missing Sentinel-2 L1C granules (clip + manifest) for full parity coverage

## 1. Goal
Ingest the manually-downloaded Sentinel-2 L1C granules that are absent from the clipped
archive but required to **pixel-match every S2 timestep** of the GEE reference patches, so
the S2 adapter (TASK-013) can be validated bit-for-bit on **all** timesteps, not just the
one covered date per patch it already passes. Run the existing AOI clip stage over the new
raw granules and rebuild the S2 clip manifest, leaving the rest of the archive untouched.

## 2. Context & Why
- TASK-013 landed with **bit-exact B4 parity** (signed median 0, nearest + −1000 DN) on one
  covered date per patch. It is **not** blocked — this task only *extends* coverage.
- The clipped S2 archive has a **3,7,3,7… day cadence**: the 7-day gaps are dates where the
  real ≤5-day S2 acquisitions were never downloaded. Each reference patch carries S2 data on
  timesteps whose dates fall inside those gaps. Same coverage-gap class as **TASK-012b**.
- The coverage-validation test
  (`tests/test_local_sources/test_s2_adapter.py::test_every_patch_has_a_covered_s2_date`)
  **xfails** with the exact backlog below; integrating these dates flips it to pass and
  unlocks per-timestep S2 parity.

## 3. Granules to download (manual — Copernicus / USGS, L1C Collection-1, baseline N0511)
For each (patch, date), grab whichever **T11U** MGRS tile** covers the patch footprint
(centre coords below; pick the tile as TASK-012b did — verify it covers the patch on
download). Product family: **`S2{A,B}_MSIL1C_…_N0511_…_T11U**`**, the same already in the
archive (raw JP2 + `MTD_MSIL1C.xml` + `MSK_CLASSI`).

| Patch | Missing date(s) | Patch centre (lon, lat) | Patch centre UTM 11N (x, y) |
|---|---|---|---|
| PR_20250406 | 2025-04-05 | −116.104, 51.026 | 562865, 5653085 |
| PR_20250414 | 2025-04-08 | −116.407, 51.855 | 540865, 5745085 |
| PR_20250423 | 2025-04-18 | −115.630, 51.138 | 595865, 5666085 |
| PR_20250502 | 2025-04-25, 2025-04-28 | −116.194, 50.748 | 556865, 5622085 |
| PR_20250510 | 2025-05-05, 2025-05-08 | −114.970, 51.830 | 639865, 5744085 |
| PR_20250519 | 2025-05-15, 2025-05-18 | −115.168, 51.977 | 625860, 5760085 |

9 acquisition dates total. A single date may need more than one MGRS tile if the patch sits
on a tile seam — grab every T11U** tile intersecting the patch footprint for that date.

## 4. Integration steps (after download lands)
1. Place new raw granules in `data/bow_valley_selection_raw/sentinel2/`.
2. Re-run the AOI clip stage for **S2 only**:
   `uv run python scripts/developer_scripts/clip_dataset.py clip-all --only sentinel2`
   (re-walks the dir, idempotent; do not re-clip other modalities).
3. The S2 `clip_manifest.csv` is rebuilt automatically by the clip stage.
4. Verify each new clipped granule has valid (>0) **B04** pixels over its target patch
   footprint (inspect the array, not the manifest count — [[s3-clip-scale-factor-bug]]).
5. Extend `test_s2_adapter.py::_PARITY_CASES` with the now-covered (patch, timestep, date)
   triples; the coverage test should flip from xfail to pass.

## 5. Acceptance Criteria
- [ ] The 9 missing dates' covering T11U** granules are clipped into the archive
      (EPSG:32611 L1C, `MTD_MSIL1C.xml` + JP2 bands present).
- [ ] S2 `clip_manifest.csv` includes the new granules; other modalities untouched.
- [ ] `test_every_patch_has_a_covered_s2_date` passes (no xfail) — every patch fully covered.
- [ ] Added per-timestep B4 parity cases stay **bit-exact** (signed median 0).
- [ ] No new failures vs `TEST_BASELINE.md` (clip + adapter delta).

## 6. Completion Protocol
1. Verify ACs. 2. Commit the new clipped granules + manifest + test updates.
3. Notify the user.
