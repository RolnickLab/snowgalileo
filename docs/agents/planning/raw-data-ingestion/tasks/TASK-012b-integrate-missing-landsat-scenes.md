# TASK-012b: Integrate missing Landsat L1TP scenes (clip + manifest) for parity coverage

## 1. Goal
Ingest the manually-downloaded Landsat 8/9 L1TP scenes that are absent from the clipped
archive but required to **pixel-match the GEE reference patches** in TASK-012, so the
Landsat adapter can be validated bit-for-bit (like DEM/ERA5/MODIS/VIIRS) instead of by
structure alone. Run the existing AOI clip stage over the new raw scenes and rebuild the
Landsat clip manifest, leaving the rest of the archive untouched.

## 2. Context & Why
- TASK-012 parity stalled on a **data-coverage gap**, not an adapter bug. The adapter
  mechanics were confirmed correct: scenes are **EPSG:32612** (cross-zone to the
  EPSG:32611 cell), MTL carries `REFLECTANCE_MULT/ADD_BAND_*` + `SUN_ELEVATION`, and the
  GEE `LANDSAT/LC0{8,9}/C02/T1_TOA` formula is `ρ = (M·DN + A) / sin(sun_elevation)`.
- For every reference patch, the timestep where GEE has Landsat data lands on a date
  whose **imaging WRS-2 path/row is missing from the archive** (e.g. `PR_20250406` t3
  needs path/row **043/024 on 2025-04-02**; the archive's 043024 has only Mar01/Mar17/
  Apr18). The clip/download stage skipped these acquisitions.
- **3 of 6 patches already have a covering product** and can be validated now
  (`PR_20250423` t2 = 043024 2025-04-18 L9; `PR_20250502` t0/t1 = 044024 2025-04-25 L9 /
  043025 2025-04-26 L8; `PR_20250519` t1 = 042024 2025-05-13 L9). The downloads below
  unlock the remaining 3 patches (`PR_20250406`, `PR_20250414`, `PR_20250510`).
- **Patch/window alignment confirmed:** all 6 patches are dated 2025-04-06 → 2025-05-19
  (inside the archive's 2025-03→06 span) and centred in the Bow Valley AOI
  (lon −114.97…−116.41, lat 50.7…52.0). They are valid in-window parity targets — the
  caveat about pre-feature patches misaligning temporally/spatially does **not** apply
  to these six.

## 3. Scenes to download (manual, USGS EarthExplorer / M2M — Collection 2 Level-1)
Required to pixel-match the three uncovered patches. Satellite inferred from the 16-day
WRS-2 repeat cycle (verify on download; grab whichever satellite actually imaged the
path/row on the date — the patch only needs **one** covering product per (date)).

| Patch / timestep | Acquisition date | WRS-2 path/row (imaging) | Likely satellite |
|---|---|---|---|
| PR_20250406 t3 | **2025-04-02** | **043/024** (alt 043/025) | L9 |
| PR_20250414 t2 | **2025-04-09** | **043/024** (alt 044/024) | L8 or L9 (verify) |
| PR_20250510 t1 | **2025-05-04** | **043/024** (alt 042/024) | L9 |

Notes:
- **043/024 on all three dates** is the single most valuable tile — it images all three
  uncovered patches. Downloading just the three 043/024 scenes (2025-04-02, 2025-04-09,
  2025-05-04) is sufficient; the alternates are fallbacks if a 043/024 scene is cloud-
  filled or unavailable.
- Product type: **L1TP, Collection 2, Tier 1** (`LC0{8,9}_L1TP_043024_<date>_..._02_T1`),
  the same product family already in the archive (raw DN + `_MTL.json` + `QA_PIXEL`).
- Drop the downloaded `.tar` (or extracted scene dir) into the raw staging the clip stage
  reads (the same place `data/bow_valley_selection_raw/landsat{8,9}/` came from).

## 4. Integration steps (after download lands)
1. Place new raw scenes in the Landsat raw staging dir.
2. Re-run the AOI clip stage for **Landsat only** over the new scenes (do not re-clip the
   other modalities); output to `data/clipped_bow_valley_selection_raw/landsat{8,9}/`.
3. Rebuild/append the Landsat `clip_manifest.csv` rows for the new products.
4. Verify each new clipped scene has valid (>0) pixels over its target patch footprint
   (inspect the **B4 array**, not the manifest count — see [[s3-clip-scale-factor-bug]]
   lesson on inflated counts).
5. Hand back to TASK-012 to add the bit-exact GEE parity assertion for the now-covered
   patches.

## 5. Acceptance Criteria
- [ ] The three (date, 043/024) scenes (or validated alternates) are clipped into the
      archive in **EPSG:32612** L1TP form with `_MTL.json` + `QA_PIXEL` present.
- [ ] Landsat `clip_manifest.csv` includes the new products; other modalities untouched.
- [ ] For each of `PR_20250406` / `PR_20250414` / `PR_20250510`, the matching clipped
      scene yields >50 % valid B4 pixels over the patch footprint (array-level check).
- [ ] No new failures vs `TEST_BASELINE.md` (clip tests delta).

## 6. Completion Protocol
1. Verify ACs. 2. Commit the new clipped scenes + manifest update.
3. Notify the user; resume TASK-012 to add bit-exact Landsat parity for the covered patches.
