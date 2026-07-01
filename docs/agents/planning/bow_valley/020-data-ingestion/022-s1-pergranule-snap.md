# PLAN — Sentinel-1 per-granule SNAP processing (process-then-clip)

*Formerly `PLAN-S1-PERGRANULE-SNAP.md`.*

## Problem

The current S1 chain processes SNAP **per `(granule, cell)`** against the **clipped**
SAFEs, and subsets **in radar geometry** (the `Subset` node sits between
`Apply-Orbit-File` and `Terrain-Correction`). Two consequences:

1. **Fragile.** A cell whose bbox overlaps the granule footprint but whose
   GCP-clipped scene has no measurement pixels there makes SNAP exit with
   `Error: Empty region!`. Pre-fix, the first such cell aborted the whole 344-cell
   build. (Now tolerated per-cell — but it should not happen at all.)
2. **Slow / wrong-ordered.** Thousands of cold-start SNAP JVM invocations
   (32 granules × ~344 cells, footprint-gated). The radar-geometry subset also
   violates S1 GRD best practice — radiometric/noise/calibration/TC should run on
   the full product, and the geoRegion crop should be applied **after**
   Terrain-Correction (in map geometry, a clean raster window).

## Proof (2026-06-11)

One **raw** 1.7 GB granule, SNAP graph with `Subset` moved to **after**
Terrain-Correction, AOI-bbox geoRegion:

- **3 m 09 s** wall, ~18 GB peak RSS, 993 MB output → **32 granules ≈ ~1h40m serial**.
- **Zero** `NullPointerException` / `Empty region` / errors.
- Output: EPSG:32611, 10 m, 3 bands (Sigma0_VH linear, Sigma0_VV linear,
  incidenceAngleFromEllipsoid 0–46°), ~70 % nonzero over the AOI bbox (the swath
  covers most of it; corners the diagonal swath misses are zero — expected).

Conclusion: process the **full raw granule once → AOI-wide terrain-corrected dB tif**,
then clip/window per cell as a pure raster read. No per-cell SNAP, no empty-region NPE.

## Design

A standalone **`s1_snap` pre-clip stage** (NOT inside `clip_sentinel1` — keeps SNAP's
heavyweight JVM/orbit/SRTM cost out of the fast parallel clip pool; clip stays
pure-raster and runs even if SNAP is absent for other sensors).

```
raw granule (radar geom, GCPs, 1.7 GB)
   │  SNAP: orbit → TNR → border-noise → calibration(σ⁰ linear)
   │        → Terrain-Correction(EPSG:32611, 10 m, +ellipsoid angle)
   │        → Subset(geoRegion = AOI bbox, in MAP geometry)        [ONCE per granule]
   ▼
s1_grd_<granule>.tif   (AOI-wide, EPSG:32611, 10 m, 3 bands, dB-in-adapter)
   │  S1Adapter: per-cell window via reproject_to_cell (pure raster read)
   ▼
per-cell [VV, VH, angle] block
```

The cache key drops from `(granule, cell)` to **`granule`** — one AOI-wide tif per
granule (32 tifs), not thousands of per-cell tifs.

## Changes

| #   | Component                      | Change                                                                                                                                                                                                                                 | Contract impact                           |
| --- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| 1   | `s1_grd_graph.xml`             | Move `Subset` to **after** `Terrain-Correction`; source `Apply-Orbit-File` → `ThermalNoiseRemoval` directly. `${region}` becomes the **AOI bbox** (map geom), not per-cell.                                                            | Graph only; output bands/order unchanged. |
| 2   | `s1_snap.py` `cache_tif_name`  | Drop the `_cell{id}` suffix → `s1_grd_<stem>.tif`.                                                                                                                                                                                     | Filename contract.                        |
| 3   | `s1_snap.py` build fns         | `build_granule_cache` runs SNAP **once** per granule over the AOI bbox (no per-cell loop). `cells` param → single `aoi_4326` region. Reads from the **raw** archive.                                                                   | API of build fns.                         |
| 4   | `s1_snap.py` `ensure_s1_cache` | Key needed/missing by **granule** (footprint gate vs AOI stays). Per-cell `failed_ids` logic removed (no per-cell runs).                                                                                                               | Internal.                                 |
| 5   | `s1.py` `S1Adapter`            | `_GRANULE_RE`: drop `_cell(?P<cell>\d+)`. `_cached_for`: drop the `g.cell_id == cell.cell_id` filter (any AOI-wide granule on `day` is read; `reproject_to_cell` already windows to the cell). `cache_root` now the per-granule cache. | Adapter cache contract.                   |
| 6   | `build_bow_valley_s1_cache.py` | Read raw archive (`raw_root/sentinel1`), pass AOI not grid cells. Output dir = the per-granule cache the adapter reads.                                                                                                                | CLI.                                      |
| 7   | Tests                          | `test_s1_adapter`, `test_s1_parity`, `test_s1_ensure_cache`, `test_exporter_parity`: per-granule fixtures (filenames without `_cell`, no cell filter).                                                                                 | Test fixtures.                            |

## Cache location

The processed per-granule tifs are an **intermediate** product (like clipped SAFEs).
Proposed: `clipped_root/sentinel1_snap/s1_grd_<granule>.tif` (where the adapter
already looks via `archive_root/sentinel1_snap`), but sourced from `raw_root/sentinel1`.
The exporter's `s1_cache_dir = archive_root / "sentinel1_snap"` is unchanged — only
*what* lands there changes (per-granule, not per-cell). **No exporter edit needed.**

## Parity guard

The `s1-adapter-snap-cache-and-angle` memory note proved per-cell parity on
`PR_20250519`. After the refactor, re-run that parity check: the AOI-wide tif
windowed to the same cell must reproduce the same `[VV, VH, angle]` block (the only
change is WHERE the geoRegion crop happens — post-TC vs pre-TC — and post-TC is the
correct order, so values should match or improve at cell edges). **Do not merge
until S1 parity is re-confirmed.**

## Clip-stage interaction

> **SUPERSEDED (2026-06-11).** The audit below concluded `clip_sentinel1` should stay
> because the viewer read its raw-DN output. The user then decided the viewer should
> read the **processed** S1 too — there is **no use for raw-unprocessed S1 anywhere**.
> So `clip_sentinel1` / `_clip_s1_measurement` were **removed**, `sentinel1` dropped from
> the clip `SOURCES`, and the viewer's S1 renderer + discovery switched to the
> `sentinel1_snap/` processed tifs (`viewer/manifest._discover_s1_products` +
> `_Sentinel1Renderer` reading the EPSG:32611 tif). The per-granule SNAP cache is now the
> single S1 product for both cube and viewer. The audit is kept for the rationale trail.

`clip_sentinel1` did **more** than the GCP slice the cube path superseded. Audited
consumers of its output (clipped S1 SAFE `.zip`, raw-DN range-geometry measurement
TIFFs + GCPs):

1. **Viewer `_Sentinel1Renderer`** (`viewer/renderers.py:449`) reads the clipped
   SAFE's `-vv-`/`-vh-` measurement TIFF directly, GCP-warps it, and displays raw
   backscatter as dB for visual QA. **This is a live consumer** — it needs the
   clipped SAFE to exist with raw-DN measurement TIFFs inside.
2. `clip_manifest.csv` S1 row (footprint / valid_pixel_count / action). Nothing
   asserts its presence (no directory-contract or audit check requires an S1 row),
   so no completeness break — but the row keeps S1 visible in the manifest.

**The cube `S1Adapter` reads the SNAP cache, NOT the clipped SAFE** — so the two paths
serve different consumers with different data domains:

| Path                            | Data domain                                      | Consumer                                                         |
| ------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------- |
| `clip_sentinel1` → clipped SAFE | **raw DN**, radar geometry, GCPs                 | viewer quicklook (visual QA) — *correct to show raw backscatter* |
| `s1_snap` (NEW) → AOI dB tif    | **calibrated σ⁰**, EPSG:32611, terrain-corrected | cube adapter — *correct to match GEE S1_GRD*                     |

Therefore:

- **`clip_sentinel1` / `_clip_s1_measurement` STAY, untouched.** They are NOT dead
  code — the viewer's raw-DN S1 quicklook genuinely depends on the clipped SAFE.
  (Earlier draft said "dead for cube, flag for removal"; the audit found the live
  viewer consumer — revised.)
- The `s1_snap` stage is **additive alongside** clip, reading the **raw** archive
  (`raw_root/sentinel1`). It touches nothing clip or the viewer depends on.
- **Structural parity holds.** Every other modality stays `clip → adapter`. S1 is
  the one source with a second, independent `raw → SNAP → adapter` path — required
  because S1's GEE value domain needs the full SNAP chain, which clip (a pure-raster,
  SNAP-free, parallel stage) must not host.

## Out of scope

- Mode B (1km tiling): same per-granule cache serves it (AOI-wide tif windows to
  any cell). No extra build.

## Validation order

1. Update graph (#1) → re-run the proof granule through `build_granule_cache` to
   confirm the wired path matches the manual proof.
2. Adapter (#5) + tests → `test_s1_adapter`, `test_s1_parity` green.
3. `ensure_s1_cache` (#4) + `test_s1_ensure_cache` green.
4. Full ruff + mypy + the S1 test subset.
5. Build all 32 granules (~1h40m). Re-export the diagnosed cube. Confirm VV/VH
   populate in the viewer + S1 parity on `PR_20250519`.

## STATUS — DONE (2026-06-11)

Implemented and verified. The seven components landed; `ensure_s1_cache` became
**verify-only** (building stays in the offline driver, keeping SNAP out of the export
path); the exporter's `auto_build_s1_cache` → `verify_s1_cache` (default `True`, a cheap
glob, fails loud on a missing cache). The clip CLIs were renamed `process_raw_*` and
gained `process-s1` / `process-all`; the audit gained an S1-cache coverage check.

**Idempotency + raw-safety** (per user follow-up): SNAP writes to a `.partial` inside the
cache dir and atomically renames on success (a crash leaves no false cache hit); the raw
archive is read-only (read-mode `ZipFile`, extract to system tmp, only the cache dir is
written); a changed graph rebuilds in place via `--overwrite`.

**Parity — all three reference patches now pass** (median |Δ|, tol 1.0 dB / 1.0°):

| Patch       | VV    | VH    | angle  | note                                                                                                                |
| ----------- | ----- | ----- | ------ | ------------------------------------------------------------------------------------------------------------------- |
| PR_20250519 | 0.401 | 0.421 | 0.240° | long-proven                                                                                                         |
| PR_20250423 | 0.427 | 0.468 | 0.343° | was "Empty region!" (pre-TC) — **fixed**                                                                            |
| PR_20250406 | 0.579 | 0.509 | 0.784° | was a ~10 dB "anomaly" — it was OUR old clipped+pre-TC processing, **not GEE's data**; raw + post-TC reproduces GEE |

Both prior xfails promoted to real passing assertions. Full local_sources suite: 218
passed, 2 skipped. ruff + mypy clean.

**Still to run (operator, heavy):** the full 32-granule cache build
(`process_raw_dataset.py process-s1`, ~1h40m) and the cube re-export to populate VV/VH in
the real cube + viewer. Not run here (hours of SNAP); the pipeline is proven on the parity
granules.

## FOLLOW-UP — S1 is now processed-only, never clipped (2026-06-11)

Per the user: there is **no use for raw-unprocessed S1 anywhere**, so the per-granule SNAP
cache became the *single* S1 product for both cube and viewer:

- `process-all` reordered to **process-then-clip**: `process-s1` first, then `clip-all` of
  every other modality (S1 is not a clip source).
- `clip_sentinel1` / `_clip_s1_measurement` **removed**; `sentinel1` dropped from clip
  `SOURCES`/`MODALITIES`; the now-dead `gcp_buffer_pixels` ClipSetting removed.
- The viewer reads processed S1: `viewer/manifest._discover_s1_products` synthesizes S1
  `ProductRow`s from `sentinel1_snap/s1_grd_*.tif`; `_Sentinel1Renderer` reads band 2 (VV
  linear σ⁰) from the EPSG:32611 tif → dB (no zip, no GCP warp). Dead `_read_gcp_band_4326`
  removed. 3 new viewer tests.
- 221 passed, 2 skipped; ruff + mypy clean on all touched files.
