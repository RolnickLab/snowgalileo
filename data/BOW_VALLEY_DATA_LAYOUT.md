# Bow Valley local-pipeline data layout

> **Reference** — the path contract for the direct-source (non-GEE) Bow Valley
> pipeline. For the legacy pre-training / FSC-eval GEE export flow, see
> [`README.md`](README.md). For *how* the stages run, see
> [`../docs/local_data_processing.md`](../docs/local_data_processing.md).

The local pipeline moves data through three roots, in order:

```
raw archive  ──clip──▶  clipped archive  ──assemble──▶  processing tree
(read-only)             (clip output)                   (cache + 8-day cubes + daily FSC)
```

Where these roots physically live is **configurable per machine** (see §3) — by
default they resolve to the repo's own `data/` folder, and each can be redirected
to any drive without editing code. None of their *contents* are tracked in git
(`data/*` is gitignored); only this document and `README.md` ship in the repo.

---

## 1. The roots

| Path (under `data/`)               | Role                          | Stage owner            | Access  |
|------------------------------------|-------------------------------|------------------------|---------|
| `bow_valley_inference_aoi.geojson` | Authoritative AOI boundary    | all (clip + inference) | read    |
| `bow_valley_selection_raw/`        | Raw per-modality archive      | input (placed by hand) | read    |
| `clipped_bow_valley_selection_raw/`| AOI-clipped archive           | clip stage (TASK-002)  | r/w     |
| `bow_valley_processing/`           | Cube-assembly tree (cache, cubes, daily FSC) | assembly (Stage 2) | r/w |

### `bow_valley_inference_aoi.geojson`
Single-Polygon GeoJSON, **EPSG:4326**. The one source of AOI truth — the clip
gate, the grid generator, and the viewer all read it; nothing hardcodes bounds.
Loaded by `src.data.local_sources.clip.settings.load_aoi_polygon`.

### `bow_valley_selection_raw/` — raw input
The untouched per-modality download archive, **placed here manually**, never
written by pipeline code. Per-modality subdirs (`dem/`, `worldcover/`, `era5/`,
`landsat8/`, `landsat9/`, `modis/`, `viirs/`, `sentinel1/`, `sentinel2/`,
`sentinel3/`). Layout and per-sensor formats are detailed in
[`../docs/local_data_processing.md` §1](../docs/local_data_processing.md).
Read by: `clip_dataset.py`, `create_stac_catalog.py`.

### `clipped_bow_valley_selection_raw/` — clip output
Output of the clip stage: every raw product cropped to the AOI, same
per-modality subdir layout, plus a per-source `clip_manifest.csv` (and a combined
manifest at the root). This is the **single archive root every downstream adapter
reads** — adapters never reach back into the raw archive. Written by
`clip_dataset.py`; read by `clip_audit.py` and the viewer
(`src/data/local_sources/viewer/`).

### `bow_valley_processing/` — assembly tree
Stage-2 working tree for cube assembly. Holds the per-(cell, day) `.npz`
intermediate cache (`cube_cache/{cell_id}/{day}_{modality}.npz`), the assembled
8-day cubes (`cubes/`), daily FSC COGs (`daily_fsc/`), and per-process
subdirectories. See
[`../docs/agents/planning/raw-data-ingestion/SPEC_BOW_VALLEY_DATA.md`](../docs/agents/planning/raw-data-ingestion/SPEC_BOW_VALLEY_DATA.md)
§"Storage".

The assembled cubes under `cubes/` are the durable end-product — there is **no
separate "final cube archive" root**. The `cube_cache/` and `scratch/` subtrees
are intermediate and safe to wipe and regenerate; `cubes/` and `daily_fsc/` are
the outputs to keep. (To place `cubes/` on different storage from the rest of the
tree, symlink `bow_valley_processing/cubes` at the target — the code resolves it
as `processing_root/"cubes"`.)

---

## 2. Stage flow

```
bow_valley_selection_raw/        (raw, manual)
        │  clip_dataset.py  (AOI intersect gate + per-modality clip)
        ▼
clipped_bow_valley_selection_raw/  + clip_manifest.csv
        │  clip_audit.py   (zero-signal QA)   ── viewer (visual QA)
        │  Stage 2 assembly (adapters → cube_cache → cubes)
        ▼
bow_valley_processing/           (cube_cache/, cubes/ ← 8-day cubes, daily_fsc/)
```

---

## 3. Where the roots physically live (portability)

The four roots resolve from `src.data.local_sources.paths.LocalPaths` (env prefix
`LOCAL_`, see §4). Defaults are **repo-relative** (`data/<name>`), so the layout
is portable: nothing machine-specific is baked into the repo or the code. Pick
whichever of the three tiers fits the machine — they can be mixed per root.

### Tier 0 — zero config (default)
Set nothing. Every root resolves under the repo's own `data/` folder, and the
pipeline creates the directories on first write. Best for a fresh clone, a demo,
or a small run that fits on the repo's disk. Fully portable: any clone works
identically with no setup.

### Tier 1 — `.env` redirect (recommended for real deployments)
When the data lives on dedicated storage, point each `LOCAL_*` variable at an
**absolute path** in a repo-root `.env` (gitignored — copy `.env.example`):

```dotenv
# .env  — per machine, never committed
LOCAL_RAW_ROOT=/mnt/archive/bow_valley/selection_raw
LOCAL_CLIPPED_ROOT=/mnt/archive/bow_valley/clipped_selection_raw
LOCAL_PROCESSING_ROOT=/mnt/fast_scratch/bow_valley/processing
LOCAL_AOI_PATH=/mnt/archive/bow_valley/inference_aoi.geojson
```

Each root is independent and may sit on a **different filesystem** (e.g. raw on
archive, processing on fast scratch). A collaborator on another machine clones
the repo, copies `.env.example` → `.env`, edits these four lines — no symlinks,
no code edits. Because `processing_root` etc. *are* the real absolute paths, all
nested subdirectories (`cube_cache/{cell_id}/…`, `cubes/`, `daily_fsc/`) are
created directly on the designated drive.

### Tier 2 — symlinks (single-machine convenience)
Leave the defaults and make `data/<name>` a **symlink** to the real storage:

```bash
ln -s /mnt/fast_scratch/bow_valley/processing data/bow_valley_processing
```

Then `data/bow_valley_processing` transparently lands on the target drive with no
`.env`. This works for a directory you fully own, but it is per-machine (the
symlink is not in git) and is the **least portable** option — prefer Tier 1 when
sharing across machines or users. (Note: a symlinked *directory* redirects all
files and subdirs created inside it; symlinking only a *parent* will not send new
sibling directories elsewhere — use a per-root `.env` path for that.)

> **AOI per region.** Whichever tier you use, `LOCAL_AOI_PATH` must point at the
> region's AOI polygon (EPSG:4326). For a different region, supply a different
> AOI file and a fresh set of roots.

---

## 4. Configuration surface

| Setting              | Default                                | Override                                          |
|----------------------|----------------------------------------|---------------------------------------------------|
| raw root             | `data/bow_valley_selection_raw`        | `LOCAL_RAW_ROOT` env, or `--input-dir` (clip CLI) |
| clipped root         | `data/clipped_bow_valley_selection_raw`| `LOCAL_CLIPPED_ROOT` env, or `--output-dir`; viewer: `VIEWER_CLIPPED_ROOT` |
| processing root      | `data/bow_valley_processing`           | `LOCAL_PROCESSING_ROOT` env                       |
| AOI path             | `data/bow_valley_inference_aoi.geojson`| `LOCAL_AOI_PATH` env, or `--aoi` (clip CLI); viewer: `VIEWER_AOI_PATH` |

Defaults are centralized in `src.data.local_sources.paths.LocalPaths`
(pydantic-settings, env prefix `LOCAL_`) and are **repo-relative** — see §3 for
the three ways to point them at real storage (zero-config, `.env`, or symlink). A
repo-root `.env` is loaded automatically (see `.env.example`). The clip CLIs
additionally accept explicit `--input-dir` / `--output-dir` / `--aoi` flags that
win over the settings defaults for one-off runs.

Precedence (highest first): CLI flag → `LOCAL_*` env / `.env` → repo-relative
default. The viewer also honours its own `VIEWER_*` prefix, which wins over
`LOCAL_*` for the viewer only.
