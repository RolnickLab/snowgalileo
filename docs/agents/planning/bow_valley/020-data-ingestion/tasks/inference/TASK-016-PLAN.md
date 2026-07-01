# TASK-016 — Entry-point scripts + directory-contract test + full-stack parity gate (PLAN)

> Implementation plan for `TASK-016-entrypoints-directory-contract-parity-gate.md`.
> This is the **last** task of the Bow Valley direct-source pipeline; landing it closes
> Phase 3. Sourced from FDD §4.8, SPEC FR-20b/AC-27/AC-32, PLAN §4/§7.

## DOWNSTREAM IS SACRED (binding constraint, restated)

We are **additive**. The two new scripts wire together components that already exist
(`LocalSourceExporter`, `InferenceGridDriver`, `DailyMosaicWriter`, `build_grid`,
`CubeSettings`) and build a model via the **existing, unchanged** load path
(`Encoder(**config["model"]["encoder"])` → `EncoderWithHead(...)` → `load_state_dict`,
exactly as `scripts/eval_only.py` / `predict_and_generate_output.py` do). No edit to
`src/fsc/*`, `src/snowgalileo/*`, `src/data/earthengine/*`, or any adapter. The GEE
`predict_and_store_output` runner keeps working in parallel.

## Resolved open questions

- **Q6 (checkpoint path) — RESOLVED.** Real finetuned `EncoderWithHead` weights are on
  disk: `logging_checkpoints/snowgalileo_finetune/*.pth` (~23 MB each, full encoder+head
  state, strict-loadable). Encoder size = `ai4snow_tiny` (192/12/3,
  `max_sequence_length=24`) — build with `Encoder(**load_check_config("ai4snow_tiny.json")["model"]["encoder"])`.
  See memory `snowgalileo-checkpoints-available` + `scripts/eval_only.py:83-92`. The
  `inference.yaml` `checkpoint` field defaults to a sensible finetuned `.pth` but is
  overridable; if the file is **absent** the script fails loudly (no silent random init).
- **Q3 (mode A/B), Q5 (output destination), Q7 (budget):** config only; default mode A,
  default output `processing_root/daily_fsc/`. Surfaced to the user at completion, not
  blocking.

## Files

### 1. `src/data/local_sources/settings.py` — add `InferenceSettings` (MODIFY, additive)

A second `BaseSettings` sibling to `CubeSettings`, same `from_yaml` / env-precedence
pattern (`env_prefix="INFER_"`), loaded from `inference.yaml`. Fields:

- `checkpoint: Path` — finetuned `EncoderWithHead` `.pth` (Q6). **Required at run time**
  (validated to exist in the script, not at import — keeps tests/import cheap).
- `eval_config_name: str = "fsc_inference_bow_river_tiny.json"` — the eval JSON (drives
  encoder size token + `sigmoid_slope` + head `eval_config`), read from `configs/eval/`.
- `decoder_mode: str = "finetune"`.
- `batch_size: int = 8`, `device: str = "cpu"`.
- `out_dir: Path | None = None` — defaults to `CubeSettings().daily_fsc_dir` in the script
  (Q5 override point). Inference window + mode are **read from `cube.yaml`** (one source of
  truth for the sweep), so `InferenceSettings` does not duplicate them.

Rationale: keep `cube.yaml` the authority for *what* the sweep is (window, mode, roots);
`inference.yaml` only carries *how to run the model* (checkpoint, batch, device, output).

### 2. `scripts/developer_scripts/bow_valley_inference_local/export_bow_valley_cube.py` (NEW, Typer)

`--config configs/bow_valley/cube.yaml`, `--limit N` (cap cells for a smoke run),
`--window-end YYYY-MM-DD` (optional; default = `cube.yaml` `window_end`). Builds the grid
(`build_grid(mode=settings.mode)`), a real-adapter `LocalSourceExporter(placeholder=False, archive_root=settings.archive_root, out_dir=settings.cubes_dir)`, and exports one cube per
`(cell, window_end)` for the (optionally limited) grid. `structlog` JSON. This is the cube
half (AC-3, first script).

### 3. `scripts/developer_scripts/bow_valley_inference_local/infer_bow_valley_daily_fsc.py` (NEW, Typer)

`--cube-config cube.yaml`, `--config inference.yaml`, `--limit N`. Loads both settings,
builds the model from the checkpoint via the **existing** path, builds the grid, constructs
`InferenceGridDriver(exporter=real_exporter, model=model, grid=grid[:limit], window_start=cube.window_start, window_end=cube.window_end, out_dir=infer.out_dir or cube.daily_fsc_dir, device=infer.device, batch_size=infer.batch_size)`, calls `.run()`. Fails loudly if `checkpoint` missing
(AC: flag unset Q6). This is the inference half (AC-3, second script).

Helper `_build_model(infer: InferenceSettings) -> EncoderWithHead` lifted verbatim from
`eval_only.py` (load_check_config → Encoder → EncoderWithHead → load_state_dict) — no new
model logic.

### 4. `configs/bow_valley/inference.yaml` (NEW)

`checkpoint`, `eval_config_name`, `decoder_mode`, `batch_size`, `device`, `out_dir` (commented
default). Documents Q6/Q5.

### 5. `tests/test_local_sources/test_directory_contract.py` (NEW, Red → AC-32)

Run a **tiny** cube+inference (1–2 cells, 1 day, `placeholder=True` exporter + tiny untrained
`EncoderWithHead`, all under `tmp_path` `processing_root`):

- snapshot `{path: mtime+size}` of both archive roots before/after → assert **identical**
  (zero writes/mods under `clipped_*` and `*_selection_raw`).
- assert cube tif landed in `cubes/`, COG in `daily_fsc/`, `.npz` (if any) in `cube_cache/`.
- create dummy files in `cube_cache/` + `scratch/`, delete those two dirs, assert
  `cubes/` + `daily_fsc/` files still present (intermediate/deliverable separation).
  The exporter/driver are pointed at a `tmp_path` processing root via injected `out_dir`s, so
  the test never touches the real archives. **No SNAP/real-archive** — placeholder cube only
  (AC-32 is about *write boundaries*, not parity).

### 6. `tests/test_local_sources/test_exporter_parity.py` (NEW, Red → AC-27)

Full-stack per-source numeric parity: real-adapter `LocalSourceExporter` over the cells
backing the 6 `tests/fixtures/gee_reference_patches/PR_*.tif`, diff each source's bands vs the
reference within that source's **already-documented** tolerance (S2 ≤50 DN, S1 ≤1 dB,
Landsat/MODIS/etc. per their adapter tasks). Marked `@pytest.mark.slow` +
`@pytest.mark.xdist_group("slow_archive")` (real-archive GDAL decode — same serialization
rule as the other parity tests, KNOWLEDGE.md). **Coordinate reconciliation (AC-27):** pair
cube↔reference by the **shared cube-CSV row** the patch came from (the patch filename encodes
the UTM center via the reference; resolve the matching `GridCell` by center, not by filename
string). Skips cleanly if a source's clipped archive / SNAP cache is absent (S1 needs the SNAP
cache; document the skip like `test_s1_parity`). This is the gate, not a per-source unit test —
it reuses each adapter's documented tolerance constant, never invents new ones.

### 7. `docs/agents/KNOWLEDGE.md` — add the **five** flagged entries (AC-4)

MODIS `-28672` sentinel load-bearing; ERA5 temp-sign preserved (out of scope to fix); S3
identity-norm intentional; `PR` filename prefix supported/unused; per-cell inference has **no**
cross-cell context (each cell is an independent forward). Several already exist as memories —
mirror them as KNOWLEDGE bullets so AC-4 is literally satisfied in that file.

### 8. `pyproject.toml` — pin `typer` into the `dev` group (MODIFY)

`typer` is currently only transitive (0.26.3 resolves). The two scripts import it directly →
declare it explicitly (`typer>=0.15`) so the dep is not accidental. Register nothing else.

## Test / validation strategy

- TDD: write `test_directory_contract.py` + `test_exporter_parity.py` first (Red), then the
  scripts/config/settings (Green).
- `uv run pytest tests/test_local_sources/test_directory_contract.py -v` (fast, always runs).
- `uv run pytest tests/test_local_sources/test_exporter_parity.py -v` (slow; skips without
  archive — document which sources are exercised on this box).
- Smoke: both scripts `--limit 2` end-to-end from config (AC-3).
- `ruff` + `mypy` on the two scripts + settings.
- **Full-suite delta** per `TEST_BASELINE.md` (NEW-failures list MUST be empty; NOT
  `pytest -x`). Slow parity tests serialized on the `slow_archive` xdist group.

## Approval gates (CLAUDE.md — incremental)

Suggested order, **stop + summarize after each, commit only on explicit approval**:

1. `InferenceSettings` + `inference.yaml` + `typer` pin.
2. `test_directory_contract.py` (Red) → `export_bow_valley_cube.py` +
   `infer_bow_valley_daily_fsc.py` (Green for the contract test).
3. `test_exporter_parity.py` (Red) → confirm green on available sources / documented skips.
4. KNOWLEDGE.md entries + check off subtasks/ACs.
5. Final delta run + commit (closes Phase 3).

## Risks / flags

- **Parity gate is archive-dependent.** On a box missing a clipped source (esp. S1's SNAP
  cache), those sub-checks **skip**, not fail — AC-27 is "within tolerance *where exercised*".
  I will report exactly which sources ran vs skipped, not claim blanket parity.
- **Real adapters are slow.** The cube smoke run (`--limit 2`) and parity test do real GDAL
  decodes; both are bounded (≤6 cells) and the parity test is `slow`-gated.
- **Checkpoint absence** must fail loudly in the inference script (no silent random init,
  unlike the GEE script's `else` branch) — an all-random sweep would produce a plausible-but-
  meaningless COG.

```
```
