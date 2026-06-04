# TASK-016: Add entry-point scripts, the directory-contract test, and the full-stack parity gate

## 1. Goal
Ship the two operator entry points (`export_bow_valley_cube.py`,
`infer_bow_valley_daily_fsc.py`) wired to `cube.yaml`/`inference.yaml`, the
directory-contract test that enforces write boundaries, and the full-stack numeric
parity gate against the Phase-0 GEE reference patches.

## 2. Context & References
- **FDD step:** §4.8 — "Add entry-point scripts" + directory-contract test (AC-32).
- **SPEC:** FR-20b, AC-27, AC-32; Verification Plan steps 8 (full-stack parity) & 7
  (directory contract); Storage NFR.
- **PLAN:** §4 module layout (scripts + configs), §3 Directory layout (write-boundary
  contract), §7 Phase 3 Step 5.
- **Upstream tasks:** TASK-001 (reference patches), TASK-004 (exporter), TASK-006…014
  (adapters), TASK-015 (driver + mosaic).
- **Config wiring (PLAN §4):**
  - `cube.yaml`: `archive_root` (`…/clipped_bow_valley_selection_raw`),
    `processing_root` (`…/bow_valley_processing`), AOI bbox, date range, CRS, mode A/B,
    cache cap.
  - `inference.yaml`: checkpoint path, batch size, output dir (default
    `processing_root/daily_fsc`), days.
- **Directory contract (AC-32):** after a cube+inference run, all new files live under
  `data/bow_valley_processing/` in the correct subdirs (cubes→`cubes/`, COGs→`daily_fsc/`,
  npz→`cube_cache/`); **no** file created/modified under
  `data/clipped_bow_valley_selection_raw` or `data/bow_valley_selection_raw`; deleting
  `cube_cache/` + `scratch/` does not remove any file in `cubes/` or `daily_fsc/`.
- **Full-stack parity (AC-27):** `test_exporter_parity.py` — per-source diff between the
  direct-source cube and the Phase-0 GEE reference patches within each source's
  documented tolerance.
- **Relevant skills:** `software-dev` (Typer, pydantic-settings, config), `geospatial`,
  `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_directory_contract.py` (Red, AC-32): run a small cube+inference,
      snapshot the two archive roots before/after, assert zero writes there; assert all
      outputs land in the correct `processing_root` subdir; assert deleting
      `cube_cache/`+`scratch/` leaves `cubes/`+`daily_fsc/` intact.
- [ ] 2. Write `test_exporter_parity.py` (Red, AC-27): full-stack per-source numeric diff
      vs `tests/fixtures/gee_reference_patches/` within documented tolerances.
- [ ] 3. Implement `scripts/export_bow_valley_cube.py` (Typer, reads `cube.yaml`) and
      `scripts/infer_bow_valley_daily_fsc.py` (Typer, reads `inference.yaml`).
- [ ] 4. Finalize `configs/bow_valley/cube.yaml` and `inference.yaml`.
- [ ] 5. Green + Refactor.
- [ ] 6. Add the `KNOWLEDGE.md` entries flagged in PLAN §6 (MODIS `-28672` sentinel;
      ERA5 temp-sign preserved; S3 identity-norm intentional; `PR` prefix; per-cell
      no cross-cell context).

## 4. Requirements & Constraints
- **Technical:** Typer CLIs; pydantic-settings config loading; no hardcoded paths/secrets
  (archive paths via config). `structlog` JSON logging.
- **Business:** Write boundaries (AC-32) are non-negotiable — Stage 2 never writes into
  either archive. Parity tolerances are the ones documented per source (TASK-005 +
  per-adapter tasks). Checkpoint path (Q6) is required to actually run inference — flag
  if unset.
- **Out of scope:** Mode-B production sizing (Q3/Q7 — compute budget), object-storage
  output (Q5 — config switch only). Model retraining.

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-32): no file created/modified under either archive by the
      cube/inference run; outputs in correct subdirs; deleting `cube_cache/`+`scratch/`
      leaves `cubes/`+`daily_fsc/` intact.
- [ ] AC-2 (SPEC AC-27): full-stack per-source parity within documented tolerance for
      every source.
- [ ] AC-3: both entry-point scripts run end-to-end from config on a small cell subset.
- [ ] AC-4: `KNOWLEDGE.md` contains the five flagged entries.
- [ ] AC-5: ruff + mypy clean; targeted new tests green; full suite introduces NO new
      failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`); `uv run pre-commit
      run --all-files` passes for the new/changed files (pre-existing baseline failures
      excepted).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_directory_contract.py -v
uv run pytest tests/test_local_sources/test_exporter_parity.py -v

# End-to-end smoke from config (small cell subset)
uv run python scripts/export_bow_valley_cube.py --config configs/bow_valley/cube.yaml --limit 4
uv run python scripts/infer_bow_valley_daily_fsc.py --config configs/bow_valley/inference.yaml --limit 4

uv run ruff check scripts/export_bow_valley_cube.py scripts/infer_bow_valley_daily_fsc.py
uv run mypy scripts/export_bow_valley_cube.py scripts/infer_bow_valley_daily_fsc.py
uv run pre-commit run --all-files
```
Expected: directory-contract + parity tests green; both scripts complete on the subset;
ruff/mypy/pre-commit exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify ACs. 2. Run Section 6 commands.
3. Commit:
   ```bash
   git add scripts/export_bow_valley_cube.py scripts/infer_bow_valley_daily_fsc.py \
           configs/bow_valley/cube.yaml configs/bow_valley/inference.yaml \
           tests/test_local_sources/test_directory_contract.py \
           tests/test_local_sources/test_exporter_parity.py docs/agents/KNOWLEDGE.md
   git commit -m "feat(bow-valley): entry-point scripts + directory contract + full-stack parity gate — closes TASK-016"
   ```
4. Check off subtasks/ACs; note any unmet parity tolerances or the unset checkpoint (Q6).
5. Notify the user — this closes Phase 3. Surface remaining config-gated questions (Q3
   mode, Q6 checkpoint, Q7 budget, Q5 output destination) before a production run.
