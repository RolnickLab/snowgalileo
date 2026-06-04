# TASK-005: Run S1/S2 parity spikes (throwaway de-risk) — go/no-go decision point

## 1. Goal
Quantify value-domain drift for the two highest-interchange-risk sources (S1 GRD,
S2 L1C) against the Phase-0 GEE reference patches, using minimal throwaway
download+reproject scripts. This is a decision gate: if drift cannot be recovered by
processing, escalate **before** building the full adapter stack.

## 2. Context & References
- **FDD step:** §4.5 — "Run S1/S2 parity spikes (throwaway, de-risk)".
- **SPEC:** AC-14, AC-15 (drift quantified here, promoted to production in TASK-013/012);
  Verification Plan step 5.
- **PLAN:** §7 Phase 3 Step 0, §6 (S1/S2 highest interchange risk), §9.
- **FDD §3 Known Risks:** "Value-domain drift (S1, S2 highest)" — mitigated by running
  these spikes first as a go/no-go point.
- **Upstream tasks:** TASK-001 (reference patches in `tests/fixtures/gee_reference_patches/`),
  TASK-003 (`GridCell`, resampler).
- **Source semantics (DATA_ANALYSIS.md):**
  - **S1:** `COPERNICUS/S1_GRD` target; bands `[VV, VH, angle]`; IW; mask pixels
    `< -30.0`; expected domain ≈ `[-50, 1]` dB for VV/VH. Preprocessing (orbit, noise,
    calibration, terrain correction) materially shifts values.
  - **S2:** `COPERNICUS/S2_HARMONIZED` target; bands `[B2,B3,B4,B8,B11,B12]`; **all
    archive granules are L1C baseline N0511 (04.00+)** → subtract **1000 DN** to match
    the harmonized domain; ÷10000 downstream.
- **Relevant skills:** `geospatial` (SAR preprocessing, S2 baseline, reprojection),
  `tdd` (parity thresholds).

## 3. Subtasks
- [ ] 1. Write `test_s1_parity.py` and `test_s2_parity.py` (Red): per-band numeric diff
      between the spike output and the reference patch within a **documented tolerance**
      (record the chosen tolerance in the test docstring + audit).
- [ ] 2. Implement a minimal S1 GRD read→IW→calibrate→terrain-correct→reproject spike
      (throwaway script, not productionized); apply the `< -30.0` edge mask.
- [ ] 3. Implement a minimal S2 L1C read→baseline-check→−1000 DN→reproject spike.
- [ ] 4. Run both spikes over the same cells as the reference patches; compute and log
      per-band drift (`structlog`).
- [ ] 5. **Decision point:** record drift vs tolerance in the audit; if drift is
      unrecoverable, raise it to the user before TASK-006. Otherwise proceed.

## 4. Requirements & Constraints
- **Technical:** SNAP/`pyroSAR`-equivalent or `rasterio`+`sarsen`-style calibration for
  S1 (document which); `rasterio` for S2 JP2. Tolerances are explicit constants, logged.
- **Business:** These scripts are **throwaway** — they de-risk, they are not the
  production adapters. The real adapters are TASK-012 (Landsat is separate), TASK-013
  (S2), TASK-014 (S1). Drift numbers and the chosen tolerances feed those tasks.
- **Out of scope:** No coalesce/mosaic, no full exporter integration, no other sources.

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-14, spike form): S1 spike output bands `[VV, VH, angle]`, pixels
      `< -30.0` masked, domain ≈ `[-50, 1]`; per-band drift vs reference recorded against
      a stated tolerance.
- [ ] AC-2 (SPEC AC-15, spike form): S2 spike subtracts 1000 DN for N0511 granules;
      reflectance domain matches `S2_HARMONIZED`; per-band drift recorded.
- [ ] AC-3: drift-vs-tolerance verdict written to the Phase-0 audit (or a new
      `PARITY_SPIKE_NOTES.md`); go/no-go stated.
- [ ] AC-4: ruff + mypy clean on the spike scripts; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_s1_parity.py -v
uv run pytest tests/test_local_sources/test_s2_parity.py -v

# Run the spikes and emit drift report
uv run python scripts/spikes/s1_parity_spike.py --ref tests/fixtures/gee_reference_patches
uv run python scripts/spikes/s2_parity_spike.py --ref tests/fixtures/gee_reference_patches

uv run ruff check scripts/spikes/
uv run mypy scripts/spikes/
```
Expected: both parity tests green within stated tolerance (or an explicit, recorded
escalation if not); drift report written; ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify every AC in Section 5.
2. Run all Section 6 commands; confirm expected output.
3. Commit:
   ```bash
   git add scripts/spikes/ tests/test_local_sources/test_s1_parity.py \
           tests/test_local_sources/test_s2_parity.py
   git commit -m "spike(bow-valley): S1/S2 parity de-risk + drift report — closes TASK-005"
   ```
4. Check off subtasks/ACs; record the go/no-go verdict.
5. **Decision gate:** notify the user with drift numbers and the go/no-go before TASK-006.
