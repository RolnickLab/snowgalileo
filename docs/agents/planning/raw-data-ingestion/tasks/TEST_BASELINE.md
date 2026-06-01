# Pre-existing Test Baseline

The test suite is **already red on a clean checkout**, independent of any Bow Valley
work. During implementation of these tasks, act on **new** failures only — never treat
a baseline failure as a regression, and never let one block a task's approval.

## Baseline (captured at commit `021b4540`, branch `raw_data_prep`)

`uv run pytest -q --tb=no` → **6 failed, 32 passed, 17 subtests passed**.

Pre-existing failures (do NOT attempt to fix as part of Bow Valley tasks; do NOT count
as regressions):

```
tests/test_dataset.py::TestDataset::test_tif_to_array
tests/test_retrieve_cloud_state.py::TestRetrieveCloudState::test_end_to_end
tests/test_retrieve_season_from_filename.py::TestRetrieveSeasonFromFilename::test_map_int_to_cloud_states
tests/test_sklearn_preprocessing.py::TestSklearn::test_aggregation
tests/test_sklearn_preprocessing.py::TestSklearn::test_forward_filling
tests/test_sklearn_preprocessing.py::TestSklearn::test_median_replace
```

These touch the existing dataset loader, cloud-state mapping, and sklearn preprocessing —
none of which the Bow Valley pipeline modifies (downstream code is unchanged by design).
Some fail simply because expected fixture files are absent on this machine; others have
unknown causes. Either way they are out of scope.

## Validation rule for every task (overrides the per-task "no regressions" wording)

1. **Never use `pytest -x`** at the suite level — it stops on the first (pre-existing)
   failure before reaching new tests. Run the **targeted** new test files with `-x`
   (those should be clean), but run the **full suite without `-x`**.
2. A task's regression check is the **delta against this baseline**, not "zero failures".
   Use the helper below.
3. If a task legitimately changes the baseline set (it should not — downstream code is
   untouched), update this file in the same PR and explain why.

### Delta check (copy-paste)

```bash
cd /home/dev/projects/presto-v3
# Run full suite, list current failures, diff against the baseline.
uv run pytest -q -p no:cacheprovider --tb=no 2>/dev/null \
  | grep -E '^FAILED' | sed 's/^FAILED //; s/ -.*$//' | sort > /tmp/current_failures.txt

cat > /tmp/baseline_failures.txt <<'EOF'
tests/test_dataset.py::TestDataset::test_tif_to_array
tests/test_retrieve_cloud_state.py::TestRetrieveCloudState::test_end_to_end
tests/test_retrieve_season_from_filename.py::TestRetrieveSeasonFromFilename::test_map_int_to_cloud_states
tests/test_sklearn_preprocessing.py::TestSklearn::test_aggregation
tests/test_sklearn_preprocessing.py::TestSklearn::test_forward_filling
tests/test_sklearn_preprocessing.py::TestSklearn::test_median_replace
EOF
sort -o /tmp/baseline_failures.txt /tmp/baseline_failures.txt

echo "=== NEW failures introduced by this task (must be empty to pass) ==="
comm -23 /tmp/current_failures.txt /tmp/baseline_failures.txt

echo "=== Baseline failures that newly PASS (informational, fine) ==="
comm -13 /tmp/current_failures.txt /tmp/baseline_failures.txt
```

**Pass condition:** the "NEW failures" list is empty. The task's own new test files
pass on their own (`pytest tests/test_local_sources/<file> -v` green).
